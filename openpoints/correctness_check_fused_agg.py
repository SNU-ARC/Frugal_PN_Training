import torch
from cpp import pointnet2_cuda
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple


class GroupingOperation(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample, device=features.device)

        pointnet2_cuda.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


# Fused aggregation with positional encoding
class FusedGroupAndReduceOperationPE(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :pe: (B, C, npoint, nsample) tensor of positional encoding
        :return:
            output: (B, C, npoint) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert pe.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures)
        source_idx = torch.cuda.IntTensor(B, C, nfeatures)
        source_idx_pe = torch.cuda.IntTensor(B, C, nfeatures)

        # For now, nsample must be power of 2. Other cases will be supported in the future.
        assert nsample & (nsample-1) == 0, "nsample must be power of 2"

        pointnet2_cuda.fused_group_and_reduce_pe_wrapper(B, C, N, nfeatures, nsample, features, idx, pe, output, source_idx, source_idx_pe)

        ctx.for_backwards = (source_idx, source_idx_pe, N, nsample)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
            grad_pe: (B, C, npoint, nsample) gradient of the positional encoding
        """
        source_idx, source_idx_pe, N, nsample = ctx.for_backwards

        B, C, npoint = grad_out.size()
        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_pe = torch.zeros([B, C, npoint, nsample], dtype=torch.float, device=grad_out.device, requires_grad=True)

        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.fused_group_and_reduce_pe_grad_wrapper(B, C, N, npoint, grad_out_data, source_idx, source_idx_pe, grad_features.data, grad_pe.data)
        return grad_features, None, grad_pe


fused_group_and_reduce_pe = FusedGroupAndReduceOperationPE.apply


if __name__ == "__main__":
    import time

    B, C, N = 8, 128, 6000
    npoint = 1500
    nsample = 32
    device = 'cuda:0'
    features1 = torch.rand(B, C, N, requires_grad=True).float().to(device)
    features1.retain_grad()
    pe1 = torch.rand(B, C, npoint, nsample, requires_grad=True).float().to(device)
    pe1.retain_grad()
    grad1 = torch.rand(B, C, npoint).float().to(device)
    features2 = features1.clone().detach().requires_grad_(True)
    pe2 = pe1.clone().detach().requires_grad_(True)
    grad2 = grad1.clone().detach()

    nns_idx = torch.randint(0, 6000, (B, npoint, nsample)).int().cuda()

    # Fused aggregation
    pooled_features1 = fused_group_and_reduce_pe(features1, nns_idx, pe1)
    pooled_features1.backward(gradient=grad1, retain_graph=True)
    grad_out1 = features1.grad
    grad_pe_out1 = pe1.grad

    # Original version
    grouped_features = grouping_operation(features2, nns_idx)
    pooled_features2 = torch.max(grouped_features + pe2, dim=-1, keepdim=False)[0]
    pooled_features2.backward(gradient=grad2, retain_graph=True)
    grad_out2 = features2.grad
    grad_pe_out2 = pe2.grad
    
    print(grad_out1 == grad_out2)
    print(grad_pe_out1 == grad_pe_out2)

