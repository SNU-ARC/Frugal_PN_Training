import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample
from openpoints.models.layers import ball_query
from openpoints.models.layers import offline_fps


def main(gpu, cfg, profile=False):
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    # Currently, only modelnet40 is supported.
    stride_list = cfg.model.encoder_args.strides
    stride = ''.join(str(e) for e in list(filter(lambda x: x != 1, stride_list)))

    save_path = "data/ModelNet40Ply2048/fps_results_"+stride+"_epoch"+str(cfg.epochs)+".dat"

    if os.path.exists(save_path):
        print("Precomputed fps result file already exists!")
        exit(0)

    # Batch size must be 1 for fps extration.
    # Training datset must not be shuffled when extracting fps results.
    # Caution!! Batch size must be greater than 1 in real training. I manually set shuffle=False when batch_size==1.
    train_loader = build_dataloader_from_cfg(1,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             precompute_fps=2,
                                             stride_list = cfg.model.encoder_args.strides,
                                             )
    fps_list = run_one_epoch(train_loader, cfg)

    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(fps_list, f)

def run_one_epoch(train_loader, cfg):
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), ascii=True)

    strides = cfg.model.encoder_args.strides
    fps_list = [None] * len(train_loader)
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        points = data['x']
        data['pos'] = points[:, :, :3].contiguous()
        xyz = data['pos']

        for i, stride in enumerate(strides):
            if stride != 1:
                num_sample = xyz.shape[1]//stride;
                fps_idx = furthest_point_sample(xyz, num_sample).squeeze(0).long()
                threshold = float((xyz.squeeze(0)[fps_idx] - xyz.squeeze(0)[fps_idx[-1]]).square().sum(-1).sqrt().sort()[0][1]) 
                break

        # Build filter matrix using ball query.
        nfilter = 16    # Sufficient amount of filters that can cover the threshold.
        filter_matrix = ball_query(threshold, nfilter, xyz, xyz).squeeze(0)

        # Generate "number of epoch" versions of fps indices in a flash speed!
        fps_indices = offline_fps(filter_matrix, num_sample, cfg.epochs)
        fps_list[idx] = fps_indices.cpu().clone().detach()
    return fps_list

