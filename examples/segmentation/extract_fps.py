"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
Author: Guocheng Qian @ 2022, guocheng.qian@kaust.edu.sa
"""
import __init__
import argparse, yaml, os, logging, numpy as np, csv, wandb, glob
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.dataset.data_util import voxelize
from openpoints.dataset.semantic_kitti.semantickitti import load_label_kitti, load_pc_kitti, remap_lut_read, remap_lut_write, get_semantickitti_file_list
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample
from openpoints.models.layers import ball_query
from openpoints.models.layers import offline_fps
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def main(gpu, cfg):
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    stride_list = cfg.model.encoder_args.strides
    stride = ''.join(str(e) for e in list(filter(lambda x: x != 1, stride_list)))
    if 's3dis' in cfg.dataset.common.NAME.lower():
        save_path = "data/S3DIS/fps_results_"+stride+"_epoch"+str(cfg.epochs)+".dat"
    elif 'scannet' in cfg.dataset.common.NAME.lower():
        save_path = "data/ScanNet/fps_results_"+stride+"_epoch"+str(cfg.epochs)+".dat"

    if os.path.exists(save_path):
        print("Precomputed fps result file already exists!")
        exit(0)

    # Batch size and loop must be 1 for fps extraction
    cfg.dataset.train.loop = 1
    train_loader = build_dataloader_from_cfg(1,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             precompute_fps=2,
                                             stride_list=cfg.model.encoder_args.strides,
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
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        xyz = data['pos']
        for i, stride in enumerate(strides):
            if stride != 1:
                num_sample = max(xyz.shape[1] // stride, min(cfg.dataset.train.voxel_max // stride, xyz.shape[1]))
                break   # only generate fps index for the first layer

        # First perform farthest point sampling once to find out the threshold.
        fps_idx = furthest_point_sample(xyz, num_sample).squeeze(0).long()
        threshold = float((xyz.squeeze(0)[fps_idx] - xyz.squeeze(0)[fps_idx[-1]]).square().sum(-1).sqrt().sort()[0][1]) 

        # Build filter matrix using ball query.
        nfilter = 16    # Sufficient amount of filters that can cover the threshold.
        #print(num_sample)
        filter_matrix = ball_query(threshold, nfilter, xyz, xyz).squeeze(0)

        # Generate "number of epoch" versions of fps indices in a flash speed!
        fps_indices = offline_fps(filter_matrix, num_sample, cfg.epochs)
        fps_list[idx] = fps_indices.cpu().clone().detach()
    return fps_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    
    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']

    # wandb config
    main(0, cfg)
