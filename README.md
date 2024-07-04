# Frugal_PN_Training
This is an implementation of our paper "Frugal 3D Point Cloud Model Training via Progressive Near Point Filtering and Fused Aggregation".

## Install
```
source install.sh
```
Note:  

   We recommend using CUDA 11.x; check your CUDA version by: `nvcc --version` before using the bash file;

## Usage
### L-FPS
Can be easily applied to other models and datasets by changing few lines of dataloader. Please refer to ```openpoints/dataset/s3dis/s3dis.py```, ```openpoints/dataset/scannetv2/scannet.py```, and ```openpoints/dataset/modelnet/modelnet40_ply_2048_loader.py```.
We also register ```offlinefps``` operation in OpenPoints library. 
Implementation can be found in ```subsample.py``` and ```openpoints/cpp/pointnet2_batch/src/sampling_gpu.cu```.

### Fused Aggregation
We register ```fused_group_and_reduce``` and ```fused_group_and_reduce_pe``` operation in OpenPoints library. Implementation can be found in ```group.py``` and ```openpoints/cpp/pointnet2_batch/src```. They can be used just like other operations in OpenPoints.
```
# Without positional encoding
idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz)
features = fused_group_and_reduce(features, idx)

# With positional encoding
idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz)
features = fused_group_and_reduce_pe(features, idx, grouped_pos_enc)
```

## Prepare Dataset

### ModelNet40
ModelNet40 dataset is downloaded automatically.

### S3DIS
```
mkdir -p data/S3DIS/
cd data/S3DIS
gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y
tar -xvf s3disfull.tar
```

### ScanNet
```
cd data
gdown https://drive.google.com/uc?id=1uWlRPLXocqVbJxPvA2vcdQINaZzXf1z_
tar -xvf ScanNet.tar
```

## Extract L-FPS results
### ModelNet40
```
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnet++da-lfps.yaml mode=extract_fps wandb.use_wandb=False
```

### S3DIS
```
CUDA_VISIBLE_DEVICES=0 bash script/extract_fps_segmentation.sh cfgs/s3dis/pointmetabase-l-lfps.yaml wandb.use_wandb=False
```

### ScanNet
```
CUDA_VISIBLE_DEVICES=0 bash script/extract_fps_segmentation.sh cfgs/scannet/pointmetabase-l-lfps.yaml wandb.use_wandb=False
```

## Train
### ModelNet40 (PointNet++)
```
# Run baseline
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnet++da.yaml wandb.use_wandb=False

# Run with L-FPS
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnet++da-lfps.yaml wandb.use_wandb=False

# Run with fused aggregation
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnet++da-fused.yaml wandb.use_wandb=False

# Run with all optimizations
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnet++da-all.yaml wandb.use_wandb=False
```

### S3DIS, ScanNet (PointNet++, PointMetaBase)
```
# Run baseline
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/[s3dis, scannet]/[pointnet++da, pointmetabase-l, pointmetabase-xl].yaml wandb.use_wandb=False

# Run with L-FPS
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/[s3dis, scannet]/[pointnet++da, pointmetabase-l, pointmetabase-xl]-lfps.yaml wandb.use_wandb=False

# Run with fused aggregation
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/[s3dis, scannet]/[pointnet++da, pointmetabase-l, pointmetabase-xl]-fused.yaml wandb.use_wandb=False

# Run with all optimizations
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/[s3dis, scannet]/[pointnet++da, pointmetabase-l, pointmetabase-xl]-all.yaml wandb.use_wandb=False
```

## Acknowledgment
This repository is built on reusing codes of [PointMetaBase](https://github.com/linhaojia13/PointMetaBase), [OpenPoints](https://github.com/guochengqian/openpoints) and [PointNeXt](https://github.com/guochengqian/PointNeXt). 

## Citation
```tex
@inproceedings {frugaltraining,
    title={Frugal 3D Point Cloud Model Training via Progressive Near Point Filtering and Fused Aggregation},
    author={Donghyun Lee and Yejin Lee and Hongil Yoon and Jae W. Lee},
    booktitle = {European Conference on Computer Vision ({ECCV} 24)},
    year={2024},
}
```
