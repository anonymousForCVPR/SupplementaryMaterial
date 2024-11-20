# Anonymous GitHub Repository for CVPR

This is a anonymous gitHub repository included CVPR2025-3553 Supplementary Material.

## Installation

### Environment

* Basic Environment

Please use the setup.sh to install requirements.

```
bash setup.sh
```

* Flash Attention

Please follow the offical guide in repository of Flash Attention.

## Quick Start

* Our model file is location in ./sp2t or ./pointcept.
* The training log and val/test result is location in ./submit. (Anonymous GitHub Repository Only)
* The config is location in ./config.

## Data Prepare

Please follow the offical guide of Pointcept fot prepare Scannet, Scannet200, S3DIS and nuScenes.

## Model Zoo

### Indoor semantic segmentation

Example running scripts are as follows:

```
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-sppt-0-base -n semseg-sppt-0-base

# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-sppt-0-base -n semseg-sppt-0-base

# S3DIS Area5
sh scripts/train.sh -g 4 -d s3dis -c semseg-sppt-area5-rpe -n semseg-sppt-area5-rpe

# S3DIS 6-fold (Take Area1 as a example)
sh scripts/train.sh -g 4 -d s3dis -c semseg-sppt-6fold-T1 -n semseg-sppt-6fold-T1
```

For Test, the example scirpts are as follows:

```
# ScanNet
sh scripts/test.sh -g 4 -d scannet -c semseg-sppt-0-base -n semseg-sppt-0-base

# ScanNet200
sh scripts/test.sh -g 4 -d scannet200 -c semseg-sppt-0-base -n semseg-sppt-0-base

# S3DIS Area5
sh scripts/test.sh -g 4 -d s3dis -c semseg-sppt-area5-rpe -n semseg-sppt-area5-rpe

# S3DIS 6-fold (Take Area1 as a example)
sh scripts/test.sh -g 4 -d s3dis -c semseg-sppt-6fold-T1 -n semseg-sppt-6fold-T1
```

### Indoor instance segmentation

Example running scripts are as follows:

```
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c insseg-sppt-0-base -n insseg-sppt-0-base

# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c insseg-sppt-0-base -n insseg-sppt-0-base
```

### Outdoor semantic segmentation

Example running scripts are as follows:

```
# nuScenes
sh scripts/train.sh -g 4 -d scannet -c semseg-sppt-1-base -n semseg-sppt-1-base
```

## Train log, Test file and Test Result

We have provided the train log, test file and test result in the ./submit. We will open the weight of our model if the paper is accepted.

```
├─others
│  ├─scannet-semseg-octformer-v1m1-0-base
│  ├─scannet-semseg-pt-v3m1-0-base
│  ├─scannet-semseg-swin3d-v1m1-0-small
│  └─scannet200-semseg-pt-v3m1-0-base
└─sppt
    ├─nuscenes-semseg-sppt
    ├─s3dis-area5-semseg-sppt
    ├─scannet-insseg-sppt
    ├─scannet-semseg-sppt
    ├─scannet200-insseg-sppt
    └─scannet200-semseg-sppt
```