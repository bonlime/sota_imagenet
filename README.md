# SOTA Imagenet Training
## Overview
This code can be used to train PyTorch models on Imagenet to SOTA accuracy. This code was also used extensively for writing my Master Thesis. This repository contains only code to run training and configs for experiments. Models and other utils are defined in separate repository: [Pytorch-Tools](https://github.com/bonlime/pytorch-tools). See [installation instructions](https://github.com/bonlime/pytorch-tools#installation) for more details. 

## Run training
Download and build docker container. set `NUM_NODES` to number of available gpus. 
```
git clone https://github.com/bonlime/sota_imagenet
cd sota_imagenet
chmod +x docker/build.sh
docker/build.sh
export NUM_NODES=4
export IMAGENET_DIR=[path to imagenet data on your machine]
```
This code expects the following folder structure for Imagenet:
```
$PATH_TO_IMAGENET
├── 320
│   ├── train
|   |   ├── n01440764
│   │   ...
│   │   └── n01443537
│   └── val
│       ├── n01440764
│       ...
|       └── n01518878
└── raw-data
│   ├── train
|   |   ├── n01440764
│   │   ...
│   │   └── n01443537
│   └── val
│       ├── n01440764
│       ...
|       └── n01518878
```
Smaller version of images are used in some configurations to speed up training. You can get them by running `src/resize_imagenet.py`.
To start training run:
```
docker run --rm -it --shm-size 8G --gpus=all -v $IMAGENET_DIR:/workdir/data/:ro -v `pwd`/logs/:/workdir/logs bonlime/imagenet:latest python3 -m torch.distributed.launch --nproc_per_node=$NUM_NODES train.py -c configs/resnet50_baseline.yaml
```
This will train default torchvision version of Resnet50 to results close to `Acc@1 77.100 Acc@5 93.476`

If you want a higher accuracy use `-c configs/_old_configs/BResNet50_encoder.yaml` to train a ResNet50-like model to `Acc@1 81.420 Acc@5 95.654` with almost the same speed as original model.