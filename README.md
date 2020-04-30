# SOTA Imagenet Training
## Overview
This code can be used to train PyTorch models on Imagenet to SOTA accuracy. 

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
docker run --rm -it --shm-size 8G --gpus=all -v $IMAGENET_DIR:/workdir/data/ -v `pwd`/logs/:/workdir/logs bonlime/imagenet:latest python3 -m torch.distributed.launch --nproc_per_node=$NUM_NODES train.py -c configs/resnet50_baseline.yaml
```
This will train default torchvision version of Resnet50 to results close to `Acc@1 77.100 Acc@5 93.476`