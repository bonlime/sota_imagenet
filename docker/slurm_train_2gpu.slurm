#!/bin/sh
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 8
#SBATCH --partition gpu_small
#SBATCH --gpus 4
#SBATCH --mem-per-gpu=8G
#SBATCH --time=0-05:00:00     # 30 minutes
## --mail-user=emil.zakirov@skoltech.ru
## --mail-type=ALL
#SBATCH --job-name="imagenet experiments"


# it's commented because this module is loaded on login
# module load python/pytorch-1.6.0  

# python3 train.py -c configs/resnet50_baseline.yaml --data_dir $IMAGENET_DIR

python3 -m torch.distributed.launch --nproc_per_node=4 train.py -c $@
