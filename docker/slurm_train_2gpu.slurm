#!/bin/sh
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 12
#SBATCH --partition gpu_small
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=16G
#SBATCH --time=0-20:00:00
#SBATCH --job-name="imagenet experiments"

# it's commented because this module is loaded on login
# module load python/pytorch-1.6.0  

python3 -m torch.distributed.launch --nproc_per_node=2 train.py -c $@ --use_tfrecords
