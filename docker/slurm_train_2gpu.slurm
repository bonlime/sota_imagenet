#!/bin/sh
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 12
#SBATCH --partition gpu
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=16G
#SBATCH --time=0-24:00:00
#SBATCH --job-name="imagenet experiments"

python3 -m torch.distributed.launch --nproc_per_node=2 --use_env train.py loader.use_tfrecords=True val_loader.use_tfrecords=True +hydra_exp=$@
