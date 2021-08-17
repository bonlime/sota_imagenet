#!/bin/sh
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 12
#SBATCH --partition gpu
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=16G
#SBATCH --time=1-12:00:00
#SBATCH --job-name="imagenet experiments"
#SBATCH --output=logs/slurm-%j.out

python3 -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env \
    --max_restarts 0 \
    --master_port 11112 \
    train.py loader.use_tfrecords=True val_loader.use_tfrecords=True +hydra_exp=$@
