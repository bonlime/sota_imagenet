#!/bin/sh
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 4
#SBATCH --partition gpu_devel
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=16Gb
#SBATCH --time=0-00:20:00
#SBATCH --job-name="imagenet experiments test"

python3 train.py loader.use_tfrecords=True val_loader.use_tfrecords=True +hydra_exp=$@
