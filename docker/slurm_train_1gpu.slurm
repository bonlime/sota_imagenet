#!/bin/sh
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 6
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=16Gb
#SBATCH --time=2-00:00:00
#SBATCH --job-name="imagenet experiments"

python3 train.py loader.use_tfrecords=True val_loader.use_tfrecords=True +hydra_exp=$@
