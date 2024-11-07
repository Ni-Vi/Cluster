#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=train-model
#SBATCH --output=jobs/train-model.%J.out
#SBATCH --gres gpu:1

flight env activate gridware
module load libs/nvidia-cuda


pdm run python -m wandb agent lewidi/cluster/01y3zt6s
