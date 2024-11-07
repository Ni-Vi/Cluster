#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=25G
#SBATCH --time=00:30:00
#SBATCH --partition=nodes
#SBATCH --job-name=run-clustering
#SBATCH --output=jobs/clusterings/run-clustering.%J.out

pdm run python -m wandb agent lewidi/cluster-performance/"$SWEEP_ID"
