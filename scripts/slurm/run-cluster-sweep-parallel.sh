#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=run-clustering
#SBATCH --output=jobs/clusterings/run-clustering.%J.out

pdm run python -m wandb agent lewidi/cluster-performance/"$SWEEP_1_ID" &
pdm run python -m wandb agent lewidi/cluster-performance/"$SWEEP_2_ID" &
pdm run python -m wandb agent lewidi/cluster-performance/"$SWEEP_3_ID" &
pdm run python -m wandb agent lewidi/cluster-performance/"$SWEEP_4_ID" &

echo "Waiting for processes to finish"
wait
echo "All processes finished"
