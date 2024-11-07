#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=run-clustering
#SBATCH --output=jobs/run-clustering.%J.out

flight env activate gridware
module load libs/nvidia-cuda

# sweep ids for the train_encoder_only_model.py script
RUN_IDS=(
	"" # Decoder Only  - Anon
	"" # Decoder Only - Babe
	"" # Decoder Only - GWSD
	"" # Decoder Only - MBIC
	"" # Encoder Decoder Pretrained - Anon
	"" # Encoder Decoder Pretrained - Babe
	"" # Encoder Decoder Pretrained - GWSD
	"" # Encoder Decoder Pretrained - MBIC
	"" # Classifier - Anon
	"" # Classifier - Babe
	"" # Classifier - GWSD
	"" # Classifier - MBIC
)
# sweep ids for the train_model.py script
OTHER_RUN_IDS=(
	'' # Buffalo - Anon
	'' # Buffalo - Babe
	'' # Buffalo - GWSD
	'' # Buffalo - MBIC
	'' # Enc Enc - Anon
	'' # Enc Enc Babe
	'' # Enc Enc - GWSD
	'' # Enc Enc MBIC
	'' # Cross Attention - Anon
	'' # Cross Attention - Babe
	'' # Cross Attention - GWSD
	'' # Cross Attention - MBIC
	# '' # Downsampled Cross Attention - GWSD
	# '' # Downsampled Cross Attention - Anon
	# '' #  Cross Attention Pooled No Downsampling - GWSD
	# '' # Cross Attention Pooled No Downsampling - Anon
)
flight env activate gridware
module load libs/nvidia-cuda
pdm run python src/mtl_cluster/clustering.py
