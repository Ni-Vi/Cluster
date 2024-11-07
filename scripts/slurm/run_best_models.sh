#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=run-best-model
#SBATCH --output=jobs/run-best-model.%J.out
#SBATCH --gres gpu:1

flight env activate gridware
module load libs/nvidia-cuda

# sweep ids for the train_encoder_only_model.py script
SWEEP_IDS=(
	"mt7unnyh" # Decoder Only  - Anon
	"941qcipw" # Decoder Only - Babe
	"33fqags8" # Decoder Only - GWSD
	"k1y0wchk" # Decoder Only - MBIC
	"rtqco6l5" # Encoder Decoder Pretrained - Anon
	"siiv7m0l" # Encoder Decoder Pretrained - Babe
	"3nyc2udf" # Encoder Decoder Pretrained - GWSD
	"9zxi7jwz" # Encoder Decoder Pretrained - MBIC
	"r4a3sbxq" # Classifier - Anon
	"3h8opzbm" # Classifier - Babe
	"35k1g9vr" # Classifier - GWSD
	"dkaarbek" # Classifier - MBIC
)

# sweep ids for the train_model.py script
OTHER_SWEEP_IDS=(
	'64vmdp9v' # Buffalo - Anon
	'vuqy95v5' # Buffalo - Babe
	'6invxuva' # Buffalo - GWSD
	'muist94w' # Buffalo - MBIC
	'ccmolb18' # Enc Enc - Anon
	'01y3zt6s' # Enc Enc Babe
	'iq1mjo5j' # Enc Enc - GWSD
	'dksk1yje' # Enc Enc MBIC
	'o035uys1' # Cross Attention - Anon
	'aa46nd9p' # Cross Attention - Babe
	'4ydum9e6' # Cross Attention - GWSD
	'nexcve1x' # Cross Attention - MBIC
	# '6ctq5919' # Downsampled Cross Attention - GWSD
	# 'goeyp9sk' # Downsampled Cross Attention - Anon
	# 'ezqw2yy1' #  Cross Attention Pooled No Downsampling - GWSD
	# 'zjfcbmh6' # Cross Attention Pooled No Downsampling - Anon
)

for SWEEP_ID in "${SWEEP_IDS[@]}"; do
	SWEEP_ID="$SWEEP_ID" ENABLE_CHECKPOINTING=1 pdm run python src/mtl_cluster/train_encoder_only_model.py
done

for SWEEP_ID in "${OTHER_SWEEP_IDS[@]}"; do
	SWEEP_ID="$SWEEP_ID" ENABLE_CHECKPOINTING=1 pdm run python src/mtl_cluster/train_model.py
done
