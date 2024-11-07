#! /bin/bash

SWEEP_IDS=(
# # MBIC decoder_only pca kmeans
# "vqaxgtp4"
# # MBIC decoder_only umap kmeans
# "jkmwm8dh"
# # MBIC encoder_decoder_pretrained pca kmeans
# "1l2h1ah2"
# # MBIC encoder_decoder_pretrained umap kmeans
# "haxl8ztl"
# # MBIC cross_attention_pooled pca kmeans
# "u8bjsibr"
# # MBIC cross_attention_pooled umap kmeans
# "2r55aihw"
# # MBIC encoder_encoder pca kmeans
# "4htasgu8"
# # MBIC encoder_encoder umap kmeans
# "zwfk46yp"
# # MBIC cross_attention_unpooled pca kmeans
# "rbe38ccj"
# # MBIC cross_attention_unpooled umap kmeans
# "8m0565mn"
# # GWSD decoder_only pca kmeans
# "tlke7rs8"
# # GWSD decoder_only umap kmeans
# "gqhvodtd"
# # GWSD encoder_decoder_pretrained pca kmeans
# "wvedisqh"
# # GWSD encoder_decoder_pretrained umap kmeans
# "pnze3q48"
# # GWSD cross_attention_pooled pca kmeans
# "a195wmai"
# # GWSD cross_attention_pooled umap kmeans
# "z9dh8qhl"
# # GWSD encoder_encoder pca kmeans
# "z0mgs98d"
# # GWSD encoder_encoder umap kmeans
# "25k6asa5"
# # GWSD cross_attention_unpooled pca kmeans
# "ef9a0488"
# # GWSD cross_attention_unpooled umap kmeans
# "gar0c6hg"
# MBIC classifier pca kmeans
"v7xegp05"
# MBIC classifier umap kmeans
"hcc1wvn6"
# GWSD classifier pca kmeans
"hgr95sr1"
# GWSD classifier umap kmeans
"wz3hfqgw"
# MBIC decoder_only none kmeans
"cmwgvebz"
# MBIC encoder_decoder_pretrained none kmeans
"aie6zuvv"
# MBIC classifier none kmeans
"298u7llw"
# MBIC cross_attention_pooled none kmeans
"tbb7tel1"
# MBIC encoder_encoder none kmeans
"lw616i77"
# MBIC cross_attention_unpooled none kmeans
"dscp5ds3"
# GWSD decoder_only none kmeans
"gvdvil15"
# GWSD encoder_decoder_pretrained none kmeans
"xhtzb1lt"
# GWSD classifier none kmeans
"4sah6a7t"
# GWSD cross_attention_pooled none kmeans
"ism3c5ob"
# GWSD encoder_encoder none kmeans
"ubskczlx"
# GWSD cross_attention_unpooled none kmeans
"c9p9k81v"
# MBIC decoder_only pca hdbscan
"t0fnbm2q"
# MBIC decoder_only umap hdbscan
"bqm5469x"
# MBIC encoder_decoder_pretrained pca hdbscan
"93iergua"
# MBIC encoder_decoder_pretrained umap hdbscan
"poqxpu9z"
# MBIC classifier pca hdbscan
"0pkctk74"
# MBIC classifier umap hdbscan
"w7jetst4"
# MBIC cross_attention_pooled pca hdbscan
"pkzm8t1m"
# MBIC cross_attention_pooled umap hdbscan
"q7xolhcl"
# MBIC encoder_encoder pca hdbscan
"kpkopkqf"
# MBIC encoder_encoder umap hdbscan
"9l26616y"
# MBIC cross_attention_unpooled pca hdbscan
"e4r8doze"
# MBIC cross_attention_unpooled umap hdbscan
"xa2ldsdg"
# GWSD decoder_only pca hdbscan
"uq3pfti2"
# GWSD decoder_only umap hdbscan
"ho84m1mf"
# GWSD encoder_decoder_pretrained pca hdbscan
"gingjhww"
# GWSD encoder_decoder_pretrained umap hdbscan
"cs7z0sml"
# GWSD classifier pca hdbscan
"5668kh2f"
# GWSD classifier umap hdbscan
"5ytictq6"
# GWSD cross_attention_pooled pca hdbscan
"oxoxszyo"
# GWSD cross_attention_pooled umap hdbscan
"ezmrvokr"
# GWSD encoder_encoder pca hdbscan
"iogrf2dx"
# GWSD encoder_encoder umap hdbscan
"xrpr2sg5"
# GWSD cross_attention_unpooled pca hdbscan
"hrcis4qw"
# GWSD cross_attention_unpooled umap hdbscan
"rgwsi8x9"
# MBIC decoder_only none hdbscan
"sujz4div"
# MBIC encoder_decoder_pretrained none hdbscan
"ussgman6"
# MBIC classifier none hdbscan
"ccjcxtlr"
# MBIC cross_attention_pooled none hdbscan
"vxnhy3x5"
# MBIC encoder_encoder none hdbscan
"xcg4crji"
# MBIC cross_attention_unpooled none hdbscan
"6x5k2lfv"
# GWSD decoder_only none hdbscan
"yrzrw8hn"
# GWSD encoder_decoder_pretrained none hdbscan
"89c15pps"
# GWSD classifier none hdbscan
"pe8jt1f0"
# GWSD cross_attention_pooled none hdbscan
"y44q78wd"
# GWSD encoder_encoder none hdbscan
"xedlshxf"
# GWSD cross_attention_unpooled none hdbscan
"k1oov8vm"
)

# for SWEEP_ID in "${SWEEP_IDS[@]}"
# do
# 	pdm run python -m wandb sweep --stop lewidi/cluster-performance/"$SWEEP_ID"
# 		# sbatch --export=SWEEP_ID="$SWEEP_ID" scripts/slurm/run-cluster-sweep.sh
# done



# Counter to keep track of variable index
counter=1

# Iterate over the array in chunks of four
for ((i = 0; i < ${#SWEEP_IDS[@]}; i += 4)); do
    for ((j = 0; j < 4 && i + j < ${#SWEEP_IDS[@]}; j++)); do
        index=$((i + j))
				var_name="SWEEP_$((j+1))_ID"
        var_value="${SWEEP_IDS[index]}"
				export "$var_name"="$var_value"
    done
		echo "1: $SWEEP_1_ID"
		echo "2: $SWEEP_2_ID"
		echo "3: $SWEEP_3_ID"
		echo "4: $SWEEP_4_ID"
		echo "---"

		# Submit the job
		sbatch --export=SWEEP_1_ID="$SWEEP_1_ID",SWEEP_2_ID="$SWEEP_2_ID",SWEEP_3_ID="$SWEEP_3_ID",SWEEP_4_ID="$SWEEP_4_ID" scripts/slurm/run-cluster-sweep-parallel.sh

    ((counter++))
done
