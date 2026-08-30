#!/bin/bash
# ds4 v3 第三种子（seed 45）：与 seed43/44 完全同配方，仅换种子/输出目录。
# 4090 estimate: ~13-14 h total. Log: train_v3s45.log
set -e
cd "$(dirname "$0")"
export GM_PAGEABLE_HOST=1
${PY:-python3} code/train_graphmixer_jt.py --dataset dataset4 --data_dir . --heuristics_dir . \
  --dense_id_remap --idmap ./idmap_ds4.npz --use_official_split --hist_pos_time_only \
  --neg_mode test_template --unseen_ratio 0.234 --neg_freq_table ./cand_freq_pairs.npz \
  --use_heuristics --use_recency_feats --use_cf_feats --cf_no_cooc \
  --use_cooc --cooc_table ./cooc_ds4.npz \
  --hard_neg_ratio 0.5 --hardnbr_table ./hardnbr_ds4.npy \
  --src_features ./node_features_ds4_src.npy --dst_features ./node_features_ds4_dst.npy \
  --save_dir ./saved_gm_d4_v3s45 --hidden_dim 256 --history_length 40 --batch_size 4096 \
  --epochs 3 --lr 1e-3 --num_negatives 31 --val_samples 4000 --num_layers 2 \
  --mlp_ratio 2.0 --dropout 0.1 --seed 45 --train_all 2>&1 | tee train_v3s45.log
gzip -kf saved_gm_d4_v3s45/dataset4.csv
echo TRAIN_DONE
