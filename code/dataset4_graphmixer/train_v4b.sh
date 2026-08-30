#!/bin/bash
# ds4 v4-B（②号卡 4090，s44 收完后跑）：同 v4-A 的全宽 99 配方，seed 3408。
# 4090 24G 显存：100 候选打分量是 v3 的 3.1 倍，batch 降到 1536 防 OOM。
# 预计：~2.2h/epoch x 3 + ~40min 推理 = 7-8h。Log: train_v4b.log
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
  --save_dir ./saved_gm_d4_v4b --hidden_dim 256 --history_length 40 --batch_size 1536 \
  --epochs 3 --lr 1e-3 --num_negatives 99 --val_samples 4000 --num_layers 2 \
  --mlp_ratio 2.0 --dropout 0.1 --seed 3408 --train_all 2>&1 | tee train_v4b.log
echo TRAIN_DONE
