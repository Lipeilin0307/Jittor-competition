#!/bin/bash
# train_honest.sh — 诚实版 v4a：与 train_v4a.sh 完全同配方，唯一改动 = 去掉 --train_all
# （官方 split，val 不进训练）→ 产出"见过 train、没见过 val"的 checkpoint，
# 供 valhot/官方 val 做本地无泄漏评分器，筛查 rollout 剂量/PL 剂量/新成员。
# 预计 ~1.5-2h/epoch x 3 + 推理 30min ≈ 5-6h。Log: train_honest.log
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
  --save_dir ./saved_gm_d4_honest --hidden_dim 256 --history_length 40 --batch_size 4096 \
  --epochs 3 --lr 1e-3 --num_negatives 99 --val_samples 4000 --num_layers 2 \
  --mlp_ratio 2.0 --dropout 0.1 --seed 3407 2>&1 | tee train_honest.log
echo HONEST_DONE
