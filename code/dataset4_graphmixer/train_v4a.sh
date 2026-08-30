#!/bin/bash
# ds4 v4-A（新卡 A100）：与 v3/s44 完全同配方，唯一改动 = listwise 全宽 99 负样本
# （训练候选宽度 32 -> 100，与线上打分布对齐；LTX129 全候选 CE 思路）。
# A100 40G：batch 保持 4096（32 候选时的显存 x3 仍有余量）。seed 3407。
# 预计：~1.5-2h/epoch x 3 + ~30min 推理 = 5-7h。Log: train_v4a.log
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
  --save_dir ./saved_gm_d4_v4a --hidden_dim 256 --history_length 40 --batch_size 4096 \
  --epochs 3 --lr 1e-3 --num_negatives 99 --val_samples 4000 --num_layers 2 \
  --mlp_ratio 2.0 --dropout 0.1 --seed 3407 --train_all 2>&1 | tee train_v4a.log
echo TRAIN_DONE
