#!/bin/bash
# ds3 三种子连跑（seed 1/2/3）：与本地 run_d3_v3_seed*.bat 同配方。
# 每种子两段式：20 epochs 后从 checkpoint 续训 10 epochs，最后 gzip csv。
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
for S in 1 2 3; do
  echo "===== SEED $S stage1 (20ep) ====="
  $PY train_graphmixer_jt.py --dataset dataset3 --data_dir . --shared_nodes --use_heuristics \
    --use_official_split --hist_pos_time_only --neg_mode test_template --use_known_flag \
    --use_recency_feats --use_cf_feats --node_features ./node_features_dataset3_n2v256.npy \
    --save_dir ./saved_cf_s$S --hidden_dim 256 --history_length 40 --batch_size 2048 \
    --epochs 20 --lr 1e-3 --num_negatives 31 --val_samples 4000 --num_layers 2 \
    --mlp_ratio 2.0 --dropout 0.1 --seed $S
  echo "===== SEED $S stage2 (+10ep) ====="
  $PY train_graphmixer_jt.py --dataset dataset3 --data_dir . --shared_nodes --use_heuristics \
    --use_official_split --hist_pos_time_only --neg_mode test_template --use_known_flag \
    --use_recency_feats --use_cf_feats --node_features ./node_features_dataset3_n2v256.npy \
    --save_dir ./saved_cf_s$S --init_checkpoint ./saved_cf_s$S/dataset3_graphmixer.pkl \
    --hidden_dim 256 --history_length 40 --batch_size 2048 --epochs 10 --lr 1e-3 \
    --num_negatives 31 --val_samples 4000 --num_layers 2 --mlp_ratio 2.0 --dropout 0.1 --seed $S
  gzip -kf saved_cf_s$S/dataset3.csv
  echo "SEED $S DONE"
done
echo ALL_D3_SEEDS_DONE
