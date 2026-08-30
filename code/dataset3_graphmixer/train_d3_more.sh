#!/bin/bash
# ds3 通宵队列：seed 4/5（同三种子配方）+ 异配置成员（hidden384 + hist100）
# 每个成员两段式：20 epochs 后续训 10 epochs，产出即 gzip。
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
for S in 4 5; do
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
echo "===== H384H100 stage1 (20ep) ====="
$PY train_graphmixer_jt.py --dataset dataset3 --data_dir . --shared_nodes --use_heuristics \
  --use_official_split --hist_pos_time_only --neg_mode test_template --use_known_flag \
  --use_recency_feats --use_cf_feats --node_features ./node_features_dataset3_n2v256.npy \
  --save_dir ./saved_cf_h384 --hidden_dim 384 --history_length 100 --batch_size 2048 \
  --epochs 20 --lr 1e-3 --num_negatives 31 --val_samples 4000 --num_layers 2 \
  --mlp_ratio 2.0 --dropout 0.1 --seed 11
echo "===== H384H100 stage2 (+10ep) ====="
$PY train_graphmixer_jt.py --dataset dataset3 --data_dir . --shared_nodes --use_heuristics \
  --use_official_split --hist_pos_time_only --neg_mode test_template --use_known_flag \
  --use_recency_feats --use_cf_feats --node_features ./node_features_dataset3_n2v256.npy \
  --save_dir ./saved_cf_h384 --init_checkpoint ./saved_cf_h384/dataset3_graphmixer.pkl \
  --hidden_dim 384 --history_length 100 --batch_size 2048 --epochs 10 --lr 1e-3 \
  --num_negatives 31 --val_samples 4000 --num_layers 2 --mlp_ratio 2.0 --dropout 0.1 --seed 11
gzip -kf saved_cf_h384/dataset3.csv
echo ALL_D3_MORE_DONE
