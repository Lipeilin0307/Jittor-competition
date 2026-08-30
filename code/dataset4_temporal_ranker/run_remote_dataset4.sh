#!/usr/bin/env bash
set -euo pipefail

ROOT="/cache/ma-user/dataset4_temporal_ranker_v2_20260817_p32271"
CODE="$ROOT/code"
MODEL_CODE="$CODE/dataset2_temporal_ranker"
TRAIN_CSV="/home/ma-user/work/d4pkg/pkg/train.csv"
TEST_CSV="/home/ma-user/work/d4pkg/pkg/dataset4/test.csv"

mkdir -p "$ROOT/data/dataset2" "$ROOT/artifacts" "$ROOT/reports" "$ROOT/scratch"
ln -sfn "$TRAIN_CSV" "$ROOT/data/dataset2/train.csv"
ln -sfn "$TEST_CSV" "$ROOT/data/dataset2/test.csv"

source /home/ma-user/anaconda3/bin/activate PyTorch-2.1.0
export PYTHONPATH="$MODEL_CODE${PYTHONPATH:+:$PYTHONPATH}"
export JITTOR_BACKEND=acl
export GRAPH_FIT_WORKERS=32
export SVD_THREADS=32
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

exec > >(tee -a "$ROOT/run.log") 2>&1
echo "run_started=$(date --iso-8601=seconds)"
echo "root=$ROOT"
echo "python=$(command -v python)"

python "$CODE/build_timesafe_dataset4_cache.py" \
  --code-root "$MODEL_CODE" \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --out-root "$ROOT" \
  --train-rows 200000 \
  --valid-rows 40000 \
  --history-frac 0.70 \
  --workers 32 \
  --seed 20260817 \
  --svd-dim 64 \
  --cos-hard-count 24 \
  --cos-pool-size 320 \
  --pair-decoy-count 6 \
  --final-eval-rows 5000 \
  --final-eval-seed 42

python -m src.pipeline \
  --data-dir "$ROOT/data" \
  --artifacts "$ROOT/artifacts" \
  --reports "$ROOT/reports" \
  --scratch-dir "$ROOT/scratch" \
  --seeds 3407 \
  --hidden 256 \
  --epochs 10 \
  --batch-size 4096 \
  --lr 5e-4 \
  --margin-weight 0.10 \
  --margin 0.25 \
  --profile-calibration-weight 0.075 \
  --anchor-seed 20260813 \
  --anchor-epochs 8 \
  --anchor-batch-size 4096 \
  --anchor-lr 2e-2 \
  train

python "$CODE/eval_temporal_ranker_testlike.py" \
  --code-root "$MODEL_CODE" \
  --data-dir "$ROOT/data" \
  --graph "$ROOT/artifacts/valid_graph.pkl" \
  --checkpoint "$ROOT/artifacts/candidate_mlp_seed3407/feature_mlp.pkl" \
  --norm "$ROOT/artifacts/candidate_mlp_seed3407/feature_norm.npz" \
  --anchor-weights "$ROOT/artifacts/honest_anchor_weights.npz" \
  --output "$ROOT/reports/honest_mrr.json" \
  --analysis-output "$ROOT/reports/honest_analysis.json" \
  --dump-output "$ROOT/reports/honest_dump.npz" \
  --num-rows 5000 \
  --seed 42 \
  --hidden 256 \
  --predict-batch-size 1024 \
  --residual-weights "0,1.0" \
  --profile-calibration-weights "0.075" \
  --primary-profile-weight 0.075

echo "run_finished=$(date --iso-8601=seconds)"
