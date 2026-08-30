#!/usr/bin/env bash
# 自定义配置版本 - 数据路径 /home/ma-user/work, 15 epochs
# 说明：本流水线只处理 dataset2；dataset1 由 code/dataset1_graphmixer/ 负责。
set -euo pipefail

cd "$(dirname "$0")/.."
# 激活 conda 环境（如果未激活）
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "Jittor-1.3.11.0" ]; then
  source /home/ma-user/anaconda3/bin/activate Jittor-1.3.11.0 || true
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
ACTION="${ACTION:-all}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export SVD_THREADS="${SVD_THREADS:-32}"

# ========== 你要求修改的参数 ==========
DATA_DIR="/home/ma-user/work/github_others"           # 数据路径：/home/ma-user/work/dataset1 和 dataset2
EPOCHS="15"                             # 训练 15 epochs
# =======================================

BASELINE_ROOT="${BASELINE_ROOT:-baseline_artifacts}"
ARTIFACTS="${ARTIFACTS:-artifacts}"
REPORTS="${REPORTS:-reports}"
SUBMISSION="${SUBMISSION:-submission}"

BUILD_BASELINE="${BUILD_BASELINE:-1}"
STABLE_SEED="${STABLE_SEED:-2026}"
STABLE_SVD_DIM="${STABLE_SVD_DIM:-128}"
STABLE_RECENT_LIMIT="${STABLE_RECENT_LIMIT:-160}"
STABLE_TRANSITION_WINDOW="${STABLE_TRANSITION_WINDOW:-16}"
STABLE_TRANSITION_TOPK="${STABLE_TRANSITION_TOPK:-384}"
STABLE_MAX_VALID_EVENTS="${STABLE_MAX_VALID_EVENTS:-30000}"
STABLE_SEARCH_ROUNDS="${STABLE_SEARCH_ROUNDS:-5}"
STABLE_FEATURE_WORKERS="${STABLE_FEATURE_WORKERS:-64}"
STABLE_PREDICT_WORKERS="${STABLE_PREDICT_WORKERS:-64}"
STABLE_PREDICT_BATCH_SIZE="${STABLE_PREDICT_BATCH_SIZE:-16384}"
TRAIN_STABLE_MLP="${TRAIN_STABLE_MLP:-1}"
STABLE_MLP_TRAIN_ROWS="${STABLE_MLP_TRAIN_ROWS:-80000}"
STABLE_MLP_HIDDEN="${STABLE_MLP_HIDDEN:-192}"
STABLE_MLP_EPOCHS="${STABLE_MLP_EPOCHS:-8}"
STABLE_MLP_BATCH_SIZE="${STABLE_MLP_BATCH_SIZE:-2048}"
STABLE_MLP_LR="${STABLE_MLP_LR:-8e-4}"
STABLE_MLP_OUTPUT_WEIGHT="${STABLE_MLP_OUTPUT_WEIGHT:-0.20}"

SEED="${SEED:-3026}"
WORKERS="${WORKERS:-96}"
HISTORY_FRAC="${HISTORY_FRAC:-0.70}"
TRAIN_ROWS="${TRAIN_ROWS:-400000}"
VALID_ROWS="${VALID_ROWS:-60000}"
TEMPLATE_TRAIN_ROWS="${TEMPLATE_TRAIN_ROWS:-400000}"
TEMPLATE_VALID_ROWS="${TEMPLATE_VALID_ROWS:-100000}"
MAX_POOL="${MAX_POOL:-700}"
SVD_DIM="${SVD_DIM:-128}"
FIT_EDGE_LIMIT="${FIT_EDGE_LIMIT:-0}"
SRC_SEQ_LEN="${SRC_SEQ_LEN:-64}"
DST_SEQ_LEN="${DST_SEQ_LEN:-64}"
SEEDS="${SEEDS:-3407,42,256}"
HIDDEN="${HIDDEN:-384}"
# EPOCHS 已在顶部设置为 15
BATCH_SIZE="${BATCH_SIZE:-4096}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-4096}"
LR="${LR:-8e-4}"
REUSE_BASELINE_FEATURES="${REUSE_BASELINE_FEATURES:-0}"
REUSE_STABLE_GRAPHS="${REUSE_STABLE_GRAPHS:-1}"
BLEND_WEIGHT="${BLEND_WEIGHT:-0.10}"
OUTPUT_NAME="${OUTPUT_NAME:-temporal_ranker_blend_0p10}"
SWEEP_BLENDS="${SWEEP_BLENDS:-0.02,0.05,0.10,0.20,0.35,1.00}"

mkdir -p "$ARTIFACTS" "$REPORTS" "$SUBMISSION" logs

run_final() {
  local runner=()
  if command -v numactl >/dev/null 2>&1 && command -v taskset >/dev/null 2>&1; then
    runner=(numactl --interleave=all taskset -c "${CPUSET:-64-191}")
  elif command -v taskset >/dev/null 2>&1; then
    runner=(taskset -c "${CPUSET:-64-191}")
  fi
  "${runner[@]}" "$PYTHON_BIN" -m src.pipeline \
    --data-dir "$DATA_DIR" \
    --baseline-root "$BASELINE_ROOT" \
    --artifacts "$ARTIFACTS" \
    --reports "$REPORTS" \
    --submission "$SUBMISSION" \
    --build-baseline "$BUILD_BASELINE" \
    --stable-seed "$STABLE_SEED" \
    --stable-svd-dim "$STABLE_SVD_DIM" \
    --stable-recent-limit "$STABLE_RECENT_LIMIT" \
    --stable-transition-window "$STABLE_TRANSITION_WINDOW" \
    --stable-transition-topk "$STABLE_TRANSITION_TOPK" \
    --stable-max-valid-events "$STABLE_MAX_VALID_EVENTS" \
    --stable-search-rounds "$STABLE_SEARCH_ROUNDS" \
    --stable-feature-workers "$STABLE_FEATURE_WORKERS" \
    --stable-predict-workers "$STABLE_PREDICT_WORKERS" \
    --stable-predict-batch-size "$STABLE_PREDICT_BATCH_SIZE" \
    --train-stable-mlp "$TRAIN_STABLE_MLP" \
    --stable-mlp-train-rows "$STABLE_MLP_TRAIN_ROWS" \
    --stable-mlp-hidden "$STABLE_MLP_HIDDEN" \
    --stable-mlp-epochs "$STABLE_MLP_EPOCHS" \
    --stable-mlp-batch-size "$STABLE_MLP_BATCH_SIZE" \
    --stable-mlp-lr "$STABLE_MLP_LR" \
    --stable-mlp-output-weight "$STABLE_MLP_OUTPUT_WEIGHT" \
    --seed "$SEED" \
    --workers "$WORKERS" \
    --history-frac "$HISTORY_FRAC" \
    --train-rows "$TRAIN_ROWS" \
    --valid-rows "$VALID_ROWS" \
    --template-train-rows "$TEMPLATE_TRAIN_ROWS" \
    --template-valid-rows "$TEMPLATE_VALID_ROWS" \
    --max-pool "$MAX_POOL" \
    --svd-dim "$SVD_DIM" \
    --fit-edge-limit "$FIT_EDGE_LIMIT" \
    --src-seq-len "$SRC_SEQ_LEN" \
    --dst-seq-len "$DST_SEQ_LEN" \
    --seeds "$SEEDS" \
    --hidden "$HIDDEN" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --predict-batch-size "$PREDICT_BATCH_SIZE" \
    --lr "$LR" \
    --reuse-baseline-features "$REUSE_BASELINE_FEATURES" \
    --reuse-stable-graphs "$REUSE_STABLE_GRAPHS" \
    --blend-weight "$BLEND_WEIGHT" \
    --output-name "$OUTPUT_NAME" \
    --sweep-blends "$SWEEP_BLENDS" \
    "$1"
}

echo "[temporal-ranker] action=$ACTION start=$(date -Is)"
echo "[temporal-ranker] DATA_DIR=$DATA_DIR EPOCHS=$EPOCHS"
case "$ACTION" in
  all|prepare|neural|baseline|build|train|predict|package|package-sweep)
    run_final "$ACTION"
    ;;
  *)
    echo "Unknown ACTION=$ACTION" >&2
    echo "Use one of: prepare, neural, all, baseline, build, train, predict, package, package-sweep" >&2
    exit 2
    ;;
esac
echo "[temporal-ranker] action=$ACTION finish=$(date -Is)"