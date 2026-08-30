#!/usr/bin/env bash
# =============================================================================
# Temporal Graph Ranker —— dataset2 主流水线入口
# -----------------------------------------------------------------------------
# 本脚本只处理 dataset2（动态图链路预测的二部图数据集）。
#   - 稳定基线（图特征 + 权重搜索 + 稳定 MLP）
#   - 上下文排序器（hard-negative + sequence/audience 特征 + 候选 MLP）
#   - 融合（baseline × (1-w) + context × w）并写出 dataset2.csv
#
# dataset1 由独立的 GraphMixer 流水线负责，见 code/dataset1_graphmixer/。
#
# 用法：
#   # 1) 准备数据：把官方 dataset2/train.csv、dataset2/test.csv 放到 DATA_DIR 下
#   #    （DATA_DIR 默认 ./data_A，即 $DATA_DIR/dataset2/{train,test}.csv）
#   # 2) 一键全流程（baseline + build + train + predict + package）
#   ACTION=all bash scripts/run_pipeline.sh
#   #    或分两步执行（CPU baseline 先跑，NPU/GPU 训练后跑）
#   ACTION=prepare bash scripts/run_pipeline.sh   # 仅 CPU 产物
#   ACTION=neural  bash scripts/run_pipeline.sh   # 训练 + 预测 + 打包
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

# ---------- 环境激活（按需修改为你的 conda 环境） ----------
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "Jittor-1.3.11.0" ]; then
  source /home/ma-user/anaconda3/bin/activate Jittor-1.3.11.0 2>/dev/null || true
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
ACTION="${ACTION:-all}"

# 单线程 BLAS，避免与 fork worker 争抢核心；SVD 阶段单独放开
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export SVD_THREADS="${SVD_THREADS:-32}"

# ---------- 路径与参数 ----------
# DATA_DIR：包含 dataset2/ 子目录的父目录
DATA_DIR="${DATA_DIR:-data_A}"
BASELINE_ROOT="${BASELINE_ROOT:-baseline_artifacts}"
ARTIFACTS="${ARTIFACTS:-artifacts}"
REPORTS="${REPORTS:-reports}"
SUBMISSION="${SUBMISSION:-submission}"

# 稳定基线参数
BUILD_BASELINE="${BUILD_BASELINE:-1}"
STABLE_SEED="${STABLE_SEED:-2026}"
STABLE_SVD_DIM="${STABLE_SVD_DIM:-128}"
STABLE_RECENT_LIMIT="${STABLE_RECENT_LIMIT:-160}"
STABLE_TRANSITION_WINDOW="${STABLE_TRANSITION_WINDOW:-16}"
STABLE_TRANSITION_TOPK="${STABLE_TRANSITION_TOPK:-384}"
STABLE_MAX_VALID_EVENTS="${STABLE_MAX_VALID_EVENTS:-30000}"
STABLE_SEARCH_ROUNDS="${STABLE_SEARCH_ROUNDS:-5}"
STABLE_FEATURE_WORKERS="${STABLE_FEATURE_WORKERS:-32}"
STABLE_PREDICT_WORKERS="${STABLE_PREDICT_WORKERS:-32}"
STABLE_PREDICT_BATCH_SIZE="${STABLE_PREDICT_BATCH_SIZE:-32768}"
TRAIN_STABLE_MLP="${TRAIN_STABLE_MLP:-1}"
STABLE_MLP_TRAIN_ROWS="${STABLE_MLP_TRAIN_ROWS:-80000}"
STABLE_MLP_HIDDEN="${STABLE_MLP_HIDDEN:-192}"
STABLE_MLP_EPOCHS="${STABLE_MLP_EPOCHS:-15}"
STABLE_MLP_BATCH_SIZE="${STABLE_MLP_BATCH_SIZE:-4096}"
STABLE_MLP_LR="${STABLE_MLP_LR:-8e-4}"
STABLE_MLP_OUTPUT_WEIGHT="${STABLE_MLP_OUTPUT_WEIGHT:-0.20}"

# 上下文排序器参数
SEED="${SEED:-3026}"
WORKERS="${WORKERS:-48}"
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
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-4096}"
LR="${LR:-8e-4}"
REUSE_BASELINE_FEATURES="${REUSE_BASELINE_FEATURES:-0}"
REUSE_STABLE_GRAPHS="${REUSE_STABLE_GRAPHS:-1}"
BLEND_WEIGHT="${BLEND_WEIGHT:-0.10}"
OUTPUT_NAME="${OUTPUT_NAME:-temporal_ranker_blend_0p10}"
SWEEP_BLENDS="${SWEEP_BLENDS:-0.02,0.05,0.10,0.20,0.35,1.00}"

mkdir -p "$ARTIFACTS" "$REPORTS" "$SUBMISSION" logs

run_pipeline() {
  local runner=()
  if command -v numactl >/dev/null 2>&1 && command -v taskset >/dev/null 2>&1; then
    runner=(numactl --interleave=all taskset -c "${CPUSET:-0-191}")
  elif command -v taskset >/dev/null 2>&1; then
    runner=(taskset -c "${CPUSET:-0-191}")
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

echo "[temporal-ranker] dataset2-only pipeline"
echo "[temporal-ranker] action=$ACTION start=$(date -Is)"
echo "[temporal-ranker] DATA_DIR=$DATA_DIR  EPOCHS=$EPOCHS  BLEND_WEIGHT=$BLEND_WEIGHT"

case "$ACTION" in
  all|prepare|neural|baseline|build|train|predict|package|package-sweep)
    run_pipeline "$ACTION"
    ;;
  *)
    echo "Unknown ACTION=$ACTION" >&2
    echo "Use one of: prepare, neural, all, baseline, build, train, predict, package, package-sweep" >&2
    exit 2
    ;;
esac

echo "[temporal-ranker] action=$ACTION finish=$(date -Is)"
echo "[temporal-ranker] 提交文件: $SUBMISSION/$OUTPUT_NAME/dataset2.csv"
