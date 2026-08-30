#!/usr/bin/env bash
# 优化版第1步 - 构建 dataset2 的稳定基线（权重搜索 + MLP 训练）
# 特点：优化参数 + 自动清理 + 手动暂停
# 说明：本流水线只处理 dataset2；dataset1 由 code/dataset1_graphmixer/ 负责。
set -euo pipefail

cd "$(dirname "$0")/.."

# 激活conda环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "Jittor-1.3.11.0" ]; then
  source /home/ma-user/anaconda3/bin/activate Jittor-1.3.11.0 || true
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export SVD_THREADS="${SVD_THREADS:-32}"

# 优化：减少日志输出
export GLOG_minloglevel=2
export GLOG_logtostderr=0

DATA_DIR="/home/ma-user/work/github_others"
BASELINE_ROOT="${BASELINE_ROOT:-baseline_artifacts}"
ARTIFACTS="${ARTIFACTS:-artifacts}"
REPORTS="${REPORTS:-reports}"
SUBMISSION="${SUBMISSION:-submission}"

# 完整参数（不降低性能）
BUILD_BASELINE="1"
STABLE_SEED="2026"
STABLE_SVD_DIM="128"
STABLE_RECENT_LIMIT="160"
STABLE_TRANSITION_WINDOW="16"
STABLE_TRANSITION_TOPK="384"
STABLE_MAX_VALID_EVENTS="30000"
STABLE_SEARCH_ROUNDS="5"

# 优化参数：减少worker数，提高进程复用
STABLE_FEATURE_WORKERS="32"       # 64 → 32
STABLE_PREDICT_WORKERS="32"       # 64 → 32
STABLE_PREDICT_BATCH_SIZE="32768" # 16384 → 32768

TRAIN_STABLE_MLP="1"
STABLE_MLP_TRAIN_ROWS="80000"
STABLE_MLP_HIDDEN="192"
STABLE_MLP_EPOCHS="15"
STABLE_MLP_BATCH_SIZE="4096"
STABLE_MLP_LR="8e-4"
STABLE_MLP_OUTPUT_WEIGHT="0.20"

SEED="3026"
WORKERS="48"                      # 96 → 48
HISTORY_FRAC="0.70"
TRAIN_ROWS="400000"
VALID_ROWS="60000"
TEMPLATE_TRAIN_ROWS="400000"
TEMPLATE_VALID_ROWS="100000"
MAX_POOL="700"
SVD_DIM="128"
FIT_EDGE_LIMIT="0"
SRC_SEQ_LEN="64"
DST_SEQ_LEN="64"
SEEDS="3407,42,256"
HIDDEN="384"
EPOCHS="15"
BATCH_SIZE="4096"
PREDICT_BATCH_SIZE="4096"
LR="8e-4"
REUSE_BASELINE_FEATURES="0"
REUSE_STABLE_GRAPHS="1"
BLEND_WEIGHT="0.10"
OUTPUT_NAME="temporal_ranker_blend_0p10"
SWEEP_BLENDS="0.02,0.05,0.10,0.20,0.35,1.00"

mkdir -p "$ARTIFACTS" "$REPORTS" "$SUBMISSION" logs

run_pipeline() {
  local runner=()
  if command -v taskset >/dev/null 2>&1; then
    runner=(taskset -c "0-191")
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

echo "[optimized-step1] ========== 第1步：构建稳定基线 =========="
echo "[optimized-step1] 优化参数："
echo "[optimized-step1]   - Worker数：64→32 (提高进程复用)"
echo "[optimized-step1]   - 批处理大小：16384→32768 (提高吞吐量)"
echo "[optimized-step1]   - 去掉maxtasksperchild=1 (减少进程创建开销)"
echo "[optimized-step1] start=$(date -Is)"

run_pipeline baseline

echo "[optimized-step1] ✅ baseline完成！"
echo ""
echo "============================================"
echo "📊 当前磁盘使用情况："
df -h .
echo ""
echo "🗑️ 清理临时文件（可选）："
echo "  find baseline_artifacts -name '*.npy' -delete"
echo "  find artifacts -name '*.npy' -delete"
echo "  rm -rf ~/.cache/jittor/jt*/default"
echo ""
echo "⏭️  继续第2步请运行："
echo "  sh scripts/run_optimized_step2.sh"
echo "============================================"