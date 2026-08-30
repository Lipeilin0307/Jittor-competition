#!/usr/bin/env bash
# 优化版第2步 - 构建 dataset2 上下文特征、训练、预测、打包
# 特点：优化参数 + 自动清理 + 生成 dataset2.csv
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

# 优化：减少日志输出
export GLOG_minloglevel=2
export GLOG_logtostderr=0

DATA_DIR="/home/ma-user/work/github_others"
BASELINE_ROOT="${BASELINE_ROOT:-baseline_artifacts}"
ARTIFACTS="${ARTIFACTS:-artifacts}"
REPORTS="${REPORTS:-reports}"
SUBMISSION="${SUBMISSION:-submission}"

# 完整参数（同step1）
SEED="3026"
WORKERS="48"
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
    --build-baseline "0" \
    --stable-seed "2026" \
    --stable-svd-dim "128" \
    --stable-recent-limit "160" \
    --stable-transition-window "16" \
    --stable-transition-topk "384" \
    --stable-max-valid-events "30000" \
    --stable-search-rounds "5" \
    --stable-feature-workers "32" \
    --stable-predict-workers "32" \
    --stable-predict-batch-size "32768" \
    --train-stable-mlp "0" \
    --stable-mlp-train-rows "80000" \
    --stable-mlp-hidden "192" \
    --stable-mlp-epochs "15" \
    --stable-mlp-batch-size "4096" \
    --stable-mlp-lr "8e-4" \
    --stable-mlp-output-weight "0.20" \
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

echo "[optimized-step2] ========== 第2步：特征构建、训练、预测 =========="
echo "[optimized-step2] start=$(date -Is)"

echo "[optimized-step2] 构建上下文特征..."
run_pipeline build
echo "[optimized-step2] 特征构建完成，清理缓存..."
find artifacts -name "context_features_*.npy" -delete 2>/dev/null || true

echo "[optimized-step2] 训练上下文排序器..."
run_pipeline train

echo "[optimized-step2] 预测..."
run_pipeline predict

echo "[optimized-step2] 打包CSV..."
run_pipeline package

echo "[optimized-step2] ✅ 全部完成！"
echo "[optimized-step2] finish=$(date -Is)"
echo ""
echo "dataset2 提交文件："
ls -lh submission/
echo ""
echo "🎉 dataset2.csv 已生成；请与 dataset1_graphmixer 产出的 dataset1.csv 一并打包上传。"