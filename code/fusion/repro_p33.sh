#!/usr/bin/env bash
# =============================================================================
# repro_p33.sh — B 榜最优成绩（1.1803393647383136，第 24 名）终稿融合层复现
#
# 输入：13 个成员的秩矩阵（rank-percentile 空间，行=测试行序，列=100 候选列序）
# 输出：dataset3.csv / dataset4.csv（与线上提交文件逐字节一致，md5 见
#       final_submission/md5.txt，本队已本地验证 IDENTICAL）
#
# 用法：把 13 个秩矩阵按下表路径放好后，在本目录执行：
#   bash repro_p33.sh <rank_dir>
# <rank_dir> 下应有：
#   d3s1_rank.npy d3s2_rank.npy d3s3_rank.npy d3s4_rank.npy d3s5_rank.npy
#   d3h384_rank.npy
#   p5mix_h_rank.npy   （= v3 0.15 + honest 0.15 + rolloutA 0.15 + rolloutB 0.15
#                        + craft_e15 0.30 + heur2 0.10 的 ds4 六成员内层融合，
#                        由同一套 emit_nway.py 生成）
#   tm4_rank.npy       （队友 temporal ranker v2 的 ds4 输出转秩矩阵）
#
# 环境：仅需 python + numpy（融合层纯 CPU，分钟级完成）。
# =============================================================================
set -euo pipefail
R="${1:-./ranks}"

# ---- dataset3 = 五种子各 0.17 + h384 0.15（六路秩融合）----
python emit_nway.py \
  --members "$R/d3s1_rank.npy,$R/d3s2_rank.npy,$R/d3s3_rank.npy,$R/d3s4_rank.npy,$R/d3s5_rank.npy,$R/d3h384_rank.npy" \
  --weights 0.17,0.17,0.17,0.17,0.17,0.15 \
  --out dataset3.csv

# ---- dataset4 = p5mix_h x 0.70 + tm4 x 0.30 ----
python blend_emit.py --a "$R/p5mix_h_rank.npy" --b "$R/tm4_rank.npy" --w 0.7 --out dataset4.csv

# ---- 校验 ----
python - <<'PY'
import hashlib
def md5(p):
    h = hashlib.md5()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<26), b''): h.update(c)
    return h.hexdigest()
expect = {'dataset3.csv':'b1828cda9f71e79c28e77e33be564999',
          'dataset4.csv':'80694df1f404908946194429627e0584'}
for k,v in expect.items():
    got = md5(k)
    print(k, got, 'OK' if got==v else 'MISMATCH(期望 '+v+')')
PY
