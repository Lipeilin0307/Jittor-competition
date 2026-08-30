# Temporal Graph Ranker —— dataset2 流水线

独立 Jittor 项目，专门处理赛道一的 **dataset2**（二部图、`src` 与 `dst`
不共享节点集合、以"流行度/趋势型"模式为主）。读取 `dataset2/train.csv` 与
`dataset2/test.csv`，输出 `dataset2.csv`。

> dataset1 由 `code/dataset1_graphmixer/` 中的 GraphMixer 流水线独立处理，
> 两者互不依赖；最终提交时把两条流水线各自产出的 `dataset1.csv`、`dataset2.csv`
> 合并打包即可。

## 设计概要

两阶段级联：

1. **稳定基线（stable baseline）** —— 纯 CPU，无需 GPU
   - `GraphFeatureModel`：基于训练边构建时序统计 + TruncatedSVD 嵌入 +
     一阶转移概率表，共 21 维手工特征（`rule / pop / recent_pop / trend /
     recency / src_recent_exact / pair_log / dst_known / degree_cap /
     candidate_seen_in_test / svd / profile / transition` 及对应行内 rank）。
   - 在多个验证集（hard-negative、test-injection、low-pop injection）上做
     坐标下降式权重搜索（`search_weights_multi`），得到线性融合权重。
   - 一个浅层残差 MLP（`stable_mlp`）在验证集特征上再学一份 logits，
     与线性分数按 `feature_logits + 0.20 * mlp_logits` 融合，作为基线 logits。
2. **上下文排序器（context ranker）** —— 需 GPU/Ascend
   - 以 70% 历史边拟合图模型，构造两类训练样本：
     (a) **hard-negative**：从源节点历史、热门目标、转移表Top-K 中采负例；
     (b) **test-template**：把已知正例插入真实测试候选模板，逼近线上候选分布。
   - 在 21 维基础特征上再追加 14 维 **sequence/audience** 上下文特征
     （历史目标均值/最大/末次向量与候选的点积、audience 均值点积、
     源活跃度、源历史覆盖率、pop×source_activity 交叉等）。
   - 3 个 seed 训练残差 MLP（`hidden=384`，`epochs=15`），z-score 后平均。
3. **融合输出**
   - `logits = z(baseline) * (1 - w) + z(context) * w`，默认 `w=0.10`；
   - softmax 成概率、行内归一 + 8 位小数，写 `dataset2.csv`。

## 目录结构

```
dataset2_temporal_ranker/
├── src/
│   ├── pipeline.py          # 入口：argparse + 子命令分发（baseline/prepare/neural/all/...）
│   ├── io_data.py           # 数据读写、CSV 校验、scratch 目录管理
│   ├── temporal_graph.py    # GraphFeatureModel：21 维手工特征 + SVD + 转移表
│   ├── stable_stage.py      # 稳定基线：train_stable_dataset2 / predict_stable_dataset2
│   ├── context_stage.py     # 上下文排序器：build/train/predict/package（dataset2-only）
│   ├── candidate_ranker.py  # hard-negative 构造 + 残差 MLP 训练/推理（Jittor）
│   ├── evaluation.py        # 验证集构造、权重搜索、MRR 评估
│   └── parallel_features.py # fork 共享只读图模型的多进程特征生成
└── scripts/
    ├── run_pipeline.sh          # 主入口：ACTION=all/prepare/neural/...
    ├── run_optimized_step1.sh   # 仅 baseline（CPU）
    └── run_optimized_step2.sh   # build + train + predict + package（GPU）
```

## 运行

### 数据准备

把官方 `dataset2/train.csv`、`dataset2/test.csv` 放到 `$DATA_DIR/dataset2/`
下。默认 `DATA_DIR=data_A`（即 `data_A/dataset2/{train,test}.csv`），可用环境变量覆盖：

```bash
DATA_DIR=/path/to/parent bash scripts/run_pipeline.sh
```

### 一键全流程

```bash
ACTION=all bash scripts/run_pipeline.sh
```

`ACTION=all` 会依次执行：构建稳定基线 → 构建上下文特征 → 训练上下文 MLP →
预测 → 融合打包，最终产出 `submission/temporal_ranker_blend_0p10/dataset2.csv`。

### 分两步执行（推荐，便于资源调度）

当 NPU/GPU 被其它任务占用时，可先跑纯 CPU 的基线，再跑 GPU 训练：

```bash
# 第1步：CPU 稳定基线（图特征 + 权重搜索 + 稳定 MLP），不占 GPU
ACTION=prepare bash scripts/run_pipeline.sh
# 等同于：
bash scripts/run_optimized_step1.sh

# 第2步：GPU 训练上下文排序器 + 预测 + 打包
ACTION=neural bash scripts/run_pipeline.sh
# 等同于：
bash scripts/run_optimized_step2.sh
```

### 关键超参数（均可通过环境变量覆盖）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `STABLE_SVD_DIM` | 128 | 稳定基线 SVD 维度 |
| `STABLE_MLP_EPOCHS` | 15 | 稳定 MLP 训练轮数 |
| `STABLE_MLP_OUTPUT_WEIGHT` | 0.20 | 稳定 MLP logits 融合权重 |
| `WORKERS` | 48 | 特征生成 fork worker 数 |
| `HISTORY_FRAC` | 0.70 | 用前 70% 边拟合历史图 |
| `TRAIN_ROWS` | 400000 | hard-negative 训练样本数 |
| `HIDDEN` | 384 | 上下文 MLP 隐藏维 |
| `EPOCHS` | 15 | 上下文 MLP 训练轮数 |
| `SEEDS` | 3407,42,256 | 多 seed 训练后平均 |
| `BLEND_WEIGHT` | 0.10 | 上下文 logits 融合权重 |

### 输出

- `submission/temporal_ranker_blend_0p10/dataset2.csv`：行数 = 测试集行数，
  每行 100 列概率（和为 1，8 位小数），即赛题要求格式。
- `reports/*.json`：各阶段报告（split、权重、MRR、top1 统计等）。
- `artifacts/*.npy` / `*.pkl`：中间产物（图模型、logits、MLP checkpoint）。

## 注意事项

- **Linux 推荐**：`parallel_features.py`、`candidate_ranker.py`、
  `temporal_graph.py` 的多进程优化依赖 `fork`，Windows 下会回退到单进程
  （结果一致但显著变慢）。
- **scratch 目录**：大张量分片默认写 `/dev/shm/temporal_ranker_<pid>`，
  不占磁盘；无 `/dev/shm` 时回退到系统 tmp。
- **路径需纯 ASCII**：Jittor JIT 编译在中文路径下会失败。
- 本流水线**只读 dataset2**，不依赖也不产出 dataset1 的任何文件。
