# GraphMixer —— dataset1 流水线

Jittor 实现的 GraphMixer，专门处理赛道一的 **dataset1**（非二部图、`src`/`dst`
共享节点集合、重复边比例高、以"记忆型"模式为主）。读取 `dataset1/train.csv` 与
`dataset1/test.csv`，输出 `dataset1.csv`。

> dataset2 由 `code/dataset2_temporal_ranker/` 中的 Temporal Graph Ranker
> 流水线独立处理，两者互不依赖。

## 设计概要

- **节点特征**：Node2Vec（`p=q=1` 即经典 DeepWalk 设定：每节点 200 条 × 30 步
  随机游走 + skip-gram Word2Vec，`window=10`、`min_count=1`，256 维），L2 归一化
  后作为模型初始化特征。
- **模型 GraphMixer**：每个源节点最近 40 条交互目标编码为序列，经正弦位置编码 +
  2 层 MLP-Mixer（token-mixing + channel-mixing，双残差）提取时序模式，再用
  "位置指数衰减 × 时间间隔 recency" 加权平均池化得到历史记忆向量；查询向量 =
  源节点嵌入 + 历史投影；候选打分用 MLP scorer，融合 3 维启发式特征（候选节点
  degree、popularity、(src,dst) 历史边计数）。
- **训练**：listwise 交叉熵（温度 0.1）+ 0.2 × BPR margin loss，时间切分 85%/15%
  验证选优；两阶段训练——第一阶段从头 20 epochs，第二阶段加载最优 checkpoint 续训 10 epochs。
- **推理**：对 100 个候选打分后转为行内分位数（同分按候选 ID 打破平局，保证确定性），
  按赛题格式输出。

## 目录结构

```
dataset1_graphmixer/
├── gen_features.py          # Node2Vec 节点特征生成 → node_features_dataset1_n2v256.npy
├── precompute_heuristics.py # 启发式特征预计算：degree / popularity / edge_count
├── model_graphmixer_jt.py   # GraphMixer 模型定义（Jittor）：MixerBlock / MLPScorer / GraphMixerModel
├── train_graphmixer_jt.py   # 训练 + 验证选优 + 测试推理主流程 → dataset1.csv
└── compare_results.py       # （辅助）两个结果文件一致性对比，非流水线必需
```

## 运行

### 数据准备

把官方 `dataset1/train.csv`、`dataset1/test.csv` 放到工作目录的 `dataset1/`
下，本目录脚本放工作目录根：

```
工作目录/
├── dataset1/
│   ├── train.csv
│   └── test.csv
├── gen_features.py
├── precompute_heuristics.py
├── model_graphmixer_jt.py
└── train_graphmixer_jt.py
```

### 三步运行

```bash
# 步骤1：生成节点特征（CPU，node2vec 游走较慢，约 1-2 小时）
python gen_features.py --dataset dataset1 --data_dir .

# 步骤2：预计算启发式特征（约 1 分钟）
python precompute_heuristics.py --dataset dataset1 --data_dir . --shared_nodes

# 步骤3：第一阶段训练（GPU，约 20 分钟）
python train_graphmixer_jt.py \
  --dataset dataset1 --data_dir . --shared_nodes --use_heuristics \
  --node_features ./node_features_dataset1_n2v256.npy \
  --save_dir ./saved_gm_d1_final \
  --hidden_dim 256 --history_length 40 --batch_size 2048 --epochs 20 \
  --lr 1e-3 --num_negatives 31 --val_samples 4000 \
  --num_layers 2 --mlp_ratio 2.0 --dropout 0.1

# 步骤4：第二阶段续训（加载第一阶段最优 checkpoint 再训 10 epochs，约 10 分钟）
python train_graphmixer_jt.py \
  --dataset dataset1 --data_dir . --shared_nodes --use_heuristics \
  --node_features ./node_features_dataset1_n2v256.npy \
  --save_dir ./saved_gm_d1_final \
  --init_checkpoint ./saved_gm_d1_final/dataset1_graphmixer.pkl \
  --hidden_dim 256 --history_length 40 --batch_size 2048 --epochs 10 \
  --lr 1e-3 --num_negatives 31 --val_samples 4000 \
  --num_layers 2 --mlp_ratio 2.0 --dropout 0.1
```

### 输出

- `saved_gm_d1_final/dataset1.csv`：61,051 行 × 100 列，8 位小数，即 A 榜提交格式。
- `saved_gm_d1_final/dataset1_graphmixer.pkl`：最优 checkpoint（**已按官方要求从提交
  zip 中剔除**，可用 `--init_checkpoint` 直接加载复现）。
- `saved_gm_d1_final/dataset1_graphmixer_meta.npy`：记录最优验证 MRR。

### 关键超参数

`hidden_dim=256`、`history_length=40`、`num_layers=2`、`mlp_ratio=2.0`、
`dropout=0.1`、`temperature=0.1`、`time_decay=0.5`、`batch_size=2048`、
`lr=1e-3`、`num_negatives=31`、两阶段 `epochs=20+10`。

期望本地验证 MRR：第一阶段 ≈0.988，第二阶段 ≈0.991
（实测从头两阶段复现 0.990792）。

## 注意事项

- **路径需纯 ASCII**：Jittor JIT 编译在中文路径下会失败。
- Jittor 首次运行若提示 "jit_utils updated, please rerun"，重新执行同一命令即可。
- 本项目未使用 JittorGeometric（dataset1 方案不依赖该库）。
