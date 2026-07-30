# 第六届计图挑战赛 · 赛道一 dataset1 方案

队伍：**飞带快友队** ｜ A 榜排名：**第 31 名**（总分 1.4081）

本仓库为赛道一（基于图学习的动态推荐任务）**dataset1** 部分的完整解决方案，基于 **Jittor** 实现，可从原始数据独立复现。

- dataset1 本地验证集 **MRR = 0.9909**（时间切分 85%/15%）
- 方案：Node2Vec 结构特征 + GraphMixer 时序建模 + 启发式记忆特征 + 两阶段训练

> dataset2 部分由队友负责，见其对应仓库。

## 方案概述

赛题：给定历史时序交互 (src, dst, time)，对测试集中每个 (src, time) 的 100 个候选 dst 输出交互概率分布，评测指标为 MRR。

dataset1 的特点：src/dst 共享节点集合（非二部图）、重复边比例高、"记忆型"交互模式主导。方案四个核心组件：

1. **节点结构特征**：Node2Vec（p=q=1，每节点 200 条 × 30 步游走，skip-gram Word2Vec window=10，256 维），L2 归一化；
2. **GraphMixer 时序建模**：将每个源节点最近 **40** 条交互目标编码为序列，经正弦位置编码 + 2 层 MLP-Mixer（token/channel mixing，双残差），再以"位置指数衰减 × 时间间隔 recency"加权池化得到历史记忆向量；
3. **启发式记忆特征**：候选节点 degree、popularity、(src, dst) 历史边计数，与神经打分融合；
4. **两阶段训练**：listwise 交叉熵（温度 0.1）+ BPR margin loss；先从头训练 20 epochs，再加载最优 checkpoint 续训 10 epochs。

## 目录结构

```
├── code/
│   ├── gen_features.py           # Node2Vec 节点特征生成
│   ├── precompute_heuristics.py  # 启发式特征预计算（degree / popularity / edge_count）
│   ├── model_graphmixer_jt.py    # GraphMixer 模型定义（Jittor）
│   ├── train_graphmixer_jt.py    # 训练 + 验证选优 + 推理主流程
│   └── compare_results.py        # 结果一致性对比工具
├── docs/
│   └── 提交说明文档.pdf           # 完整方案与复现说明
├── requirements.txt
└── README.md
```

## 环境

- Ubuntu 22.04 + CUDA 12.4 + Python 3.10 + Jittor ≥ 1.3.10（亦在 Windows 11 + RTX 3070 Laptop + Jittor 1.3.8.5 验证通过）
- `pip install -r requirements.txt`
- 注意：项目路径必须为纯 ASCII（中文路径会导致 Jittor JIT 编译失败）

## 快速开始

将官方 `dataset1/train.csv`、`dataset1/test.csv` 放入 `dataset1/` 目录：

```bash
# 1. 生成节点特征（CPU，约 1–2 小时）
python code/gen_features.py --dataset dataset1 --data_dir .

# 2. 预计算启发式特征（约 1 分钟）
python code/precompute_heuristics.py --dataset dataset1 --data_dir . --shared_nodes

# 3. 第一阶段训练（GPU，约 20 分钟）
python code/train_graphmixer_jt.py \
  --dataset dataset1 --data_dir . --shared_nodes --use_heuristics \
  --node_features ./node_features_dataset1_n2v256.npy \
  --save_dir ./saved_gm_d1_final \
  --hidden_dim 256 --history_length 40 --batch_size 2048 --epochs 20 \
  --lr 1e-3 --num_negatives 31 --val_samples 4000 \
  --num_layers 2 --mlp_ratio 2.0 --dropout 0.1

# 4. 第二阶段续训（约 10 分钟）
python code/train_graphmixer_jt.py \
  --dataset dataset1 --data_dir . --shared_nodes --use_heuristics \
  --node_features ./node_features_dataset1_n2v256.npy \
  --save_dir ./saved_gm_d1_final \
  --init_checkpoint ./saved_gm_d1_final/dataset1_graphmixer.pkl \
  --hidden_dim 256 --history_length 40 --batch_size 2048 --epochs 10 \
  --lr 1e-3 --num_negatives 31 --val_samples 4000 \
  --num_layers 2 --mlp_ratio 2.0 --dropout 0.1
```

输出：`saved_gm_d1_final/dataset1.csv`（61,051 × 100，提交格式）。

期望指标：一阶段 val MRR ≈ 0.988，二阶段 ≈ 0.991（±0.002 波动正常）。

## 复现说明

- 全部特征仅由 `train.csv` 构建，不使用测试集标签，不使用任何外部数据；
- node2vec 采样（workers=4）与 Jittor CUDA 训练存在固有非确定性，两次独立训练的模型对候选尾部排序会有差异，**复现成功以验证集 MRR 为准**（与 0.990958 差 <0.002）；
- 加载同一 checkpoint 推理可逐位复现提交结果（top-1 一致率 99.98%）。

## 参考

- Cong et al., *GraphMixer: A Simple Yet Effective Framework for Temporal Graph Learning*
- Grover & Leskovec, *node2vec: Scalable Feature Learning for Networks* (KDD 2016)
- [Jittor](https://github.com/Jittor/jittor) 深度学习框架

## 声明

本仓库代码仅供学习交流。比赛相关数据请从官方渠道获取，本仓库不包含数据文件。

## License

[MIT](LICENSE)
