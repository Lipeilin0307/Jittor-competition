# 第六届计图人工智能挑战赛 · 赛道一完整方案

队伍：**飞带快友队** ｜ A 榜排名：**第 31 名**（1.4081054563093174）｜ B 榜排名：**第 24 名**（1.1803393647383136）

本仓库为赛道一（基于图学习的动态推荐任务）的完整开源方案，基于 **Jittor** 实现，覆盖 A 榜（dataset1 / dataset2）与 B 榜（dataset3 / dataset4）全部四个数据集的代码、文档与复现链。

## 赛题

给定历史时序交互 (src, dst, time)，对测试集中每个 (src, time) 的 100 个候选 dst 输出交互概率分布，评测指标为 MRR。总分为两个数据集得分直接相加。

## 方案总览：数据集感知的分治

| 数据集 | 图结构 | 主导模式 | 方案 |
|---|---|---|---|
| dataset1（A） | 非二部图，src/dst 共享节点 | 记忆型 | Node2Vec + GraphMixer（Jittor） |
| dataset2（A） | 二部图，src/dst 不共享 | 流行度/趋势型 | Temporal Graph Ranker 两阶段级联 |
| dataset3（B） | 同 dataset1 族 | 记忆型 | GraphMixer 五种子 + 异构成员秩融合（0.17×5 + 0.15） |
| dataset4（B） | 二部图，1640 万边 | 趋势型 + 90 天时间断层 + 23.4% UNK 候选 | **双系统融合**：GraphMixer 主线（70%，含 rollout 推理期自注入）× Temporal Ranker v2（30%） |

B 榜两大新挑战与对策：

1. **约 90 天训练-测试时间断层** → 推理期 rollout 自注入（转导学习）：按测试时间序打分，每行 top1 预测实时写回该 src 的历史流，剂量曲线实测 21% 注入为峰值（+0.002）；
2. **23.4% UNK + 对抗性候选** → test_template 负样本 + unseen_ratio=0.234 对齐线上候选分布；纯记忆/热度类启发式（EdgeBank、门控）实测精准踩雷，全部弃用。

关键技术点：时间加权池化（位置指数衰减 × exp(−time_gap)）、listwise CE + BPR margin loss、两段式训练（20+10 epochs）、双塔 id 重映射、多成员 rank-percentile 秩融合（确定性、可逐字节复现）。

## 目录结构

```
├── code/
│   ├── dataset1_graphmixer/        # A 榜 dataset1：Node2Vec + GraphMixer
│   ├── dataset2_temporal_ranker/   # A 榜 dataset2：Temporal Graph Ranker 两阶段级联
│   ├── dataset3_graphmixer/        # B 榜 dataset3：GraphMixer 五种子 + h384 异构成员
│   ├── dataset4_graphmixer/        # B 榜 dataset4：GraphMixer 主线（训练/rollout 推理/启发式成员）
│   ├── dataset4_temporal_ranker/   # B 榜 dataset4：Temporal Ranker v2（tm4，Ascend NPU）
│   └── fusion/                     # B 榜秩融合三件套 + 终稿复现脚本 repro_p33.sh
├── docs/
│   ├── 提交说明文档_A榜.pdf         # A 榜完整方案与复现说明
│   ├── 提交说明文档_B榜.pdf         # B 榜完整方案、A→B 改动、四级复现链
│   ├── 改动说明_A榜到B榜.md
│   └── 复现说明.md                 # B 榜最优成绩复现（L1~L4）
├── requirements.txt
└── README.md
```

## 环境

- Ubuntu 22.04 + Python 3.9/3.10 + Jittor 1.3.8.5（CUDA 11.2/11.8 验证通过；亦在 Windows 11 + RTX 3070 Laptop 验证通过）
- `pip install -r requirements.txt`
- 注意：项目路径必须为纯 ASCII（中文路径会导致 Jittor JIT 编译失败）
- tm4（dataset4_temporal_ranker）运行于 Ascend NPU（ModelArts，`JITTOR_BACKEND=acl`）

## 复现

各数据集的完整训练/推理命令见 `docs/提交说明文档_A榜.pdf`、`docs/提交说明文档_B榜.pdf` 与各子目录内脚本（`train_*.sh`）。

**B 榜最优成绩（1.1803393647383136）的融合层复现为确定性计算**（纯 CPU、分钟级）：13 个成员秩矩阵经 `code/fusion/repro_p33.sh` 加权融合后，与线上提交文件 md5 逐字节一致（dataset3.csv = `b1828cda…`，dataset4.csv = `80694df1…`，详见 `docs/复现说明.md`）。

大文件（checkpoint、秩矩阵、终稿 CSV）与数据集不入库：数据请从比赛官方渠道获取；checkpoint 可按文档中的训练命令复训（单成员 GPU 6~14 小时），训练存在 ±0.002 MRR 级固有波动，多种子秩融合正是为此设计。

## 参考

- Cong et al., *GraphMixer: A Simple Yet Effective Framework for Temporal Graph Learning*
- Grover & Leskovec, *node2vec: Scalable Feature Learning for Networks* (KDD 2016)
- [Jittor](https://github.com/Jittor/jittor) 深度学习框架

## 声明

本仓库代码仅供学习交流。比赛相关数据请从官方渠道获取，本仓库不包含数据文件。

## License

[MIT](LICENSE)
