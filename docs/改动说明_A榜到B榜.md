# 相对 A 榜算法的改动说明（B 榜要求项）

**队伍**：飞带快友队　**赛道**：赛道一
**A 榜**：dataset1 + dataset2，最优 1.4081054563093174（#31）
**B 榜**：dataset3 + dataset4，最优 1.1803393647383136（#24）

B 榜更换了两个数据集（dataset3/dataset4），评测协议与 A 榜一致
（每 (src,time) 对 100 候选 dst 输出概率分布，MRR 评测），但 B 榜数据呈现两个新的
结构性挑战，驱动了以下改动。

## 0. B 榜新挑战（改动动机）

1. **训练窗与测试窗之间存在约 90 天的时间断层**：训练后期模式与测试期分布漂移显著，
   冻结模型在测试窗退化明显。
2. **测试候选池约 23.4% 是训练期从未出现的 dst（UNK）**，且候选池混入了大量
   "历史上出现过但非正解"的对抗性干扰项——纯记忆/热度类启发式（EdgeBank、门控
   筛选等）会精准踩雷（我队实测多组此类探针显著负收益）。

## 1. dataset1 → dataset3 的改动

A 榜 dataset1 方案为 Node2Vec + GraphMixer 单模型。B 榜 dataset3 沿用同一
GraphMixer 主架构（hidden 256 / history 40 / 2 层 Mixer / 时间加权池化），改动如下：

| 维度 | A 榜 dataset1 | B 榜 dataset3 |
|---|---|---|
| 负样本 | 31 随机负例 | test_template 模式（模拟线上候选分布）+ known_flag |
| 特征 | Node2Vec 256 维 + 3 维启发式 | 新增 CF 协同特征（--use_cf_feats）、recency 时间特征（--use_recency_feats） |
| 训练切分 | 时间尾部 15% 自划 val | 使用官方 split（--use_official_split）+ hist_pos_time_only |
| 训练轮数 | 20 + 10 两段式 | 沿用两段式 |
| 输出 | 单模型 | **五种子（seed 1~5）+ 异构成员 h384（hidden 384 / history 100，seed 11）共 6 路 rank-percentile 秩融合**（0.17×5 + 0.15） |

动机：dataset3 为"记忆型"数据集（与 dataset1 同族），单模型方差是主要损耗源；
多种子 + 异构容量的秩融合在不改变单模型配方的前提下稳定增收（线上实测 +0.0126）。

## 2. dataset2 → dataset4 的改动

A 榜 dataset2 为队友的 Temporal Graph Ranker 两阶段级联（手工图特征基线 + 上下文
MLP）。B 榜 dataset4 改为**双系统融合**：我方 GraphMixer 主线（70%）+ 队友升级后的
Temporal Ranker v2（tm4，30%）。

### 2.1 GraphMixer 主线相对 A 榜 dataset1 版的扩展

dataset4 为二部图（src/dst 不同空间、61 万+86 万节点、1640 万边），主要扩展：

1. **双塔 id 重映射**（--dense_id_remap + idmap_ds4.npz）：src/dst 分别映射到独立
   稠密 id 空间，UNK dst 使用专用 padding 位；
2. **双路节点特征**：src/dst 各自独立训练 Node2Vec 256 维
   （node_features_ds4_src/dst.npy，gen_features_ds4.py）；
3. **候选侧特征栈**：heuristics（degree/popularity/edge_count）+ recency（6 维）
   \+ CF 协同特征（lite，4 维）+ 近似共现（3 维，build_cooc_ds4.py）+ 硬负例表
   （build_hardnbr_ds4.py，hard_neg_ratio 0.5）+ 候选频率表（cand_freq_pairs.npz）；
4. **训练候选对齐线上**：test_template 负样本 + unseen_ratio=0.234（与线上 23.4%
   UNK 比例一致）；v4a/v4b 进一步用全宽 99 负样本（训练候选宽度 = 线上 100 列）；
5. **train_all 训练**：v4a/v4b 将官方 val 一并纳入训练（最后冲榜阶段），honest
   成员保留官方 val 不进训练作多样性来源；
6. **rollout 推理期自注入（转导学习）**：推理按测试时间序推进，每行打分的 top1
   预测实时写回该 src 的历史流（rollout_predict_d4.py --mode rollout），缓解 90 天
   时间断层导致的冻结历史失真。线上实测剂量曲线 0%→10.5%→21% 对应
   +0.000/+0.001/+0.002（21% 为峰值），是本队在转导方向上的最小有效实现。

### 2.2 tm4（队友系统）

A 榜 dataset2 的 Temporal Graph Ranker 升级为 dataset4 版 v2：time-safe 特征缓存
（build_timesafe_dataset4_cache.py）、test-like 评估器（eval_temporal_ranker_testlike.py），
在 Ascend NPU 环境运行。与我方 GNN 主线相关性 0.74~0.78（半异构），外层 30% 融合
实测 +0.003~0.014。

### 2.3 融合层

- 全部成员输出统一转为 **rank-percentile 秩空间**（rank_prep.py），在秩空间做线性
  加权（blend_emit.py / emit_nway.py），按 1e-4 网格、8 位小数文本写出；
- 融合层为确定性纯 CPU 计算，终稿两 CSV 已验证可从成员秩矩阵逐字节重现
  （md5 校验见 final_submission/md5.txt）。

## 3. 试过并证伪的方向（防止误解为未做尝试）

TGN 记忆网络（测试集上相关性 -0.29，90 天断层下记忆僵死）、SVD 共现嵌入、
hard_neg 错误挖掘续训、recent 短窗微调、h10 时间窗收窄、EdgeBank/热度门控、
stacking 元学习、伪标签自训练（与 rollout 等值但未更强）。以上均有实验记录，
未进入终稿。
