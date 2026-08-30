"""
train_graphmixer_jt.py
=====================
GraphMixer training script for Jittor (Jittor CUDA / NPU).
Based on the proven TemporalBPR v3 training pipeline.

Usage (dataset1, A-list winning configuration, two-stage training):
  # Stage 1: train 20 epochs from scratch
  python train_graphmixer_jt.py \
    --dataset dataset1 --data_dir . --shared_nodes --use_heuristics \
    --node_features ./node_features_dataset1_n2v256.npy \
    --save_dir ./saved_gm_d1_final \
    --hidden_dim 256 --history_length 40 --batch_size 2048 --epochs 20 \
    --lr 1e-3 --num_negatives 31 --val_samples 4000 \
    --num_layers 2 --mlp_ratio 2.0 --dropout 0.1
  # Stage 2: continue 10 epochs from the stage-1 best checkpoint
  python train_graphmixer_jt.py \
    --dataset dataset1 --data_dir . --shared_nodes --use_heuristics \
    --node_features ./node_features_dataset1_n2v256.npy \
    --save_dir ./saved_gm_d1_final \
    --init_checkpoint ./saved_gm_d1_final/dataset1_graphmixer.pkl \
    --hidden_dim 256 --history_length 40 --batch_size 2048 --epochs 10 \
    --lr 1e-3 --num_negatives 31 --val_samples 4000 \
    --num_layers 2 --mlp_ratio 2.0 --dropout 0.1

Prerequisites (run once before training):
  python gen_features.py --dataset dataset1 --data_dir .
  python precompute_heuristics.py --dataset dataset1 --data_dir . --shared_nodes

The script trains with a temporal 85%/15% split, keeps the checkpoint with
the best validation MRR, then runs inference on {data_dir}/{dataset}/test.csv
and writes the submission file to {save_dir}/{dataset}.csv
(100 comma-separated probabilities per row, 8 decimals).

dataset3 adaptation switches (all default OFF = original dataset1 behavior):
  --use_official_split   use split column: split=0 train, split=1 val
  --hist_pos_time_only   histories + gap stats use time>0 edges only
  --neg_mode test_template  negatives ~91% seen-dst (popularity-weighted)
                            + ~9% unseen-dst, matching test candidate mix
  --use_known_flag       append dst_known 4th heuristic dim to the scorer
  --use_recency_feats    append 6-dim RecencyStats block (recent popularity /
                         7d-30d trend / dst,pair,src last-seen gaps), computed
                         strictly before each row's time from positive-time edges
  --use_cf_feats         append 8-dim CFStats block (item-item co-occurrence
                         sums / deg-normalized sums / coverage / max cooc /
                         n2v mean & last-20 cosine similarities / hist length),
                         concatenated AFTER the recency block. CF is
                         time-agnostic: it uses the same edge set as
                         RecencyStats but keeps time=0 edges too.

dataset4 adaptation switches (all default OFF = ds3 behavior):
  --dense_id_remap       remap sparse raw ids to dense model ids: seen dst ->
                         1..N_dst, seen src -> 1..N_src, 0 = padding,
                         N_dst+1 = UNK (any dst never seen in train; zero-init,
                         trainable; matches the 23.4% unseen test candidates).
                         Required for ds4: raw-id embedding tables + Adam would
                         need ~15.6 GB VRAM (8 GB GPU). Use with --src_features/
                         --dst_features and idmap_ds4.npz from gen_features_ds4.
  --src_features / --dst_features  separate per-tower feature tables (non-shared
                         mode), rows aligned to the model id space (dense when
                         --dense_id_remap is on). --node_features stays the
                         shared/single-table option.
  --cf_no_cooc           CFStats lite: skip the exact co-occurrence matrix
                         (~44 GB at ds4 scale); 4-dim CF block (n2v cosines +
                         hist length) instead of 8.
  --heuristics_dir DIR   where heuristics_{dataset}_*.{npy,pkl} live
                         (default: --data_dir; ds4 keeps them in the work dir
                         because the dataset dir is read-only).
  --unseen_ratio R       test_template unseen-negative fraction (default 0.09;
                         ds4 uses 0.234).
  --max_test_rows N      score only the first N test rows (smoke tests).

dataset4 v2 upgrades (independent ablation switches, both default OFF):
  --use_cooc / --cooc_table PATH
                         append a 3-dim APPROXIMATE co-occurrence block (offline
                         table from build_cooc_ds4.py: hit_frac / max_w / wsum of
                         the candidate's top-128 cooc neighbors vs the model
                         history matrix) AFTER the CF block. Stacks with
                         --cf_no_cooc lite; scorer widens by 3 -> train from scratch.
  --hard_neg_ratio R     fraction of SEEN negatives drawn from the positive dst's
                         top-256 cos neighbors (--hardnbr_table, built by
                         build_hardnbr_ds4.py) instead of the popularity pool;
                         hard draws exclude the positive and the src history dsts.
                         unseen_ratio / UNK logic unchanged. 0 = old behavior.
  --max_train_batches N  stop each epoch after N batches (30-batch smoke tests).
"""
import argparse
import os
import random
from collections import deque

import numpy as np
import pandas as pd
from tqdm import tqdm

import jittor as jt
if os.environ.get('GM_PAGEABLE_HOST', '0') == '1':
    # ds4 v2 keeps ~16-20GB of host feature tables alive; pinned staging
    # (cudaMallocHost) for the 0.88GB dst-embedding param then FAILS at
    # checkpoint-load time (measured, and the post-hoc flag switch is too
    # late -- the allocator is chosen at CUDA init). Pageable H2D costs only
    # a few ms/batch here (~10MB transferred per batch).
    jt.flags.use_cuda_host_allocator = 0
jt.flags.use_cuda = 1  # Enable GPU on local CUDA backend
from jittor import nn

from model_graphmixer_jt import GraphMixerModel


# ============================================================================
# Data utilities (identical to train_jittor_v3.py)
# ============================================================================

def build_histories_with_time(source_ids, destination_ids, time_values, source_min, history_length, split_index,
                              pos_time_only=False):
    """Build history sequences with time gaps.

    pos_time_only=True: only time>0 edges enter the history sequences and the
    gap-scale statistics; time=0 rows remain training/validation samples but
    never appear inside any history (they still feed heuristics / Node2Vec).
    Default False reproduces the original behavior exactly.
    """
    num_sources = int(source_ids.max() - source_min + 1)
    # int32 histories: dense dst ids are <= 862247 and jittor's nn.Embedding
    # accepts int32 indices (validated); halves the 4.5GB train history block.
    train_histories = np.zeros((split_index, history_length), dtype=np.int32)
    train_time_gaps = np.zeros((split_index, history_length), dtype=np.float32)
    val_histories = np.zeros((len(source_ids) - split_index, history_length), dtype=np.int32)
    val_time_gaps = np.zeros((len(source_ids) - split_index, history_length), dtype=np.float32)
    histories = [deque(maxlen=history_length) for _ in range(num_sources)]
    history_times = [deque(maxlen=history_length) for _ in range(num_sources)]

    # Compute gap scale as p90 of PER-SOURCE consecutive positive-time gaps.
    # Why not the global stream: on ds4 the merged edge stream is ~95 edges/s,
    # so the global p90 clamps to 1s and exp(-log1p(gap)/scale) collapses the
    # weight of every history slot except the last one (typical per-src gap is
    # hours). ds3's winning behavior had recency modulation nearly OFF
    # (scale=283s vs typical log1p(gap)~12 => weight~0.96); per-src p90
    # reproduces that regime here.
    last_time = np.full(num_sources, -1.0, dtype=np.float64)
    time_diffs = []
    for index in range(len(source_ids)):
        t = float(time_values[index])
        if pos_time_only and t <= 0.0:
            continue  # time=0 edges never enter histories nor gap stats
        src_idx = int(source_ids[index] - source_min)
        prev = last_time[src_idx]
        if prev >= 0.0 and t > prev:
            time_diffs.append(t - prev)
        last_time[src_idx] = t
    gap_scale = float(np.percentile(time_diffs, 90)) if time_diffs else 1.0
    gap_scale = max(gap_scale, 1.0)

    for index, (source, destination, time) in enumerate(zip(source_ids, destination_ids, time_values)):
        src_idx = int(source - source_min)
        history = histories[src_idx]
        hist_times = history_times[src_idx]
        target = train_histories[index] if index < split_index else val_histories[index - split_index]
        target_gaps = train_time_gaps[index] if index < split_index else val_time_gaps[index - split_index]
        if history:
            target[-len(history):] = history
            # Compute time gaps: current_time - hist_time, log1p normalized
            gaps = [np.log1p(max(float(time) - float(t), 0.0)) / gap_scale for t in hist_times]
            target_gaps[-len(gaps):] = gaps
        if pos_time_only and float(time) <= 0.0:
            continue  # time=0 edges never enter history sequences
        history.append(int(destination))
        history_times[src_idx].append(float(time))

    return train_histories, train_time_gaps, val_histories, val_time_gaps, histories, history_times, gap_scale


def build_histories(source_ids, destination_ids, source_min, history_length, split_index):
    """Backwards compatible wrapper without time."""
    num_sources = int(source_ids.max() - source_min + 1)
    train_histories = np.zeros((split_index, history_length), dtype=np.int64)
    val_histories = np.zeros((len(source_ids) - split_index, history_length), dtype=np.int64)
    histories = [deque(maxlen=history_length) for _ in range(num_sources)]

    for index, (source, destination) in enumerate(zip(source_ids, destination_ids)):
        history = histories[int(source - source_min)]
        target = train_histories[index] if index < split_index else val_histories[index - split_index]
        if history:
            target[-len(history):] = history
        history.append(int(destination))

    return train_histories, val_histories, histories


def compute_svd_features_shared(source_ids, destination_ids, split_index, num_nodes, hidden_dim, seed=42):
    """Build shared-node graph SVD features (dataset1) using only train split."""
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    src = source_ids[:split_index]
    dst = destination_ids[:split_index]
    row = src.tolist() + dst.tolist()
    col = dst.tolist() + src.tolist()
    data = np.ones(len(row), dtype=np.float32)

    A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A.data = np.log1p(A.data)

    svd = TruncatedSVD(n_components=hidden_dim, random_state=seed)
    features = svd.fit_transform(A)

    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features = features / norms
    if features.shape[0] > 0:
        features[0] = 0.0
    return features.astype(np.float32)


def compute_svd_features_bipartite(source_ids, destination_ids, split_index,
                                    src_min, src_count, dst_count, hidden_dim, seed=42):
    """Build bipartite graph SVD features (dataset2) using only train split."""
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    src = source_ids[:split_index] - src_min
    dst = destination_ids[:split_index]
    row = src.tolist()
    col = dst.tolist()
    data = np.ones(len(row), dtype=np.float32)

    A = csr_matrix((data, (row, col)), shape=(src_count, dst_count))
    A.data = np.log1p(A.data)

    svd = TruncatedSVD(n_components=hidden_dim, random_state=seed)
    source_features = svd.fit_transform(A).astype(np.float32)
    destination_features = svd.components_.T.astype(np.float32)

    src_norms = np.linalg.norm(source_features, axis=1, keepdims=True) + 1e-8
    source_features = source_features / src_norms
    dst_norms = np.linalg.norm(destination_features, axis=1, keepdims=True) + 1e-8
    destination_features = destination_features / dst_norms

    max_node_id = max(src_min + src_count - 1, dst_count - 1)
    features = np.zeros((max_node_id + 1, hidden_dim), dtype=np.float32)
    features[src_min:src_min + src_count] = source_features
    features[:dst_count] = destination_features
    if features.shape[0] > 0:
        features[0] = 0.0
    return features


# ============================================================================
# Evaluation (identical to train_jittor_v3.py)
# ============================================================================

def evaluate_mrr(model, source_values, destination_values, history_values, time_gap_values,
                 sample_indices, num_destinations, batch_size, seed,
                 recency_stats=None, time_values=None, cf_stats=None, cooc_stats=None):
    """Validation MRR with 99 random negatives.

    recency_stats/time_values: optional RecencyStats + per-row times (same
    indexing as source_values) enabling the 6-dim recency feature block.
    cf_stats: optional CFStats enabling the 8-dim CF block (concatenated
    AFTER the recency block, same order as the training loop).
    cooc_stats: optional CoocTable (approx co-occurrence, 3 dims AFTER the CF
    block); the per-row history matrix is used as H.
    """
    rng = np.random.default_rng(seed)
    reciprocal_ranks = []
    model.eval()

    for start in range(0, len(sample_indices), batch_size):
        indices = sample_indices[start:start + batch_size]
        positives = destination_values[indices]

        negatives = rng.integers(1, num_destinations, size=(len(indices), 99), dtype=np.int64)
        collisions = negatives == positives[:, np.newaxis]
        while collisions.any():
            negatives[collisions] = rng.integers(
                1, num_destinations, size=int(collisions.sum()), dtype=np.int64
            )
            collisions = negatives == positives[:, np.newaxis]

        candidates = np.concatenate([positives[:, np.newaxis], negatives], axis=1)
        sources = source_values[indices]
        histories = history_values[indices]
        time_gaps = time_gap_values[indices] if time_gap_values is not None else None
        extra = None
        blocks = []
        if recency_stats is not None:
            blocks.append(recency_stats.batch_features(sources, time_values[indices], candidates))
        if cf_stats is not None:
            blocks.append(cf_stats.batch_features(sources, candidates))
        if cooc_stats is not None:
            blocks.append(cooc_stats.batch_features(histories, candidates))
        if blocks:
            extra = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=2)

        with jt.no_grad():
            scores = model.execute(sources, histories, candidates, time_gaps,
                                   extra_feats=extra).numpy()

        ranks = 1 + (scores[:, 1:] >= scores[:, :1]).sum(axis=1)
        reciprocal_ranks.extend((1.0 / ranks).tolist())

    return float(np.mean(reciprocal_ranks))


def map_ids_to_dense(seen_ids, raw_ids, default):
    """Map raw ids to dense positions (index in seen_ids + 1); misses -> default."""
    pos = np.searchsorted(seen_ids, raw_ids)
    pos_clip = np.minimum(pos, len(seen_ids) - 1)
    found = seen_ids[pos_clip] == raw_ids
    return np.where(found, pos_clip + 1, default).astype(np.int64)


def rank_percentiles(raw_scores, candidates):
    """Convert raw scores to unique percentiles in [0.01, 1.00]."""
    result = np.empty_like(raw_scores, dtype=np.float64)
    for row_index in range(len(raw_scores)):
        order = np.lexsort((candidates[row_index], raw_scores[row_index]))
        result[row_index, order] = np.arange(1, raw_scores.shape[1] + 1, dtype=np.float64)
    return result / raw_scores.shape[1]


def check_embedding_weights(model):
    """Return True if any embedding weight is NaN/Inf."""
    if hasattr(model, 'src_emb'):
        w1 = model.src_emb.weight.numpy()
        if np.isnan(w1).any() or np.isinf(w1).any():
            return True
    if hasattr(model, 'dst_emb') and model.dst_emb is not model.src_emb:
        w2 = model.dst_emb.weight.numpy()
        if np.isnan(w2).any() or np.isinf(w2).any():
            return True
    return False


def verify_checkpoint_shapes(model, ckpt_path):
    """Strictly verify checkpoint parameter shapes against the constructed model.

    Jittor's Module.load only LOGS shape mismatches (no exception) and keeps
    mismatched parameters at their random init -- a silently corrupted run.
    This matters for --use_cf_feats: a CF-enabled scorer has a wider input
    than a v2 scorer, so loading a v2 checkpoint must fail loudly instead of
    silently re-initializing the scorer first layer. Raises RuntimeError on
    any mismatch.
    """
    saved = jt.load(ckpt_path)
    if not isinstance(saved, dict):
        raise RuntimeError(
            f'{ckpt_path}: expected a pickled state dict, got {type(saved).__name__}')
    current = model.state_dict()
    problems = []
    for key, value in current.items():
        if key not in saved:
            problems.append(f'    missing key in checkpoint: {key}')
            continue
        saved_shape = tuple(saved[key].shape) if hasattr(saved[key], 'shape') else None
        if saved_shape != tuple(value.shape):
            problems.append(f'    shape mismatch: {key} model={tuple(value.shape)} '
                            f'vs checkpoint={saved_shape}')
    if problems:
        raise RuntimeError(
            f'Checkpoint {ckpt_path} does not match the constructed model architecture '
            f'({len(problems)} problem(s)):\n' + '\n'.join(problems[:20]) +
            '\nCheck that --hidden_dim/--history_length/--num_layers/--mlp_ratio/'
            '--use_heuristics/--use_known_flag/--use_recency_feats/--use_cf_feats/'
            '--shared_nodes and --node_features match the run that produced the '
            'checkpoint. In particular, a checkpoint trained WITHOUT --use_cf_feats '
            'cannot initialize a CF-enabled model (the scorer input width differs); '
            'train from scratch instead.')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset1')
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--node_features', type=str, default=None)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--history_length', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_negatives', type=int, default=31)
    parser.add_argument('--val_samples', type=int, default=4000)
    parser.add_argument('--shared_nodes', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--init_checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    # GraphMixer-specific args
    parser.add_argument('--num_layers', type=int, default=2, help='Number of MLP-Mixer layers')
    parser.add_argument('--mlp_ratio', type=float, default=2.0, help='MLP hidden ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for listwise loss')
    parser.add_argument('--time_decay', type=float, default=0.5, help='Exponential decay factor for time weighting')
    parser.add_argument('--use_heuristics', action='store_true', help='Use heuristic features (degree, popularity, edge_count) in scorer')
    # dataset3 adaptation switches (all default to original behavior)
    parser.add_argument('--use_official_split', action='store_true',
                        help='Use split column in train.csv: split=0 -> train, split=1 -> val. '
                             'Falls back to 85%%/15%% temporal split when no split column exists.')
    parser.add_argument('--hist_pos_time_only', action='store_true',
                        help='Build history sequences and time-gap stats from time>0 edges only; '
                             'time=0 edges stay in heuristic stats and the Node2Vec graph.')
    parser.add_argument('--neg_mode', type=str, default='uniform', choices=['uniform', 'test_template'],
                        help='uniform: original random negatives. test_template: ~(1-unseen_ratio) seen-dst '
                             '(popularity-weighted) + ~unseen_ratio unseen-dst negatives, matching test candidate mix.')
    parser.add_argument('--unseen_ratio', type=float, default=0.09,
                        help='Fraction of negatives drawn from the unseen-dst pool in test_template mode.')
    parser.add_argument('--neg_freq_table', type=str, default=None,
                        help='Optional cand_freq_pairs.npz (raw_id + count, from test.csv candidate '
                             'columns). In test_template mode the SEEN negative pool is weighted by '
                             'these empirical candidate frequencies instead of train popularity; '
                             'pool ids absent from the table (weight 0) fall back to train '
                             'popularity. Mirrors eval_testlike --candidate_mode empirical.')
    parser.add_argument('--use_known_flag', action='store_true',
                        help='Append dst_known (candidate seen as dst in train) as an extra heuristic dim.')
    parser.add_argument('--use_recency_feats', action='store_true',
                        help='Append the 6-dim RecencyStats block (recent dst popularity, 7d/30d trend, '
                             'dst/pair/src last-seen gaps) after the heuristic block. Stats are built '
                             'from ALL positive-time edges of train.csv; features are computed strictly '
                             'before each row\'s own time (leak-free).')
    parser.add_argument('--use_cf_feats', action='store_true',
                        help='Append the CFStats block (item-item co-occurrence from the train '
                             'graph + n2v similarity of the candidate to the src history) after the '
                             'recency block. Built from the same edge set as RecencyStats (all train.csv '
                             'edges) but time-agnostic: time=0 edges are used too. Uses the dst-side '
                             'node features for the n2v similarity dims. NOTE: the scorer input width '
                             'changes, so CF-enabled models must train from scratch (no v2 init '
                             'checkpoint).')
    # dataset4 adaptation switches
    parser.add_argument('--dense_id_remap', action='store_true',
                        help='Remap sparse raw ids to dense model ids (seen dst -> 1..N, seen src -> '
                             '1..M, 0 = padding, N+1 = trainable UNK row for unseen dsts). Requires '
                             'non-shared mode. Cuts ds4 embedding+Adam VRAM from ~15.6 GB to ~4.7 GB.')
    parser.add_argument('--idmap', type=str, default=None,
                        help='Optional idmap_ds4.npz from gen_features_ds4.py; when given with '
                             '--dense_id_remap its seen id arrays are verified against train.csv.')
    parser.add_argument('--idmap_allow_subset', action='store_true',
                        help='Allow train.csv to cover only a subset of the idmap ids (recent-window '
                             'fine-tune). The idmap id sets become the canonical mapping so dense '
                             'feature tables built from the FULL train stay aligned.')
    parser.add_argument('--src_features', type=str, default=None,
                        help='Non-shared mode: per-src feature table .npy aligned to model id space.')
    parser.add_argument('--dst_features', type=str, default=None,
                        help='Non-shared mode: per-dst feature table .npy aligned to model id space.')
    parser.add_argument('--cf_no_cooc', action='store_true',
                        help='CFStats lite (use_cooc=False): skip the exact co-occurrence matrix; '
                             '4-dim CF block (mean/max/top5 n2v cosine + log hist length).')
    parser.add_argument('--use_cooc', action='store_true',
                        help='Append the 3-dim APPROXIMATE co-occurrence block (offline table '
                             'from build_cooc_ds4.py; hit_frac / max_w / wsum against the model '
                             'history matrix) AFTER the CF block. Stacks with --cf_no_cooc lite.')
    parser.add_argument('--cooc_table', type=str, default='./cooc_ds4.npz',
                        help='Path to cooc_ds4.npz (nbr_ids + nbr_w) for --use_cooc.')
    parser.add_argument('--hard_neg_ratio', type=float, default=0.0,
                        help='Fraction of SEEN negatives drawn from the positive dst top-K cos '
                             'neighbors (--hardnbr_table) instead of the popularity pool; '
                             'hard draws exclude the positive and the src history dsts. '
                             '0 (default) = old behavior. Requires --neg_mode test_template.')
    parser.add_argument('--hardnbr_table', type=str, default='./hardnbr_ds4.npy',
                        help='Path to hardnbr_ds4.npy (N x K int32 cos-neighbor table).')
    parser.add_argument('--max_train_batches', type=int, default=None,
                        help='Stop each epoch after N training batches (debug/smoke; '
                             'default: full epoch).')
    parser.add_argument('--heuristics_dir', type=str, default=None,
                        help='Directory holding heuristics_{dataset}_*.{npy,pkl} (default: data_dir).')
    parser.add_argument('--max_test_rows', type=int, default=None,
                        help='Score only the first N rows of test.csv (smoke tests; default: all).')
    parser.add_argument('--profile', action='store_true',
                        help='Print per-stage ms/batch + RSS every 200 training batches.')
    args = parser.parse_args()

    # Single seed source: --seed drives every random stream in this script.
    #   * python `random` and the numpy legacy API (any incidental use)
    #   * jittor global RNG: non-SVD weight init, dropout masks, CUDA RNG
    #   * training rng below (epoch shuffle + negative sampling) = default_rng(args.seed)
    #   * validation negatives: default_rng(args.seed + epoch)
    # Default 42 reproduces the historical behavior exactly. Jittor CUDA is
    # inherently non-deterministic, so equal seeds are not bit-reproducible,
    # but different seeds do produce different random streams (different
    # shuffles / negatives / init / dropout) for ensemble diversity.
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_seed(args.seed)
    print(f'Random seed: {args.seed} (python-random / numpy / jittor set; '
          f'train shuffle + negatives + val negatives derived from it)')

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    train_path = f'{args.data_dir}/{args.dataset}/train.csv'
    test_path = f'{args.data_dir}/{args.dataset}/test.csv'
    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    # RAM: candidate columns (col 2+) are node ids < 2**31; int32 halves the
    # test frame (~0.9GB saved over the whole run). Downcast via concat —
    # in-place iloc assignment would keep the int64 dtype.
    test_frame = pd.concat([test_frame.iloc[:, :2],
                            test_frame.iloc[:, 2:].astype(np.int32)], axis=1)

    source_ids = train_frame['src'].to_numpy(np.int64)
    destination_ids = train_frame['dst'].to_numpy(np.int64)
    # Read time column if available, else use row index as pseudo-timestamp
    if 'time' in train_frame.columns:
        time_values = train_frame['time'].to_numpy(np.float64)
    else:
        time_values = np.arange(len(train_frame), dtype=np.float64)
    # test_max_node_id is only needed to size the id space in shared/non-remap
    # modes; compute it lazily to avoid a ~1.9GB int64 temp copy of the
    # candidate block in dense_id_remap mode.
    test_max_node_id = None

    if args.shared_nodes:
        test_max_node_id = int(test_frame.iloc[:, 2:].to_numpy(np.int64).max())
        source_min = 0
        source_count = int(max(source_ids.max(), destination_ids.max(), test_max_node_id)) + 1
        destination_count = source_count
        remap = None
    elif args.dense_id_remap:
        # ds4 dense remap: sorted seen ids -> dense 1..N; 0 = padding;
        # N_dst+1 = UNK row for any dst that never appeared in train.
        seen_src = np.unique(source_ids)
        seen_dst = np.unique(destination_ids)
        unk_dst_id = len(seen_dst) + 1
        if args.idmap and os.path.exists(args.idmap):
            im = np.load(args.idmap)
            if getattr(args, 'idmap_allow_subset', False):
                # Subset mode (e.g. recent-window fine-tune): train.csv may cover
                # only a subset of the idmap ids; the idmap id sets are canonical
                # so dense feature tables (built from the FULL train) stay aligned.
                if not (np.isin(seen_src, im['seen_src_ids']).all()
                        and np.isin(seen_dst, im['seen_dst_ids']).all()):
                    raise RuntimeError(
                        f'{args.idmap} is not a superset of the train.csv id sets; '
                        'cannot use subset mode.')
                print(f'  idmap subset mode: train covers {len(seen_src)}/{len(im["seen_src_ids"])} src, '
                      f'{len(seen_dst)}/{len(im["seen_dst_ids"])} dst; using idmap id sets.')
                seen_src = im['seen_src_ids'].astype(np.int64)
                seen_dst = im['seen_dst_ids'].astype(np.int64)
                unk_dst_id = int(im['unk_dst_id'])
            elif not (np.array_equal(im['seen_src_ids'], seen_src)
                    and np.array_equal(im['seen_dst_ids'], seen_dst)
                    and int(im['unk_dst_id']) == unk_dst_id):
                raise RuntimeError(
                    f'{args.idmap} does not match the train.csv id sets; the dense '
                    'feature tables would be misaligned. Regenerate gen_features_ds4.')
            else:
                print(f'  idmap {args.idmap} verified against train.csv id sets.')
        source_ids = np.searchsorted(seen_src, source_ids).astype(np.int64) + 1
        destination_ids = np.searchsorted(seen_dst, destination_ids).astype(np.int64) + 1
        remap = (seen_src, seen_dst, unk_dst_id)
        source_min = 0
        source_count = len(seen_src) + 1
        destination_count = len(seen_dst) + 2
        print(f'  dense_id_remap: src->[1..{len(seen_src)}], dst->[1..{len(seen_dst)}], '
              f'UNK_DST={unk_dst_id} (padding=0)')
    else:
        test_max_node_id = int(test_frame.iloc[:, 2:].to_numpy(np.int64).max())
        source_min = int(source_ids.min())
        source_count = int(source_ids.max() - source_min + 1)
        destination_count = int(max(destination_ids.max(), test_max_node_id)) + 1
        remap = None

    has_split_col = 'split' in train_frame.columns
    if args.train_all:
        split_index = len(train_frame)
    elif args.use_official_split and has_split_col:
        split_values = train_frame['split'].to_numpy(np.int64)
        split_index = int((split_values == 0).sum())
        # The history builder requires train rows to be exactly the first
        # split_index rows so that val rows only see earlier edges.
        contiguous = bool((split_values[:split_index] == 0).all()
                          and (split_values[split_index:] != 0).all())
        if contiguous:
            print(f'  Using official split column: split=0 -> train ({split_index}), '
                  f'split=1 -> val ({len(train_frame) - split_index})')
        else:
            print('[WARNING] split column is not contiguous (split=0 rows are not a prefix); '
                  'falling back to 85%/15% temporal split.')
            split_index = len(train_frame) - int(len(train_frame) * 0.15)
    else:
        if args.use_official_split and not has_split_col:
            print('[WARNING] --use_official_split set but train.csv has no split column; '
                  'falling back to 85%/15% temporal split.')
        split_index = len(train_frame) - int(len(train_frame) * 0.15)

    print(f'Dataset: {args.dataset}, shared_nodes={args.shared_nodes}')
    print(f'  source_count={source_count}, destination_count={destination_count}')
    print(f'  train edges={split_index}, val edges={len(train_frame) - split_index}')
    print(f'  GraphMixer: layers={args.num_layers}, mlp_ratio={args.mlp_ratio}, dropout={args.dropout}')
    print(f'  neg_mode={args.neg_mode}, hist_pos_time_only={args.hist_pos_time_only}, '
          f'use_known_flag={args.use_known_flag}')

    # train_frame (~0.5GB) is no longer needed; everything downstream works on
    # the extracted numpy arrays. split_values is prelude-only too.
    del train_frame
    if 'split_values' in dir():
        del split_values
    import gc
    gc.collect()

    # ------------------------------------------------------------------
    # Build temporal histories with time gaps
    # ------------------------------------------------------------------
    train_histories, train_time_gaps, val_histories, val_time_gaps, full_histories, full_history_times, gap_scale = build_histories_with_time(
        source_ids, destination_ids, time_values, source_min, args.history_length, split_index,
        pos_time_only=args.hist_pos_time_only)
    print(f'  Time gap scale: {gap_scale:.2f}')

    # ------------------------------------------------------------------
    # Node features (per-tower tables, single shared table, or SVD fallback)
    # ------------------------------------------------------------------
    src_features = dst_features = None
    node_features = None
    if args.src_features or args.dst_features:
        if args.shared_nodes:
            raise ValueError('--src_features/--dst_features are for non-shared mode only.')
        if not (args.src_features and args.dst_features):
            raise ValueError('--src_features and --dst_features must be given together.')
        src_features = np.load(args.src_features).astype(np.float32)
        dst_features = np.load(args.dst_features).astype(np.float32)
        # Row 0 is the padding row and must be exactly zero (jittor's in-model
        # weight[0].update zeroing is a silent no-op; the caller owns this).
        src_features[0] = 0.0
        dst_features[0] = 0.0
        print(f'Loaded src features from {args.src_features}, shape={src_features.shape}')
        print(f'Loaded dst features from {args.dst_features}, shape={dst_features.shape}')
        if src_features.shape[0] < source_count or dst_features.shape[0] < destination_count:
            raise ValueError(
                f'feature tables smaller than the model id space: src '
                f'{src_features.shape[0]} < {source_count} or dst '
                f'{dst_features.shape[0]} < {destination_count}')
    elif args.node_features and os.path.exists(args.node_features):
        node_features = np.load(args.node_features).astype(np.float32)
        print(f'Loaded node features from {args.node_features}, shape={node_features.shape}')
    else:
        if args.shared_nodes:
            num_nodes = source_count
            print(f'Computing shared-node SVD features for {num_nodes} nodes...')
            node_features = compute_svd_features_shared(
                source_ids, destination_ids, split_index, num_nodes, args.hidden_dim,
                seed=args.seed)
        else:
            print(f'Computing bipartite SVD features: src={source_count}, dst={destination_count}...')
            node_features = compute_svd_features_bipartite(
                source_ids, destination_ids, split_index,
                source_min, source_count, destination_count, args.hidden_dim,
                seed=args.seed)
        if args.node_features:
            np.save(args.node_features, node_features)
            print(f'Saved SVD features to {args.node_features}')
        else:
            print('SVD features computed (not saved).')

    if node_features is not None and node_features.shape[0] > 0:
        node_features[0] = 0.0

    # ------------------------------------------------------------------
    # Heuristic features (optional)
    # ------------------------------------------------------------------
    heuristic_kwargs = {}
    if args.use_heuristics:
        heur_dir = args.heuristics_dir if args.heuristics_dir else args.data_dir
        degree_path = f'{heur_dir}/heuristics_{args.dataset}_degree.npy'
        popularity_path = f'{heur_dir}/heuristics_{args.dataset}_popularity.npy'
        edge_count_path = f'{heur_dir}/heuristics_{args.dataset}_edge_count.pkl'
        if os.path.exists(degree_path) and os.path.exists(popularity_path) and os.path.exists(edge_count_path):
            heuristic_degree = np.load(degree_path)
            heuristic_popularity = np.load(popularity_path)
            import pickle
            with open(edge_count_path, 'rb') as f:
                edge_count_dict, edge_count_max = pickle.load(f)
            heuristic_kwargs = {
                'heuristic_degree': heuristic_degree,
                'heuristic_popularity': heuristic_popularity,
                'heuristic_edge_count': edge_count_dict,
                'edge_count_max': edge_count_max,
            }
            ec_form = ('vectorized int64-key table' if isinstance(edge_count_dict, tuple)
                       else f'dict ({len(edge_count_dict)} pairs)')
            print(f'Loaded heuristic features from {heur_dir}: degree={heuristic_degree.shape}, '
                  f'edge_count={ec_form}, edge_count_max={edge_count_max:.0f}')
        else:
            print('[WARNING] --use_heuristics enabled but precomputed files not found.')
            print(f'  Run: python precompute_heuristics.py --dataset {args.dataset} '
                  f'--data_dir {args.data_dir} --out_dir {heur_dir}')
            print('  Falling back to no heuristics.')

    # dst_known flag (optional, independent of --use_heuristics):
    # 1 if candidate id appeared as dst in train.csv, else 0.
    if args.use_known_flag:
        dst_known = np.zeros(destination_count, dtype=bool)
        uniq_dst = np.unique(destination_ids)
        dst_known[uniq_dst[uniq_dst < destination_count]] = True
        heuristic_kwargs['use_known_flag'] = True
        heuristic_kwargs['heuristic_dst_known'] = dst_known
        print(f'dst_known flag enabled: {int(dst_known.sum())} known dst nodes out of {destination_count}')

    # Recency features (optional): 6-dim block from RecencyStats over the FULL
    # train.csv positive-time edges. Queries are strictly before each row's own
    # time, so training rows cannot see themselves or the future. src ids are
    # shifted into model space (source_min offset) for both build and query.
    recency_stats = None
    if args.use_recency_feats:
        from recency_features import RecencyStats
        recency_stats = RecencyStats(source_ids - source_min, destination_ids, time_values)
        print(f'Recency features enabled: {recency_stats.num_edges} positive-time edges indexed '
              f'(6 extra scorer dims).')

    # CF features (optional): 8-dim block from CFStats over the SAME edge set as
    # RecencyStats (full train.csv; CF is time-agnostic so time=0 edges are used
    # too). Same id-space convention as RecencyStats: src shifted by source_min
    # for build and query; dst/candidate ids are raw node ids (which also index
    # the node_features rows). Concatenated AFTER the recency block.
    cf_stats = None
    if args.use_cf_feats:
        from cf_features import CFStats
        cf_base = dst_features if dst_features is not None else node_features
        if cf_base is None:
            raise ValueError('--use_cf_feats requires dst-side node features.')
        cf_stats = CFStats(source_ids - source_min, destination_ids, cf_base,
                           use_cooc=not args.cf_no_cooc)
        print(f'CF features enabled: {cf_stats.num_edges} edges indexed, '
              f'cooc_nnz={cf_stats.cooc_nnz}, build={cf_stats.build_seconds:.1f}s '
              f'({cf_stats.dim} extra scorer dims, lite={args.cf_no_cooc}).')

    # Approximate co-occurrence block (optional, independent of CFStats): 3-dim
    # block from the OFFLINE-built neighbor table (build_cooc_ds4.py). Queries
    # use the model history matrix (B, history_length) as H, so the caller must
    # pass it through make_extra/evaluate_mrr. Concatenated AFTER the CF block.
    cooc_stats = None
    if args.use_cooc:
        from cf_features import CoocTable
        cooc_stats = CoocTable(args.cooc_table)
        print(f'Approx cooc features enabled: {args.cooc_table} '
              f'nbr_ids{cooc_stats.nbr_ids.shape} (3 extra scorer dims).')

    def make_extra(src_model, times, cands, hists=None):
        """Combined caller-supplied scorer block, fixed order
        [recency(6) | cf(4/8) | approx-cooc(3)].

        src_model: src ids in model space (shifted by source_min); cands: raw
        dst ids; hists: (B, history_length) history matrix (needed only when
        --use_cooc is on). Returns None when all feature groups are off.
        """
        blocks = []
        if recency_stats is not None:
            blocks.append(recency_stats.batch_features(src_model, times, cands))
        if cf_stats is not None:
            blocks.append(cf_stats.batch_features(src_model, cands))
        if cooc_stats is not None:
            blocks.append(cooc_stats.batch_features(hists, cands))
        if not blocks:
            return None
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=2)

    # ------------------------------------------------------------------
    # Model: Full GraphMixer
    # ------------------------------------------------------------------
    model = GraphMixerModel(
        src_count=source_count,
        dst_count=destination_count,
        hidden_dim=args.hidden_dim,
        initial_features=(dst_features if dst_features is not None else node_features),
        initial_src_features=src_features,
        initial_dst_features=dst_features,
        shared_nodes=args.shared_nodes,
        num_layers=args.num_layers,
        max_history_length=args.history_length,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        temperature=args.temperature,
        time_decay=args.time_decay,
        extra_feat_dim=((6 if recency_stats is not None else 0)
                        + (cf_stats.dim if cf_stats is not None else 0)
                        + (cooc_stats.dim if cooc_stats is not None else 0)),
        **heuristic_kwargs,
    )

    optimizer = nn.Adam(list(model.parameters()), lr=args.lr, weight_decay=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, f'{args.dataset}_graphmixer.pkl')

    # ------------------------------------------------------------------
    # Optional: load initial checkpoint for stage-2 fine-tuning
    # ------------------------------------------------------------------
    best_mrr = 0.0
    if args.init_checkpoint and os.path.exists(args.init_checkpoint):
        # Strict shape check first: Jittor's Module.load only LOGS shape
        # mismatches (silent random re-init), so a v2 (no-CF) checkpoint loaded
        # into a CF-enabled model would corrupt the scorer silently.
        verify_checkpoint_shapes(model, args.init_checkpoint)
        model.load(args.init_checkpoint)
        print(f'Loaded initial checkpoint: {args.init_checkpoint}')
        meta_path = args.init_checkpoint.replace('.pkl', '_meta.npy')
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True).item()
            best_mrr = meta.get('best_mrr', 0.0)
            print(f'  -> initial best_mrr: {best_mrr:.6f}')

    # ------------------------------------------------------------------
    # Validation setup
    # ------------------------------------------------------------------
    if not args.train_all:
        val_sources = source_ids[split_index:] - source_min
        val_destinations = destination_ids[split_index:]
        val_indices = np.linspace(0, len(val_sources) - 1, args.val_samples, dtype=np.int64)

    # ------------------------------------------------------------------
    # Negative sampling pools (test_template mode)
    # ------------------------------------------------------------------
    # seen pool  = node ids that appeared as dst in train.csv (popularity-weighted)
    # unseen pool = node ids in [1, destination_count) never seen as dst
    # Both pools exclude 0 (padding). Mixture ~ (1-unseen_ratio) / unseen_ratio
    # matches the test candidate template (~91 seen / ~9 unseen per row).
    neg_sampler = None
    if args.neg_freq_table and args.neg_mode != 'test_template':
        raise ValueError('--neg_freq_table requires --neg_mode test_template')
    if args.neg_mode == 'test_template':
        dst_counts = np.bincount(destination_ids, minlength=destination_count)
        seen_pool = np.nonzero(dst_counts)[0]
        seen_pool = seen_pool[seen_pool > 0]
        pop_weights = dst_counts[seen_pool].astype(np.float64)
        if args.neg_freq_table:
            # Weight the seen pool by the REAL test candidate distribution
            # (raw-id keyed frequency table) instead of train popularity.
            freq = np.load(args.neg_freq_table)
            f_ids = freq['raw_id'].astype(np.int64)
            f_cnt = freq['count'].astype(np.float64)
            if not (len(f_ids) and np.all(np.diff(f_ids) > 0)):
                raise ValueError(f'{args.neg_freq_table}: raw_id must be strictly increasing.')
            pool_raw = remap[1][seen_pool - 1] if remap is not None else seen_pool
            f_pos = np.minimum(np.searchsorted(f_ids, pool_raw), len(f_ids) - 1)
            hit = f_ids[f_pos] == pool_raw
            n_miss = int((~hit).sum())
            if n_miss:
                print(f'  [neg_freq_table] WARNING: {n_miss}/{len(seen_pool)} seen dst ids '
                      'absent from the frequency table; train popularity fallback.')
            freq_w = np.where(hit, f_cnt[f_pos], 0.0)
            seen_weights = np.where(freq_w > 0, freq_w, pop_weights)
            cov = float(freq_w.sum() / max(f_cnt.sum(), 1.0))
            print(f'  neg_freq_table {args.neg_freq_table}: weighted '
                  f'{int(hit.sum())}/{len(seen_pool)} seen pool ids '
                  f'(candidate mass coverage {cov:.4f}); zeros -> train popularity.')
        else:
            seen_weights = pop_weights
        seen_cdf = np.cumsum(seen_weights)
        seen_cdf /= seen_cdf[-1]
        unseen_mask = np.ones(destination_count, dtype=bool)
        unseen_mask[0] = False
        unseen_mask[seen_pool] = False
        unseen_pool = np.nonzero(unseen_mask)[0]
        print(f'test_template negatives: seen_pool={len(seen_pool)}, unseen_pool={len(unseen_pool)}, '
              f'unseen_ratio={args.unseen_ratio:.3f}')
        if len(seen_pool) == 0:
            raise ValueError('test_template neg_mode: seen dst pool is empty.')

        def neg_sampler(rng, n):
            """Draw n negatives from the test-template mixture."""
            out = np.empty(n, dtype=np.int64)
            if len(unseen_pool) > 0:
                use_unseen = rng.random(n) < args.unseen_ratio
            else:
                use_unseen = np.zeros(n, dtype=bool)
            n_seen = int((~use_unseen).sum())
            if n_seen:
                idx = np.searchsorted(seen_cdf, rng.random(n_seen))
                out[~use_unseen] = seen_pool[np.minimum(idx, len(seen_pool) - 1)]
            n_unseen = int(use_unseen.sum())
            if n_unseen:
                out[use_unseen] = unseen_pool[rng.integers(0, len(unseen_pool), size=n_unseen)]
            return out

    # cos-hard negatives (optional): hard_neg_ratio of the SEEN negatives are
    # drawn from the positive dst's top-K cos-neighbor table instead of the
    # popularity pool. Hard draws exclude the positive itself and every dst in
    # the src's (model-window) history; unseen ratio and UNK logic unchanged.
    neg_sampler_hard = None
    if args.hard_neg_ratio > 0.0:
        if args.neg_mode != 'test_template':
            raise ValueError('--hard_neg_ratio requires --neg_mode test_template')
        hard_table = np.load(args.hardnbr_table, mmap_mode='r')
        if hard_table.ndim != 2:
            raise ValueError(f'{args.hardnbr_table}: expected a 2-D (N, K) table')
        KH = hard_table.shape[1]
        print(f'cos-hard negatives enabled: ratio={args.hard_neg_ratio:.2f} of seen negs, '
              f'table {args.hardnbr_table} {hard_table.shape}')

        def _draw_pop(rng, k):
            idx = np.searchsorted(seen_cdf, rng.random(k))
            return seen_pool[np.minimum(idx, len(seen_pool) - 1)]

        def _draw_hard(rng, pos_sub):
            """Top-K cos-neighbor draws; padding-0 results (zero-norm rows)
            fall back to popularity draws."""
            v = hard_table[pos_sub, rng.integers(0, KH, size=len(pos_sub))]
            z = v == 0
            while z.any():
                v[z] = _draw_pop(rng, int(z.sum()))
                z = v == 0
            return v

        def neg_sampler_hard(rng, pos_arr, n_neg, hist_arr):
            """(B,) positives + (B, H) histories -> (B, n_neg) negatives.

            Per slot: unseen w.p. unseen_ratio; else hard w.p. hard_neg_ratio;
            else popularity. Collisions with the positive (all slots) and with
            the src history (hard slots only) are resampled.
            """
            B = len(pos_arr)
            n = B * n_neg
            out = np.empty(n, dtype=np.int64)
            if len(unseen_pool) > 0:
                use_unseen = rng.random(n) < args.unseen_ratio
            else:
                use_unseen = np.zeros(n, dtype=bool)
            is_seen = ~use_unseen
            is_hard = is_seen & (rng.random(n) < args.hard_neg_ratio)
            is_pop = is_seen & ~is_hard
            n_unseen = int(use_unseen.sum())
            if n_unseen:
                out[use_unseen] = unseen_pool[rng.integers(0, len(unseen_pool), size=n_unseen)]
            n_pop = int(is_pop.sum())
            if n_pop:
                out[is_pop] = _draw_pop(rng, n_pop)
            n_hard = int(is_hard.sum())
            if n_hard:
                out[is_hard] = _draw_hard(rng, np.repeat(pos_arr, n_neg)[is_hard])
            out = out.reshape(B, n_neg)
            is_hard = is_hard.reshape(B, n_neg)

            def _collisions(neg):
                m = neg == pos_arr[:, None]
                if is_hard.any():
                    m = m | (is_hard &
                             (neg[:, :, None] == hist_arr[:, None, :]).any(axis=2))
                return m

            col = _collisions(out)
            for _ in range(20):
                if not col.any():
                    break
                k = int(col.sum())
                pos_of = np.repeat(pos_arr, n_neg)[col.ravel()]
                hard_col = is_hard[col]
                new = np.empty(k, dtype=np.int64)
                nh = int(hard_col.sum())
                if nh:
                    new[hard_col] = _draw_hard(rng, pos_of[hard_col])
                if (~hard_col).any():
                    new[~hard_col] = neg_sampler(rng, int((~hard_col).sum()))
                out[col] = new
                col = _collisions(out)
            return out

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    global_step = 0

    # --profile: per-stage timing accumulators (seconds), printed every 200
    # batches with process RSS. Semantics are identical either way.
    import time as _time
    prof = {'gather': 0.0, 'neg': 0.0, 'recency': 0.0, 'cf': 0.0, 'cooc': 0.0,
            'model': 0.0, 'step': 0.0, 'batches': 0, 't0': _time.perf_counter()}

    def _prof_report(epoch, process=None):
        nb = max(prof['batches'], 1)
        total = (_time.perf_counter() - prof['t0'])
        acc = sum(prof[k] for k in ('gather', 'neg', 'recency', 'cf', 'cooc', 'model', 'step'))
        rss = ''
        if process is not None:
            rss = f', RSS={process.memory_info().rss / 2**30:.2f}GB'
        print(f'[profile ep{epoch + 1}] ms/batch over {nb} batches: '
              f'gather={1000 * prof["gather"] / nb:.0f}, neg={1000 * prof["neg"] / nb:.0f}, '
              f'recency={1000 * prof["recency"] / nb:.0f}, cf={1000 * prof["cf"] / nb:.0f}, '
              f'cooc={1000 * prof["cooc"] / nb:.0f}, '
              f'model={1000 * prof["model"] / nb:.0f}, step={1000 * prof["step"] / nb:.0f} '
              f'(accounted {1000 * acc / nb:.0f} / wall {1000 * total / nb:.0f}){rss}')
        for k in ('gather', 'neg', 'recency', 'cf', 'cooc', 'model', 'step'):
            prof[k] = 0.0
        prof['batches'] = 0
        prof['t0'] = _time.perf_counter()

    _process = None
    if args.profile:
        import psutil
        _process = psutil.Process()

    for epoch in range(args.epochs):
        permutation = rng.permutation(split_index)
        model.train()
        loss_values = []
        skip_count = 0
        batch_count = 0

        pbar = tqdm(range(0, len(permutation), args.batch_size),
                    total=(len(permutation) + args.batch_size - 1) // args.batch_size,
                    ncols=120, desc=f'Epoch {epoch + 1}')

        for start in pbar:
            if args.max_train_batches is not None and batch_count >= args.max_train_batches:
                print(f'  [max_train_batches] stopping epoch early at {batch_count} batches')
                break
            batch_count += 1
            _t = _time.perf_counter() if args.profile else None
            indices = permutation[start:start + args.batch_size]
            sources = source_ids[indices] - source_min
            positives = destination_ids[indices]
            histories = train_histories[indices]
            time_gaps = train_time_gaps[indices]
            if args.profile:
                prof['gather'] += _time.perf_counter() - _t
                _t = _time.perf_counter()

            # Negatives (avoid 0 = padding and collision with positive)
            if neg_sampler_hard is not None:
                # cos-hard mode does its own positive/history collision handling
                negatives = neg_sampler_hard(rng, positives, args.num_negatives, histories)
            else:
                if neg_sampler is not None:
                    negatives = neg_sampler(rng, len(indices) * args.num_negatives).reshape(
                        len(indices), args.num_negatives)
                else:
                    negatives = rng.integers(
                        1, destination_count, size=(len(indices), args.num_negatives), dtype=np.int64
                    )
                collisions = negatives == positives[:, np.newaxis]
                while collisions.any():
                    if neg_sampler is not None:
                        negatives[collisions] = neg_sampler(rng, int(collisions.sum()))
                    else:
                        negatives[collisions] = rng.integers(
                            1, destination_count, size=int(collisions.sum()), dtype=np.int64
                        )
                    collisions = negatives == positives[:, np.newaxis]

            candidates = np.concatenate([positives[:, np.newaxis], negatives], axis=1)
            if args.profile:
                prof['neg'] += _time.perf_counter() - _t
                _t = _time.perf_counter()

            # Recency/CF feature block for [pos | negs] at this row's time
            # (recency part is leak-free; CF is time-agnostic by design)
            if args.profile:
                blocks = []
                if recency_stats is not None:
                    blocks.append(recency_stats.batch_features(
                        sources, time_values[indices], candidates))
                prof['recency'] += _time.perf_counter() - _t
                _t = _time.perf_counter()
                if cf_stats is not None:
                    blocks.append(cf_stats.batch_features(sources, candidates))
                prof['cf'] += _time.perf_counter() - _t
                _t = _time.perf_counter()
                if cooc_stats is not None:
                    blocks.append(cooc_stats.batch_features(histories, candidates))
                prof['cooc'] += _time.perf_counter() - _t
                _t = _time.perf_counter()
                extra = (None if not blocks else
                         (blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=2)))
            else:
                extra = make_extra(sources, time_values[indices], candidates,
                                   hists=histories)

            # Forward + loss
            loss, loss_dict = model.calculate_loss(sources, histories, candidates, time_gaps,
                                                   extra_feats=extra)

            # NaN/Inf guard: skip bad batch
            loss_val = float(loss.item())
            if args.profile:
                prof['model'] += _time.perf_counter() - _t
                _t = _time.perf_counter()
            if np.isnan(loss_val) or np.isinf(loss_val):
                skip_count += 1
                continue

            # Stable Jittor pattern: no manual grad clip
            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()
            if args.profile:
                prof['step'] += _time.perf_counter() - _t
                prof['batches'] += 1
                if prof['batches'] % 200 == 0:
                    _prof_report(epoch, _process)

            loss_values.append(loss_val)
            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'lw': f'{loss_dict["listwise"]:.4f}',
                'bpr': f'{loss_dict["bpr"]:.4f}'
            })
            global_step += 1

        if args.profile and prof['batches'] % 200 != 0:
            _prof_report(epoch, _process)

        avg_loss = float(np.mean(loss_values)) if loss_values else 0.0

        if skip_count > 0:
            print(f'  -> Skipped {skip_count} bad batches this epoch')

        # CRITICAL: check embedding weights after each epoch
        if check_embedding_weights(model):
            print('  -> [WARNING] NaN/Inf detected in embedding weights after epoch!')
            print('  -> Reduce learning rate or check for gradient explosion.')

        if args.train_all:
            model.save(checkpoint_path)
            meta_path = checkpoint_path.replace('.pkl', '_meta.npy')
            np.save(meta_path, {'best_mrr': best_mrr})
            snap_path = checkpoint_path.replace('.pkl', f'_epoch{epoch + 1}.pkl')
            model.save(snap_path)
            print(f'Epoch {epoch + 1}, loss={avg_loss:.4f}, saved full-train model '
                  f'(+ snapshot {snap_path}).')
        else:
            mrr = evaluate_mrr(
                model, val_sources, val_destinations, val_histories, val_time_gaps,
                val_indices, destination_count, args.batch_size, args.seed + epoch,
                recency_stats=recency_stats,
                time_values=time_values[split_index:] if recency_stats is not None else None,
                cf_stats=cf_stats,
                cooc_stats=cooc_stats,
            )
            print(f'Epoch {epoch + 1}, loss={avg_loss:.4f}, val_mrr={mrr:.6f}')
            if mrr > best_mrr:
                best_mrr = mrr
                model.save(checkpoint_path)
                meta_path = checkpoint_path.replace('.pkl', '_meta.npy')
                np.save(meta_path, {'best_mrr': best_mrr})
                print(f'  -> New best MRR: {best_mrr:.6f}, saved.')

    # ------------------------------------------------------------------
    # Test inference
    # ------------------------------------------------------------------
    # Free the big train-time arrays BEFORE the checkpoint reload: jittor's
    # model.load stages every parameter through a pinned host buffer
    # (cudaMallocHost), which fails when the 32 GB host already holds ~16 GB
    # of feature tables + history blocks (measured: RSS 15.9 GB ->
    # cudaErrorMemoryAllocation on the 0.88 GB dst embedding param). The four
    # history arrays (~5.2 GB) and the hard-negative table (~0.9 GB) are dead
    # after the epoch loop; test histories come from full_histories below.
    del train_histories, train_time_gaps, val_histories, val_time_gaps
    if args.hard_neg_ratio > 0.0:
        neg_sampler_hard = None
        del hard_table
    import gc
    gc.collect()

    print('\nGenerating test predictions...')
    if not args.train_all and os.path.exists(checkpoint_path):
        try:
            model.load(checkpoint_path)
        except RuntimeError as exc:
            raise RuntimeError(
                f'Checkpoint load failed (pinned-host staging?): {exc}\n'
                'Rerun with GM_PAGEABLE_HOST=1 set in the environment '
                '(pageable H2D; required for ds4 v2 scale).') from exc
        print(f'Loaded best checkpoint (MRR={best_mrr:.6f})')

    if args.max_test_rows:
        test_frame = test_frame.iloc[:args.max_test_rows]
        print(f'[max_test_rows] scoring only the first {len(test_frame)} test rows')

    test_sources_global = test_frame['src'].to_numpy(np.int64)
    test_candidates_raw = test_frame.iloc[:, 2:].to_numpy(np.int64)
    if remap is not None:
        seen_src_arr, seen_dst_arr, unk_dst = remap
        test_sources = map_ids_to_dense(seen_src_arr, test_sources_global, 0)
        test_candidates = map_ids_to_dense(seen_dst_arr, test_candidates_raw, unk_dst)
        n_unk = int((test_candidates == unk_dst).sum())
        print(f'  dense remap applied: UNK candidate positions '
              f'{n_unk}/{test_candidates.size} ({100.0 * n_unk / test_candidates.size:.2f}%)')
    else:
        test_sources = test_sources_global - source_min
        test_candidates = test_candidates_raw

    test_histories = np.zeros((len(test_frame), args.history_length), dtype=np.int32)
    test_time_gaps = np.zeros((len(test_frame), args.history_length), dtype=np.float32)
    # Read test time column if available
    if 'time' in test_frame.columns:
        test_time_values = test_frame['time'].to_numpy(np.float64)
    else:
        test_time_values = np.zeros(len(test_frame), dtype=np.float64)
    for index, (source, time) in enumerate(zip(test_sources, test_time_values)):
        history = full_histories[int(source)]
        hist_times = full_history_times[int(source)]
        if history:
            test_histories[index, -len(history):] = history
            gaps = [np.log1p(max(float(time) - float(t), 0.0)) / gap_scale for t in hist_times]
            test_time_gaps[index, -len(gaps):] = gaps

    predictions = []
    model.eval()
    for start in tqdm(range(0, len(test_sources), args.batch_size), ncols=120, desc='Testing'):
        stop = min(start + args.batch_size, len(test_sources))
        extra = make_extra(
            test_sources[start:stop],
            test_time_values[start:stop],
            test_candidates[start:stop],
            hists=test_histories[start:stop],
        )
        with jt.no_grad():
            scores = model.execute(
                test_sources[start:stop],
                test_histories[start:stop],
                test_candidates[start:stop],
                test_time_gaps[start:stop],
                extra_feats=extra,
            ).numpy()
        if np.isnan(scores).any():
            print(f'[WARNING] NaN scores in test rows [{start}:{stop})')
        # Rank ties are broken by the RAW candidate ids (submission semantics).
        predictions.append(rank_percentiles(scores, test_candidates_raw[start:stop]))

    output = np.concatenate(predictions, axis=0)
    output_path = os.path.join(args.save_dir, f'{args.dataset}.csv')
    np.savetxt(output_path, output, delimiter=',', fmt='%.8f')

    print(f'\nSaved: {output_path}, shape={output.shape}, range=[{output.min():.4f}, {output.max():.4f}]')
    print(f'Best validation MRR: {best_mrr:.6f}')


if __name__ == '__main__':
    main()
