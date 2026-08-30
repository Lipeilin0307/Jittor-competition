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
"""
import argparse
import os
import random
from collections import deque

import numpy as np
import pandas as pd
from tqdm import tqdm

import jittor as jt
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
    train_histories = np.zeros((split_index, history_length), dtype=np.int64)
    train_time_gaps = np.zeros((split_index, history_length), dtype=np.float32)
    val_histories = np.zeros((len(source_ids) - split_index, history_length), dtype=np.int64)
    val_time_gaps = np.zeros((len(source_ids) - split_index, history_length), dtype=np.float32)
    histories = [deque(maxlen=history_length) for _ in range(num_sources)]
    history_times = [deque(maxlen=history_length) for _ in range(num_sources)]

    # Compute global gap scale (90th percentile of time gaps)
    time_diffs = []
    for i in range(1, len(time_values)):
        if pos_time_only and (time_values[i] <= 0.0 or time_values[i - 1] <= 0.0):
            continue  # time=0 edges excluded from gap normalization stats
        if time_values[i] > time_values[i-1]:
            time_diffs.append(float(time_values[i] - time_values[i-1]))
    gap_scale = np.percentile(time_diffs, 90) if time_diffs else 1.0
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
                 recency_stats=None, time_values=None, cf_stats=None):
    """Validation MRR with 99 random negatives.

    recency_stats/time_values: optional RecencyStats + per-row times (same
    indexing as source_values) enabling the 6-dim recency feature block.
    cf_stats: optional CFStats enabling the 8-dim CF block (concatenated
    AFTER the recency block, same order as the training loop).
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
        if blocks:
            extra = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=2)

        with jt.no_grad():
            scores = model.execute(sources, histories, candidates, time_gaps,
                                   extra_feats=extra).numpy()

        ranks = 1 + (scores[:, 1:] >= scores[:, :1]).sum(axis=1)
        reciprocal_ranks.extend((1.0 / ranks).tolist())

    return float(np.mean(reciprocal_ranks))


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
    parser.add_argument('--neg_mode', type=str, default='uniform', choices=['uniform', 'test_template', 'self_hard'],
                        help='uniform: original random negatives. test_template: ~(1-unseen_ratio) seen-dst '
                             '(popularity-weighted) + ~unseen_ratio unseen-dst negatives, matching test candidate mix. '
                             'self_hard: per-row mixture of --hardneg_ratio mined hard negatives from --hardneg_pool '
                             '(self-error mining, see mine_hardneg_d3.py) + the rest from the test_template pools; '
                             'rows without a pool entry fall back to test_template entirely.')
    parser.add_argument('--unseen_ratio', type=float, default=0.09,
                        help='Fraction of negatives drawn from the unseen-dst pool in test_template mode.')
    parser.add_argument('--hardneg_pool', type=str, default=None,
                        help='Path to hardneg_pool npz (src_ids in MODEL space + neg_ids [n_src, N] raw dst ids, '
                             'zero-padded) produced by mine_hardneg_d3.py. Required when --neg_mode self_hard.')
    parser.add_argument('--hardneg_ratio', type=float, default=0.5,
                        help='In self_hard mode, per training row round(num_negatives * hardneg_ratio) negatives '
                             'are drawn from that row\'s mined pool (with replacement, over its non-padded entries); '
                             'the remainder come from the test_template mixture.')
    parser.add_argument('--use_known_flag', action='store_true',
                        help='Append dst_known (candidate seen as dst in train) as an extra heuristic dim.')
    parser.add_argument('--use_recency_feats', action='store_true',
                        help='Append the 6-dim RecencyStats block (recent dst popularity, 7d/30d trend, '
                             'dst/pair/src last-seen gaps) after the heuristic block. Stats are built '
                             'from ALL positive-time edges of train.csv; features are computed strictly '
                             'before each row\'s own time (leak-free).')
    parser.add_argument('--use_cf_feats', action='store_true',
                        help='Append the 8-dim CFStats block (item-item co-occurrence from the train '
                             'graph + n2v similarity of the candidate to the src history) after the '
                             'recency block. Built from the same edge set as RecencyStats (all train.csv '
                             'edges) but time-agnostic: time=0 edges are used too. Uses --node_features '
                             'for the n2v similarity dims. NOTE: the scorer input width changes, so '
                             'CF-enabled models must train from scratch (no v2 init checkpoint).')
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

    source_ids = train_frame['src'].to_numpy(np.int64)
    destination_ids = train_frame['dst'].to_numpy(np.int64)
    # Read time column if available, else use row index as pseudo-timestamp
    if 'time' in train_frame.columns:
        time_values = train_frame['time'].to_numpy(np.float64)
    else:
        time_values = np.arange(len(train_frame), dtype=np.float64)
    test_max_node_id = int(test_frame.iloc[:, 2:].to_numpy(np.int64).max())

    if args.shared_nodes:
        source_min = 0
        source_count = int(max(source_ids.max(), destination_ids.max(), test_max_node_id)) + 1
        destination_count = source_count
    else:
        source_min = int(source_ids.min())
        source_count = int(source_ids.max() - source_min + 1)
        destination_count = int(max(destination_ids.max(), test_max_node_id)) + 1

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

    # ------------------------------------------------------------------
    # Build temporal histories with time gaps
    # ------------------------------------------------------------------
    train_histories, train_time_gaps, val_histories, val_time_gaps, full_histories, full_history_times, gap_scale = build_histories_with_time(
        source_ids, destination_ids, time_values, source_min, args.history_length, split_index,
        pos_time_only=args.hist_pos_time_only)
    print(f'  Time gap scale: {gap_scale:.2f}')

    # ------------------------------------------------------------------
    # Node features (SVD or provided)
    # ------------------------------------------------------------------
    if args.node_features and os.path.exists(args.node_features):
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

    if node_features.shape[0] > 0:
        node_features[0] = 0.0

    # ------------------------------------------------------------------
    # Heuristic features (optional)
    # ------------------------------------------------------------------
    heuristic_kwargs = {}
    if args.use_heuristics:
        degree_path = f'{args.data_dir}/heuristics_{args.dataset}_degree.npy'
        popularity_path = f'{args.data_dir}/heuristics_{args.dataset}_popularity.npy'
        edge_count_path = f'{args.data_dir}/heuristics_{args.dataset}_edge_count.pkl'
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
            print(f'Loaded heuristic features: degree={heuristic_degree.shape}, edge_count_max={edge_count_max:.0f}')
        else:
            print('[WARNING] --use_heuristics enabled but precomputed files not found.')
            print(f'  Run: python precompute_heuristics.py --dataset {args.dataset} --data_dir {args.data_dir}')
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
        cf_stats = CFStats(source_ids - source_min, destination_ids, node_features)
        print(f'CF features enabled: {cf_stats.num_edges} edges indexed, '
              f'cooc_nnz={cf_stats.cooc_nnz}, build={cf_stats.build_seconds:.1f}s '
              f'(8 extra scorer dims).')

    def make_extra(src_model, times, cands):
        """Combined caller-supplied scorer block, fixed order [recency(6) | cf(8)].

        src_model: src ids in model space (shifted by source_min); cands: raw
        dst ids. Returns None when both feature groups are off (v2 behavior).
        """
        blocks = []
        if recency_stats is not None:
            blocks.append(recency_stats.batch_features(src_model, times, cands))
        if cf_stats is not None:
            blocks.append(cf_stats.batch_features(src_model, cands))
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
        initial_features=node_features,
        shared_nodes=args.shared_nodes,
        num_layers=args.num_layers,
        max_history_length=args.history_length,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        temperature=args.temperature,
        time_decay=args.time_decay,
        extra_feat_dim=((6 if recency_stats is not None else 0)
                        + (8 if cf_stats is not None else 0)),
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
    if args.neg_mode in ('test_template', 'self_hard'):
        dst_counts = np.bincount(destination_ids, minlength=destination_count)
        seen_pool = np.nonzero(dst_counts)[0]
        seen_pool = seen_pool[seen_pool > 0]
        seen_cdf = np.cumsum(dst_counts[seen_pool].astype(np.float64))
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

    # ------------------------------------------------------------------
    # Self-error-mined hard negatives (self_hard mode only). The npz comes
    # from mine_hardneg_d3.py: src_ids are in MODEL space (raw - source_min;
    # for ds3 --shared_nodes source_min=0), neg_ids are raw dst ids,
    # zero-padded. Per-row mixture: round(num_negatives * hardneg_ratio) draws
    # from the row's mined pool, the rest from the test_template sampler;
    # rows without a pool entry fall back to test_template for ALL slots.
    # Collision/padding handling afterwards is unchanged (collided slots are
    # redrawn from the test_template pools by the existing loop; mined rows
    # store only non-zero dst ids in their first hardneg_len entries, so 0 is
    # never emitted).
    # ------------------------------------------------------------------
    neg_sampler_self_hard = None
    if args.neg_mode == 'self_hard':
        if not args.hardneg_pool or not os.path.exists(args.hardneg_pool):
            raise FileNotFoundError(
                f'--neg_mode self_hard requires --hardneg_pool (got {args.hardneg_pool}). '
                'Run mine_hardneg_d3.py first.')
        hn = np.load(args.hardneg_pool)
        hardneg_src = hn['src_ids'].astype(np.int64)
        hardneg_neg = hn['neg_ids'].astype(np.int64)
        if hardneg_neg.ndim != 2 or len(hardneg_src) != hardneg_neg.shape[0]:
            raise ValueError(f'{args.hardneg_pool}: expected src_ids (n,) and neg_ids (n, N), '
                             f'got {hardneg_src.shape} vs {hardneg_neg.shape}')
        if len(hardneg_src) > 1 and (np.diff(hardneg_src) <= 0).any():
            raise ValueError(f'{args.hardneg_pool}: src_ids must be strictly increasing '
                             '(mine_hardneg_d3.py emits them sorted).')
        hardneg_len = (hardneg_neg != 0).sum(axis=1)
        n_hard_per_row = int(round(args.num_negatives * args.hardneg_ratio))
        n_hard_per_row = min(max(n_hard_per_row, 0), args.num_negatives)
        train_src_model = source_ids[:split_index] - source_min
        covered = int(np.isin(train_src_model, hardneg_src).sum()) if len(hardneg_src) else 0
        print(f'self_hard negatives: pool={args.hardneg_pool} ({len(hardneg_src)} srcs, '
              f'top-{hardneg_neg.shape[1]}), hardneg_ratio={args.hardneg_ratio} '
              f'-> {n_hard_per_row}/{args.num_negatives} hard per row; '
              f'train-row pool coverage {covered}/{split_index} '
              f'({100.0 * covered / max(split_index, 1):.1f}%)')
        del train_src_model

        def neg_sampler_self_hard(rng, src_model_arr, n_neg):
            """Row-wise negatives: (B, n_neg) int64. Found rows draw their first
            n_hard_per_row slots (with replacement) from their mined pool's
            non-padded entries; every remaining slot (and all slots of rows
            without a usable pool) comes from the test_template mixture."""
            B = len(src_model_arr)
            out = np.empty((B, n_neg), dtype=np.int64)
            template_mask = np.ones((B, n_neg), dtype=bool)
            if n_hard_per_row > 0 and len(hardneg_src):
                pos = np.searchsorted(hardneg_src, src_model_arr)
                pos = np.minimum(pos, len(hardneg_src) - 1)
                found = hardneg_src[pos] == src_model_arr
                usable = found & (hardneg_len[pos] > 0)
                if usable.any():
                    rows = np.nonzero(usable)[0]
                    lens = hardneg_len[pos[rows]]                    # (F,)
                    draw = rng.integers(0, lens[:, np.newaxis],
                                        size=(len(rows), n_hard_per_row))
                    out[rows[:, np.newaxis], np.arange(n_hard_per_row)] = \
                        hardneg_neg[pos[rows][:, np.newaxis], draw]
                    template_mask[rows, :n_hard_per_row] = False
            n_template = int(template_mask.sum())
            if n_template:
                out[template_mask] = neg_sampler(rng, n_template)
            return out

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    global_step = 0

    for epoch in range(args.epochs):
        permutation = rng.permutation(split_index)
        model.train()
        loss_values = []
        skip_count = 0

        pbar = tqdm(range(0, len(permutation), args.batch_size),
                    total=(len(permutation) + args.batch_size - 1) // args.batch_size,
                    ncols=120, desc=f'Epoch {epoch + 1}')

        for start in pbar:
            indices = permutation[start:start + args.batch_size]
            sources = source_ids[indices] - source_min
            positives = destination_ids[indices]
            histories = train_histories[indices]
            time_gaps = train_time_gaps[indices]

            # Negatives (avoid 0 = padding and collision with positive)
            if args.neg_mode == 'self_hard':
                negatives = neg_sampler_self_hard(rng, sources, args.num_negatives)
            elif neg_sampler is not None:
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

            # Recency/CF feature block for [pos | negs] at this row's time
            # (recency part is leak-free; CF is time-agnostic by design)
            extra = make_extra(sources, time_values[indices], candidates)

            # Forward + loss
            loss, loss_dict = model.calculate_loss(sources, histories, candidates, time_gaps,
                                                   extra_feats=extra)

            # NaN/Inf guard: skip bad batch
            loss_val = float(loss.item())
            if np.isnan(loss_val) or np.isinf(loss_val):
                skip_count += 1
                continue

            # Stable Jittor pattern: no manual grad clip
            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()

            loss_values.append(loss_val)
            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'lw': f'{loss_dict["listwise"]:.4f}',
                'bpr': f'{loss_dict["bpr"]:.4f}'
            })
            global_step += 1

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
            print(f'Epoch {epoch + 1}, loss={avg_loss:.4f}, saved full-train model.')
        else:
            mrr = evaluate_mrr(
                model, val_sources, val_destinations, val_histories, val_time_gaps,
                val_indices, destination_count, args.batch_size, args.seed + epoch,
                recency_stats=recency_stats,
                time_values=time_values[split_index:] if recency_stats is not None else None,
                cf_stats=cf_stats,
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
    print('\nGenerating test predictions...')
    if not args.train_all and os.path.exists(checkpoint_path):
        model.load(checkpoint_path)
        print(f'Loaded best checkpoint (MRR={best_mrr:.6f})')

    test_sources_global = test_frame['src'].to_numpy(np.int64)
    test_sources = test_sources_global - source_min
    test_candidates = test_frame.iloc[:, 2:].to_numpy(np.int64)

    test_histories = np.zeros((len(test_frame), args.history_length), dtype=np.int64)
    test_time_gaps = np.zeros((len(test_frame), args.history_length), dtype=np.float32)
    # Read test time column if available
    if 'time' in test_frame.columns:
        test_time_values = test_frame['time'].to_numpy(np.float64)
    else:
        test_time_values = np.zeros(len(test_frame), dtype=np.float64)
    for index, (source, time) in enumerate(zip(test_sources_global, test_time_values)):
        history = full_histories[int(source - source_min)]
        hist_times = full_history_times[int(source - source_min)]
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
        )
        with jt.no_grad():
            scores = model.execute(
                test_sources[start:stop],
                test_histories[start:stop],
                test_candidates[start:stop],
                test_time_gaps[start:stop],
                extra_feats=extra,
            ).numpy()
        predictions.append(rank_percentiles(scores, test_candidates[start:stop]))

    output = np.concatenate(predictions, axis=0)
    output_path = os.path.join(args.save_dir, f'{args.dataset}.csv')
    np.savetxt(output_path, output, delimiter=',', fmt='%.8f')

    print(f'\nSaved: {output_path}, shape={output.shape}, range=[{output.min():.4f}, {output.max():.4f}]')
    print(f'Best validation MRR: {best_mrr:.6f}')


if __name__ == '__main__':
    main()
