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
"""
import argparse
import os
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

def build_histories_with_time(source_ids, destination_ids, time_values, source_min, history_length, split_index):
    """Build history sequences with time gaps."""
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


def compute_svd_features_shared(source_ids, destination_ids, split_index, num_nodes, hidden_dim):
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

    svd = TruncatedSVD(n_components=hidden_dim, random_state=42)
    features = svd.fit_transform(A)

    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features = features / norms
    if features.shape[0] > 0:
        features[0] = 0.0
    return features.astype(np.float32)


def compute_svd_features_bipartite(source_ids, destination_ids, split_index,
                                    src_min, src_count, dst_count, hidden_dim):
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

    svd = TruncatedSVD(n_components=hidden_dim, random_state=42)
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
                 sample_indices, num_destinations, batch_size, seed):
    """Validation MRR with 99 random negatives."""
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

        with jt.no_grad():
            scores = model.execute(sources, histories, candidates, time_gaps).numpy()

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
    args = parser.parse_args()

    np.random.seed(args.seed)

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

    split_index = len(train_frame) if args.train_all else len(train_frame) - int(len(train_frame) * 0.15)

    print(f'Dataset: {args.dataset}, shared_nodes={args.shared_nodes}')
    print(f'  source_count={source_count}, destination_count={destination_count}')
    print(f'  train edges={split_index}, val edges={len(train_frame) - split_index}')
    print(f'  GraphMixer: layers={args.num_layers}, mlp_ratio={args.mlp_ratio}, dropout={args.dropout}')

    # ------------------------------------------------------------------
    # Build temporal histories with time gaps
    # ------------------------------------------------------------------
    train_histories, train_time_gaps, val_histories, val_time_gaps, full_histories, full_history_times, gap_scale = build_histories_with_time(
        source_ids, destination_ids, time_values, source_min, args.history_length, split_index)
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
                source_ids, destination_ids, split_index, num_nodes, args.hidden_dim)
        else:
            print(f'Computing bipartite SVD features: src={source_count}, dst={destination_count}...')
            node_features = compute_svd_features_bipartite(
                source_ids, destination_ids, split_index,
                source_min, source_count, destination_count, args.hidden_dim)
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

            # Random negatives (avoid 0 = padding)
            negatives = rng.integers(
                1, destination_count, size=(len(indices), args.num_negatives), dtype=np.int64
            )
            collisions = negatives == positives[:, np.newaxis]
            while collisions.any():
                negatives[collisions] = rng.integers(
                    1, destination_count, size=int(collisions.sum()), dtype=np.int64
                )
                collisions = negatives == positives[:, np.newaxis]

            candidates = np.concatenate([positives[:, np.newaxis], negatives], axis=1)

            # Forward + loss
            loss, loss_dict = model.calculate_loss(sources, histories, candidates, time_gaps)

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
                val_indices, destination_count, args.batch_size, args.seed + epoch
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
        with jt.no_grad():
            scores = model.execute(
                test_sources[start:stop],
                test_histories[start:stop],
                test_candidates[start:stop],
                test_time_gaps[start:stop],
            ).numpy()
        predictions.append(rank_percentiles(scores, test_candidates[start:stop]))

    output = np.concatenate(predictions, axis=0)
    output_path = os.path.join(args.save_dir, f'{args.dataset}.csv')
    np.savetxt(output_path, output, delimiter=',', fmt='%.8f')

    print(f'\nSaved: {output_path}, shape={output.shape}, range=[{output.min():.4f}, {output.max():.4f}]')
    print(f'Best validation MRR: {best_mrr:.6f}')


if __name__ == '__main__':
    main()
