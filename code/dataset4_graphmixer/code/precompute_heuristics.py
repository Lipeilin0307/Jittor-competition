"""
precompute_heuristics.py
========================
Precompute heuristic features for link prediction:
  1. degree[candidate]     - candidate node degree (as dst)
  2. popularity[candidate] - candidate popularity (normalized occurrence)
  3. edge_count[src, dst]  - number of historical edges between src and dst

ds4 changes (all default OFF = ds3 behavior):
  --dense_id_remap   remap ids exactly like train_graphmixer_jt.py (seen dst ->
                     1..N_dst, seen src -> 1..N_src, 0 = padding, N_dst+1 =
                     UNK with zero stats). Heuristic arrays are indexed by the
                     DENSE id space.
  --out_dir DIR      where to write heuristics_{dataset}_* files (default:
                     --data_dir; ds4's data_dir is read-only, so use the work dir)
  edge_count is stored in the vectorized int64-key form
    ((sorted_keys int64, counts float32, shift int), edge_count_max)
  with key = (src << shift) | dst, built via sort + unique (14M pairs at ds4
  scale; a Python defaultdict of tuples would cost minutes + GBs of RAM).
  GraphMixerModel consumes either this tuple form or the legacy dict form.

Usage:
  python precompute_heuristics.py --dataset dataset2 --data_dir .
  python precompute_heuristics.py --dataset dataset1 --data_dir . --shared_nodes
  python precompute_heuristics.py --dataset dataset4 --data_dir F:\\download\\data_B \\
      --out_dir . --dense_id_remap
"""
import argparse
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd


def precompute(dataset, data_dir, out_dir=None, shared_nodes=False, dense_id_remap=False):
    out_dir = out_dir or data_dir
    train_path = f'{data_dir}/{dataset}/train.csv'
    train = pd.read_csv(train_path)
    src = train['src'].to_numpy(np.int64)
    dst = train['dst'].to_numpy(np.int64)

    if dense_id_remap:
        if shared_nodes:
            raise ValueError('--dense_id_remap is only for bipartite (non-shared) mode')
        seen_src = np.unique(src)
        seen_dst = np.unique(dst)
        src = np.searchsorted(seen_src, src).astype(np.int64) + 1
        dst = np.searchsorted(seen_dst, dst).astype(np.int64) + 1
        num_nodes = len(seen_dst) + 2           # + padding row 0 + UNK row
        print(f'  dense_id_remap: src->[1..{len(seen_src)}], dst->[1..{len(seen_dst)}], '
              f'UNK={len(seen_dst) + 1}')
    else:
        num_dst = int(dst.max()) + 1
        if shared_nodes:
            num_nodes = max(int(src.max()) + 1, num_dst)
        else:
            num_nodes = num_dst  # heuristic arrays indexed by node ID
    print(f'  Heuristic num_nodes={num_nodes} (shared_nodes={shared_nodes})')

    # 1. degree (count as dst in training set)
    degree = np.bincount(dst, minlength=num_nodes).astype(np.float32)[:num_nodes]
    degree_max = degree.max()
    if degree_max > 0:
        degree = degree / degree_max

    # 2. popularity (same as degree for dst, but kept separate for clarity)
    popularity = degree.copy()

    # 3. edge_count (directed edge count)
    if dense_id_remap:
        shift = max(1, int(num_nodes).bit_length())
        keys = (src << np.int64(shift)) | dst
        keys.sort()
        uniq, counts = np.unique(keys, return_counts=True)
        edge_count = (uniq, counts.astype(np.float32), shift)
        edge_count_max = float(counts.max()) if len(counts) else 1.0
        print(f'  edge_count: {len(uniq)} unique pairs (vectorized int64-key form, '
              f'shift={shift})')
    else:
        ec = defaultdict(int)
        for s, d in zip(src, dst):
            ec[(int(s), int(d))] += 1
        edge_count = dict(ec)
        edge_count_max = max(ec.values()) if ec else 1.0

    # Save
    np.save(f'{out_dir}/heuristics_{dataset}_degree.npy', degree)
    np.save(f'{out_dir}/heuristics_{dataset}_popularity.npy', popularity)
    with open(f'{out_dir}/heuristics_{dataset}_edge_count.pkl', 'wb') as f:
        pickle.dump((edge_count, float(edge_count_max)), f)

    print(f'Heuristics for {dataset}:')
    print(f'  num_nodes={num_nodes}, degree_max={degree_max:.0f}, edge_count_max={edge_count_max:.0f}')
    print(f'  Saved to {out_dir}/heuristics_{dataset}_*.npy/pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--shared_nodes', action='store_true')
    parser.add_argument('--dense_id_remap', action='store_true')
    args = parser.parse_args()
    precompute(args.dataset, args.data_dir, args.out_dir, args.shared_nodes,
               args.dense_id_remap)
