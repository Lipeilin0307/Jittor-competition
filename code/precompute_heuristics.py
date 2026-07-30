"""
precompute_heuristics.py
=======================
Precompute heuristic features for link prediction:
  1. degree[candidate]     - candidate node degree (as dst)
  2. popularity[candidate] - candidate popularity (normalized occurrence)
  3. edge_count[src, dst]  - number of historical edges between src and dst

Usage:
  python precompute_heuristics.py --dataset dataset2 --data_dir .
  python precompute_heuristics.py --dataset dataset1 --data_dir . --shared_nodes
"""
import argparse
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd


def precompute(dataset, data_dir, shared_nodes=False):
    train_path = f'{data_dir}/{dataset}/train.csv'
    train = pd.read_csv(train_path)
    src = train['src'].to_numpy(np.int64)
    dst = train['dst'].to_numpy(np.int64)

    num_dst = int(dst.max()) + 1
    if shared_nodes:
        num_nodes = max(int(src.max()) + 1, num_dst)
    else:
        num_nodes = num_dst  # heuristic arrays indexed by node ID
    print(f'  Heuristic num_nodes={num_nodes} (shared_nodes={shared_nodes})')

    # 1. degree (count as dst in training set)
    degree = np.zeros(num_nodes, dtype=np.float32)
    for d in dst:
        degree[d] += 1.0
    degree_max = degree.max()
    if degree_max > 0:
        degree = degree / degree_max

    # 2. popularity (same as degree for dst, but kept separate for clarity)
    popularity = degree.copy()

    # 3. edge_count (directed edge count)
    edge_count = defaultdict(int)
    for s, d in zip(src, dst):
        edge_count[(int(s), int(d))] += 1
    edge_count_max = max(edge_count.values()) if edge_count else 1.0

    # Save
    np.save(f'{data_dir}/heuristics_{dataset}_degree.npy', degree)
    np.save(f'{data_dir}/heuristics_{dataset}_popularity.npy', popularity)
    with open(f'{data_dir}/heuristics_{dataset}_edge_count.pkl', 'wb') as f:
        pickle.dump((dict(edge_count), float(edge_count_max)), f)

    print(f'Heuristics for {dataset}:')
    print(f'  num_nodes={num_nodes}, degree_max={degree_max:.0f}, edge_count_max={edge_count_max:.0f}')
    print(f'  Saved to {data_dir}/heuristics_{dataset}_*.npy/pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--shared_nodes', action='store_true')
    args = parser.parse_args()
    precompute(args.dataset, args.data_dir, args.shared_nodes)
