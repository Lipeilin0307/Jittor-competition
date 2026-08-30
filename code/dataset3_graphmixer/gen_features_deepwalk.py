"""
gen_features_deepwalk.py
=========================
Manual DeepWalk (uniform random walks) without node2vec library.
Generates node features for the GraphMixer model (dataset1, shared-node graph).

Reproducibility notes:
  - numpy random seed is fixed via --seed (walk generation is deterministic).
  - Word2Vec is trained with workers=1 by default so that results are
    bit-reproducible across runs and machines. Use --workers 8 only if you
    accept run-to-run variation (gensim multi-threading is non-deterministic).

Usage:
  python gen_features_deepwalk.py --dataset dataset1 --data_dir . --seed 42

Output:
  {data_dir}/node_features_{dataset}_n2v256.npy   (L2-normalized, row 0 = 0)
"""
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec


def generate_deepwalk(dataset_name, data_dir='.', output_dim=256,
                      walk_length=10, num_walks=50, window=5, negative=5,
                      sg=0, workers=1, seed=42):
    np.random.seed(seed)

    train_path = f'{data_dir}/{dataset_name}/train.csv'
    df = pd.read_csv(train_path)
    src = df['src'].to_numpy(np.int64)
    dst = df['dst'].to_numpy(np.int64)

    max_node_id = max(src.max(), dst.max())
    print(f'{dataset_name}: max_node_id={max_node_id}, edges={len(src)}')

    # [1] Build adjacency list (undirected)
    print('[1/3] Building adjacency list...')
    adj = [[] for _ in range(max_node_id + 1)]
    for s, d in zip(src, dst):
        adj[s].append(d)
        adj[d].append(s)

    # [2] Generate random walks (uniform, deterministic under np seed)
    print(f'[2/3] Generating {num_walks} walks per node (length={walk_length})...')
    walks = []
    nodes = [i for i in range(max_node_id + 1) if len(adj[i]) > 0]

    for node in tqdm(nodes, desc='Walks', ncols=100):
        for _ in range(num_walks):
            walk = [str(node)]
            current = node
            for _ in range(walk_length - 1):
                neighbors = adj[current]
                if not neighbors:
                    break
                current = neighbors[np.random.randint(0, len(neighbors))]
                walk.append(str(current))
            walks.append(walk)

    # [3] Train Word2Vec (CBOW)
    print(f'[3/3] Training Word2Vec (sg={sg}, workers={workers})...')
    model = Word2Vec(
        walks, vector_size=output_dim, window=window, min_count=1,
        sg=sg, workers=workers, epochs=1, negative=negative, seed=seed
    )

    # Extract embeddings
    print('Extracting embeddings...')
    features = np.zeros((max_node_id + 1, output_dim), dtype=np.float32)
    for node in tqdm(range(max_node_id + 1), desc='Extraction', ncols=100):
        key = str(node)
        if key in model.wv:
            features[node] = model.wv[key]

    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features = features / norms
    features[0] = 0.0

    out_path = f'{data_dir}/node_features_{dataset_name}_n2v256.npy'
    np.save(out_path, features)
    print(f'Saved: {out_path}, shape={features.shape}')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset1')
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--output_dim', type=int, default=256)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--num_walks', type=int, default=50)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--negative', type=int, default=5)
    parser.add_argument('--sg', type=int, default=0, help='0=CBOW (default), 1=skip-gram')
    parser.add_argument('--workers', type=int, default=1,
                        help='workers=1 for full reproducibility; >1 is faster but non-deterministic')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    generate_deepwalk(args.dataset, args.data_dir, args.output_dim,
                      args.walk_length, args.num_walks, args.window,
                      args.negative, args.sg, args.workers, args.seed)
    print('Done.')
