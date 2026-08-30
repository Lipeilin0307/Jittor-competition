"""
gen_features.py
=============
Generate Node2Vec 256 features for dataset1 and dataset2.
Usage:
  python gen_features.py                        # both datasets (original behavior)
  python gen_features.py --dataset dataset1     # dataset1 only
  python gen_features.py --dataset dataset1 --data_dir .
"""
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec


def generate_node2vec(dataset_name, data_dir='.', output_dim=256, walk_length=30, num_walks=200):
    train_path = f'{data_dir}/{dataset_name}/train.csv'
    df = pd.read_csv(train_path)
    src = df['src'].to_numpy(np.int64)
    dst = df['dst'].to_numpy(np.int64)
    
    # Build graph (edges are undirected for Node2Vec)
    G = nx.Graph()
    for s, d in zip(src, dst):
        G.add_edge(int(s), int(d))
    
    print(f'{dataset_name}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}')
    
    # Node2Vec
    n2v = Node2Vec(G, dimensions=output_dim, walk_length=walk_length,
                   num_walks=num_walks, workers=4, quiet=True, p=1.0, q=1.0)
    model = n2v.fit(window=10, min_count=1, batch_words=4, epochs=1)
    
    # Extract embeddings
    max_node_id = max(src.max(), dst.max())
    features = np.zeros((max_node_id + 1, output_dim), dtype=np.float32)
    
    for node in G.nodes():
        key = str(node)
        if key in model.wv:
            features[node] = model.wv[key]
    
    # L2 normalize + zero padding
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features = features / norms
    features[0] = 0.0
    
    out_path = f'{data_dir}/node_features_{dataset_name}_n2v{output_dim}.npy'
    np.save(out_path, features)
    print(f'Saved: {out_path}, shape={features.shape}')
    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', choices=['dataset1', 'dataset2', 'all'])
    parser.add_argument('--data_dir', default='.')
    args = parser.parse_args()
    targets = ['dataset1', 'dataset2'] if args.dataset == 'all' else [args.dataset]
    for name in targets:
        generate_node2vec(name, args.data_dir, 256)
    print('All features generated.')
