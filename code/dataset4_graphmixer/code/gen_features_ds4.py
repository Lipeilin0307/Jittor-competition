# -*- coding: utf-8 -*-
"""
gen_features_ds4.py
===================
Node features for the dataset4 (track-2) GraphMixer temporal pipeline.

Method (validated by ds4_signal_check.py, cos-MRR 0.2168 vs popularity 0.0617):
  * Treat each src's dst interaction list as a "sentence" and train a gensim
    Word2Vec skip-gram model on the dst vocabulary. Sentences are built from
    ALL train.csv edges (split=0 and split=1), preserving file order within
    each src (the file is time-ordered).
  * dst embedding table: one row per dst id that appeared in train.csv;
    ids missing from the word2vec vocab (min_count) stay zero vectors.
  * src embedding table: L2-normalized mean of the src's (normalized)
    historical dst embeddings.

DENSE ID REMAP (ds4 deviation, forced by hardware)
--------------------------------------------------
ds4 raw ids are sparse: dst ids live in [23, 4068785] with only 862,246 used,
src ids in [4068791, 5052904] with 680,640 used. Raw-id embedding tables would
need (4.07M + 5.05M) x 256 x 4B = ~9.3 GB of weights (~15.6 GB with Adam
states) -- impossible on the 8 GB GPU. All ds4 scripts therefore share a dense
remap (--dense_id_remap):

  seen dst ids (sorted) -> dense 1..N_dst ; dense 0      = padding
                                            dense N_dst+1 = UNK (any dst never
                                              seen in train, e.g. 23.4% of the
                                              test candidates; zero-init,
                                              trainable)
  seen src ids (sorted) -> dense 1..N_src ; dense 0      = padding / unseen src

This module writes the dense tables directly:
  node_features_ds4_dst.npy : (N_dst + 2, dim) float32, rows L2-normalized
  node_features_ds4_src.npy : (N_src + 1, dim) float32, rows L2-normalized
  idmap_ds4.npz             : seen_src_ids / seen_dst_ids / unk_dst_id
                              (dense id = position in seen_*_ids + 1)

Usage:
  python gen_features_ds4.py --dataset dataset4 --data_dir F:\\download\\data_B \
      --out_dir . --workers 4
"""
import argparse
import os
import time

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset4')
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--out_dir', type=str, default='.')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--min_count', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--negative', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    train_path = f'{args.data_dir}/{args.dataset}/train.csv'
    print(f'[1/5] reading {train_path} ...')
    frame = pd.read_csv(train_path)
    src_raw = frame['src'].to_numpy(np.int64)
    dst_raw = frame['dst'].to_numpy(np.int64)
    n_edges = len(frame)
    del frame
    print(f'      edges={n_edges:,}  ({time.time() - t_start:.0f}s)')

    # ------------------------------------------------------------------
    # Dense remap (shared contract with train/eval/predict --dense_id_remap)
    # ------------------------------------------------------------------
    t0 = time.time()
    print('[2/5] building dense id remap + per-src sentences ...')
    seen_src = np.unique(src_raw)                      # sorted
    seen_dst = np.unique(dst_raw)                      # sorted
    n_src, n_dst = len(seen_src), len(seen_dst)
    unk_dst = n_dst + 1
    src_dense = np.searchsorted(seen_src, src_raw) + 1
    dst_dense = np.searchsorted(seen_dst, dst_raw) + 1
    print(f'      seen src={n_src:,}  seen dst={n_dst:,}  UNK_DST={unk_dst}')

    # Sentences: dst id strings grouped by src, file order preserved
    # (stable argsort keeps the original temporal order inside each src).
    order = np.argsort(src_raw, kind='stable')
    dst_sorted = dst_raw[order]
    src_sorted = src_raw[order]
    _, starts = np.unique(src_sorted, return_index=True)
    ends = np.append(starts[1:], n_edges)
    assert len(starts) == n_src
    sentences = []
    for lo, hi in zip(starts, ends):
        sentences.append([str(x) for x in dst_sorted[lo:hi]])
    del order, dst_sorted, src_sorted, starts, ends
    total_tokens = sum(len(s) for s in sentences)
    print(f'      sentences={len(sentences):,}  tokens={total_tokens:,}  '
          f'({time.time() - t0:.0f}s)')

    # ------------------------------------------------------------------
    # Word2Vec (parameters validated by ds4_signal_check.py)
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f'[3/5] word2vec: dim={args.dim}, window={args.window}, sg=1, '
          f'min_count={args.min_count}, workers={args.workers}, '
          f'epochs={args.epochs}, negative={args.negative}, seed={args.seed} ...')
    from gensim.models import Word2Vec
    w2v = Word2Vec(sentences, vector_size=args.dim, window=args.window, sg=1,
                   min_count=args.min_count, workers=args.workers,
                   epochs=args.epochs, negative=args.negative, seed=args.seed)
    del sentences
    print(f'      vocab={len(w2v.wv):,}  ({time.time() - t0:.0f}s)')

    # ------------------------------------------------------------------
    # dst table (dense rows, L2-normalized; vocab-missing ids stay zero)
    # ------------------------------------------------------------------
    t0 = time.time()
    print('[4/5] writing dst table ...')
    dim = args.dim
    dst_table = np.zeros((n_dst + 2, dim), dtype=np.float32)
    keys = [str(x) for x in seen_dst]
    present = np.array([k in w2v.wv for k in keys])
    idx = np.nonzero(present)[0]
    vecs = w2v.wv[[keys[i] for i in idx]].astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-30)
    dst_table[idx + 1] = vecs                            # dense id = position + 1
    # row 0 = padding, row unk_dst = UNK: both stay zero
    cov = len(idx)
    print(f'      dst coverage {cov}/{n_dst} ({100.0 * cov / n_dst:.1f}%), '
          f'nonzero rows={int((np.linalg.norm(dst_table, axis=1) > 0).sum())}')
    out_dst = os.path.join(args.out_dir, 'node_features_ds4_dst.npy')
    np.save(out_dst, dst_table)
    print(f'      saved {out_dst} shape={dst_table.shape}')

    # ------------------------------------------------------------------
    # src table: normalized mean of historical dst vectors (all train edges)
    # ------------------------------------------------------------------
    t0 = time.time()
    print('[5/5] writing src table (mean of dst embeddings) ...')
    from scipy.sparse import csr_matrix
    A = csr_matrix((np.ones(n_edges, dtype=np.float32), (src_dense, dst_dense)),
                   shape=(n_src + 1, n_dst + 2))
    sums = A @ dst_table                                 # (N_src+1, dim)
    counts = np.bincount(src_dense, minlength=n_src + 1).astype(np.float32)
    mean = sums / np.maximum(counts, 1.0)[:, None]
    mn = np.linalg.norm(mean, axis=1, keepdims=True)
    src_table = np.where(mn > 0, mean / np.maximum(mn, 1e-30), 0.0).astype(np.float32)
    src_table[0] = 0.0                                   # padding row
    del A, sums, mean
    out_src = os.path.join(args.out_dir, 'node_features_ds4_src.npy')
    np.save(out_src, src_table)
    nz = float((np.linalg.norm(src_table, axis=1) > 0).mean())
    print(f'      saved {out_src} shape={src_table.shape}, nonzero-row ratio={nz:.4f}')

    np.savez(os.path.join(args.out_dir, 'idmap_ds4.npz'),
             seen_src_ids=seen_src, seen_dst_ids=seen_dst,
             unk_dst_id=np.int64(unk_dst))
    print(f'      saved idmap_ds4.npz (unk_dst_id={unk_dst})')
    print(f'done in {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
