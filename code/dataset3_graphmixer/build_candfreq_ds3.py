"""
build_candfreq_ds3.py
=====================
Build the candidate-frequency table for dataset3, same format as ds4's
cand_freq_pairs.npz (consumed by train_graphmixer_jt.py --neg_freq_table and
eval_testlike.py --cand_freq):

  raw_id: (M,) int32, STRICTLY INCREASING raw node ids
  count:  (M,) int32, number of appearances of that id in test.csv c1..c100

Semantics (mirrors heuristic_scorer_d4.py / valhot_d4.py): empirical frequency
of each dst among the REAL test candidate columns -- "test-period popularity",
a legal signal because test.csv candidate sets are given. In --shared_nodes
mode the training script matches these raw ids directly against the seen-dst
pool (no remap), so this table is keyed by RAW node ids.

Method: chunked pd.read_csv of test.csv candidate columns + np.bincount into a
(num_nodes,) accumulator, then keep ids with count > 0 (np.nonzero returns
them in increasing order, satisfying the strict-increasing contract).

Usage:
  python build_candfreq_ds3.py            # full build + built-in checks
  python build_candfreq_ds3.py --verify   # extra brute-force cross-check
"""
import argparse
import os
import time as _time

import numpy as np


def detect_num_nodes(node_features_path):
    x = np.load(node_features_path, mmap_mode='r')
    return int(x.shape[0])


def build(test_csv, num_nodes, chunksize=50_000):
    import pandas as pd
    t0 = _time.time()
    freq = np.zeros(num_nodes, dtype=np.int64)
    n_rows = 0
    for chunk in pd.read_csv(test_csv, chunksize=chunksize):
        cand = chunk.iloc[:, 2:].to_numpy(np.int64).ravel()
        assert cand.size == 0 or cand.max() < num_nodes, \
            f'candidate id {int(cand.max())} out of node space ({num_nodes})'
        freq += np.bincount(cand, minlength=num_nodes)
        n_rows += len(chunk)
    print(f'  test rows={n_rows}, candidate slots={n_rows * 100}, '
          f'read+count {_time.time() - t0:.1f}s')
    return freq, n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_csv', default=r'D:\work_d3\dataset3\test.csv')
    ap.add_argument('--node_features', default='./node_features_dataset3_n2v256.npy',
                    help='used only to detect the shared node space size')
    ap.add_argument('--num_nodes', type=int, default=None,
                    help='override node space size (default: node_features rows)')
    ap.add_argument('--out', default='./cand_freq_pairs_ds3.npz')
    ap.add_argument('--chunksize', type=int, default=50_000)
    ap.add_argument('--verify', action='store_true',
                    help='brute-force cross-check of 200 random ids')
    args = ap.parse_args()

    num_nodes = args.num_nodes or detect_num_nodes(args.node_features)
    print(f'node space: num_nodes={num_nodes} (raw ids 0..{num_nodes - 1})')

    freq, n_rows = build(args.test_csv, num_nodes, args.chunksize)
    total = int(freq.sum())
    assert total == n_rows * 100, f'count mismatch: {total} != {n_rows * 100}'

    raw_id = np.nonzero(freq)[0].astype(np.int32)   # inherently increasing
    count = freq[raw_id].astype(np.int32)
    assert len(raw_id) and np.all(np.diff(raw_id.astype(np.int64)) > 0)
    np.savez(args.out, raw_id=raw_id, count=count)
    sz = os.path.getsize(args.out) / 2 ** 20
    print(f'Saved {args.out}: raw_id{raw_id.shape} int32 + count{count.shape} '
          f'int32, {sz:.2f}MB, ids={len(raw_id)}, count range '
          f'[{int(count.min())}, {int(count.max())}], total mass={total}')

    if args.verify:
        import pandas as pd
        rng = np.random.default_rng(0)
        frame = pd.read_csv(args.test_csv)
        cand = frame.iloc[:, 2:].to_numpy(np.int64)
        pick = rng.choice(raw_id, size=min(200, len(raw_id)), replace=False)
        bad = 0
        for rid in pick:
            ref = int((cand == int(rid)).sum())
            got = int(freq[int(rid)])
            if ref != got:
                bad += 1
                print(f'  MISMATCH id={rid}: ref={ref} got={got}')
        assert bad == 0, 'brute-force cross-check failed'
        print(f'verify OK: {len(pick)} random ids match brute-force counts')


if __name__ == '__main__':
    main()
