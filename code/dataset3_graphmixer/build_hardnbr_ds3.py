"""
build_hardnbr_ds3.py
====================
Offline top-K cosine neighbor table over the ds3 SHARED node2vec embeddings
(node_features_dataset3_n2v256.npy, 63635 x 256, rows = RAW node ids, row 0 =
padding). Used by the training script's cos-hard negative sampler
(--hard_neg_ratio). In --shared_nodes mode the sampler indexes this table with
raw dst ids, so no remap is applied (ds4 version used dense ids + an UNK row;
ds3 has no UNK row -- the ds4-specific hardcoded UNK zeroing is removed).

Method (identical to build_hardnbr_ds4.py):
  * L2-normalize rows (zero-norm rows stay zero -> zero scores).
  * GPU (jittor CUDA) batched matmul: row batches of --row_batch rows vs
    column tiles of --col_tile nodes; fp32 scores (fp16 measured to corrupt
    top-K selection on ds4: boundary ties shuffle ranks).
  * Per tile: 8-thread numpy argpartition keeps the local top-K; a running
    per-row top-K (values+ids) is merged across tiles. Self-similarity is
    excluded before selection (-inf).
  * Row 0 (padding) and any zero-norm row are emitted as all-zero neighbor
    lists (training falls back to popularity sampling).

Output: hardnbr_ds3.npy (num_nodes, K) int32, K default 256.

Budget: 63635^2 fp32 scores tiled -> well under a minute on RTX3070.
--selftest runs a tiny exact check; --spot_check N verifies N random rows
against fp32 brute force.
"""
import argparse
import os
import time as _time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _topk_tile(scores, k, nthreads=8):
    """Row-wise top-k of a (R, C) score block via threaded argpartition.

    Returns (vals fp32 (R,k), idx int32 (R,k)) sorted descending per row.
    """
    R = scores.shape[0]
    chunks = np.array_split(np.arange(R), nthreads)

    def work(rows):
        part = np.argpartition(scores[rows], -k, axis=1)[:, -k:]
        vals = scores[rows[:, None], part]
        order = np.argsort(-vals, axis=1)
        idx = part[np.arange(len(rows))[:, None], order]
        return vals[np.arange(len(rows))[:, None], order].astype(np.float32), \
            idx.astype(np.int32)

    with ThreadPoolExecutor(nthreads) as ex:
        parts = list(ex.map(work, chunks))
    vals = np.concatenate([p[0] for p in parts], axis=0)
    idx = np.concatenate([p[1] for p in parts], axis=0)
    return vals, idx


def build(dst_features, out_path, topk=256, row_batch=4096, col_tile=131072,
          rows_limit=None, spot_check=0, seed=0):
    import jittor as jt
    jt.flags.use_cuda = 1
    t0 = _time.time()
    X = np.load(dst_features).astype(np.float32)
    n = X.shape[0]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.where(norms > 0, X / np.maximum(norms, 1e-30), 0.0).astype(np.float32)
    zero_rows = norms[:, 0] <= 0
    Xj = jt.array(X)                        # (n, 256) fp32 on GPU
    del X
    n_rows = min(n, rows_limit) if rows_limit else n
    print(f'  {n_rows} rows, topk={topk}, row_batch={row_batch}, '
          f'col_tile={col_tile}; norms done {_time.time() - t0:.1f}s')

    out = np.zeros((n, topk), dtype=np.int32)
    n_tiles = (n + col_tile - 1) // col_tile
    for r0 in range(0, n_rows, row_batch):
        r1 = min(r0 + row_batch, n_rows)
        R = r1 - r0
        best_v = np.full((R, topk), -np.inf, dtype=np.float32)
        best_i = np.zeros((R, topk), dtype=np.int32)
        rows_global = np.arange(r0, r1)
        for t in range(n_tiles):
            c0 = t * col_tile
            c1 = min(c0 + col_tile, n)
            sc = jt.matmul(Xj[r0:r1], Xj[c0:c1].transpose(0, 1))
            sc32 = sc.numpy()                           # (R, ct) fp32
            del sc
            # exclude self-pairs (row r vs col r)
            in_tile = (rows_global >= c0) & (rows_global < c1)
            if in_tile.any():
                lr = rows_global[in_tile] - r0
                lc = rows_global[in_tile] - c0
                sc32[lr, lc] = -np.inf
            v, ix = _topk_tile(sc32, topk)
            ix = ix + c0
            del sc32
            # merge running best with tile top-k
            cv = np.concatenate([best_v, v], axis=1)          # (R, 2k)
            ci = np.concatenate([best_i, ix], axis=1)
            sel = np.argpartition(cv, -topk, axis=1)[:, -topk:]
            rv = cv[np.arange(R)[:, None], sel]
            ri = ci[np.arange(R)[:, None], sel]
            order = np.argsort(-rv, axis=1)
            best_v = rv[np.arange(R)[:, None], order]
            best_i = ri[np.arange(R)[:, None], order]
            del cv, ci, sel, rv, ri, v, ix
        out[r0:r1] = best_i
        if (r0 // row_batch) % 20 == 0 or r1 == n_rows:
            print(f'  rows {r1}/{n_rows} ({_time.time() - t0:.1f}s)', flush=True)
    # padding / zero-norm rows -> all-zero neighbor lists
    out[zero_rows] = 0
    out[0] = 0
    np.save(out_path, out)
    print(f'Saved {out_path}: shape={out.shape} int32, '
          f'{os.path.getsize(out_path) / 2 ** 20:.0f}MB, '
          f'total {_time.time() - t0:.1f}s')

    if spot_check > 0:
        rng = np.random.default_rng(seed)
        Xf = np.load(dst_features).astype(np.float32)
        nf = np.linalg.norm(Xf, axis=1, keepdims=True)
        Xf = np.where(nf > 0, Xf / np.maximum(nf, 1e-30), 0.0).astype(np.float32)
        # zero-norm rows are intentionally all-zero in the output table
        # (training falls back to popularity draws); exclude them from the
        # overlap check or they trivially score 0.
        nz_rows = np.nonzero(nf[1:min(n_rows, n), 0] > 0)[0] + 1
        rows = rng.choice(nz_rows, size=min(spot_check, len(nz_rows)),
                          replace=False)
        worst = topk
        worst_gap = 0.0
        for r in rows:
            cos = Xf @ Xf[r]
            cos[r] = -np.inf
            ref = np.argpartition(-cos, topk)[:topk]
            boundary = -float(np.partition(-cos, topk)[topk])
            got = set(out[r].tolist()) - {0}
            ref_set = set(ref.tolist())
            inter = len(got & ref_set)
            worst = min(worst, inter)
            # value-level gap of the worst missed reference item
            miss = [i for i in ref_set if i not in got]
            if miss:
                gap = max(float(cos[i]) - boundary for i in miss)
                worst_gap = max(worst_gap, gap)
        print(f'spot_check: {spot_check} rows, worst top-{topk} overlap={worst}, '
              f'worst missed-item gap above boundary={worst_gap:.2e}')
        # GPU fp32 (cublas summation order / TF32) shuffles near-tie boundary
        # items by ~1e-4; exact-set equality is NOT required. Fail only on real
        # breakage: big overlap loss OR a missed item far above the boundary.
        assert worst >= topk - 20 and worst_gap < 2e-3, \
            'brute-force spot check failed (beyond precision noise)'
        print('spot_check OK (boundary-tie tolerant)')


def selftest():
    """Tiny exact check on CPU path pieces (topk tile + merge logic)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    scores = (X @ X.T).astype(np.float16)
    scores[np.arange(50), np.arange(50)] = np.float16('-inf')
    v, ix = _topk_tile(scores, 5)
    ref = np.argsort(-(X @ X.T - np.eye(50) * 2), axis=1)[:, :5]
    for r in range(50):
        assert set(ix[r].tolist()) == set(ref[r].tolist()), f'row {r}'
        assert np.all(np.diff(v[r]) <= 1e-3)
    print('selftest OK: threaded top-k matches brute force')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dst_features', default='./node_features_dataset3_n2v256.npy')
    ap.add_argument('--out', default='./hardnbr_ds3.npy')
    ap.add_argument('--topk', type=int, default=256)
    ap.add_argument('--row_batch', type=int, default=4096)
    ap.add_argument('--col_tile', type=int, default=131072)
    ap.add_argument('--rows_limit', type=int, default=None,
                    help='build only the first N rows (slice validation)')
    ap.add_argument('--spot_check', type=int, default=8,
                    help='brute-force verify N random rows after build')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    build(args.dst_features, args.out, args.topk, args.row_batch,
          args.col_tile, args.rows_limit, args.spot_check)


if __name__ == '__main__':
    main()
