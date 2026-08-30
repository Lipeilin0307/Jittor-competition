"""
build_cooc_ds3.py
=================
Offline approximate item-item co-occurrence builder for dataset3, adapted from
build_cooc_ds4.py. Difference vs the ds4 version: ds3 trains with
--shared_nodes (NO dense remap), so the table is keyed by RAW node ids and no
idmap is applied. Row count and the pair-key base are parameterized from the
shared node space (default: shape of node_features_dataset3_n2v256.npy ->
num_nodes = 63635, raw ids 0..63634; row 0 = padding).

Method (memory-bounded):
  * edges: train.csv rows with split==0 AND time>0 (positive-time train edges),
    raw src/dst ids used as-is.
  * per-src history list = DISTINCT dsts, most-recent-first cap of 64
    (dedupe keeps the LAST occurrence; file is time-ordered so "last" ==
    "most recent").
  * emit every ORDERED pair (dst_i, dst_j), i != j, inside each capped list as
    an int64 key a*BASE+b (BASE=num_nodes, unique for raw ids < num_nodes).
  * sort + unique -> pair counts; keep count >= min_count (default 2).
  * per dst a keep the top-K (default 128) neighbors b by count.

Output: cooc_ds3.npz
  nbr_ids: (num_nodes, K) int32   - neighbor raw dst ids, 0-padded
  nbr_w:   (num_nodes, K) float32 - log1p(count), row-max normalized
Plus memmap sidecars cooc_ds3_nbr_ids.npy / cooc_ds3_nbr_w.npy (CoocTable
prefers these for true mmap).

Use --max_edges N for a sliced correctness/speed validation, --selftest for a
hand-checkable tiny example, --verify for an exact dict-based cross-check on a
small subsample.
"""
import argparse
import os
import time as _time

import numpy as np

_CHUNK_TILES = 64_000_000   # max tiled (i,j) elements per emission chunk


def detect_num_nodes(node_features_path):
    """Shared-node space size = rows of the n2v feature table."""
    x = np.load(node_features_path, mmap_mode='r')
    return int(x.shape[0])


def emit_pairs(dst_lists, off, base, chunk_tiles=_CHUNK_TILES):
    """Vectorized ordered-pair emission over grouped capped dst lists.

    dst_lists: (M,) int32 capped distinct dst ids, grouped by src.
    off: (G+1,) int64 group offsets into dst_lists.
    Returns int64 array of keys a*base+b (a != b within each group).
    """
    G = len(off) - 1
    sizes = np.diff(off)
    m = sizes.astype(np.int64)
    tiled = m * m                        # (i,j) incl. self per group
    total_tiled = int(tiled.sum())
    total_pairs = int((tiled - m).sum())
    print(f'  groups={G}, capped elems={int(m.sum())}, '
          f'ordered pairs={total_pairs / 1e6:.1f}M')
    if total_pairs == 0:
        return np.empty(0, dtype=np.int64)

    tstart = np.zeros(G + 1, dtype=np.int64)
    np.cumsum(tiled, out=tstart[1:])
    keys_out = []
    produced = 0
    g0 = 0
    t0 = _time.time()
    while g0 < G:
        # chunk = maximal group range with tiled sum <= chunk_tiles
        budget = chunk_tiles
        cum = np.cumsum(tiled[g0:])
        g1 = g0 + int(np.searchsorted(cum, budget, side='right'))
        if g1 == g0:
            g1 = g0 + 1                     # single group bigger than budget
        lo_t, hi_t = int(tstart[g0]), int(tstart[g1])
        p = np.arange(lo_t, hi_t, dtype=np.int64) - lo_t
        gid = np.repeat(np.arange(g0, g1, dtype=np.int64),
                        tiled[g0:g1].astype(np.int64)) - g0
        mloc = m[g0:g1]
        m_g = mloc[gid]
        j = p - (tstart[g0:g1] - lo_t)[gid]
        i = j // m_g
        jj = j - i * m_g
        base_idx = off[g0:g1][gid]
        a = dst_lists[base_idx + i]
        b = dst_lists[base_idx + jj]
        keep = (i != jj) & (a != b)
        k = (a[keep].astype(np.int64) * base) + b[keep]
        keys_out.append(k)
        produced += len(k)
        del p, gid, m_g, j, i, jj, base_idx, a, b, keep, k
        g0 = g1
    keys = (keys_out[0] if len(keys_out) == 1
            else np.concatenate(keys_out))
    print(f'  emitted {produced / 1e6:.1f}M pairs in {_time.time() - t0:.1f}s '
          f'({len(keys_out)} chunks)')
    return keys


def build_tables(src, dst, cap, topk, min_count, n_rows, base):
    """Full pipeline from (src, dst) raw-id edge arrays to nbr tables."""
    t0 = _time.time()
    # ---- group edges by src, preserving time order (stable sort) ----
    order = np.argsort(src, kind='stable')
    s = src[order]
    d = dst[order].astype(np.int32)
    del order
    # ---- per-src dedupe dst, keep LAST (most recent) occurrence ----
    # (must handle NON-consecutive repeats: same dst interacted again later)
    key2 = s * np.int64(base) + d
    rev = key2[::-1]
    _, first_idx = np.unique(rev, return_index=True)
    keep_pos = np.sort(len(key2) - 1 - first_idx)
    del key2, rev, first_idx
    keep_mask = np.zeros(len(s), dtype=bool)
    keep_mask[keep_pos] = True
    s = s[keep_mask]
    d = d[keep_mask]
    del keep_mask, keep_pos
    print(f'  distinct (src,dst) pairs: {len(s) / 1e6:.2f}M '
          f'({_time.time() - t0:.1f}s)')
    # ---- cap: last `cap` entries per src ----
    uniq_s, gstart = np.unique(s, return_index=True)
    off = np.append(gstart, len(s)).astype(np.int64)
    cnt = np.diff(off)
    idx_in = np.arange(len(s), dtype=np.int64) - off[:-1].repeat(cnt)
    from_end = cnt.repeat(cnt) - idx_in
    capped = from_end <= cap
    d_cap = d[capped]
    cnt_cap = np.minimum(cnt, cap)
    off_cap = np.zeros(len(off), dtype=np.int64)
    np.cumsum(cnt_cap, out=off_cap[1:])
    del s, d, idx_in, from_end, capped, cnt
    print(f'  capped lists (cap={cap}): {len(d_cap) / 1e6:.2f}M elems, '
          f'{len(uniq_s)} srcs ({_time.time() - t0:.1f}s)')
    # ---- emit ordered pairs, sort, count ----
    keys = emit_pairs(d_cap, off_cap, base)
    del d_cap
    if len(keys) == 0:
        return (np.zeros((n_rows, topk), np.int32),
                np.zeros((n_rows, topk), np.float32), 0)
    t1 = _time.time()
    keys.sort()
    change = np.ones(len(keys), dtype=bool)
    change[1:] = keys[1:] != keys[:-1]
    uidx = np.nonzero(change)[0]
    uk = keys[uidx]
    cnts = np.diff(np.append(uidx, len(keys))).astype(np.int32)
    del keys, change, uidx
    print(f'  unique pairs: {len(uk) / 1e6:.1f}M (sort+count '
          f'{_time.time() - t1:.1f}s)')
    keep = cnts >= min_count
    uk = uk[keep]
    cnts = cnts[keep]
    print(f'  pairs with count>={min_count}: {len(uk) / 1e6:.1f}M')
    a = (uk // base).astype(np.int32)
    b = (uk % base).astype(np.int32)
    del uk
    # ---- per-a top-K by count ----
    t1 = _time.time()
    order = np.lexsort((-cnts, a))
    a_s = a[order]
    b_s = b[order]
    c_s = cnts[order]
    del a, b, cnts, order
    ua, ustart = np.unique(a_s, return_index=True)
    ucnt = np.diff(np.append(ustart, len(a_s)))
    take = np.minimum(ucnt, topk)
    nbr_ids = np.zeros((n_rows, topk), dtype=np.int32)
    nbr_w = np.zeros((n_rows, topk), dtype=np.float32)
    # positions of the first `take` entries of each group in the sorted stream
    pos = np.ones(int(take.sum()), dtype=np.int64)
    pos[0] = ustart[0]
    bounds = np.cumsum(take)[:-1]
    pos[bounds] = ustart[1:] - ustart[:-1] - take[:-1] + 1
    pos = np.cumsum(pos)
    row_rep = np.repeat(ua, take)
    col_rep = np.concatenate([np.arange(t, dtype=np.int64) for t in take]) \
        if len(take) else np.empty(0, dtype=np.int64)
    nbr_ids[row_rep, col_rep] = b_s[pos]
    w = np.log1p(c_s[pos].astype(np.float64)).astype(np.float32)
    nbr_w[row_rep, col_rep] = w
    # row-max normalization of log1p weights
    rmax = nbr_w.max(axis=1, keepdims=True)
    nz = rmax[:, 0] > 0
    nbr_w[nz] /= rmax[nz]
    print(f'  top-{topk} tables built ({_time.time() - t1:.1f}s); '
          f'rows with >=1 nbr: {int(nz.sum())}; total {_time.time() - t0:.1f}s')
    return nbr_ids, nbr_w, int(len(ua))


def load_edges(train_csv, num_nodes, max_edges=None):
    """ds3 shared-nodes mode: RAW ids, no remap. split==0 & time>0 only."""
    import pandas as pd
    t0 = _time.time()
    frame = pd.read_csv(train_csv, usecols=['src', 'dst', 'time', 'split'])
    frame = frame[(frame['split'] == 0) & (frame['time'] > 0)]
    if max_edges:
        frame = frame.iloc[:max_edges]
    print(f'  edges split==0 & time>0: {len(frame) / 1e6:.2f}M '
          f'(read {_time.time() - t0:.1f}s)')
    src = frame['src'].to_numpy(np.int64)
    dst = frame['dst'].to_numpy(np.int64)
    assert len(src) == 0 or (src.max() < num_nodes and dst.max() < num_nodes), \
        f'raw id out of node space (num_nodes={num_nodes})'
    return src, dst


def selftest():
    """Hand-checkable tiny example (n_rows=8, base=8).

    src1: [1,2,3], src2: [1,2], src3: [1,4], src4: [1,2] (with a duplicate
    NON-consecutive edge 4->1 ... ->2 ->1 to exercise keep-last dedupe).
    Ordered counts: (1,2)=3,(2,1)=3, (1,3)=1,(3,1)=1,(2,3)=1,(3,2)=1,
                    (1,4)=1,(4,1)=1.
    min_count=2 keeps only (1,2),(2,1) with count 3.
    """
    src = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 4], dtype=np.int64)
    dst = np.array([1, 2, 3, 1, 2, 1, 4, 1, 2, 1], dtype=np.int64)
    nbr_ids, nbr_w, _ = build_tables(src, dst, cap=64, topk=128, min_count=2,
                                     n_rows=8, base=8)
    assert nbr_ids[1, 0] == 2 and nbr_ids[2, 0] == 1
    assert abs(nbr_w[1, 0] - 1.0) < 1e-6 and abs(nbr_w[2, 0] - 1.0) < 1e-6
    assert (nbr_ids[1, 1:] == 0).all() and (nbr_ids[2, 1:] == 0).all()
    assert (nbr_ids[3] == 0).all() and (nbr_ids[4] == 0).all()
    # dedupe check: src4 had dst 1 twice; count(1,2) counts src4 once -> 3
    print('selftest OK: counts/topk/normalization/dedupe verified')


def verify_slice(src, dst, cap, min_count, num_nodes, n_edges=200_000, seed=0):
    """Exact dict-based cross-check of pair counts on a small subsample."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(1, len(src) - n_edges)))
    s = src[start:start + n_edges]
    d = dst[start:start + n_edges]
    lists = defaultdict(list)
    seen = defaultdict(set)
    for si, di in zip(s.tolist(), d.tolist()):
        if di in seen[si]:
            lists[si].remove(di)
        else:
            seen[si].add(di)
        lists[si].append(di)
    ref = defaultdict(int)
    for si, lst in lists.items():
        lst = lst[-cap:]
        for x in range(len(lst)):
            for y in range(len(lst)):
                if x != y:
                    ref[(lst[x], lst[y])] += 1
    nbr_ids, nbr_w, _ = build_tables(s, d, cap=cap, topk=128,
                                     min_count=min_count,
                                     n_rows=num_nodes, base=num_nodes)
    got = {}
    for a in range(nbr_ids.shape[0]):
        for j in range(nbr_ids.shape[1]):
            bb = int(nbr_ids[a, j])
            if bb:
                got[(a, bb)] = nbr_w[a, j]
    mism = 0
    checked = 0
    for (a, bb), w in got.items():
        rc = ref.get((a, bb), 0)
        checked += 1
        if rc < min_count:
            mism += 1
    # every kept pair must have ref count >= min_count
    print(f'verify: kept pairs={checked}, mismatches={mism}')
    assert mism == 0, 'dict cross-check failed'
    print('verify OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', default=r'D:\work_d3\dataset3\train.csv')
    ap.add_argument('--node_features', default='./node_features_dataset3_n2v256.npy',
                    help='used only to detect the shared node space size')
    ap.add_argument('--num_nodes', type=int, default=None,
                    help='override node space size (default: node_features rows)')
    ap.add_argument('--out', default='./cooc_ds3.npz')
    ap.add_argument('--cap', type=int, default=64)
    ap.add_argument('--topk', type=int, default=128)
    ap.add_argument('--min_count', type=int, default=2)
    ap.add_argument('--max_edges', type=int, default=None,
                    help='use only the first N filtered edges (slice validation)')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--verify', action='store_true',
                    help='dict cross-check on a 200k-edge subsample')
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    num_nodes = args.num_nodes or detect_num_nodes(args.node_features)
    print(f'node space: num_nodes={num_nodes} (raw ids 0..{num_nodes - 1}, '
          f'base={num_nodes})')

    t0 = _time.time()
    src, dst = load_edges(args.train_csv, num_nodes, args.max_edges)
    if args.verify:
        verify_slice(src, dst, args.cap, args.min_count, num_nodes)
    nbr_ids, nbr_w, nrows = build_tables(src, dst, args.cap, args.topk,
                                         args.min_count,
                                         n_rows=num_nodes, base=num_nodes)
    np.savez(args.out, nbr_ids=nbr_ids, nbr_w=nbr_w)
    sz = os.path.getsize(args.out) / 2 ** 20
    stem = args.out[:-4] if args.out.endswith('.npz') else args.out
    np.save(stem + '_nbr_ids.npy', nbr_ids)
    np.save(stem + '_nbr_w.npy', nbr_w)
    print(f'Saved {args.out}: nbr_ids{nbr_ids.shape} int32 + nbr_w{nbr_w.shape} '
          f'float32, {sz:.0f}MB, rows_with_nbr={nrows}, '
          f'total {_time.time() - t0:.1f}s')
    print(f'Also saved memmap sidecars: {stem}_nbr_ids.npy / {stem}_nbr_w.npy '
          '(CoocTable prefers these for true mmap).')


if __name__ == '__main__':
    main()
