# -*- coding: utf-8 -*-
"""
craft_ds4_score.py
==================
Offline scoring for the CRAFT dataset4 port -- the decision-gate evaluator.

--mode val : sample --num_rows (default 5000) split=1 rows with the SAME
    protocol as eval_testlike.py (time-uniform linspace after mergesort by
    time; seed only drives candidate sampling), build 100 test-like candidates
    per row with the EMPIRICAL recipe (1 positive + (99-k) seen negatives drawn
    WITH replacement ~ cand_freq empirical frequencies [hot-row Polya urn] +
    k unseen drawn WITH replacement, k ~ B_hist_unseen histogram), score with
    CRAFT, rank by STRICT-GREATER counts (duplicates allowed), report
    MRR / Hit@10 overall and grouped by pair_seen (positive dst occurred for
    this src anywhere in split=0).
    Reference targets (GraphMixer v2): overall 0.366 / pair_seen=False 0.29.

--mode test: score test.csv rows (col0=src raw id, col1=time,
    col2..col101=100 candidate dst raw ids). Raw scores -> npz; submission
    csv = per-row min-max map to [0.01, 1.0] written with %.8f
    (--submit_map rankpct reproduces GraphMixer's unique-percentile map
    instead). Use --max_rows for a throughput smoke (do NOT run full 2.32M
    rows locally).

Id space: dense (src 1..N_src, dst 1..N_dst, 0=padding, UNK=862247); sampler
src ids are offset by SRC_OFFSET = unk_dst_id (see craft_ds4_train.py header).
Model hyper-parameters are read from <checkpoint root>_meta.json written by
craft_ds4_train.py.
"""
import argparse
import json
import os
import sys
import time
import types

# pymetis stub (see craft_ds4_train.py)
_stub = types.ModuleType('pymetis')


def _pymetis_missing(*a, **k):
    raise ImportError('pymetis is not installed (stubbed by craft_ds4); '
                      'graph partitioning is unused by the CRAFT port.')


_stub.part_graph = _pymetis_missing
sys.modules.setdefault('pymetis', _stub)

VENDOR = os.environ.get('CRAFT_VENDOR', r'D:\work_d4\vendor\JittorGeometric')
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

import numpy as np
import pandas as pd
import jittor as jt
from jittor_geometric.data.temporal import TemporalData
from jittor_geometric.nn.models.craft import CRAFT
from jittor_geometric.dataloader.temporal_dataloader import get_neighbor_sampler


def map_ids_to_dense(seen_ids, raw_ids, default):
    """Map raw ids to dense positions (index in seen_ids + 1); misses -> default."""
    pos = np.searchsorted(seen_ids, raw_ids)
    pos_clip = np.minimum(pos, len(seen_ids) - 1)
    found = seen_ids[pos_clip] == raw_ids
    return np.where(found, pos_clip + 1, default).astype(np.int64)


# ---------------------------------------------------------------------------
# Test-like candidate construction -- verbatim protocol from eval_testlike.py
# (lines 142-146, 213-307). RAW id space; remapped to dense afterwards.
# ---------------------------------------------------------------------------

def build_candidate_pools(dst0_raw, space_max):
    """Seen pool = split=0 dst ids; unseen = ids in [1, space_max] never a dst."""
    seen_ids, seen_counts = np.unique(dst0_raw, return_counts=True)
    unseen_ids = np.setdiff1d(np.arange(1, space_max + 1, dtype=np.int64), seen_ids)
    return seen_ids, seen_counts.astype(np.float64), unseen_ids


def construct_candidates_empirical(positives, seen_ids, seen_emp_w, unseen_ids,
                                   hist_k, rng, hot_frac=0.11, urn_kappa=600.0):
    """Empirical test-distribution candidates (eval_testlike.py recipe).

    Per row: k ~ hist_k unseen count; k unseen negatives uniform WITH
    replacement; 99-k seen negatives WITH replacement (hot rows ~hot_frac use a
    Polya/CRP urn over the row's own p_emp draws -> within-row duplicate
    clusters; cold rows iid ~ p_emp). Positive NOT excluded from the seen pool.
    All 100 slots shuffled. Returns (candidates, pos_col, dup_stats).
    """
    n = len(positives)
    ks = rng.choice(len(hist_k), size=n, p=hist_k / hist_k.sum())
    n_seen = 99 - ks

    u_off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(ks, out=u_off[1:])
    s_off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(n_seen, out=s_off[1:])
    flat_unseen = rng.integers(0, len(unseen_ids), size=int(u_off[-1]))

    p_seen = seen_emp_w / seen_emp_w.sum()
    hot = rng.random(n) < hot_frac
    iid_pool = rng.choice(len(seen_ids), size=int(s_off[-1]) + 16, p=p_seen)
    pool_off = 0
    flat_seen = np.empty(int(s_off[-1]), dtype=np.int64)
    for i in range(n):
        lo, hi = int(s_off[i]), int(s_off[i + 1])
        m = hi - lo
        if m == 0:
            continue
        if not hot[i]:
            flat_seen[lo:hi] = iid_pool[pool_off:pool_off + m]
            pool_off += m
            continue
        row = flat_seen[lo:hi]
        for t in range(m):
            if t == 0 or rng.random() < urn_kappa / (urn_kappa + t):
                row[t] = iid_pool[pool_off]
                pool_off += 1
            else:
                row[t] = row[rng.integers(0, t)]
    flat_seen = seen_ids[flat_seen]

    candidates = np.empty((n, 100), dtype=np.int64)
    for i in range(n):
        ns, nu = int(n_seen[i]), int(ks[i])
        row = candidates[i]
        row[0] = positives[i]
        row[1:1 + ns] = flat_seen[s_off[i]:s_off[i + 1]]
        row[1 + ns:] = unseen_ids[flat_unseen[u_off[i]:u_off[i + 1]]]
    order = np.argsort(rng.random(candidates.shape), axis=1)
    candidates = np.take_along_axis(candidates, order, axis=1)
    pos_col = (candidates == positives[:, None]).argmax(axis=1).astype(np.int64)

    srt = np.sort(candidates, axis=1)
    dup_slots = (np.diff(srt, axis=1) == 0).sum(axis=1).astype(np.float64)
    dup_stats = (float((dup_slots > 0).mean()), float(dup_slots.mean()))
    return candidates, pos_col, dup_stats


def select_val_rows(train_frame, split_values, num_rows):
    """Time-uniform sampling of split=1 rows (linspace over time-sorted segment)."""
    val_frame = train_frame[split_values == 1].sort_values('time', kind='mergesort')
    num_rows = min(num_rows, len(val_frame))
    sel = np.linspace(0, len(val_frame) - 1, num_rows, dtype=np.int64)
    return val_frame.iloc[sel].reset_index(drop=True)


def group_mrr(name, mask, ranks, out):
    n = int(mask.sum())
    if n == 0:
        print(f'  {name:<28s}: n=0 (skipped)')
        return
    rr = 1.0 / ranks[mask]
    mrr = float(rr.mean())
    h10 = float((ranks[mask] <= 10).mean())
    print(f'  {name:<28s}: n={n:<6d} MRR={mrr:.6f}  Hit@10={h10:.6f}')
    out[name.strip()] = {'n': n, 'mrr': mrr, 'hit10': h10}


def patch_craft_unary_mlps(model):
    """WORKAROUND (vendor files untouched): jittor 1.3.8.5 CUDA
    Linear(in_features=1) silently returns inf / wrong values at some batch
    sizes (verified in craft_ds4_train.py preflight). CRAFT's time_projection
    / repeat_times_projection are MLP(num_layers=1, input_dim=1); replace
    their execute with the mathematically identical broadcast multiply-add
    (exact, maxerr=0 vs numpy). Must match craft_ds4_train.py.
    """
    for name in ('time_projection', 'repeat_times_projection'):
        mlp = getattr(model, name, None)
        if mlp is None or getattr(mlp, 'num_layers', None) != 1:
            continue
        lin = mlp.lins[0]
        if lin.weight.shape[1] != 1:
            continue
        w, b = lin.weight, lin.bias

        def _execute(x, w=w, b=b):
            y = x * w.t()
            if b is not None:
                y = y + b
            return y

        mlp.execute = _execute
        print(f'[patch] {name}: CUDA Linear(1,{w.shape[0]}) -> broadcast '
              'multiply-add (unary-Linear kernel bug workaround)', flush=True)


def load_meta(checkpoint):
    meta_path = checkpoint.replace('.pkl', '_meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'meta json not found next to checkpoint: {meta_path}')
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_model(meta):
    m = meta['model']
    model = CRAFT(n_layers=m['n_layers'], n_heads=m['n_heads'],
                  hidden_size=m['hidden_size'],
                  hidden_dropout_prob=m['hidden_dropout_prob'],
                  attn_dropout_prob=m['attn_dropout_prob'],
                  hidden_act=m['hidden_act'], layer_norm_eps=m['layer_norm_eps'],
                  initializer_range=m['initializer_range'], n_nodes=m['n_nodes'],
                  max_seq_length=m['max_seq_length'], loss_type=m['loss_type'],
                  use_pos=m['use_pos'],
                  input_cat_time_intervals=m['input_cat_time_intervals'],
                  output_cat_time_intervals=m['output_cat_time_intervals'],
                  output_cat_repeat_times=m['output_cat_repeat_times'],
                  num_output_layer=m['num_output_layer'],
                  emb_dropout_prob=m['emb_dropout_prob'],
                  skip_connection=m['skip_connection'])
    model.set_min_idx(m['n_nodes'] + 1, 1)  # src_min_idx unused; dst_min_idx=1
    return model


def score_block(model, sampler, src_query_ids, times, cand_dense, num_neighbors):
    """Score (rows, 100) dense candidates. Returns float32 scores (rows, 100).

    src_query_ids: sampler-space src ids (dense + SRC_OFFSET; 0 = no history).
    """
    b = len(src_query_ids)
    ts0 = time.time()
    neigh_seq, _, neigh_times = sampler.get_historical_neighbors_left(
        node_ids=src_query_ids, node_interact_times=times,
        num_neighbors=num_neighbors)
    neigh_len = (neigh_seq != 0).sum(axis=1)
    flat = cand_dense.reshape(-1)
    tt = np.broadcast_to(times[:, None], cand_dense.shape).reshape(-1)
    last_nbr, _, last_t = sampler.get_historical_neighbors_left(
        node_ids=flat, node_interact_times=tt, num_neighbors=1)
    last_t = last_t.reshape(b, -1).astype(np.float32)
    last_t[last_nbr.reshape(b, -1) == 0] = -100000
    t_samp = time.time() - ts0

    assert neigh_seq.shape == (b, num_neighbors)
    assert cand_dense.shape[0] == b
    tm0 = time.time()
    with jt.no_grad():
        pos, neg = model.predict(
            src_neighb_seq=jt.array(neigh_seq),
            src_neighb_seq_len=jt.array(neigh_len),
            src_neighb_interact_times=jt.array(neigh_times),
            cur_pred_times=jt.array(times.astype(np.float32)),
            test_dst=jt.array(cand_dense),
            dst_last_update_times=jt.array(last_t))
        K = cand_dense.shape[1]
        assert tuple(pos.shape) == (b,), f'pos {pos.shape} != ({b},)'
        assert tuple(neg.shape) == (b * (K - 1),), f'neg {neg.shape}'
        # stitch back preserving the original column order (pos=col0, neg=col1..)
        scores = np.concatenate([pos.numpy()[:, None],
                                 neg.numpy().reshape(b, K - 1)], axis=1)
    t_model = time.time() - tm0
    return scores.astype(np.float32), t_samp, t_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', type=str, required=True, choices=['val', 'test'])
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_dir', type=str, default=r'F:\download\data_B\dataset4')
    p.add_argument('--idmap', type=str, default=r'D:\work_d4\idmap_ds4.npz')
    p.add_argument('--cand_freq', type=str, default=r'D:\work_d4\cand_freq_pairs.npz')
    p.add_argument('--unseen_hist', type=str, default=r'D:\work_d4\B_hist_unseen.npy')
    p.add_argument('--space_max_cache', type=str,
                   default=r'D:\work_d4\craft\test_cand_max.npy',
                   help='cache for max raw candidate id in test.csv (unseen pool size)')
    p.add_argument('--num_rows', type=int, default=5000, help='val mode rows')
    p.add_argument('--max_rows', type=int, default=0, help='test mode: 0=all, N=first N rows')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--submit_map', type=str, default='minmax',
                   choices=['minmax', 'rankpct'],
                   help='minmax: per-row min-max to [0.01,1.0] (spec); '
                        'rankpct: GraphMixer-style unique percentiles (ties by raw id).')
    p.add_argument('--out_csv', type=str, default=None)
    p.add_argument('--out_npz', type=str, default=None)
    p.add_argument('--metrics_json', type=str, default=None)
    p.add_argument('--cpu', action='store_true')
    return p.parse_args(argv)


def main():
    args = parse_args()
    t_start = time.time()
    jt.flags.use_cuda = 0 if args.cpu else 1
    print(f'[config] {vars(args)}', flush=True)

    meta = load_meta(args.checkpoint)
    SRC_OFFSET = int(meta['data']['src_offset'])
    unk_dst_id = int(meta['data']['unk_dst_id'])
    num_neighbors = int(meta['model']['max_seq_length'])
    print(f'[meta] {os.path.basename(args.checkpoint)}: SRC_OFFSET={SRC_OFFSET} '
          f'UNK={unk_dst_id} L={num_neighbors} hidden={meta["model"]["hidden_size"]}', flush=True)

    # ------------------------------------------------------------------
    # Load train.csv (all edges -> sampler; split=0 -> pools / pair_seen)
    # ------------------------------------------------------------------
    t0 = time.time()
    frame = pd.read_csv(os.path.join(args.data_dir, 'train.csv'),
                        dtype={'src': np.int64, 'dst': np.int64,
                               'time': np.float64, 'split': np.int8})
    print(f'[data] train.csv {len(frame)} rows read in {time.time()-t0:.1f}s', flush=True)
    raw_src = frame['src'].to_numpy(np.int64)
    raw_dst = frame['dst'].to_numpy(np.int64)
    times_all = frame['time'].to_numpy(np.float64)
    split = frame['split'].to_numpy(np.int64)

    im = np.load(args.idmap)
    seen_src, seen_dst = im['seen_src_ids'], im['seen_dst_ids']
    assert int(im['unk_dst_id']) == unk_dst_id
    src_dense = map_ids_to_dense(seen_src, raw_src, 0)
    dst_dense = map_ids_to_dense(seen_dst, raw_dst, 0)
    n_dst_dense = len(seen_dst)

    num_edges = len(frame)
    data = TemporalData(src=(src_dense + SRC_OFFSET).astype(np.int64),
                        dst=dst_dense, t=times_all,
                        edge_ids=np.arange(num_edges, dtype=np.int64) + 1)
    t0 = time.time()
    sampler = get_neighbor_sampler(data, 'recent', seed=1)
    print(f'[sampler] build over {num_edges} edges took {time.time()-t0:.1f}s', flush=True)

    model = build_model(meta)
    patch_craft_unary_mlps(model)
    model.load_state_dict(jt.load(args.checkpoint))
    model.eval()
    print(f'[model] loaded {args.checkpoint}', flush=True)

    m0 = split == 0
    src0, dst0 = src_dense[m0], dst_dense[m0]
    dst0_raw = raw_dst[m0]

    if args.mode == 'val':
        run_val(args, frame, split, raw_dst, dst0_raw, src0, dst0,
                seen_src, seen_dst, unk_dst_id, n_dst_dense,
                model, sampler, SRC_OFFSET, num_neighbors, t_start)
    else:
        run_test(args, seen_src, seen_dst, unk_dst_id,
                 model, sampler, SRC_OFFSET, num_neighbors, t_start)


def run_val(args, frame, split, raw_dst, dst0_raw, src0, dst0,
            seen_src, seen_dst, unk_dst_id, n_dst_dense,
            model, sampler, SRC_OFFSET, num_neighbors, t_start):
    # space_max = max raw candidate space (train dst max vs test candidates max)
    if os.path.exists(args.space_max_cache):
        test_max = int(np.load(args.space_max_cache))
    else:
        tf = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
        test_max = int(tf.iloc[:, 2:].to_numpy(np.int64).max())
        np.save(args.space_max_cache, np.int64(test_max))
        del tf
    space_max = max(int(raw_dst.max()), test_max)
    print(f'[val] space_max={space_max} (test_max={test_max})', flush=True)

    val_frame = select_val_rows(frame, split, args.num_rows)
    n = len(val_frame)
    val_srcs_raw = val_frame['src'].to_numpy(np.int64)
    val_dsts_raw = val_frame['dst'].to_numpy(np.int64)
    val_times = val_frame['time'].to_numpy(np.float64)
    val_srcs = map_ids_to_dense(seen_src, val_srcs_raw, 0)
    val_dsts = map_ids_to_dense(seen_dst, val_dsts_raw, unk_dst_id)
    print(f'[val] {n} rows (time-uniform from {int((split==1).sum())} split=1), '
          f't [{val_times.min():.0f}, {val_times.max():.0f}]', flush=True)

    rng = np.random.default_rng(args.seed)
    seen_ids, _, unseen_ids = build_candidate_pools(dst0_raw, space_max)
    freq = np.load(args.cand_freq)
    f_ids = freq['raw_id'].astype(np.int64)
    f_cnt = freq['count'].astype(np.float64)
    assert len(f_ids) and np.all(np.diff(f_ids) > 0), 'cand_freq raw_id not increasing'
    f_pos = np.searchsorted(f_ids, seen_ids)
    assert np.all(f_ids[f_pos] == seen_ids), 'split=0 dst ids missing from cand_freq'
    hist_k = np.load(args.unseen_hist).astype(np.float64)
    assert hist_k.ndim == 1 and hist_k.sum() > 0
    t0 = time.time()
    candidates_raw, pos_col, dup_stats = construct_candidates_empirical(
        val_dsts_raw, seen_ids, f_cnt[f_pos], unseen_ids, hist_k, rng)
    print(f'[val] candidates built in {time.time()-t0:.1f}s: seen pool {len(seen_ids)}, '
          f'unseen pool {len(unseen_ids)}; dup rows {100*dup_stats[0]:.2f}% '
          f'(mean dup slots {dup_stats[1]:.2f})', flush=True)
    candidates = map_ids_to_dense(seen_dst, candidates_raw.ravel(), unk_dst_id
                                  ).reshape(candidates_raw.shape)

    # pair_seen: any split=0 occurrence of the (src, dst) pair (dense spaces)
    pshift = max(1, int(n_dst_dense + 2).bit_length())
    pkeys = (src0.astype(np.int64) << np.int64(pshift)) | dst0
    pkeys.sort()
    qk = (val_srcs.astype(np.int64) << np.int64(pshift)) | val_dsts
    ppos = np.minimum(np.searchsorted(pkeys, qk), len(pkeys) - 1)
    pair_seen = pkeys[ppos] == qk
    print(f'[val] pair_seen positives: {int(pair_seen.sum())}/{n} '
          f'({100.0*pair_seen.mean():.2f}%)', flush=True)

    ranks = np.empty(n, dtype=np.int64)
    t_samp = t_model = 0.0
    t_score = time.time()
    for start in range(0, n, args.batch_size):
        stop = min(start + args.batch_size, n)
        b = stop - start
        scores, ds, dm = score_block(
            model, sampler, val_srcs[start:stop] + SRC_OFFSET,
            val_times[start:stop], candidates[start:stop], num_neighbors)
        t_samp += ds
        t_model += dm
        if np.isnan(scores).any():
            print(f'[WARNING] NaN scores in rows [{start}:{stop})', flush=True)
        pos_scores = scores[np.arange(b), pos_col[start:stop]]
        # empirical protocol: STRICTLY-greater counts (duplicates share scores)
        ranks[start:stop] = (scores > pos_scores[:, None]).sum(axis=1) + 1
    score_s = time.time() - t_score
    print(f'[val] scored {n} rows in {score_s:.1f}s '
          f'({n/max(score_s,1e-9):.1f} rows/s; sampler {t_samp:.1f}s / '
          f'model {t_model:.1f}s)', flush=True)

    rr = 1.0 / ranks
    out = {'mode': 'val', 'n': n, 'mrr': float(rr.mean()),
           'hit1': float((ranks <= 1).mean()), 'hit10': float((ranks <= 10).mean()),
           'median_rank': float(np.median(ranks)),
           'seconds_scoring': score_s, 'seconds_total': time.time() - t_start,
           'checkpoint': args.checkpoint, 'seed': args.seed}
    print('\n========== CRAFT ds4 test-like validation (split=1) ==========')
    print(f'  MRR         : {out["mrr"]:.6f}')
    print(f'  Hit@1       : {out["hit1"]:.6f}')
    print(f'  Hit@10      : {out["hit10"]:.6f}')
    print(f'  Median rank : {out["median_rank"]:.1f}')
    print('\n-- Grouped by pair_seen (positive dst seen for this src in split=0) --')
    group_mrr('pair_seen = True', pair_seen, ranks, out)
    group_mrr('pair_seen = False', ~pair_seen, ranks, out)
    print('  (reference GraphMixer v2: overall 0.366 / pair_seen=False 0.29)')

    if args.metrics_json:
        with open(args.metrics_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'[val] metrics -> {args.metrics_json}')
    if args.out_npz:
        np.savez(args.out_npz, ranks=ranks, pair_seen=pair_seen,
                 val_srcs_raw=val_srcs_raw, val_dsts_raw=val_dsts_raw,
                 val_times=val_times, pos_col=pos_col)
        print(f'[val] per-row dump -> {args.out_npz}')
    print(f'\nTotal wall time: {time.time()-t_start:.1f}s', flush=True)


def run_test(args, seen_src, seen_dst, unk_dst_id,
             model, sampler, SRC_OFFSET, num_neighbors, t_start):
    t0 = time.time()
    tf = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
    if args.max_rows > 0:
        tf = tf.iloc[:args.max_rows]
    print(f'[test] test.csv read in {time.time()-t0:.1f}s; scoring {len(tf)} rows '
          f'(max_rows={args.max_rows})', flush=True)
    test_srcs_raw = tf['src'].to_numpy(np.int64)
    test_times = tf.iloc[:, 1].to_numpy(np.float64)
    test_cands_raw = tf.iloc[:, 2:].to_numpy(np.int64)
    n = len(tf)
    assert test_cands_raw.shape[1] == 100

    test_srcs = map_ids_to_dense(seen_src, test_srcs_raw, 0)  # miss -> 0 = no history
    test_cands = map_ids_to_dense(seen_dst, test_cands_raw.ravel(), unk_dst_id
                                  ).reshape(n, 100)
    n_unk = int((test_cands == unk_dst_id).sum())
    print(f'[test] UNK candidate slots: {n_unk}/{test_cands.size} '
          f'({100.0*n_unk/test_cands.size:.2f}%); '
          f'unseen src rows: {int((test_srcs==0).sum())}', flush=True)

    # sampler-space src: dense + OFFSET where seen, else 0 (padding, empty list)
    src_query = np.where(test_srcs > 0, test_srcs + SRC_OFFSET, 0).astype(np.int64)

    all_scores = np.empty((n, 100), dtype=np.float32)
    t_samp = t_model = 0.0
    t_score = time.time()
    for start in range(0, n, args.batch_size):
        stop = min(start + args.batch_size, n)
        scores, ds, dm = score_block(
            model, sampler, src_query[start:stop], test_times[start:stop],
            test_cands[start:stop], num_neighbors)
        t_samp += ds
        t_model += dm
        all_scores[start:stop] = scores
        done = stop
        if done % (args.batch_size * 10) == 0 or done == n:
            rate = done / max(time.time() - t_score, 1e-9)
            print(f'[test] {done}/{n} rows ({rate:.1f} rows/s, '
                  f'sampler {t_samp:.1f}s / model {t_model:.1f}s)', flush=True)
    score_s = time.time() - t_score
    print(f'[test] scored {n} rows in {score_s:.1f}s '
          f'({n/max(score_s,1e-9):.1f} rows/s; sampler {t_samp:.1f}s / '
          f'model {t_model:.1f}s)', flush=True)
    if n < 2322538:
        full_est = 2322538 / max(n / max(score_s, 1e-9), 1e-9)
        print(f'[test] extrapolated FULL 2,322,538-row scoring: ~{full_est/60:.1f} min '
              f'(linear in rows)', flush=True)

    if args.out_npz:
        np.savez(args.out_npz, scores=all_scores, src=test_srcs_raw,
                 times=test_times, candidates=test_cands_raw)
        print(f'[test] raw scores -> {args.out_npz} {all_scores.shape}', flush=True)

    if args.out_csv:
        if args.submit_map == 'rankpct':
            # GraphMixer-style unique percentiles in (0,1], ties by raw cand id
            out = np.empty_like(all_scores, dtype=np.float64)
            for i in range(n):
                order = np.lexsort((test_cands_raw[i], all_scores[i]))
                out[i, order] = np.arange(1, 101, dtype=np.float64)
            out /= 100.0
        else:  # minmax per spec: per-row min-max map to [0.01, 1.0]
            smin = all_scores.min(axis=1, keepdims=True)
            smax = all_scores.max(axis=1, keepdims=True)
            rng_ = np.maximum(smax - smin, 1e-12)
            out = 0.01 + 0.99 * (all_scores - smin) / rng_
        np.savetxt(args.out_csv, out, delimiter=',', fmt='%.8f')
        print(f'[test] submission ({args.submit_map}) -> {args.out_csv} '
              f'{out.shape} range=[{out.min():.4f},{out.max():.4f}]', flush=True)
    print(f'\nTotal wall time: {time.time()-t_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
