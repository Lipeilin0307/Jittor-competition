# -*- coding: utf-8 -*-
"""
craft_ds4_train.py
==================
CRAFT (NeurIPS'25, arXiv:2505.19408) ported to competition dataset4
(Taobao click bipartite graph: src=user, dst=item), Jittor implementation.

The vendor repo (D:\\work_d4\\vendor\\JittorGeometric, READ-ONLY) is used via
sys.path; nothing is pip-installed and no vendor file is modified.

Id-space design (decision-gate spec)
------------------------------------
* All ids are DENSE: src 1..680640, dst 1..862246, 0=padding, 862247=UNK dst.
* CRAFT's neighbor sampler keeps ONE undirected adjacency over the union of
  src/dst ids (get_neighbor_sampler: adj_list[src].append(dst...),
  adj_list[dst].append(src...)).  Dense src/dst spaces OVERLAP numerically,
  which would corrupt histories, so src ids are OFFSET by SRC_OFFSET =
  unk_dst_id (=862247) inside TemporalData only: sampler src id =
  src_dense + SRC_OFFSET (min 862248 > max dst dense 862246, != UNK).
  dst ids stay dense, so CRAFT's node_embedding (dst/item only) and
  set_min_idx(dst_min_idx=1) make predict()'s internal shift `x-1+1` an
  identity (padding 0 stays 0, UNK 862247 maps to itself).
* CRAFT n_nodes = unk_dst_id -> Embedding(unk_dst_id+1) rows cover
  0..862247 (0=padding, 1..862246 seen dst, 862247=UNK, UNK row never
  trained, stays at random init -- same spirit as GraphMixer's UNK row).

Data
----
train.csv (src,dst,time,split) is time-ordered: rows 1..14,006,368 split=0,
then 2,402,031 split=1 rows.  TGBSeqDataset is bypassed (it downloads from
the net); TemporalData is built directly from numpy arrays following
examples/craft_example.py lines 126-128.  The sampler adjacency contains ALL
edges (split=0 + split=1); find_neighbors_before(t) truncates by time, so
there is no leakage.  Training stream = split=0 edges only
(train_val_test_split_w_mask).

Taobao paper hyper-parameters (decision-gate spec): num_neighbors=60,
n_layers=2, hidden_size=128, n_heads=2, batch_size=400, lr=1e-4 Adam, BPR,
hidden_dropout=0.2, attn_dropout=0.1, emb_dropout=0.2, use_pos=True,
input_cat_time_intervals=False, output_cat_time_intervals=True,
output_cat_repeat_times=True (seen-dominant), num_output_layer=1,
skip_connection=True, layer_norm_eps=1e-12, initializer_range=0.02.
Training negatives: 1 uniform negative per positive (paper recipe,
TemporalDataLoader num_neg_sample=1).

Smoke usage (local):
  python craft_ds4_train.py --max_edges 150000 --epochs 2 --save_dir D:\\work_d4\\craft\\smoke_ckpt
Full usage (cloud):
  python craft_ds4_train.py --epochs 10 --save_dir ./ckpt_craft_ds4
"""
import argparse
import json
import os
import random
import sys
import time
import types

# ---------------------------------------------------------------------------
# pymetis stub: jittor_geometric/dataloader/__init__.py -> cluster_loader ->
# chunk_manager imports pymetis (not installed). CRAFT never uses graph
# partitioning, so a failing stub keeps the import chain alive WITHOUT
# touching vendor files. Must be injected before importing jittor_geometric.
# ---------------------------------------------------------------------------
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
from jittor_geometric.dataloader.temporal_dataloader import (
    TemporalDataLoader, get_neighbor_sampler)


def _np(x):
    """numpy view of either a numpy array or a jt.Var."""
    return x.numpy() if isinstance(x, jt.Var) else np.asarray(x)


def map_ids_to_dense(seen_ids, raw_ids, default):
    """Map raw ids to dense positions (index in seen_ids + 1); misses -> default."""
    pos = np.searchsorted(seen_ids, raw_ids)
    pos_clip = np.minimum(pos, len(seen_ids) - 1)
    found = seen_ids[pos_clip] == raw_ids
    return np.where(found, pos_clip + 1, default).astype(np.int64)


def patch_craft_unary_mlps(model):
    """WORKAROUND (vendor files untouched): jittor 1.3.8.5 CUDA
    Linear(in_features=1) silently returns inf / wrong values at some batch
    sizes (verified: rows=4 OK, 20 inf, 400 OK, 800 inf; explicit matmul
    (20,1)@(1,128) equally broken, rows=96 finite-but-WRONG). CRAFT's
    time_projection / repeat_times_projection are MLP(num_layers=1,
    input_dim=1), i.e. exactly this degenerate case. Replace their execute
    with the mathematically identical broadcast multiply-add (verified exact,
    maxerr=0 vs numpy at rows 20/96/800). Weights/bias Var objects are kept,
    so gradients and the optimizer are unaffected.
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
            y = x * w.t()          # (rows,1) * (1,out) -> (rows,out), exact
            if b is not None:
                y = y + b
            return y

        mlp.execute = _execute
        print(f'[patch] {name}: CUDA Linear(1,{w.shape[0]}) -> broadcast '
              'multiply-add (unary-Linear kernel bug workaround)', flush=True)


def preflight_shape_check(model, num_neighbors, n_nodes):
    """Assert CRAFT.predict I/O shapes (dimension-bug tripwire).

    src_neighb_seq [B,L], test_dst [B,K] -> pos [B], neg [B*(K-1)].
    Also exercises dst_min_idx identity shift and padding handling.
    """
    B, L, K = 4, num_neighbors, 5
    rng = np.random.default_rng(0)
    src_seq = rng.integers(0, n_nodes, size=(B, L)).astype(np.int64)  # 0 = padding
    seq_len = (src_seq != 0).sum(axis=1).astype(np.int64)
    nb_times = np.sort(rng.random((B, L)).astype(np.float32) * 1e6, axis=1)
    cur = np.full(B, 1.5e9, dtype=np.float32)
    test_dst = rng.integers(1, n_nodes + 1, size=(B, K)).astype(np.int64)
    last_up = np.full((B, K), -100000.0, dtype=np.float32)
    pos, neg = model.predict(
        src_neighb_seq=jt.array(src_seq), src_neighb_seq_len=jt.array(seq_len),
        src_neighb_interact_times=jt.array(nb_times), cur_pred_times=jt.array(cur),
        test_dst=jt.array(test_dst), dst_last_update_times=jt.array(last_up))
    assert tuple(src_seq.shape) == (B, L), f'src_neighb_seq {src_seq.shape}'
    assert tuple(test_dst.shape) == (B, K), f'test_dst {test_dst.shape}'
    assert tuple(pos.shape) == (B,), f'pos {pos.shape} != ({B},)'
    assert tuple(neg.shape) == (B * (K - 1),), f'neg {neg.shape} != ({B*(K-1)},)'
    pos_np, neg_np = pos.numpy(), neg.numpy()
    assert np.isfinite(pos_np).all() and np.isfinite(neg_np).all(), 'non-finite scores'
    print(f'[preflight] predict shapes OK: src_neighb_seq[{B},{L}] test_dst[{B},{K}] '
          f'-> pos[{B}] neg[{B*(K-1)}]; scores finite.')
    # in-place side effect check: forward() clamps seq_len==0 -> 1 on the INPUT var
    sl = jt.array(np.zeros(B, dtype=np.int64))
    _ = model.predict(jt.array(src_seq), sl, jt.array(nb_times), jt.array(cur),
                      jt.array(test_dst), jt.array(last_up))
    assert (sl.numpy() == 1).all(), 'expected forward() to clamp seq_len 0->1 in place'
    print('[preflight] seq_len in-place clamp 0->1 confirmed (vendor behavior).')
    # patched unary MLP must equal x@w.T+b exactly
    for name in ('time_projection', 'repeat_times_projection'):
        mlp = getattr(model, name, None)
        if mlp is None or mlp.num_layers != 1:
            continue
        lin = mlp.lins[0]
        x = rng.normal(0, 1000, size=(20, 1)).astype(np.float32)  # bad shape rows=20
        ref = x * lin.weight.numpy().T + lin.bias.numpy()
        err = np.abs(mlp.execute(jt.array(x)).numpy() - ref).max()
        assert err == 0.0, f'{name} patch mismatch: {err}'
    print('[preflight] patched unary MLPs exact vs numpy (rows=20, err=0).')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', type=str, default=r'F:\download\data_B\dataset4')
    p.add_argument('--idmap', type=str, default=r'D:\work_d4\idmap_ds4.npz')
    p.add_argument('--save_dir', type=str, required=True)
    p.add_argument('--max_edges', type=int, default=0,
                   help='0 = full split=0 stream; N>0 = first N split=0 edges (smoke).')
    p.add_argument('--train_all', action='store_true',
                   help='Fold split=1 into the training stream (all 16.4M edges); '
                        'kills the train/val time-decay gap for final submissions. '
                        'Val becomes meaningless (contaminated) -- score test only.')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=400)
    p.add_argument('--num_neighbors', type=int, default=60)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--n_heads', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--hidden_dropout', type=float, default=0.2)
    p.add_argument('--attn_dropout', type=float, default=0.1)
    p.add_argument('--emb_dropout', type=float, default=0.2)
    p.add_argument('--num_neg_sample', type=int, default=1)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--cpu', action='store_true')
    return p.parse_args(argv)


def main():
    args = parse_args()
    t_start = time.time()
    jt.flags.use_cuda = 0 if args.cpu else 1
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f'[config] {vars(args)}', flush=True)

    # ------------------------------------------------------------------
    # Load train.csv (+ idmap), dense remap
    # ------------------------------------------------------------------
    train_path = os.path.join(args.data_dir, 'train.csv')
    t0 = time.time()
    if args.max_edges > 0:
        # split=0 rows are the contiguous head of the time-ordered file, so the
        # first max_edges data rows are exactly the first max_edges split=0 edges.
        frame = pd.read_csv(train_path, nrows=args.max_edges,
                            dtype={'src': np.int64, 'dst': np.int64,
                                   'time': np.float64, 'split': np.int8})
        assert (frame['split'] == 0).all(), 'nrows head is not all split=0?'
    else:
        frame = pd.read_csv(train_path,
                            dtype={'src': np.int64, 'dst': np.int64,
                                   'time': np.float64, 'split': np.int8})
    print(f'[data] read {len(frame)} rows in {time.time()-t0:.1f}s', flush=True)

    raw_src = frame['src'].to_numpy(np.int64)
    raw_dst = frame['dst'].to_numpy(np.int64)
    times = frame['time'].to_numpy(np.float64)
    split = frame['split'].to_numpy(np.int64)

    im = np.load(args.idmap)
    seen_src, seen_dst = im['seen_src_ids'], im['seen_dst_ids']
    unk_dst_id = int(im['unk_dst_id'])
    n_src_dense, n_dst_dense = len(seen_src), len(seen_dst)
    assert unk_dst_id == n_dst_dense + 1
    if args.max_edges == 0:  # full run: idmap must cover the whole file
        assert np.array_equal(np.unique(raw_src), seen_src), 'idmap src mismatch'
        assert np.array_equal(np.unique(raw_dst), seen_dst), 'idmap dst mismatch'
    src_dense = map_ids_to_dense(seen_src, raw_src, 0)
    dst_dense = map_ids_to_dense(seen_dst, raw_dst, 0)
    assert (dst_dense != 0).all(), 'train dst missing from idmap'
    print(f'[data] dense: src 1..{n_src_dense}, dst 1..{n_dst_dense}, '
          f'UNK={unk_dst_id}, padding=0', flush=True)

    SRC_OFFSET = unk_dst_id  # sampler-space offset for src ids (see header)
    src_sampler = (src_dense + SRC_OFFSET).astype(np.int64)

    num_edges = len(frame)
    edge_ids = np.arange(num_edges, dtype=np.int64) + 1
    if args.max_edges > 0 or args.train_all:
        train_mask = np.arange(num_edges, dtype=np.int64)
        val_mask = np.zeros(0, dtype=np.int64)
    else:
        train_mask = np.nonzero(split == 0)[0].astype(np.int64)
        val_mask = np.nonzero(split == 1)[0].astype(np.int64)
    test_mask = np.zeros(0, dtype=np.int64)
    data = TemporalData(src=src_sampler, dst=dst_dense, t=times,
                        train_mask=train_mask, val_mask=val_mask,
                        test_mask=test_mask, edge_ids=edge_ids)
    train_data, _, _ = data.train_val_test_split_w_mask()
    print(f'[data] edges={num_edges} train_stream={len(train_mask)} '
          f'val_in_graph={len(val_mask)}', flush=True)

    # ------------------------------------------------------------------
    # Neighbor sampler over ALL loaded edges (time truncation prevents leakage)
    # ------------------------------------------------------------------
    t0 = time.time()
    full_neighbor_sampler = get_neighbor_sampler(data, 'recent', seed=1)
    sampler_build_s = time.time() - t0
    print(f'[sampler] build over {num_edges} edges ({2*num_edges} directed) '
          f'took {sampler_build_s:.1f}s', flush=True)

    train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size,
                                      num_neg_sample=args.num_neg_sample,
                                      shuffle=True, seed=args.seed)

    # ------------------------------------------------------------------
    # Model (Taobao paper hyper-parameters)
    # ------------------------------------------------------------------
    n_nodes = unk_dst_id  # Embedding(n_nodes+1) covers 0..UNK
    model = CRAFT(n_layers=args.n_layers, n_heads=args.n_heads,
                  hidden_size=args.hidden_size,
                  hidden_dropout_prob=args.hidden_dropout,
                  attn_dropout_prob=args.attn_dropout, hidden_act='gelu',
                  layer_norm_eps=1e-12, initializer_range=0.02,
                  n_nodes=n_nodes, max_seq_length=args.num_neighbors,
                  loss_type='BPR', use_pos=True, input_cat_time_intervals=False,
                  output_cat_time_intervals=True, output_cat_repeat_times=True,
                  num_output_layer=1, emb_dropout_prob=args.emb_dropout,
                  skip_connection=True)
    # src_min_idx is stored but unused by predict(); dst_min_idx=1 makes the
    # internal shift an identity on dense dst ids (0->0, UNK->UNK).
    model.set_min_idx(int(data.src.min()), 1)
    patch_craft_unary_mlps(model)
    optimizer = jt.nn.Adam(list(model.parameters()), lr=args.lr)
    n_params = sum(int(p.numel()) for p in model.parameters())
    print(f'[model] CRAFT n_nodes={n_nodes} (emb rows {n_nodes+1}), '
          f'params={n_params/1e6:.2f}M', flush=True)

    preflight_shape_check(model, args.num_neighbors, n_nodes)

    meta = {
        'model': {'n_layers': args.n_layers, 'n_heads': args.n_heads,
                  'hidden_size': args.hidden_size,
                  'hidden_dropout_prob': args.hidden_dropout,
                  'attn_dropout_prob': args.attn_dropout,
                  'emb_dropout_prob': args.emb_dropout,
                  'hidden_act': 'gelu', 'layer_norm_eps': 1e-12,
                  'initializer_range': 0.02, 'n_nodes': int(n_nodes),
                  'max_seq_length': args.num_neighbors, 'loss_type': 'BPR',
                  'use_pos': True, 'input_cat_time_intervals': False,
                  'output_cat_time_intervals': True,
                  'output_cat_repeat_times': True, 'num_output_layer': 1,
                  'skip_connection': True},
        'data': {'n_src_dense': int(n_src_dense), 'n_dst_dense': int(n_dst_dense),
                 'unk_dst_id': int(unk_dst_id), 'src_offset': int(SRC_OFFSET),
                 'dst_min_idx': 1, 'edges_loaded': int(num_edges),
                 'train_edges': int(len(train_mask)),
                 'val_edges_in_graph': int(len(val_mask)),
                 'max_edges_cap': int(args.max_edges)},
        'train': {'batch_size': args.batch_size, 'lr': args.lr,
                  'num_neg_sample': args.num_neg_sample, 'seed': args.seed,
                  'epochs_requested': args.epochs},
    }

    # ------------------------------------------------------------------
    # Training loop (mirrors examples/craft_example.py, numpy-native)
    # ------------------------------------------------------------------
    epoch_logs = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        t_ep = time.time()
        t_samp = t_model = 0.0
        n_skip = 0
        for bidx, batch in enumerate(train_loader):
            src_np = _np(batch.src).astype(np.int64)
            dst_np = _np(batch.dst).astype(np.int64)
            t_np = _np(batch.t).astype(np.float64)
            neg_np = _np(batch.neg_dst).astype(np.int64)
            bs = len(src_np)
            ts0 = time.time()
            src_neighb_seq, _, src_neighb_times = \
                full_neighbor_sampler.get_historical_neighbors_left(
                    node_ids=src_np, node_interact_times=t_np,
                    num_neighbors=args.num_neighbors)
            neighbor_num = (src_neighb_seq != 0).sum(axis=1)
            if neighbor_num.sum() == 0:
                n_skip += 1
                continue
            test_dst = np.concatenate([dst_np[:, None], neg_np[:, None]], axis=1)
            K = test_dst.shape[1]
            flat = test_dst.reshape(-1)
            tt = np.broadcast_to(t_np[:, None], test_dst.shape).reshape(-1)
            dst_last_nbr, _, dst_last_t = \
                full_neighbor_sampler.get_historical_neighbors_left(
                    node_ids=flat, node_interact_times=tt, num_neighbors=1)
            dst_last_t = dst_last_t.reshape(bs, -1).astype(np.float32)
            dst_last_t[dst_last_nbr.reshape(bs, -1) == 0] = -100000
            t_samp += time.time() - ts0

            assert src_neighb_seq.shape == (bs, args.num_neighbors)
            assert test_dst.shape == (bs, K)
            tm0 = time.time()
            loss, predicts, labels = model.calculate_loss(
                src_neighb_seq=jt.array(src_neighb_seq),
                src_neighb_seq_len=jt.array(neighbor_num),
                src_neighb_interact_times=jt.array(src_neighb_times),
                cur_pred_times=jt.array(t_np.astype(np.float32)),
                test_dst=jt.array(test_dst),
                dst_last_update_times=jt.array(dst_last_t))
            assert tuple(predicts.shape) == (2 * bs,), \
                f'predicts {predicts.shape} != ({2*bs},) [pos {bs} + neg {bs*(K-1)}]'
            assert tuple(labels.shape) == (2 * bs,)
            optimizer.zero_grad()
            optimizer.step(loss)
            lv = float(loss.item())
            t_model += time.time() - tm0
            losses.append(lv)
            if (bidx + 1) % args.log_interval == 0:
                done = bidx + 1 - n_skip
                rate = done / max(time.time() - t_ep, 1e-9)
                print(f'[e{epoch+1}] batch {bidx+1}/{len(train_loader)} '
                      f'loss={np.mean(losses[-args.log_interval:]):.4f} '
                      f'({rate:.1f} it/s, samp {t_samp:.1f}s model {t_model:.1f}s)',
                      flush=True)
        ep_s = time.time() - t_ep
        done = len(losses)
        log = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)) if losses else None,
               'batches_done': done, 'batches_skipped_empty_hist': n_skip,
               'seconds': ep_s, 'it_per_s': done / max(ep_s, 1e-9),
               'sampler_seconds': t_samp, 'model_seconds': t_model}
        epoch_logs.append(log)
        print(f'[epoch {epoch+1}] loss={log["train_loss"]:.4f} batches={done} '
              f'(skipped {n_skip}) {ep_s:.1f}s = {log["it_per_s"]:.2f} it/s '
              f'(sampler {t_samp:.1f}s / model {t_model:.1f}s)', flush=True)

        ckpt = os.path.join(args.save_dir, f'craft_ds4_epoch{epoch+1}.pkl')
        jt.save(model.state_dict(), ckpt)
        meta['train']['epochs_run'] = epoch + 1
        meta['train']['epoch_logs'] = epoch_logs
        meta['train']['sampler_build_seconds'] = sampler_build_s
        meta['train']['total_wall_seconds'] = time.time() - t_start
        with open(ckpt.replace('.pkl', '_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f'[ckpt] saved {ckpt}', flush=True)

    last = os.path.join(args.save_dir, 'craft_ds4_last.pkl')
    jt.save(model.state_dict(), last)
    with open(last.replace('.pkl', '_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'[done] total {time.time()-t_start:.1f}s; last ckpt {last}', flush=True)


if __name__ == '__main__':
    main()
