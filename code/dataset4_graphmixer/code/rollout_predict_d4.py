#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rollout_predict_d4.py
=====================
Autoregressive rollout prediction for dataset4 (GraphMixer/Jittor) -- a port of
the validated ds3 prototype (rollout_predict_d3.py) to the ds4 pipeline.

Rollout idea (identical semantics to the ds3 version)
-----------------------------------------------------
The training script's test inference builds every src history from train.csv
edges only, frozen at the train cutoff. ds4's test set spans 2.32M time-sorted
rows, so for late queries the frozen history is badly outdated.

Rows are processed in time order (test.csv is asserted time-sorted). After
scoring each batch, rows whose prediction passes a confidence gate inject
(src, top1_dst, t_test) -- optionally the top-k predicted edges, see
--inject_topk -- into a dynamic per-src history store (deque(maxlen=L), the
same truncation the static builder uses). Later rows then build their model
input from the store (train edges + injected predicted edges).

Gating score convention (same as ds3, IMPORTANT)
------------------------------------------------
The gate works on per-row T=1 softmax probabilities over the 100 candidate
scores, NOT on rank percentiles (rank_percentiles assigns unique values via
lexsort, so top1 is always 1.00 and the margin is always 1/n_cand -- both
gates would be meaningless there):
    pass  <=>  (p_top1 - p_top2 >= --gate_margin) AND (p_top1 >= --gate_top1)
Default --gate_margin 1.0 is impossible to satisfy (probs sum to 1), so the
default rollout injects NOTHING and its output degenerates to the frozen
baseline -- a safe default. Lower it explicitly (e.g. 0.10) to enable
injections. --gate_top1 0 disables the top1 floor.
--inject_topk k > 1: for a row that passes the gate, inject the top-k
candidates (score order, all stamped with the row's time). k=1 reproduces the
ds3 behavior exactly.

ds4-specific differences vs the ds3 script (all following the ds4 training
script train_graphmixer_jt.py / predict_d4_fused.py, NOT the ds3 ones)
----------------------------------------------------------------------
  * dense_id_remap id space: seen src -> 1..N_src, seen dst -> 1..N_dst,
    0 = padding, N_dst+1 = trainable UNK row (unseen test candidates map to
    it; unseen test srcs map to 0). idmap_ds4.npz is verified against
    train.csv id sets, same as the training/predict scripts.
  * Per-tower node2vec feature tables (--node_src / --node_dst), never a
    single shared table. Row 0 of both is zeroed here (caller-owned padding
    invariant, same as the training script).
  * gap_scale is the p90 of PER-SOURCE consecutive positive-time gaps (ds4
    builder), not the global-stream p90 of ds3. This script recomputes it with
    a verbatim replica (compute_gap_scale_persrc) instead of calling the full
    builder, which avoids allocating the two (n_train, 40) snapshot matrices
    (~5 GB at ds4 scale). --mode selftest asserts replica == builder on random
    data before anything else runs.
  * History matrices are int32 (ds4 builder; dense dst ids fit), gaps float32
    -- same dtypes as the training/test path.
  * hist_pos_time_only is ON by default (ds4 training runs always set it);
    time<=0 edges never enter histories / gap stats / injections.
  * Extra scorer block layout [recency(6) | cf(4 or 8) | approx-cooc(3)].
    Defaults match the saved_gm_d4_v3s45 run: recency ON, CF LITE 4-dim
    (--cf_no_cooc at training time), approx-cooc 3-dim ON (CoocTable from
    cooc_ds4.npz) -> extra_feat_dim = 13, heuristic_dim = 3 (no known_flag).
    The constructed scorer width is checked against the checkpoint BEFORE the
    expensive data/feature loading, with an explicit flag-fix hint on mismatch.
  * UNK-injection guard: when a gated top-k candidate is the UNK row
    (N_dst+1), that edge is NOT injected by default (--skip_unk_inject):
    during training UNK only ever appeared as a negative candidate, never
    inside a history sequence, so injecting it would feed the history encoder
    an out-of-distribution token. --inject_unk restores the naive behavior.
  * Streaming I/O: test.csv (1.7 GB) is read in --chunk_rows pandas chunks and
    the output csv is appended chunk by chunk; the reference comparison in
    selftest/frozen mode also streams the reference file, so peak extra RAM is
    one chunk. Progress bar with ETA (tqdm, total = fast newline count).

Static-by-design (deliberate, same as ds3 v1)
---------------------------------------------
  * ONLY the src-side history sequence + time gaps are dynamic. recency / cf /
    heuristics / approx-cooc TABLES stay static (precomputed from train.csv,
    exactly the values used by the frozen path). The approx-cooc block, when
    enabled, receives the CURRENT dynamic history rows through make_extra's
    hists argument (its lookup table is static).
  * Batching: rows inside one batch are scored against the store state at the
    START of the batch; gated injections are applied row-by-row in time order
    AFTER the batch is scored. A row never sees same-batch injections
    (batch-granularity approximation; use --batch_size 1 for strict per-row
    rollout). Test rows are processed in file order, asserted time-sorted.

Modes (--mode)
--------------
  selftest : (1) unit test -- DynamicHistory rows byte-identical to
             train_graphmixer_jt.build_histories_with_time on random small
             data (both pos_time_only settings, per-source gap-scale replica
             equality, inject/clone/truncation semantics, rank_percentiles
             equivalence incl. candidate-id tie-breaking); then
             (2) FROZEN scoring of the test set streamed side-by-side with
             the reference {save_dir}/{dataset}.csv, exact global Pearson of
             the rank percentiles + top1 agreement printed (expect >0.999).
             Writes NOTHING. Exit code 0 = pass, 1 = unit-test fail,
             2 = sanity below threshold. Use --max_rows 100000 for a smoke.
  frozen   : frozen scoring (store never injected) -> write --out csv
             (100 cols, row-wise rank percentiles in [0.01, 1.0], ties broken
             by raw candidate id = submission semantics), then the same
             reference sanity print as selftest (skipped with a warning if
             the reference is missing).
  rollout  : rollout scoring with the gate -> write --out csv, print
             injection stats. No reference comparison (the outputs are
             supposed to differ).

Prerequisites on the GPU box (same directory layout as the training run)
------------------------------------------------------------------------
Run from the ds4 work directory (the one containing train_graphmixer_jt.py,
model_graphmixer_jt.py, cf_features.py, recency_features.py), e.g.
/root/autodl-tmp/work_ds4/. Set GM_PAGEABLE_HOST=1 like the training runs
(the imported training module applies the flag at import time, before any
CUDA allocation):

  cd /root/autodl-tmp/work_ds4 && export GM_PAGEABLE_HOST=1

  # 0) quick smoke sanity (100k rows, ~minutes):
  python3 rollout_predict_d4.py --mode selftest --max_rows 100000
  # 1) full frozen sanity vs the training script's own predictions:
  python3 rollout_predict_d4.py --mode selftest \
      --save_dir ./saved_gm_d4_v3s45
  # 2) frozen baseline csv:
  python3 rollout_predict_d4.py --mode frozen \
      --save_dir ./saved_gm_d4_v3s45 --out ./rollout_d4_out/frozen.csv
  # 3) rollout, top1 injection with margin 0.10:
  python3 rollout_predict_d4.py --mode rollout \
      --save_dir ./saved_gm_d4_v3s45 --gate_margin 0.10 --inject_topk 1 \
      --out ./rollout_d4_out/dataset4_rollout_m10.csv

All feature/architecture flags must match the checkpoint's training run
(defaults = the saved_gm_d4_v3s45 recipe). The checkpoint-width precheck and
verify_checkpoint_shapes fail loudly on any mismatch.
"""
import argparse
import os
import sys
import time as _time
from collections import deque

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# NOTE: jittor / train_graphmixer_jt are imported LAZILY inside main() /
# run_unit_selftest() so that this module (DynamicHistory in particular) stays
# importable on boxes without jittor. The lazy import happens before ANY other
# jittor usage, so the training module's jittor flags setup (use_cuda=1,
# GM_PAGEABLE_HOST handling) is applied exactly as at train time.

SERVER_WORK_DIR = '/root/autodl-tmp/work_ds4'


# ============================================================================
# Args
# ============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', type=str, default='selftest',
                        choices=['selftest', 'frozen', 'rollout'],
                        help="selftest: unit test + frozen sanity vs the reference csv, "
                             "writes nothing (default, safe). frozen: frozen scoring -> "
                             "--out. rollout: gated-injection rollout scoring -> --out.")
    # data / checkpoint
    parser.add_argument('--dataset', type=str, default='dataset4')
    parser.add_argument('--data_dir', type=str, default=SERVER_WORK_DIR,
                        help='Directory containing {dataset}/train.csv and '
                             '{dataset}/test.csv.')
    parser.add_argument('--save_dir', type=str, default='./saved_gm_d4_v3s45',
                        help='Checkpoint dir: {save_dir}/{dataset}_graphmixer.pkl(+_meta.npy); '
                             'the sanity reference is {save_dir}/{dataset}.csv.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path (default: {save_dir}/{dataset}_graphmixer.pkl).')
    parser.add_argument('--ref', type=str, default=None,
                        help='Override the sanity reference csv (default: {save_dir}/{dataset}.csv; '
                             '.gz fallback is automatic).')
    # feature / artifact paths (defaults = the server work dir layout)
    parser.add_argument('--node_src', type=str,
                        default=f'{SERVER_WORK_DIR}/node_features_ds4_src.npy',
                        help='Per-src node2vec table aligned to the dense model id space.')
    parser.add_argument('--node_dst', type=str,
                        default=f'{SERVER_WORK_DIR}/node_features_ds4_dst.npy',
                        help='Per-dst node2vec table aligned to the dense model id space.')
    parser.add_argument('--idmap', type=str,
                        default=f'{SERVER_WORK_DIR}/idmap_ds4.npz',
                        help='idmap_ds4.npz from gen_features_ds4.py (verified against train.csv).')
    parser.add_argument('--heuristics_dir', type=str, default=SERVER_WORK_DIR,
                        help='Directory with heuristics_{dataset}_degree.npy / _popularity.npy / '
                             '_edge_count.pkl.')
    parser.add_argument('--cf_dir', type=str, default=SERVER_WORK_DIR,
                        help='Directory holding CF/cooc artifacts (cooc_ds4.npz); --cooc_table '
                             'overrides the cooc table path.')
    parser.add_argument('--cooc_table', type=str, default=None,
                        help='Path to cooc_ds4.npz (default: {cf_dir}/cooc_ds4.npz). Only used '
                             'when --use_cooc is on.')
    # model hyper-parameters (must match the checkpoint; defaults = v3s45 recipe)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--history_length', type=int, default=40)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--mlp_ratio', type=float, default=2.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--time_decay', type=float, default=0.5)
    # feature switches (defaults = v3s45 recipe: heuristics + recency + cf-lite + approx-cooc)
    parser.add_argument('--use_heuristics', dest='use_heuristics', action='store_true', default=True)
    parser.add_argument('--no_heuristics', dest='use_heuristics', action='store_false')
    parser.add_argument('--use_known_flag', dest='use_known_flag', action='store_true', default=False,
                        help='Append the dst_known heuristic dim (v3s45 did NOT use it).')
    parser.add_argument('--use_recency_feats', dest='use_recency_feats', action='store_true', default=True)
    parser.add_argument('--no_recency_feats', dest='use_recency_feats', action='store_false')
    parser.add_argument('--use_cf_feats', dest='use_cf_feats', action='store_true', default=True)
    parser.add_argument('--no_cf_feats', dest='use_cf_feats', action='store_false')
    parser.add_argument('--cf_full', dest='cf_full', action='store_true', default=False,
                        help='CFStats full 8-dim block (use_cooc=True). Default off = ds4 lite '
                             '4-dim block (training ran --cf_no_cooc; the exact cooc matrix is '
                             '~44 GB at ds4 scale).')
    parser.add_argument('--use_cooc', dest='use_cooc', action='store_true', default=True,
                        help='Append the 3-dim approx-cooc CoocTable block AFTER the CF block.')
    parser.add_argument('--no_cooc', dest='use_cooc', action='store_false')
    parser.add_argument('--hist_pos_time_only', dest='hist_pos_time_only', action='store_true', default=True,
                        help='Histories + gap stats from time>0 edges only (ds4 training default).')
    parser.add_argument('--no_hist_pos_time_only', dest='hist_pos_time_only', action='store_false')
    # rollout switches
    parser.add_argument('--inject_topk', type=int, default=1,
                        help='Inject the top-k predicted edges per gated row (score order, all '
                             'stamped with the row time). 1 = ds3 semantics.')
    parser.add_argument('--gate_margin', type=float, default=1.0,
                        help='Confidence gate: min top1-top2 margin on per-row T=1 softmax '
                             'probabilities. DEFAULT 1.0 is unsatisfiable -> rollout degenerates '
                             'to frozen (safe). Use e.g. 0.05/0.10/0.15 to enable injections.')
    parser.add_argument('--gate_top1', type=float, default=0.0,
                        help='Confidence gate: min top1 softmax probability. 0 = disabled.')
    parser.add_argument('--skip_unk_inject', dest='skip_unk_inject', action='store_true', default=True,
                        help='Do not inject edges whose predicted dst is the UNK row (default on; '
                             'UNK never appeared inside training histories).')
    parser.add_argument('--inject_unk', dest='skip_unk_inject', action='store_false')
    # runtime
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Scoring batch size (same as the training script\'s test inference; '
                             'rows never see same-batch injections -- 1 = strict per-row rollout).')
    parser.add_argument('--chunk_rows', type=int, default=100000,
                        help='test.csv streaming chunk size (rows).')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Score only the first N test rows (debug/smoke).')
    parser.add_argument('--out', type=str, default='./rollout_d4_out/dataset4.csv',
                        help='Output csv path (frozen/rollout modes).')
    args = parser.parse_args(argv)
    if args.inject_topk < 1:
        parser.error('--inject_topk must be >= 1')
    if args.mode in ('frozen', 'rollout') and not args.out:
        parser.error(f'--mode {args.mode} requires --out.')
    return args


# ============================================================================
# Dynamic history store
# ----------------------------------------------------------------------------
# Row-format contract, byte-identical to the ds4
# train_graphmixer_jt.build_histories_with_time:
#   * per-src deque(maxlen=L) of past DENSE dst ids (int) + parallel deque of
#     times (float); oldest dropped once past L -> "most recent L interactions";
#   * history row: np.int32, shape (L,), RIGHT-aligned (most recent at the
#     last slot), zero padding on the LEFT  (target[-len(history):] = history);
#     (ds4 builder uses int32 -- dense dst ids are <= 862247 and jittor's
#     nn.Embedding accepts int32 indices);
#   * gap row: np.float32, shape (L,), same alignment;
#     gap_j = log1p(max(row_time - hist_time_j, 0.0)) / gap_scale;
#   * gap_scale: 90th percentile of PER-SOURCE consecutive positive time diffs
#     over the whole train stream, floored at 1.0 (ds4 builder semantics --
#     NOT the global-stream p90 of the ds3 builder);
#   * pos_time_only=True (ds4 default): edges with time <= 0 never enter any
#     history (they also stay out of the gap-scale stats), exactly like the
#     static builder.
# ============================================================================

def compute_gap_scale_persrc(source_ids, time_values, source_min, pos_time_only=True):
    """90th percentile of PER-SOURCE consecutive positive time diffs (min 1.0).

    Verbatim replica of the gap-scale block in the ds4
    train_graphmixer_jt.build_histories_with_time; kept separate so main() can
    obtain gap_scale without allocating the builder's two (n_train, L)
    snapshot matrices (~5 GB at ds4 scale). --mode selftest asserts
    replica == builder on random data.
    """
    num_sources = int(source_ids.max() - source_min + 1)
    last_time = np.full(num_sources, -1.0, dtype=np.float64)
    time_diffs = []
    for index in range(len(source_ids)):
        t = float(time_values[index])
        if pos_time_only and t <= 0.0:
            continue  # time=0 edges excluded from histories and gap stats
        src_idx = int(source_ids[index] - source_min)
        prev = last_time[src_idx]
        if prev >= 0.0 and t > prev:
            time_diffs.append(t - prev)
        last_time[src_idx] = t
    gap_scale = float(np.percentile(time_diffs, 90)) if time_diffs else 1.0
    return max(gap_scale, 1.0)


class DynamicHistory:
    """src -> (dst deque, time deque), replayed from train edges, then grown
    by gated prediction injection."""

    def __init__(self, num_sources, history_length, gap_scale, pos_time_only=True):
        self.L = int(history_length)
        self.gap_scale = float(gap_scale)
        self.pos_time_only = bool(pos_time_only)
        self.histories = [deque(maxlen=self.L) for _ in range(num_sources)]
        self.history_times = [deque(maxlen=self.L) for _ in range(num_sources)]
        self.num_sources = num_sources

    @classmethod
    def from_train(cls, source_ids, destination_ids, time_values, source_min,
                   history_length, split_index, pos_time_only=True, gap_scale=None,
                   min_num_sources=0):
        """Replay train rows [0, split_index) into the store, replicating the
        append logic of the ds4 build_histories_with_time exactly (same casts,
        same pos_time_only filter, same deque(maxlen=L) truncation).

        min_num_sources: lower bound on the src-table size. The ds4 dense
        remap bounds every test src to [0, N_src], so source_count always
        suffices; the parameter is kept for parity with the ds3 version.
        """
        num_sources = max(int(source_ids.max() - source_min + 1), int(min_num_sources))
        if gap_scale is None:
            gap_scale = compute_gap_scale_persrc(source_ids, time_values, source_min,
                                                 pos_time_only)
        store = cls(num_sources, history_length, gap_scale, pos_time_only)
        for i in range(split_index):
            t = float(time_values[i])
            if pos_time_only and t <= 0.0:
                continue  # time=0 edges never enter history sequences
            src_idx = int(source_ids[i] - source_min)
            store.histories[src_idx].append(int(destination_ids[i]))
            store.history_times[src_idx].append(t)
        return store

    def clone(self):
        """Deep-enough copy (fresh deques, same content) so each rollout run
        starts from the identical train-final state."""
        other = DynamicHistory(self.num_sources, self.L, self.gap_scale, self.pos_time_only)
        other.histories = [deque(d, maxlen=self.L) for d in self.histories]
        other.history_times = [deque(d, maxlen=self.L) for d in self.history_times]
        return other

    def build_row(self, src_model, row_time):
        """One (history, gaps) row, byte-identical to the ds4 static builder's
        per-row output (int32/float32 dtypes, right alignment, gap formula)."""
        history = self.histories[src_model]
        hist_times = self.history_times[src_model]
        hist_row = np.zeros(self.L, dtype=np.int32)
        gap_row = np.zeros(self.L, dtype=np.float32)
        if history:
            hist_row[-len(history):] = list(history)
            gaps = [np.log1p(max(float(row_time) - t, 0.0)) / self.gap_scale
                    for t in hist_times]
            gap_row[-len(gaps):] = gaps
        return hist_row, gap_row

    def build_batch(self, src_models, row_times):
        """Stacked (B, L) int32 / (B, L) float32 history+gap matrices."""
        B = len(src_models)
        hist = np.zeros((B, self.L), dtype=np.int32)
        gaps = np.zeros((B, self.L), dtype=np.float32)
        for j in range(B):
            hist[j], gaps[j] = self.build_row(int(src_models[j]), float(row_times[j]))
        return hist, gaps

    def inject(self, src_model, dst, row_time):
        """Append a (predicted) edge. Same casts and same pos_time_only filter
        as the static builder's append step."""
        t = float(row_time)
        if self.pos_time_only and t <= 0.0:
            return False
        self.histories[src_model].append(int(dst))
        self.history_times[src_model].append(t)
        return True


# ============================================================================
# Rank percentiles + confidence gate
# ============================================================================

def rank_percentiles(scores, candidates):
    """Row-wise unique rank percentiles in [1/n, 1.0]; highest score -> 1.0.

    Vectorized equivalent of train_graphmixer_jt.rank_percentiles: per-row
    np.lexsort((candidates, scores)) semantics -- primary key score ASCENDING,
    exact float ties broken by the RAW candidate id ascending (submission
    semantics, same as the ds4 training/predict scripts). A stable sort by
    candidate id followed by a stable sort by score is exactly that lexsort.
    --mode selftest asserts equality with the training version, including
    engineered exact ties.
    """
    n = scores.shape[1]
    o1 = np.argsort(candidates, axis=1, kind='stable')
    s_by_cand = np.take_along_axis(scores, o1, axis=1)
    o2 = np.argsort(s_by_cand, axis=1, kind='stable')
    order = np.take_along_axis(o1, o2, axis=1)
    out = np.empty(scores.shape, dtype=np.float64)
    ranks = np.broadcast_to(np.arange(1, n + 1, dtype=np.float64) / n,
                            scores.shape)
    np.put_along_axis(out, order, ranks, axis=1)
    return out.astype(np.float32)


def gate_decisions(scores, cand_dense, gate_margin, gate_top1, topk=1, unk_id=None):
    """Per-row confidence gate on T=1 softmax probabilities over the candidate
    scores (see module docstring for why softmax, not rank_percentiles).

    scores: (B, C) raw model scores. cand_dense: (B, C) DENSE model-space dst
    ids (histories store dense ids; injected edges must be dense too).

    Returns (inject_mask (B,) bool, inject_dst (B, k) int64). A row passing
    the gate injects its top-k candidates in descending-score order; entries
    equal to unk_id are replaced by -1 (skipped at injection time) when
    unk_id is not None.
    """
    z = scores.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    order = np.argsort(-p, axis=1, kind='stable')
    ps = np.take_along_axis(p, order, axis=1)
    p_top1 = ps[:, 0]
    p_top2 = ps[:, 1] if ps.shape[1] > 1 else np.zeros_like(p_top1)
    ok = ((p_top1 - p_top2) >= gate_margin) & (p_top1 >= gate_top1)
    k = min(int(topk), scores.shape[1])
    inject_dst = np.take_along_axis(cand_dense, order[:, :k], axis=1)
    if unk_id is not None:
        inject_dst = np.where(inject_dst == unk_id, -1, inject_dst)
    return ok, inject_dst


# ============================================================================
# Selftest helpers: fast row count + streaming Pearson accumulator
# ============================================================================

def count_csv_rows(path):
    """Data-row count of a csv (newline count minus the header), one fast
    binary pass (~seconds for 1.7 GB). Used for the tqdm total / ETA."""
    n = 0
    last = b''
    with open(path, 'rb') as f:
        while True:
            buf = f.read(1 << 23)
            if not buf:
                break
            n += buf.count(b'\n')
            last = buf
    if last and not last.endswith(b'\n'):
        n += 1
    return max(n - 1, 0)  # minus header


class RefComparer:
    """Streaming exact Pearson + top1-agreement accumulator against the
    reference prediction csv ({save_dir}/{dataset}.csv). Reads the reference
    chunk-by-chunk in lockstep with our scored rows -- never holds more than
    one chunk of either side."""

    def __init__(self, ref_path, chunk_rows):
        self.ref_path = ref_path
        self.reader = pd.read_csv(ref_path, header=None, chunksize=chunk_rows)
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0
        self.top1_hits = 0

    def update(self, out_block):
        """out_block: (m, C) float32 rank percentiles we just produced."""
        m = out_block.shape[0]
        ref = self.reader.get_chunk(m).to_numpy(np.float64)
        if ref.shape != out_block.shape:
            raise RuntimeError(
                f'{self.ref_path}: reference row/shape mismatch '
                f'(got {ref.shape}, expected {out_block.shape}); wrong reference file?')
        x = out_block.astype(np.float64).ravel()
        y = ref.ravel()
        self.n += x.size
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxx += float((x * x).sum())
        self.syy += float((y * y).sum())
        self.sxy += float((x * y).sum())
        self.top1_hits += int((out_block.argmax(axis=1) == ref.argmax(axis=1)).sum())

    def report(self):
        if self.n == 0:
            return float('nan'), float('nan')
        cov = self.sxy - self.sx * self.sy / self.n
        vx = self.sxx - self.sx * self.sx / self.n
        vy = self.syy - self.sy * self.sy / self.n
        pearson = cov / float(np.sqrt(vx * vy)) if vx > 0 and vy > 0 else float('nan')
        top1 = self.top1_hits / (self.n / 100.0) if self.n else float('nan')
        return pearson, top1


def find_reference_csv(args):
    """Reference path: --ref, else {save_dir}/{dataset}.csv, with .gz fallback
    (the training run gzips its output with gzip -k)."""
    if args.ref:
        return args.ref if os.path.exists(args.ref) else None
    plain = os.path.join(args.save_dir, f'{args.dataset}.csv')
    if os.path.exists(plain):
        return plain
    gz = plain + '.gz'
    if os.path.exists(gz):
        return gz
    return None


# ============================================================================
# Unit selftest: dynamic store vs the STATIC ds4 builder, byte for byte
# ============================================================================

def run_unit_selftest():
    # Lazy import: needs jittor on the box (the training module sets flags).
    try:
        import train_graphmixer_jt as TGM
    except ImportError as exc:
        print(f'[SELFTEST FAIL] cannot import train_graphmixer_jt ({exc}); '
              'run from the ds4 work directory on a box with jittor.')
        return 1

    rng = np.random.default_rng(7)
    n_checks = 0
    for pos_only in (False, True):
        for trial, (n, L, n_src) in enumerate([(300, 1, 7), (400, 5, 7), (500, 12, 9)]):
            split = n - 80
            src = rng.integers(0, n_src, n).astype(np.int64)
            dst = rng.integers(1, 60, n).astype(np.int64)
            t = np.sort(rng.uniform(0.0, 5000.0, n)).astype(np.float64)
            t[:15] = 0.0           # time=0 edges (exercise pos_time_only filter)
            t[100:110] = t[99]     # exact time ties
            tr_h, tr_g, va_h, va_g, _, _, gs = TGM.build_histories_with_time(
                src, dst, t, 0, L, split, pos_time_only=pos_only)
            gs2 = compute_gap_scale_persrc(src, t, 0, pos_only)
            assert np.isclose(gs, gs2), \
                f'per-source gap_scale mismatch: builder={gs} vs replica={gs2}'

            dyn = DynamicHistory(num_sources=n_src, history_length=L,
                                 gap_scale=gs, pos_time_only=pos_only)
            # Stream ALL rows: build row -> compare -> inject TRUE edge.
            # Validates both the row format and the injection mechanics
            # (inject(true dst) must reproduce the builder's append exactly).
            for i in range(n):
                h_row, g_row = dyn.build_row(int(src[i]), float(t[i]))
                ref_h = tr_h[i] if i < split else va_h[i - split]
                ref_g = tr_g[i] if i < split else va_g[i - split]
                assert h_row.dtype == np.int32, f'hist dtype {h_row.dtype} (ds4 uses int32)'
                assert g_row.dtype == np.float32, f'gap dtype {g_row.dtype}'
                assert h_row.shape == (L,) and g_row.shape == (L,)
                if not np.array_equal(h_row, ref_h):
                    print(f'[SELFTEST FAIL] history mismatch pos_only={pos_only} '
                          f'trial={trial} row={i}\n  dyn={h_row}\n  ref={ref_h}')
                    return 1
                if not np.array_equal(g_row, ref_g):
                    print(f'[SELFTEST FAIL] gap mismatch pos_only={pos_only} '
                          f'trial={trial} row={i}\n  dyn={g_row}\n  ref={ref_g}')
                    return 1
                # padding-position explicit check: zero slots must form a LEFT
                # prefix (dst ids are >= 1, so 0 entries are padding)
                nz = int((ref_h != 0).sum())
                assert (ref_h[:L - nz] == 0).all() and (ref_h[L - nz:] != 0).all()
                n_checks += 1
                dyn.inject(int(src[i]), int(dst[i]), float(t[i]))

    # Predicted-edge injection: fake dst must land right-aligned with the
    # correct gap; clone() must be independent; deque truncation drops oldest.
    dyn = DynamicHistory(num_sources=3, history_length=4, gap_scale=10.0)
    dyn.inject(1, 55, 100.0)
    dyn.inject(1, 56, 200.0)
    snap = dyn.clone()
    dyn.inject(1, 57, 300.0)
    h, g = dyn.build_row(1, 400.0)
    assert h.tolist() == [0, 55, 56, 57], h                      # left zero-padding
    assert h.dtype == np.int32 and g.dtype == np.float32
    exp_g = [0.0] + [np.float32(np.log1p(400.0 - t) / 10.0) for t in (100.0, 200.0, 300.0)]
    assert np.allclose(g, np.array(exp_g, dtype=np.float32)), g
    h2, _ = snap.build_row(1, 400.0)
    assert h2.tolist() == [0, 0, 55, 56], f'clone not independent: {h2}'
    # truncation: 4 appends fill the window; the 5th drops the oldest (55)
    dyn.inject(1, 58, 350.0)
    dyn.inject(1, 59, 360.0)
    h3, g3 = dyn.build_row(1, 400.0)
    assert h3.tolist() == [56, 57, 58, 59], h3
    exp_g3 = [np.float32(np.log1p(400.0 - t) / 10.0) for t in (200.0, 300.0, 350.0, 360.0)]
    assert np.allclose(g3, np.array(exp_g3, dtype=np.float32)), g3
    # pos_time_only filter on injection: time=0 edges must NOT enter history
    dyn0 = DynamicHistory(num_sources=3, history_length=4, gap_scale=10.0, pos_time_only=True)
    assert dyn0.inject(2, 77, 0.0) is False
    h4, g4 = dyn0.build_row(2, 100.0)
    assert h4.tolist() == [0, 0, 0, 0] and g4.tolist() == [0.0, 0.0, 0.0, 0.0]

    # rank_percentiles: identical to the training script's version, INCLUDING
    # engineered exact score ties (tie-break by raw candidate id ascending).
    rs = rng.normal(size=(50, 100)).astype(np.float32)
    cands50 = np.tile(np.arange(1, 101, dtype=np.int64), (50, 1))
    ref_rp = TGM.rank_percentiles(rs, cands50)
    mine_rp = rank_percentiles(rs, cands50)
    assert np.allclose(ref_rp, mine_rp), 'rank_percentiles mismatch on tie-free data'
    tie_scores = np.array([[1.0, 1.0, 0.5, 2.0]], dtype=np.float32)
    tie_cands = np.array([[30, 10, 40, 20]], dtype=np.int64)   # id 10 < 30 -> higher rank
    ref_tie = TGM.rank_percentiles(tie_scores, tie_cands)
    mine_tie = rank_percentiles(tie_scores, tie_cands)
    assert np.array_equal(ref_tie, mine_tie), (ref_tie, mine_tie)
    assert abs(float(mine_rp.min()) - 0.01) < 1e-6 and float(mine_rp.max()) == 1.0

    # gate_decisions: top1 margin semantics + topk + UNK skip.
    sc = np.array([[5.0, 1.0, 0.0, -1.0],
                   [1.0, 0.9, 0.8, 0.7]], dtype=np.float32)    # row 1: tiny margin
    cd = np.array([[11, 12, 13, 14],
                   [21, 22, 23, 24]], dtype=np.int64)
    ok, inj = gate_decisions(sc, cd, gate_margin=0.5, gate_top1=0.0, topk=1)
    assert ok.tolist() == [True, False] and inj[0, 0] == 11
    ok, inj = gate_decisions(sc, cd, gate_margin=0.0, gate_top1=0.0, topk=3, unk_id=23)
    assert ok.tolist() == [True, True]
    assert inj[0].tolist() == [11, 12, 13]
    assert inj[1].tolist() == [21, 22, -1], inj[1]             # UNK 23 -> -1 skip

    print(f'UNIT SELFTEST PASS: {n_checks} streamed rows byte-identical to the ds4 '
          'build_histories_with_time (both pos_time_only settings), per-source '
          'gap_scale replica equal, injection/truncation/clone semantics OK, '
          'rank_percentiles matches the training version incl. candidate-id '
          'tie-breaking, gate_decisions topk/UNK semantics OK.')
    return 0


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    if args.mode == 'selftest':
        rc = run_unit_selftest()
        if rc != 0:
            sys.exit(rc)

    # Lazy imports: importing the training module applies its jittor flags
    # setup (use_cuda=1, GM_PAGEABLE_HOST) exactly as at train time, before
    # any other jittor usage below.
    try:
        import train_graphmixer_jt as TGM
    except ImportError as exc:
        raise ImportError(
            'rollout_predict_d4.py must run from the ds4 work directory that also '
            'contains train_graphmixer_jt.py / model_graphmixer_jt.py / '
            f'cf_features.py / recency_features.py (script dir: {SCRIPT_DIR}). '
            f'Original error: {exc}')
    import gc
    import pickle
    import jittor as jt
    from model_graphmixer_jt import GraphMixerModel  # same class the train script uses
    from tqdm import tqdm

    t_start = _time.time()
    ckpt_path = args.checkpoint or os.path.join(args.save_dir,
                                                f'{args.dataset}_graphmixer.pkl')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
    cooc_table = args.cooc_table or os.path.join(args.cf_dir, 'cooc_ds4.npz')
    print(f'rollout_predict_d4: mode={args.mode}, dataset={args.dataset}')
    print(f'  checkpoint={ckpt_path}')
    print(f'  save_dir={args.save_dir}, out={args.out}')
    print(f'  features: node_src={args.node_src}')
    print(f'            node_dst={args.node_dst}')
    print(f'  idmap={args.idmap}, heuristics_dir={args.heuristics_dir}, '
          f'cooc_table={cooc_table if args.use_cooc else "(off)"}')
    print(f'  GraphMixer: hidden_dim={args.hidden_dim}, history_length={args.history_length}, '
          f'layers={args.num_layers}, mlp_ratio={args.mlp_ratio}, dropout={args.dropout}, '
          f'temperature={args.temperature}, time_decay={args.time_decay}')
    print(f'  feature switches: heuristics={args.use_heuristics}, '
          f'known_flag={args.use_known_flag}, recency={args.use_recency_feats}, '
          f'cf={args.use_cf_feats} ({"full 8-dim" if args.cf_full else "lite 4-dim"}), '
          f'approx_cooc={args.use_cooc}, hist_pos_time_only={args.hist_pos_time_only}')
    if args.mode == 'rollout':
        print(f'  rollout: inject_topk={args.inject_topk}, gate_margin={args.gate_margin}, '
              f'gate_top1={args.gate_top1}, skip_unk_inject={args.skip_unk_inject}, '
              f'batch_size={args.batch_size}')
        if args.gate_margin >= 1.0 and args.gate_top1 >= 1.0:
            print('  [NOTE] gate_margin=1.0 is unsatisfiable on softmax probabilities: '
                  'expect ~0 injections (rollout output == frozen baseline).')

    # ------------------------------------------------------------------
    # Load train.csv (id space + history replay + static feature tables).
    # test.csv is NOT read here -- it is streamed chunk-wise below.
    # ------------------------------------------------------------------
    train_path = f'{args.data_dir}/{args.dataset}/train.csv'
    test_path = f'{args.data_dir}/{args.dataset}/test.csv'
    train_frame = pd.read_csv(train_path)
    source_ids = train_frame['src'].to_numpy(np.int64)
    destination_ids = train_frame['dst'].to_numpy(np.int64)
    if 'time' in train_frame.columns:
        time_values = train_frame['time'].to_numpy(np.float64)
    else:
        time_values = np.arange(len(train_frame), dtype=np.float64)
    del train_frame
    gc.collect()

    # ds4 dense remap (verbatim from the training/predict scripts):
    # sorted seen ids -> dense 1..N; 0 = padding; N_dst+1 = UNK.
    seen_src = np.unique(source_ids)
    seen_dst = np.unique(destination_ids)
    unk_dst_id = len(seen_dst) + 1
    if args.idmap and os.path.exists(args.idmap):
        im = np.load(args.idmap)
        if not (np.array_equal(im['seen_src_ids'], seen_src)
                and np.array_equal(im['seen_dst_ids'], seen_dst)
                and int(im['unk_dst_id']) == unk_dst_id):
            raise RuntimeError(
                f'{args.idmap} does not match the train.csv id sets; the dense '
                'feature tables would be misaligned.')
        print(f'  idmap {args.idmap} verified against train.csv id sets.')
    else:
        raise FileNotFoundError(f'--idmap {args.idmap} not found.')
    source_ids = np.searchsorted(seen_src, source_ids).astype(np.int64) + 1
    destination_ids = np.searchsorted(seen_dst, destination_ids).astype(np.int64) + 1
    source_min = 0
    source_count = len(seen_src) + 1
    destination_count = len(seen_dst) + 2
    print(f'  dense_id_remap: src->[1..{len(seen_src)}], dst->[1..{len(seen_dst)}], '
          f'UNK_DST={unk_dst_id} (padding=0)')
    print(f'  source_count={source_count}, destination_count={destination_count}, '
          f'train edges={len(source_ids)}')

    # ------------------------------------------------------------------
    # gap_scale via the verbatim per-source replica (avoids the builder's
    # ~5 GB snapshot matrices; replica equality asserted by the unit test).
    # ------------------------------------------------------------------
    t0 = _time.time()
    gap_scale = compute_gap_scale_persrc(source_ids, time_values, source_min,
                                         args.hist_pos_time_only)
    print(f'  Time gap scale (per-source p90): {gap_scale:.2f} '
          f'({_time.time() - t0:.1f}s)')

    # ------------------------------------------------------------------
    # Dynamic store base state: ALL train.csv edges replayed once -- exactly
    # the full_histories final-deque state the training script itself used
    # for test prediction (frozen mode must reproduce that output). Every
    # test src maps to [0, N_src] by construction, so source_count covers it.
    # ------------------------------------------------------------------
    t0 = _time.time()
    base_store = DynamicHistory.from_train(
        source_ids, destination_ids, time_values, source_min,
        args.history_length, len(source_ids),
        pos_time_only=args.hist_pos_time_only, gap_scale=gap_scale,
        min_num_sources=source_count)
    print(f'  history store replayed from {len(source_ids)} train edges '
          f'({_time.time() - t0:.1f}s)')

    # ------------------------------------------------------------------
    # Node features (per-tower tables), same as training: row 0 zeroed by the
    # caller; tables must cover the dense model id space.
    # ------------------------------------------------------------------
    src_features = np.load(args.node_src).astype(np.float32)
    dst_features = np.load(args.node_dst).astype(np.float32)
    src_features[0] = 0.0
    dst_features[0] = 0.0
    print(f'Loaded src features from {args.node_src}, shape={src_features.shape}')
    print(f'Loaded dst features from {args.node_dst}, shape={dst_features.shape}')
    if src_features.shape[0] < source_count or dst_features.shape[0] < destination_count:
        raise ValueError(
            f'feature tables smaller than the model id space: src '
            f'{src_features.shape[0]} < {source_count} or dst '
            f'{dst_features.shape[0]} < {destination_count}.')

    # ------------------------------------------------------------------
    # Heuristic features (precomputed files; REQUIRED when on -- a silent
    # fallback would corrupt scoring without any shape error).
    # ------------------------------------------------------------------
    heuristic_kwargs = {}
    if args.use_heuristics:
        degree_path = f'{args.heuristics_dir}/heuristics_{args.dataset}_degree.npy'
        popularity_path = f'{args.heuristics_dir}/heuristics_{args.dataset}_popularity.npy'
        edge_count_path = f'{args.heuristics_dir}/heuristics_{args.dataset}_edge_count.pkl'
        if not (os.path.exists(degree_path) and os.path.exists(popularity_path)
                and os.path.exists(edge_count_path)):
            raise FileNotFoundError(
                f'--use_heuristics on but files missing in {args.heuristics_dir} '
                f'(need heuristics_{args.dataset}_degree.npy / _popularity.npy / '
                '_edge_count.pkl). Refusing to score with silently different features.')
        heuristic_degree = np.load(degree_path)
        heuristic_popularity = np.load(popularity_path)
        with open(edge_count_path, 'rb') as f:
            edge_count_dict, edge_count_max = pickle.load(f)
        heuristic_kwargs = {
            'heuristic_degree': heuristic_degree,
            'heuristic_popularity': heuristic_popularity,
            'heuristic_edge_count': edge_count_dict,
            'edge_count_max': edge_count_max,
        }
        ec_form = ('vectorized int64-key table' if isinstance(edge_count_dict, tuple)
                   else f'dict ({len(edge_count_dict)} pairs)')
        print(f'Loaded heuristic features from {args.heuristics_dir}: '
              f'degree={heuristic_degree.shape}, edge_count={ec_form}, '
              f'edge_count_max={edge_count_max:.0f}')

    # dst_known flag (optional; v3s45 did NOT use it)
    if args.use_known_flag:
        dst_known = np.zeros(destination_count, dtype=bool)
        uniq_dst = np.unique(destination_ids)
        dst_known[uniq_dst[uniq_dst < destination_count]] = True
        heuristic_kwargs['use_known_flag'] = True
        heuristic_kwargs['heuristic_dst_known'] = dst_known
        print(f'dst_known flag enabled: {int(dst_known.sum())} known dst nodes '
              f'out of {destination_count}')

    # Recency features (STATIC in rollout -- built over train.csv
    # positive-time edges, never updated by injections).
    recency_stats = None
    if args.use_recency_feats:
        from recency_features import RecencyStats
        recency_stats = RecencyStats(source_ids - source_min, destination_ids, time_values)
        print(f'Recency features enabled: {recency_stats.num_edges} positive-time '
              'edges indexed (6 extra scorer dims).')

    # CF features (STATIC in rollout). ds4 default: lite 4-dim block
    # (use_cooc=False; the exact cooc matrix would need ~44 GB).
    cf_stats = None
    if args.use_cf_feats:
        from cf_features import CFStats
        cf_stats = CFStats(source_ids - source_min, destination_ids, dst_features,
                           use_cooc=args.cf_full)
        print(f'CF features enabled: {cf_stats.num_edges} edges indexed, '
              f'cooc_nnz={cf_stats.cooc_nnz}, build={cf_stats.build_seconds:.1f}s '
              f'({cf_stats.dim} extra scorer dims, lite={not args.cf_full}).')

    # Approx co-occurrence block (TABLE static in rollout, but it receives the
    # CURRENT dynamic history rows via make_extra's hists argument).
    cooc_stats = None
    if args.use_cooc:
        from cf_features import CoocTable
        cooc_stats = CoocTable(cooc_table)
        print(f'Approx cooc features enabled: {cooc_table} '
              f'nbr_ids{cooc_stats.nbr_ids.shape} (3 extra scorer dims).')

    def make_extra(src_model, times, cands, hists=None):
        """Combined caller-supplied scorer block, fixed order
        [recency(6) | cf(4/8) | approx-cooc(3)] -- verbatim from training.

        src_model: DENSE model-space src ids; cands: DENSE model-space dst ids;
        hists: (B, history_length) model-space history matrix (needed only
        when --use_cooc is on).
        """
        blocks = []
        if recency_stats is not None:
            blocks.append(recency_stats.batch_features(src_model, times, cands))
        if cf_stats is not None:
            blocks.append(cf_stats.batch_features(src_model, cands))
        if cooc_stats is not None:
            blocks.append(cooc_stats.batch_features(hists, cands))
        if not blocks:
            return None
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=2)

    # ------------------------------------------------------------------
    # Checkpoint-width precheck: the scorer input width encodes the full
    # feature configuration. Fail BEFORE the expensive feature building when
    # the flags do not match the checkpoint, with an explicit hint.
    # ------------------------------------------------------------------
    heuristic_dim = ((3 if args.use_heuristics else 0)
                     + (1 if args.use_known_flag else 0))
    extra_dim = ((6 if recency_stats is not None else 0)
                 + (cf_stats.dim if cf_stats is not None else 0)
                 + (cooc_stats.dim if cooc_stats is not None else 0))
    saved0 = jt.load(ckpt_path)
    if not isinstance(saved0, dict) or 'src_emb.weight' not in saved0:
        raise RuntimeError(f'{ckpt_path}: expected a state dict with src_emb.weight.')
    src_rows = int(saved0['src_emb.weight'].shape[0])
    dst_rows = int(saved0['dst_emb.weight'].shape[0])
    if src_rows != source_count or dst_rows != destination_count:
        del saved0
        raise RuntimeError(
            f'checkpoint embedding tables (src={src_rows}, dst={dst_rows}) do not '
            f'match the train.csv dense id space (src={source_count}, '
            f'dst={destination_count}); wrong checkpoint for this data?')
    scorer_key = next((k for k in saved0 if k.endswith('scorer.mlp.0.weight')), None)
    if scorer_key is None:
        del saved0
        raise RuntimeError(
            f'{ckpt_path}: no scorer.mlp.0.weight-like key found; not a '
            'GraphMixer checkpoint from this pipeline?')
    scorer_in = int(saved0[scorer_key].shape[1])
    expect_in = 2 * args.hidden_dim + heuristic_dim + extra_dim
    del saved0
    if scorer_in != expect_in:
        raise RuntimeError(
            f'scorer input width mismatch: checkpoint={scorer_in} vs flags={expect_in} '
            f'(2*hidden_dim={2 * args.hidden_dim} + heuristic_dim={heuristic_dim} + '
            f'extra_dim={extra_dim}). Check --use_heuristics/--use_known_flag/'
            f'--use_recency_feats/--use_cf_feats/--cf_full/--use_cooc against the '
            f'training run. The v3s45 recipe is: heuristics on, known_flag off, '
            f'recency on, cf LITE 4-dim, approx-cooc on -> extra_dim=13, total=528.')

    # ------------------------------------------------------------------
    # Build + load the model, strict shape verification (verbatim training
    # construction: per-tower features, non-shared, dense id space).
    # ------------------------------------------------------------------
    model = GraphMixerModel(
        src_count=source_count,
        dst_count=destination_count,
        hidden_dim=args.hidden_dim,
        initial_features=dst_features,
        initial_src_features=src_features,
        initial_dst_features=dst_features,
        shared_nodes=False,
        num_layers=args.num_layers,
        max_history_length=args.history_length,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        temperature=args.temperature,
        time_decay=args.time_decay,
        extra_feat_dim=extra_dim,
        **heuristic_kwargs,
    )
    TGM.verify_checkpoint_shapes(model, ckpt_path)
    try:
        model.load(ckpt_path)
    except RuntimeError as exc:
        raise RuntimeError(
            f'Checkpoint load failed (pinned-host staging?): {exc}\n'
            'Rerun with GM_PAGEABLE_HOST=1 set in the environment '
            '(pageable H2D; required at ds4 scale).') from exc
    model.eval()
    print(f'Loaded checkpoint: {ckpt_path}')
    meta_path = ckpt_path.replace('.pkl', '_meta.npy')
    if os.path.exists(meta_path):
        print(f'  checkpoint meta: {np.load(meta_path, allow_pickle=True).item()}')
    # Feature tables only initialized the embeddings; the checkpoint overwrote
    # them. Free the host copies before the scoring loop (host RAM is the
    # scarce resource at ds4 scale, cf. the training script's notes).
    del src_features, dst_features
    gc.collect()

    # ------------------------------------------------------------------
    # Streaming scoring over test.csv (src,time,c1..c100), time order asserted
    # across chunk boundaries as well.
    # ------------------------------------------------------------------
    n_total = count_csv_rows(test_path)
    n_plan = min(n_total, args.max_rows) if args.max_rows else n_total
    print(f'Test rows: {n_total} (scoring {n_plan}'
          + (f' -- [max_rows] truncated from {n_total}' if args.max_rows else '') + ')')
    if n_plan == 0:
        raise RuntimeError(f'{test_path}: no data rows.')

    rollout = (args.mode == 'rollout')
    store = base_store.clone() if rollout else base_store
    writer = None
    if args.mode in ('frozen', 'rollout'):
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        writer = open(args.out, 'w')
    comparer = None
    ref_path = find_reference_csv(args)
    if args.mode == 'selftest':
        if ref_path is None:
            if writer is not None:
                writer.close()
            raise FileNotFoundError(
                f'selftest needs the reference csv: {args.save_dir}/{args.dataset}.csv '
                '(.gz also accepted) or --ref. The reference is the training '
                "script's own test prediction for the same checkpoint.")
        comparer = RefComparer(ref_path, args.chunk_rows)
        print(f'  sanity reference: {ref_path}')
    elif args.mode == 'frozen' and ref_path is not None:
        comparer = RefComparer(ref_path, args.chunk_rows)
        print(f'  sanity reference: {ref_path}')
    elif args.mode == 'frozen':
        print('  [sanity] reference not found next to the checkpoint; skipped.')

    desc = {'selftest': 'Selftest frozen scoring',
            'frozen': 'Frozen scoring',
            'rollout': 'Rollout scoring'}[args.mode]
    n_done = 0
    n_inject_rows = 0
    n_inject_edges = 0
    n_unk_cand = 0
    vmin, vmax = float('inf'), float('-inf')
    prev_time = -np.inf
    t0 = _time.time()
    pbar = tqdm(total=n_plan, ncols=120, desc=desc)
    reader = pd.read_csv(test_path, chunksize=args.chunk_rows)
    try:
        for chunk in reader:
            if args.max_rows is not None and n_done + len(chunk) > args.max_rows:
                chunk = chunk.iloc[:args.max_rows - n_done]
            if len(chunk) == 0:
                break
            src_raw = chunk['src'].to_numpy(np.int64)
            if 'time' in chunk.columns:
                time_vals = chunk['time'].to_numpy(np.float64)
            else:
                time_vals = np.zeros(len(chunk), dtype=np.float64)
            cands_raw = chunk.iloc[:, 2:].to_numpy(np.int64)
            del chunk
            # rollout requires global time order (asserted across chunks too)
            if len(time_vals) > 0:
                if float(time_vals[0]) < prev_time or \
                        (len(time_vals) > 1 and (np.diff(time_vals) < 0).any()):
                    raise RuntimeError(
                        'test.csv is not sorted by ascending time; rollout requires '
                        'time order.')
                prev_time = float(time_vals[-1])

            src_model = TGM.map_ids_to_dense(seen_src, src_raw, 0)
            cands_dense = TGM.map_ids_to_dense(seen_dst, cands_raw, unk_dst_id)
            n_unk_cand += int((cands_dense == unk_dst_id).sum())

            for start in range(0, len(src_model), args.batch_size):
                stop = min(start + args.batch_size, len(src_model))
                src_b = src_model[start:stop]
                time_b = time_vals[start:stop]
                cand_dense_b = cands_dense[start:stop]
                cand_raw_b = cands_raw[start:stop]
                hist_b, gaps_b = store.build_batch(src_b, time_b)
                extra = make_extra(src_b, time_b, cand_dense_b, hists=hist_b)
                with jt.no_grad():
                    scores = model.execute(src_b, hist_b, cand_dense_b, gaps_b,
                                           extra_feats=extra).numpy()
                if np.isnan(scores).any():
                    print(f'[WARNING] NaN scores in test rows '
                          f'[{n_done + start}:{n_done + stop})')
                out_b = rank_percentiles(scores, cand_raw_b)
                vmin = min(vmin, float(out_b.min()))
                vmax = max(vmax, float(out_b.max()))
                if writer is not None:
                    np.savetxt(writer, out_b, delimiter=',', fmt='%.8f')
                if comparer is not None:
                    comparer.update(out_b)
                if rollout:
                    # same gate semantics as ds3; injected dst = top-k dense
                    # candidate ids, row-by-row in time order after the batch
                    ok, inj = gate_decisions(
                        scores, cand_dense_b, args.gate_margin, args.gate_top1,
                        topk=args.inject_topk,
                        unk_id=unk_dst_id if args.skip_unk_inject else None)
                    for j in np.nonzero(ok)[0]:
                        s_j = int(src_b[j])
                        t_j = float(time_b[j])
                        injected = False
                        for d in inj[j]:
                            if d >= 0 and store.inject(s_j, int(d), t_j):
                                n_inject_edges += 1
                                injected = True
                        n_inject_rows += int(injected)
                pbar.update(stop - start)
            n_done += len(src_model)
            pbar.set_postfix_str(
                f'inj={n_inject_rows}/{n_done}' if rollout else f'rows={n_done}')
            del src_raw, time_vals, cands_raw, src_model, cands_dense
            if args.max_rows is not None and n_done >= args.max_rows:
                break
    finally:
        pbar.close()
        if writer is not None:
            writer.close()
    dt = _time.time() - t0

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    n_cand = 100
    if args.mode in ('frozen', 'rollout'):
        exp_min = 1.0 / n_cand
        print(f'[selfcheck] value range [{vmin:.4f}, {vmax:.4f}] '
              f'(expected [{exp_min:.4f}, 1.0000]), rows={n_done}')
        if not (abs(vmin - exp_min) < 1e-6 and abs(vmax - 1.0) < 1e-6):
            raise RuntimeError('rank-percentile range check failed.')
        print(f'Saved: {args.out}, rows={n_done} ({dt:.1f}s scoring, '
              f'{n_done / max(dt, 1e-9):.0f} rows/s)')
    if rollout:
        print(f'[rollout] injected rows={n_inject_rows}/{n_done} '
              f'({n_inject_rows / max(n_done, 1):.3f}), edges={n_inject_edges} '
              f'(topk={args.inject_topk}, margin={args.gate_margin:g}, '
              f'top1={args.gate_top1:g}, skip_unk={args.skip_unk_inject})')
    print(f'  UNK candidate positions seen: {n_unk_cand}')
    if comparer is not None:
        pearson, top1 = comparer.report()
        flag = 'OK' if pearson > 0.999 else 'LOW -- CHECK FLAGS/FEATURES!'
        print(f'[sanity] frozen vs {ref_path}: pearson(rank-percentile)='
              f'{pearson:.6f} (expect >0.999), top1_agree={top1:.4f} [{flag}]')
        if args.mode == 'selftest':
            print(f'Total wall time: {_time.time() - t_start:.1f}s')
            sys.exit(0 if pearson > 0.999 else 2)
    print(f'Total wall time: {_time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
