"""
cf_features.py
==============
Collaborative-filtering (item-item co-occurrence + n2v similarity) features for
the dataset3 (track-1) GraphMixer temporal recommendation pipeline.

Interface (fixed — other code depends on it, do not change signatures):

    class CFStats:
        def __init__(self, src, dst, node_features, use_cooc=True):
            # src/dst: int64 training-edge arrays. Which edges are passed is the
            #   caller's decision: full train.csv edges for training/test
            #   prediction, split=0 edges only for eval (mirrors RecencyStats).
            #   ALL passed edges are used, including time=0 ones (CF is
            #   time-agnostic). src ids may live in ANY non-negative id space
            #   (raw or source_min-shifted) as long as __init__ and
            #   batch_features receive the SAME space; dst/candidate ids are
            #   always raw node ids (or dense ids under --dense_id_remap).
            # node_features: (num_nodes, 256) float32 n2v features, row i ==
            #   node id i in the SAME id space as dst/candidates.
            # use_cooc: True (default, ds3 behavior) -> full 8-dim block with
            #   the item-item co-occurrence dims; False -> "lite" 4-dim block
            #   that SKIPS the M = A^T A co-occurrence build entirely (ds4:
            #   the exact cooc matrix would need ~44 GB) and returns only the
            #   n2v similarity dims + hist length. self.dim exposes the block
            #   width (8 or 4) for scorer sizing.
        def batch_features(self, src_arr, cand_mat) -> np.ndarray
            # src_arr: [N] int64 (same id space as __init__'s src)
            # cand_mat: [N, K] int64 (dst ids in the __init__ id space)
            # returns float32 [N, K, self.dim], NaN-free; rows whose src has an
            # empty history (unseen src) are all 0.

Feature block (dims, log1p = natural log):
  use_cooc=True (8 dims, ds3 default):
  0: log1p(cf_raw), cf_raw = sum_{d in hist(src)} cooc(d, cand)
       cooc(d, c) = # distinct srcs that interacted with BOTH d and c
       (diagonal excluded: cooc(d, d) = 0)
  1: log1p(cf_cos), cf_cos = sum_{d in hist(src)} cooc(d, cand) / sqrt(deg(d)*deg(cand))
       deg(x) = # distinct srcs of dst x; terms with deg(d)==0 or deg(cand)==0
       contribute 0 (skipped)
  2: coverage = fraction of hist(src) entries with cooc(d, cand) > 0   (0..1)
  3: log1p(max_cooc), max single-history-d co-occurrence with cand
  4: cos_sim(mean_n2v_hist, cand): unit(mean of RAW n2v rows of hist dsts)
       dot unit(n2v[cand])
  5: max_cos: max cosine between cand and the LAST 20 history dsts of src
       (last 20 in the edge order passed to __init__, duplicates kept, no
       time sorting)
  6: top5_mean_cos: mean of the top min(5, len) of those last-20 cosines
  7: log1p(hist_len)
  use_cooc=False (4 dims, ds4 lite): only dims [4, 5, 6, 7] above, re-indexed
  to [0, 1, 2, 3] (mean_cos / max_cos / top5_mean_cos / log1p(hist_len)).

Definition choices (documented contract):
  * hist(src) for dims 0-4 and 7 is the DISTINCT dst set of src (duplicated
    interactions do not double-count); hist_len = size of that set.
  * dims 5-6 use the raw per-src dst SEQUENCE (last 20 entries, duplicates
    kept), exactly as specified.
  * cosine with a zero-norm vector (e.g. all-zero n2v row) is 0.

Implementation notes
--------------------
* Co-occurrence matrix M = A^T A with A the binarized (src x dst) 0/1 csr
  matrix (sum_duplicates then data=1), diagonal zeroed. M is symmetric, so
  only its upper triangle is stored. ds3 full-train scale: 32.4M nnz ->
  16.2M upper-triangular entries.
* Query structure: a static open-addressing hash table over the canonical
  packed key (min(d,c) << shift | max(d,c)), stored as ONE packed uint64
  array (key << 32 | value) at ~0.24 load factor, Fibonacci (multiplicative,
  high-bit) slot hashing, linear probing. Probing is fully vectorized and
  parallelized over a small thread pool (numpy indexing releases the GIL;
  random-access probes are memory-latency bound, so 4 threads ~2.6x).
  A per-batch cache layer was measured and REMOVED: ds3 (d,c) term streams
  are ~98% unique per 2048x32 batch, so any direct-mapped cache self-evicts.
  For id spaces too large for 32-bit keys (or tables beyond the memory
  budget) the module falls back to the upper-triangular csr + a
  compaction-based bounded binary search (recency_features._segment_ranks
  trick) — same values, slower.
* All per-term reductions (cf_raw / cf_cos / coverage / max_cooc) are
  contiguous-segment np.add.reduceat / np.maximum.reduceat passes; the
  1/sqrt(deg(cand)) factor of dim 1 is constant per group and is applied
  AFTER the reduction. The whole term path uses int32/float32/uint32 arrays.
* n2v side: unit vectors precomputed once; per-src mean-unit vector
  precomputed at init via a sparse H @ n2v product; dims 5-6 use one batched
  (m, 20, D) @ (m, D, K) matmul.

Performance reality check (ds3 full-train build): edge-sampled training rows
have mean distinct hist length ~74 (max ~974), so a 2048x32 batch expands to
~4.8M (d,c) term queries — 5.7x the volume the original ~100ms/batch target
assumed (~850k terms at hist len ~13). Measured timings are printed by the
__main__ benchmark (build ~25s one-time; per-batch well under 1s; per-term
cost ~80ns, i.e. better than the implied target throughput of ~118ns/term).
"""
import os
import time as _time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_SRC_LUT_LIMIT = 50_000_000            # direct-address src LUT allowed below this
_HASH_MAX_SLOTS = 1 << 28              # table memory safety valve (2GB as uint64)
_EMPTY32 = np.uint32(0xFFFFFFFF)       # no valid key may equal this (guarded)
_HASH_THREADS = 4                      # probe thread pool size
_THREAD_MIN_KEYS = 1 << 20             # below this, query single-threaded


def _gather_positions(offsets, gids):
    """Concatenated source positions of segments `gids` in an offsets table.

    offsets: (S+1,) int64 segment table; gids: (m,) segment ids.
    Returns (positions, counts): positions indexes the underlying concatenated
    array so that underlying[positions] == concat(underlying[lo:hi] for each gid).
    """
    lo = offsets[gids]
    hi = offsets[gids + 1]
    cnt = hi - lo
    total = int(cnt.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), cnt
    pos = np.ones(total, dtype=np.int64)
    pos[0] = lo[0]
    bounds = np.cumsum(cnt)[:-1]
    pos[bounds] = lo[1:] - lo[:-1] - cnt[:-1] + 1
    return np.cumsum(pos), cnt


class _PackedCoocTable:
    """Static open-addressing (linear probing) co-occurrence table.

    Keys are canonical packed uint32 (lo << shift) | hi with lo < hi; the
    table stores (key << 32) | value in a single uint64 array so each probe
    is one gather. EMPTY cells have key == 0xFFFFFFFF (impossible for valid
    keys; guarded by CFStats). Query is vectorized and optionally threaded
    (read-only table, disjoint output slices).
    """

    __slots__ = ('shift', 'log', 'mask', 'tab', 'slots', 'nthreads', '_pool')

    def __init__(self, keys_u32, vals, shift, nthreads=_HASH_THREADS):
        n = len(keys_u32)
        self.shift = int(shift)
        slots = 1 << max(20, (int(4.1 * n) - 1).bit_length())   # load <= ~0.25
        if slots > _HASH_MAX_SLOTS:
            raise MemoryError(f'hash table would need {slots} slots')
        self.slots = slots
        self.log = slots.bit_length() - 1
        self.mask = np.uint32(slots - 1)
        empty_cell = np.uint64(_EMPTY32) << np.uint64(32)
        self.tab = np.full(slots, empty_cell, dtype=np.uint64)
        self.nthreads = nthreads
        self._pool = None
        if n == 0:
            return
        keys_u32 = np.ascontiguousarray(keys_u32, dtype=np.uint32)
        vals = np.asarray(vals, dtype=np.int32)
        base = (keys_u32 * np.uint32(2654435761)) >> np.uint32(32 - self.log)
        # Vectorized linear-probing insertion: each round, unresolved keys
        # probe (base + probe) & mask; free cells are claimed by one winner
        # per cell (np.unique on the cell ids); losers/blocked keys advance.
        cur_k, cur_v, cur_b = keys_u32, vals, base
        probe = 0
        while len(cur_k):
            s = (cur_b + np.uint32(probe)) & self.mask
            free = (self.tab[s] >> np.uint64(32)) == _EMPTY32
            if free.any():
                kf, sf, vf = cur_k[free], s[free], cur_v[free]
                _, win = np.unique(sf, return_index=True)
                self.tab[sf[win]] = (kf[win].astype(np.uint64) << np.uint64(32)) | vf[win].astype(np.uint64)
                lose = np.ones(len(kf), dtype=bool)
                lose[win] = False
                lk, lv, lb = kf[lose], vf[lose], cur_b[free][lose]
            else:
                lk = lv = lb = None
            blocked = ~free
            cur_k = np.concatenate([cur_k[blocked]] + ([lk] if lk is not None else []))
            cur_v = np.concatenate([cur_v[blocked]] + ([lv] if lv is not None else []))
            cur_b = np.concatenate([cur_b[blocked]] + ([lb] if lb is not None else []))
            probe += 1
            if probe > slots:
                raise RuntimeError('hash insertion did not terminate')

    def _query_range(self, qk, act, out):
        """Probe loop for qk[act] -> out[act] (all uint32 bit patterns)."""
        base = (qk * np.uint32(2654435761)) >> np.uint32(32 - self.log)
        probe = 0
        while len(act):
            s = (base[act] + np.uint32(probe)) & self.mask
            t = self.tab[s]
            tk = (t >> np.uint64(32)).astype(np.uint32)
            ka = qk[act]
            hit = (tk == ka) & (ka != _EMPTY32)
            if hit.any():
                out[act[hit]] = (t[hit] & np.uint64(0xFFFFFFFF)).astype(np.int32)
            stop = hit | (tk == _EMPTY32)
            act = act[~stop]
            probe += 1

    def query(self, keys_u32):
        """Vectorized lookup; missing keys -> 0. Input: uint32 canonical keys."""
        keys_u32 = np.ascontiguousarray(keys_u32, dtype=np.uint32)
        out = np.zeros(len(keys_u32), dtype=np.int32)
        n = len(keys_u32)
        if n == 0:
            return out
        act = np.arange(n, dtype=np.int64)
        if self.nthreads > 1 and n >= _THREAD_MIN_KEYS:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(self.nthreads)
            chunks = np.array_split(act, self.nthreads)
            futs = [self._pool.submit(self._query_range, keys_u32, c, out) for c in chunks]
            for f in futs:
                f.result()
        else:
            self._query_range(keys_u32, act, out)
        return out


class CFStats:
    """Item-item CF features. See module docstring for the full contract."""

    def __init__(self, src: np.ndarray, dst: np.ndarray, node_features: np.ndarray,
                 use_cooc: bool = True):
        t0 = _time.time()
        self.use_cooc = bool(use_cooc)
        self.dim = 8 if self.use_cooc else 4
        src = np.asarray(src, dtype=np.int64).ravel()
        dst = np.asarray(dst, dtype=np.int64).ravel()
        if not (len(src) == len(dst)):
            raise ValueError('src/dst length mismatch')
        if len(src) == 0:
            raise ValueError('no edges given')
        n2v = np.asarray(node_features, dtype=np.float32)
        if n2v.ndim != 2:
            raise ValueError('node_features must be a 2-D (num_nodes, dim) array')
        if src.min() < 0 or dst.min() < 0:
            raise ValueError('negative node ids are not supported')
        num_nodes = int(n2v.shape[0])
        if int(dst.max()) >= num_nodes:
            raise ValueError('dst id out of node_features row range')
        if int(src.max()) >= _SRC_LUT_LIMIT:
            raise ValueError('src id space too large for the direct-address LUT')
        self.num_edges = int(len(src))
        self.num_nodes = num_nodes
        self.feat_dim = int(n2v.shape[1])
        self.key_shift = max(1, (num_nodes - 1).bit_length())

        # ------------------------------------------------------------------
        # Per-src DISTINCT dst sets (dims 0-4, 7): sorted (src, dst) unique
        # ------------------------------------------------------------------
        order = np.lexsort((dst, src))
        s_sorted = src[order]
        d_sorted = dst[order]
        first = np.ones(len(s_sorted), dtype=bool)
        first[1:] = (s_sorted[1:] != s_sorted[:-1]) | (d_sorted[1:] != d_sorted[:-1])
        su = s_sorted[first]
        du = d_sorted[first]
        s_change = np.ones(len(su), dtype=bool)
        s_change[1:] = su[1:] != su[:-1]
        self.uniq_src = su[s_change]
        starts = np.nonzero(s_change)[0]
        self.hist_off = np.append(starts, len(su)).astype(np.int64)   # (S+1,)
        self.hist_dst = du.astype(np.int32)                           # distinct dsts
        num_srcs = len(self.uniq_src)
        # Direct-address src id -> dense gid LUT (build/query must share the space)
        self.src_lut = np.full(int(src.max()) + 1, -1, dtype=np.int64)
        self.src_lut[self.uniq_src] = np.arange(num_srcs, dtype=np.int64)

        # ------------------------------------------------------------------
        # Per-src LAST-20 dst sequence (input edge order, duplicates kept)
        # ------------------------------------------------------------------
        so = np.argsort(src, kind='stable')          # groups edges by src, order preserved
        gid_e = self.src_lut[src[so]]
        d_so = dst[so]
        cnt_raw = np.bincount(gid_e, minlength=num_srcs)
        off_raw = np.zeros(num_srcs + 1, dtype=np.int64)
        np.cumsum(cnt_raw, out=off_raw[1:])
        idx_in = np.arange(len(so), dtype=np.int64) - off_raw[gid_e]
        take = idx_in >= (cnt_raw[gid_e] - 20)
        g20 = gid_e[take]
        c20 = np.bincount(g20, minlength=num_srcs)
        self.hist20_off = np.zeros(num_srcs + 1, dtype=np.int64)
        np.cumsum(c20, out=self.hist20_off[1:])
        self.hist20_dst = d_so[take].astype(np.int32)

        # ------------------------------------------------------------------
        # Co-occurrence M = A^T A (binarized A), diagonal zeroed, upper triangle
        # (skipped entirely in lite mode: ds4's exact cooc matrix would need
        # ~44 GB, and dims 0-3 are dropped there)
        # ------------------------------------------------------------------
        self.cooc_nnz = 0
        self.table = None
        self.cooc_ptr = self.cooc_ind = self.cooc_val = None
        self.deg = None
        self.inv_sqrt_deg = None
        if self.use_cooc:
            from scipy.sparse import csr_matrix, triu
            A = csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)),
                           shape=(int(src.max()) + 1, num_nodes))
            A.sum_duplicates()
            A.data[:] = 1.0
            # deg(x) = # distinct srcs interacting with dst x
            self.deg = np.asarray(A.getnnz(axis=0), dtype=np.int32)
            M = (A.T @ A).tocsr()
            del A
            M.sum_duplicates()
            M.sort_indices()
            M.setdiag(0)
            M.eliminate_zeros()
            self.cooc_nnz = int(M.nnz)                 # symmetric count (both triangles)
            Mt = triu(M, k=1, format='csr')            # canonical (lo, hi), lo < hi
            del M
            tri_rows = np.repeat(np.arange(num_nodes, dtype=np.int64), np.diff(Mt.indptr))
            tri_cols = Mt.indices.astype(np.int64)
            tri_vals = np.asarray(Mt.data, dtype=np.float64).astype(np.int32)
            keys64 = (tri_rows << self.key_shift) | tri_cols
            max_key = int(keys64.max(initial=0))
            if max_key < 0xFFFFFFFF:                   # 32-bit keys; EMPTY reserved
                try:
                    self.table = _PackedCoocTable(keys64.astype(np.uint32), tri_vals,
                                                  self.key_shift)
                except MemoryError:
                    self.table = None
            if self.table is None:
                # Fallback: keep the upper-triangular csr and answer via bounded
                # binary search (queries canonicalize to lo < hi first).
                self.cooc_ptr = Mt.indptr.astype(np.int64)
                self.cooc_ind = Mt.indices.astype(np.int32)
                self.cooc_val = tri_vals
            del Mt, tri_rows, tri_cols, tri_vals, keys64
            inv = np.zeros(num_nodes, dtype=np.float32)
            nz = self.deg > 0
            inv[nz] = (1.0 / np.sqrt(self.deg[nz].astype(np.float64))).astype(np.float32)
            self.inv_sqrt_deg = inv

        # ------------------------------------------------------------------
        # n2v unit vectors + per-src mean-unit vector over distinct hist dsts
        # ------------------------------------------------------------------
        from scipy.sparse import csr_matrix
        norms = np.linalg.norm(n2v, axis=1, keepdims=True)
        self.n2v_unit = np.where(norms > 0, n2v / np.maximum(norms, 1e-30), 0.0).astype(np.float32)
        row_of = np.repeat(np.arange(num_srcs, dtype=np.int64), np.diff(self.hist_off))
        H = csr_matrix((np.ones(len(self.hist_dst), dtype=np.float32), (row_of, self.hist_dst)),
                       shape=(num_srcs, num_nodes))
        sums = H @ n2v                                    # (S, dim) mean numerator
        mean = sums / np.maximum(np.diff(self.hist_off), 1)[:, None].astype(np.float32)
        mn = np.linalg.norm(mean, axis=1, keepdims=True)
        self.hist_mean_unit = np.where(mn > 0, mean / np.maximum(mn, 1e-30), 0.0).astype(np.float32)
        del H, sums, mean, row_of

        self.build_seconds = _time.time() - t0

    # ------------------------------------------------------------------
    # Fallback bounded binary search (only when the hash table was not built)
    # ------------------------------------------------------------------

    def _cooc_lookup(self, lo, hi):
        """cooc(lo, hi) for canonical pairs (lo < hi) from the upper-triangular
        csr: bounded binary search of hi inside row lo (compaction loop)."""
        n = len(lo)
        out = np.zeros(n, dtype=np.int32)
        if n == 0:
            return out
        lo0 = self.cooc_ptr[lo]
        hi0 = self.cooc_ptr[lo + 1]
        act = np.nonzero(hi0 > lo0)[0]
        if len(act) == 0:
            return out
        lo_a = lo0[act]
        hi_a = hi0[act]
        tv = hi[act]
        ind = self.cooc_ind
        while len(act):
            mid = (lo_a + hi_a) >> 1
            v = ind[mid]
            less = v < tv
            lo_a = np.where(less, mid + 1, lo_a)
            hi_a = np.where(less, hi_a, mid)
            cont = hi_a > lo_a
            if (~cont).any():
                done = act[~cont]
                pos = lo_a[~cont]
                ok = pos < hi0[done]
                if ok.any():
                    qok = done[ok]
                    pok = pos[ok]
                    match = ind[pok] == tv[~cont][ok]
                    if match.any():
                        out[qok[match]] = self.cooc_val[pok[match]]
            act = act[cont]
            lo_a, hi_a, tv = lo_a[cont], hi_a[cont], tv[cont]
        return out

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def batch_features(self, src_arr: np.ndarray, cand_mat: np.ndarray) -> np.ndarray:
        src_arr = np.asarray(src_arr, dtype=np.int64).ravel()
        cand_mat = np.asarray(cand_mat, dtype=np.int64)
        n, k = cand_mat.shape
        if len(src_arr) != n:
            raise ValueError('src_arr must have shape [N] matching cand_mat [N, K]')
        out = np.zeros((n, k, self.dim), dtype=np.float32)
        if n == 0:
            return out
        # Output column layout: full mode keeps the historical 0..7 indices;
        # lite mode re-indexes the n2v dims to 0..2 and hist_len to 3.
        d_cos, d_max, d_top5, d_len = (4, 5, 6, 7) if self.use_cooc else (0, 1, 2, 3)

        # src id -> dense gid (same id space as __init__; unseen src -> empty row)
        in_lut = (src_arr >= 0) & (src_arr < len(self.src_lut))
        gid = np.full(n, -1, dtype=np.int64)
        gid[in_lut] = self.src_lut[src_arr[in_lut]]
        has = gid >= 0
        if not has.any():
            return out
        rows = np.nonzero(has)[0]
        g = gid[rows]
        m = len(rows)
        L64 = self.hist_off[g + 1] - self.hist_off[g]     # distinct-hist len, >= 1
        L = L64.astype(np.int32)

        cands = cand_mat[rows]                            # (m, K) dst ids
        c_ok = (cands >= 0) & (cands < self.num_nodes)
        cands_clip32 = np.clip(cands, 0, self.num_nodes - 1).astype(np.int32)

        Lf = L64.astype(np.float64)
        if self.use_cooc:
            # ---------------- cooc block (dims 0-3) ----------------
            # Expand every (row, cand) pair into L_row (d, c) term queries, laid
            # out as contiguous groups in (row, k) row-major order.
            reps = np.repeat(L, k)                        # (m*K,) every group >= 1
            ngrp = m * k
            gstart = np.zeros(ngrp + 1, dtype=np.int32)
            np.cumsum(reps, out=gstart[1:], dtype=np.int64)
            total_terms = int(gstart[-1])
            pos_row, _ = _gather_positions(self.hist_off, g)
            rowcat = self.hist_dst[pos_row]               # (sumL,) int32
            row_start = np.zeros(m + 1, dtype=np.int32)
            np.cumsum(L64, out=row_start[1:], dtype=np.int64)
            rstart_grp = np.repeat(row_start[:-1], k)     # (ngrp,)
            term_group = np.repeat(np.arange(ngrp, dtype=np.int32), reps)
            j_in = np.arange(total_terms, dtype=np.int32) - gstart[:-1][term_group]
            term_d = rowcat[rstart_grp[term_group] + j_in]    # int32
            term_c = cands_clip32.ravel()[term_group]         # int32

            # canonical packed uint32 key (table holds upper triangle only)
            lo = np.minimum(term_d, term_c).astype(np.uint32)
            hi = np.maximum(term_d, term_c).astype(np.uint32)
            keys_u32 = (lo << np.uint32(self.key_shift)) | hi
            if not c_ok.all():
                # out-of-space candidate -> force a miss (no valid key equals EMPTY)
                keys_u32 = np.where(c_ok.ravel()[term_group], keys_u32, _EMPTY32)
            if self.table is not None:
                cooc = self.table.query(keys_u32)
            else:
                cooc = self._cooc_lookup(lo.astype(np.int64), hi.astype(np.int64))
                if not c_ok.all():
                    cooc = cooc * c_ok.ravel()[term_group]

            gs = gstart[:-1]
            cf_raw = np.add.reduceat(cooc, gs)                # int32 sums
            # dim 1: 1/sqrt(deg(cand)) is constant per group -> applied after the sum
            wterm_d = cooc.astype(np.float32) * self.inv_sqrt_deg[term_d]
            cf_cos = np.add.reduceat(wterm_d, gs) * self.inv_sqrt_deg[cands_clip32.ravel()]
            cov_cnt = np.add.reduceat(np.clip(cooc, 0, 1), gs)
            max_cooc = np.maximum.reduceat(cooc, gs)

            out[rows, :, 0] = np.log1p(cf_raw.reshape(m, k).astype(np.float64)).astype(np.float32)
            out[rows, :, 1] = np.log1p(
                np.maximum(cf_cos, 0.0).reshape(m, k).astype(np.float64)).astype(np.float32)
            out[rows, :, 2] = (cov_cnt.reshape(m, k).astype(np.float64) / Lf[:, None]).astype(np.float32)
            out[rows, :, 3] = np.log1p(max_cooc.reshape(m, k).astype(np.float64)).astype(np.float32)
        out[rows, :, d_len] = np.log1p(Lf)[:, None].astype(np.float32)

        # ---------------- n2v block (dims 4-6 full / 0-2 lite) ----------------
        fdim = self.feat_dim
        cand_u = self.n2v_unit[cands_clip32.ravel()].reshape(m, k, fdim)
        if not c_ok.all():
            cand_u = cand_u * c_ok[:, :, None].astype(np.float32)
        mean_u = self.hist_mean_unit[g]                   # (m, D)
        out[rows, :, d_cos] = np.einsum('md,mkd->mk', mean_u, cand_u).astype(np.float32)

        pos20, c20 = _gather_positions(self.hist20_off, g)
        d20 = self.hist20_dst[pos20]
        v20 = self.n2v_unit[d20]                          # (tot20, D)
        row20 = np.repeat(np.arange(m, dtype=np.int64), c20)
        c20_start = np.zeros(m + 1, dtype=np.int64)
        np.cumsum(c20, out=c20_start[1:])
        j20 = np.arange(len(d20), dtype=np.int64) - c20_start[row20]
        pad = np.zeros((m * 20, fdim), dtype=np.float32)
        pad[row20 * 20 + j20] = v20
        pad = pad.reshape(m, 20, fdim)
        msk = np.zeros(m * 20, dtype=bool)
        msk[row20 * 20 + j20] = True
        msk = msk.reshape(m, 20)
        cos = np.matmul(pad, cand_u.transpose(0, 2, 1))   # (m, 20, K) batched sgemm
        cos = np.where(msk[:, :, None], cos, -np.inf)
        dim5 = cos.max(axis=1)                            # (m, K); every row has >= 1 valid
        srt = np.sort(cos, axis=1)[:, ::-1, :][:, :5, :]  # (m, 5, K) descending
        cnt5 = np.minimum(5, c20).astype(np.float64)      # (m,) >= 1
        top_valid = np.arange(srt.shape[1])[None, :, None] < cnt5[:, None, None]
        dim6 = np.where(top_valid, srt, 0.0).sum(axis=1) / cnt5[:, None]
        out[rows, :, d_max] = dim5.astype(np.float32)
        out[rows, :, d_top5] = dim6.astype(np.float32)
        return out


class CoocTable:
    """Approximate item-item co-occurrence block (3 dims) from an OFFLINE-built
    neighbor table (build_cooc_ds4.py -> cooc_ds4.npz). Memory-capped stand-in
    for the exact ds4 cooc matrix; stacks ON TOP of the CFStats lite block.

    Interface:
        CoocTable(table_path, chunk_rows=512)
        batch_features(hist_mat, cand_mat) -> float32 (N, C, 3)
          hist_mat: (N, H) dst ids (0 = padding), e.g. the (B, 40) model
                    history matrix used by the training loop / eval.
          cand_mat: (N, C) dst ids in the same dense id space.

    Feature dims for (H, candidate c):
      0 hit_frac: fraction of the non-padding H entries present in nbr_ids[c]
      1 max_w:    max nbr_w[c, k] over neighbors k that hit any H entry
      2 wsum:     sum of nbr_w[c, k] over hitting neighbors, divided by |H|
    All-zero rows for empty histories / out-of-space or all-padding candidates.

    Memory: the (b, C, K, H) broadcast match is chunked over rows
    (chunk_rows=512 -> ~84MB bool at C=32, K=128, H=40).
    """

    def __init__(self, table_path, chunk_rows=512, max_k=None, nthreads=4):
        # Prefer .npy sidecars (true memmap -> the 0.88GB table stays out of
        # the committed working set). Falls back to the .npz (np.savez is
        # uncompressed, but numpy returns committed ndarrays for npz members).
        import os as _os
        stem = table_path[:-4] if table_path.endswith('.npz') else table_path
        ids_p, w_p = stem + '_nbr_ids.npy', stem + '_nbr_w.npy'
        if _os.path.exists(ids_p) and _os.path.exists(w_p):
            nbr_ids = np.load(ids_p, mmap_mode='r')
            nbr_w = np.load(w_p, mmap_mode='r')
        else:
            z = np.load(table_path, mmap_mode='r')
            nbr_ids = z['nbr_ids']
            nbr_w = z['nbr_w']
        if nbr_ids.dtype != np.int32:
            nbr_ids = np.ascontiguousarray(nbr_ids, dtype=np.int32)
        if nbr_w.dtype != np.float32:
            nbr_w = np.ascontiguousarray(nbr_w, dtype=np.float32)
        if nbr_ids.shape != nbr_w.shape or nbr_ids.ndim != 2:
            raise ValueError(f'{table_path}: nbr_ids/nbr_w shape mismatch')
        if max_k is not None and max_k < nbr_ids.shape[1]:
            nbr_ids = np.ascontiguousarray(nbr_ids[:, :max_k])
            nbr_w = np.ascontiguousarray(nbr_w[:, :max_k])
        self.nbr_ids = nbr_ids
        self.nbr_w = nbr_w
        self.dim = 3
        self.num_nodes = int(self.nbr_ids.shape[0])
        self.K = int(self.nbr_ids.shape[1])
        self.chunk_rows = int(chunk_rows)
        self.nthreads = int(nthreads)
        self._pool = None

    def _chunk_features(self, r0, r1, hist, cand, h_len_c, out):
        h = np.asarray(hist[r0:r1], dtype=np.int32)          # (b, H)
        c = np.clip(cand[r0:r1], 0, self.num_nodes - 1).astype(np.int32)
        nids = self.nbr_ids[c]                               # (b, C, K)
        nw = self.nbr_w[c]                                   # (b, C, K)
        hits = (nids[:, :, :, None] == h[:, None, None, :])
        hits &= (h != 0)[:, None, None, :]
        hit_any_k = hits.any(axis=3)                         # (b, C, K)
        hit_h = hits.any(axis=2)                             # (b, C, H)
        Lc = h_len_c[r0:r1][:, None]
        out[r0:r1, :, 0] = hit_h.sum(axis=2) / Lc
        w_hit = nw * hit_any_k
        out[r0:r1, :, 1] = w_hit.max(axis=2)
        out[r0:r1, :, 2] = w_hit.sum(axis=2) / Lc

    def batch_features(self, hist_mat, cand_mat):
        hist = np.asarray(hist_mat)
        cand = np.asarray(cand_mat, dtype=np.int64)
        n, C = cand.shape
        out = np.zeros((n, C, self.dim), dtype=np.float32)
        if n == 0:
            return out
        h_len = (hist != 0).sum(axis=1).astype(np.float32)      # (n,)
        h_len_c = np.maximum(h_len, 1.0)
        bounds = list(range(0, n, self.chunk_rows)) + [n]
        spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        if self.nthreads > 1 and len(spans) > 1:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(self.nthreads)
            futs = [self._pool.submit(self._chunk_features, r0, r1,
                                      hist, cand, h_len_c, out)
                    for r0, r1 in spans]
            for f in futs:
                f.result()
        else:
            for r0, r1 in spans:
                self._chunk_features(r0, r1, hist, cand, h_len_c, out)
        return out


if __name__ == '__main__':
    # ======================================================================
    # 1) Hand-verified tiny example (8 dims), incl. duplicate-edge and
    #    shifted-src-id cases. Co-occurrence math by hand:
    #      edges: (1,10) (1,20) (2,10) (2,20) (3,10) (1,30)
    #      deg: 10->3, 20->2, 30->1
    #      cooc: (10,20)=2 {src 1,2}, (10,30)=1, (20,30)=1, else 0, diag 0
    #      hist(distinct): 1->{10,20,30}, 2->{10,20}, 3->{10}
    #      last20(seq):    1->[10,20,30], 2->[10,20], 3->[10]
    # ======================================================================
    import math

    s = np.array([1, 1, 2, 2, 3, 1])
    d = np.array([10, 20, 10, 20, 10, 30])
    n2v = np.zeros((100, 4), dtype=np.float32)
    n2v[10] = [1.0, 0.0, 0.0, 0.0]
    n2v[20] = [1.0, 1.0, 0.0, 0.0]
    n2v[30] = [0.0, 1.0, 0.0, 0.0]
    n2v[99] = [0.0, 0.0, 1.0, 0.0]
    st = CFStats(s, d, n2v)

    q_src = np.array([1, 2, 7])                 # src 7 unseen -> all-zero row
    q_c = np.array([[20, 30, 99],
                    [10, 10, 10],
                    [10, 20, 30]])
    f = st.batch_features(q_src, q_c)
    close = lambda a, b: abs(float(a) - float(b)) <= 1e-5
    r2 = math.sqrt(2.0)

    # --- row 0 (src 1), cand 20 ---
    assert close(f[0, 0, 0], math.log1p(3))                      # cf_raw = 2+0+1
    assert close(f[0, 0, 1], math.log1p(2 / math.sqrt(6) + 1 / r2))   # 0.8165+0.7071
    assert close(f[0, 0, 2], 2.0 / 3.0)                          # coverage
    assert close(f[0, 0, 3], math.log1p(2))                      # max_cooc
    assert close(f[0, 0, 4], 1.0)                                # mean-unit == unit(v20)
    assert close(f[0, 0, 5], 1.0)                                # cos(20, 20)
    assert close(f[0, 0, 6], (1.0 + 1 / r2 + 1 / r2) / 3.0)      # top5 mean (3 valid)
    assert close(f[0, 0, 7], math.log1p(3))                      # hist_len
    # --- row 0, cand 30 ---
    assert close(f[0, 1, 0], math.log1p(2))                      # cf_raw = 1+1+0
    assert close(f[0, 1, 1], math.log1p(1 / math.sqrt(3) + 1 / r2))
    assert close(f[0, 1, 2], 2.0 / 3.0)
    assert close(f[0, 1, 3], math.log1p(1))
    assert close(f[0, 1, 4], 1 / r2)                             # mean-unit . unit(v30)
    assert close(f[0, 1, 5], 1.0)                                # cos(30, 30)
    assert close(f[0, 1, 6], (1.0 + 1 / r2 + 0.0) / 3.0)
    assert close(f[0, 1, 7], math.log1p(3))
    # --- row 0, cand 99 (unseen dst, deg 0) ---
    assert f[0, 2, 0] == 0.0 and f[0, 2, 1] == 0.0
    assert f[0, 2, 2] == 0.0 and f[0, 2, 3] == 0.0
    assert f[0, 2, 4] == 0.0 and f[0, 2, 5] == 0.0 and f[0, 2, 6] == 0.0
    assert close(f[0, 2, 7], math.log1p(3))
    # --- row 1 (src 2), cand 10 ---
    assert close(f[1, 0, 0], math.log1p(2))                      # cooc(20,10)=2
    assert close(f[1, 0, 1], math.log1p(2 / math.sqrt(6)))
    assert close(f[1, 0, 2], 0.5)                                # only d=20 has cooc>0
    assert close(f[1, 0, 3], math.log1p(2))
    assert close(f[1, 0, 4], 1.0 / math.sqrt(1.25))              # unit(mean(v10,v20)).unit(v10)
    assert close(f[1, 0, 5], 1.0)                                # cos(10, 10)
    assert close(f[1, 0, 6], (1.0 + 1 / r2) / 2.0)               # top5 mean (2 valid)
    assert close(f[1, 0, 7], math.log1p(2))
    # --- row 2 (src 7, unseen) ---
    assert np.all(f[2] == 0.0)
    assert np.isfinite(f).all()
    print('hand-check (8 dims x 3 rows) OK')

    # --- duplicate-edge semantics: base example + one extra (1,10) edge.
    #     distinct hist / deg / cooc (dims 0-4, 7) must be UNCHANGED;
    #     last-20 sequence keeps the duplicate: [10,20,30] -> [10,20,30,10] ---
    s2 = np.array([1, 1, 2, 2, 3, 1, 1])
    d2 = np.array([10, 20, 10, 20, 10, 30, 10])
    st2 = CFStats(s2, d2, n2v)
    f2 = st2.batch_features(np.array([1]), np.array([[30]]))
    assert close(f2[0, 0, 0], math.log1p(2))                     # cf_raw unchanged
    assert close(f2[0, 0, 1], math.log1p(1 / math.sqrt(3) + 1 / r2))  # deg(10)=3 not 4
    assert close(f2[0, 0, 2], 2.0 / 3.0)
    assert close(f2[0, 0, 3], math.log1p(1))
    assert close(f2[0, 0, 4], 1 / r2)
    assert close(f2[0, 0, 7], math.log1p(3))                     # distinct hist len 3
    assert close(f2[0, 0, 5], 1.0)                               # cos(30, 30) still present
    assert close(f2[0, 0, 6], (1.0 + 1 / r2 + 0.0 + 0.0) / 4.0)  # 4 seq entries now
    print('duplicate-edge check OK')

    # --- shifted src id space (ids not starting at 0): results must equal
    #     the base example exactly (src ids are pure hist keys; n2v/cooc are
    #     indexed by raw dst ids only) ---
    s3 = s + 1000
    st3 = CFStats(s3, d, n2v)
    f3 = st3.batch_features(q_src + 1000, q_c)
    assert np.array_equal(f3[[0, 1]], f[[0, 1]])                 # rows 0/1 identical
    f3b = st3.batch_features(np.array([1]), q_c[:1])             # raw-id query = unseen here
    assert np.all(f3b == 0.0)
    print('shifted-src-id check OK')

    # ======================================================================
    # 1b) Lite mode (use_cooc=False, ds4): 4-dim block must equal dims
    #     [4, 5, 6, 7] of the full block on the same data; cooc must be skipped.
    # ======================================================================
    st_lite = CFStats(s, d, n2v, use_cooc=False)
    assert st_lite.dim == 4 and st.dim == 8
    assert st_lite.cooc_nnz == 0 and st_lite.table is None and st_lite.deg is None
    fl = st_lite.batch_features(q_src, q_c)
    assert fl.shape == (3, 3, 4)
    assert np.array_equal(fl, f[:, :, 4:8])                  # n2v dims + hist_len
    assert np.all(fl[2] == 0.0)                              # unseen src -> zero row
    assert np.isfinite(fl).all()
    # shifted src id space in lite mode
    st3_lite = CFStats(s3, d, n2v, use_cooc=False)
    fl3 = st3_lite.batch_features(q_src + 1000, q_c)
    assert np.array_equal(fl3, fl)
    fl3b = st3_lite.batch_features(np.array([1]), q_c[:1])   # raw-id query = unseen
    assert np.all(fl3b == 0.0)
    # duplicate edges: distinct hist_len unchanged, last-20 keeps duplicate
    st2_lite = CFStats(s2, d2, n2v, use_cooc=False)
    fl2 = st2_lite.batch_features(np.array([1]), np.array([[30]]))
    assert close(fl2[0, 0, 3], math.log1p(3))
    assert close(fl2[0, 0, 1], 1.0)
    assert close(fl2[0, 0, 2], (1.0 + 1 / r2 + 0.0 + 0.0) / 4.0)
    print('lite-mode (use_cooc=False) checks OK')

    # ======================================================================
    # 2) 2048 x 32 benchmark. Prefer the real ds4 graph + features when
    #    present (same dense-remap wiring as the train script); otherwise a
    #    synthetic ds4-scale graph. Both modes are timed; lite is what ds4
    #    trains with.
    # ======================================================================
    here = os.path.dirname(os.path.abspath(__file__))
    ds4_train = r'F:\download\data_B\dataset4\train.csv'
    ds4_n2v = os.path.join(here, 'node_features_ds4_dst.npy')
    ds4_idmap = os.path.join(here, 'idmap_ds4.npz')
    ds3_train = os.path.join(here, 'dataset3', 'train.csv')
    ds3_n2v = os.path.join(here, 'node_features_dataset3_n2v256.npy')

    def run_bench(tag, src_all, dst_all, nf):
        # Full cooc build only on small graphs (ds4-scale exact cooc is the
        # ~44 GB matrix the lite mode exists to avoid).
        modes = (False, True) if len(src_all) <= 3_000_000 else (True,)
        for lite in modes:
            big = CFStats(src_all, dst_all, nf, use_cooc=not lite)
            print(f'[{tag}] build lite={lite}: edges={big.num_edges}, '
                  f'srcs={len(big.uniq_src)}, cooc_nnz={big.cooc_nnz}, '
                  f'dim={big.dim}, build={big.build_seconds:.1f}s')
            dst_cnt = np.bincount(dst_all, minlength=big.num_nodes).astype(np.float64)
            seen = np.nonzero(dst_cnt)[0]
            cdf = np.cumsum(dst_cnt[seen])
            cdf /= cdf[-1]

            def make_batch(seed):
                r = np.random.default_rng(seed)
                rows = r.integers(0, len(src_all), size=2048)
                qs = src_all[rows]
                pos = dst_all[rows]
                neg = seen[np.minimum(np.searchsorted(cdf, r.random((2048, 31))), len(seen) - 1)]
                return qs, np.concatenate([pos[:, None], neg], axis=1)

            def timed(qs, qc, label):
                t = _time.time()
                feats = big.batch_features(qs, qc)
                dt = (_time.time() - t) * 1000.0
                assert feats.shape == (2048, 32, big.dim) and np.isfinite(feats).all()
                print(f'[{tag}] {label}: {dt:.1f} ms')

            qs1, qc1 = make_batch(1)
            timed(qs1, qc1, 'cold batch  (2048x32, first call)')
            timed(qs1, qc1, 'warm batch  (same batch)')
            timed(*make_batch(2), 'warm batch  (new batch)')
            timed(*make_batch(3), 'warm batch  (new batch #2)')

    if os.path.exists(ds4_train) and os.path.exists(ds4_n2v) and os.path.exists(ds4_idmap):
        import pandas as pd
        frame = pd.read_csv(ds4_train)
        src_raw = frame['src'].to_numpy(np.int64)
        dst_raw = frame['dst'].to_numpy(np.int64)
        idmap = np.load(ds4_idmap)
        src_all = np.searchsorted(idmap['seen_src_ids'], src_raw) + 1
        dst_all = np.searchsorted(idmap['seen_dst_ids'], dst_raw) + 1
        nf = np.load(ds4_n2v).astype(np.float32)
        run_bench('ds4 real', src_all, dst_all, nf)
    elif os.path.exists(ds3_train) and os.path.exists(ds3_n2v):
        import pandas as pd
        frame = pd.read_csv(ds3_train)
        nf = np.load(ds3_n2v).astype(np.float32)
        run_bench('ds3 real', frame['src'].to_numpy(np.int64),
                  frame['dst'].to_numpy(np.int64), nf)
    else:
        # Synthetic ds4-scale graph: 680k srcs / 862k dsts / 16M edges,
        # Zipf dst popularity, random unit n2v rows.
        rng = np.random.default_rng(7)
        n_src_s, n_dst_s, n_edge_s = 680_000, 862_000, 16_000_000
        print('synthetic ds4-scale benchmark (real files not found): '
              f'{n_edge_s / 1e6:.0f}M edges ...')
        w = 1.0 / np.arange(1, n_dst_s + 1, dtype=np.float64)
        w /= w.sum()
        dst_all = rng.choice(n_dst_s, size=n_edge_s, p=w).astype(np.int64) + 1
        src_all = rng.integers(1, n_src_s + 1, size=n_edge_s, dtype=np.int64)
        nf = rng.standard_normal((n_dst_s + 2, 256)).astype(np.float32)
        nf /= np.linalg.norm(nf, axis=1, keepdims=True) + 1e-9
        nf[0] = 0.0
        run_bench('synthetic', src_all, dst_all, nf)

    print('cf_features self-check OK')
