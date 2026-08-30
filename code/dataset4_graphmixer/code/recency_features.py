"""
recency_features.py
===================
Recency / recent-popularity / trend features for the dataset3 (track-1)
GraphMixer temporal recommendation pipeline.

Interface (fixed — other agents depend on it, do not change signatures):

    class RecencyStats:
        def __init__(self, src, dst, time):
            # training-edge arrays; internally keeps time>0 edges only
        def batch_features(self, src_arr, time_arr, cand_mat) -> np.ndarray
            # src_arr/time_arr: [N]; cand_mat: [N, K]
            # returns float32 [N, K, 6], leak-free (strictly before t):
            #   0: log1p(cnt of cand as dst in [t-30d, t))
            #   1: log1p(cnt of cand as dst in [t-7d,  t))
            #   2: trend = log1p(cnt7+0.5) - log1p(cnt30+0.5)
            #   3: log1p(seconds since cand last appeared as dst before t, +1);
            #      never appeared -> log1p(1e10)
            #   4: log1p(seconds since (src,cand) last interacted before t, +1);
            #      never -> log1p(1e10)
            #   5: log1p(seconds since src last active before t, +1);
            #      never -> log1p(1e10)
            # rows with t<=0 get log1p(1e10) in all 6 dims; output is NaN-free.

Implementation notes
--------------------
Three group -> ascending-int32-times indexes are built at init (grouped by
dst, by src, and by (src,dst) pair; pair keys packed as
``src * 2**ceil(log2(max_node+1)) + dst``). Group segments are contiguous
ranges of one globally (group, time)-sorted array, so every query reduces to
a lower_bound inside a segment:

  * group lookup: dst/src use a direct-address LUT (node ids are small);
    pairs use one vectorized searchsorted over the sorted unique pair keys;
  * window counts: rank(t) - rank(t-window) inside the candidate's segment;
  * last-seen: element at rank(t)-1, validated against the segment start.

``_segment_ranks`` is a fully vectorized segment-bounded binary search:
two edge probes (segment fully before / fully after the key resolve the
majority of training queries, since most candidate events are old relative
to the row time) followed by a compaction loop over the unresolved queries
only (active set shrinks geometrically; segments average ~15 events for dst
and ~1.3 for pairs, vs ~20 random-access hops for a naive whole-array
np.searchsorted — measured ~6x faster end to end).

Times are truncated to int32 seconds (ds3 timestamps are integral Unix
seconds < 2**31, lossless). Build = 3 x (lexsort + unique) over ~574k
edges (~0.3s). A [2048, 32] training batch = 65,536 candidate queries is
answered in ~15-25ms; [5000, 100] validation calls in ~0.2s.
"""
import numpy as np

_SEC_30D = np.int64(30 * 86400)
_SEC_7D = np.int64(7 * 86400)
NEVER = np.float32(np.log1p(1e10))  # "never happened" constant
_LUT_LIMIT = 10_000_000             # direct-address LUT allowed below this key space


class _GroupedTimes:
    """Group key -> ascending int32 timestamps, stored as contiguous segments."""

    def __init__(self, keys, times):
        order = np.lexsort((times, keys))
        k = keys[order]
        self.times = times[order].astype(np.int32)     # global, sorted within group
        self.uniq, self.starts = np.unique(k, return_index=True)
        self.ends = np.append(self.starts[1:], len(k))
        # Direct-address group lookup when the key space is small (node ids)
        self.lut = None
        if len(self.uniq) and int(self.uniq[-1]) < _LUT_LIMIT:
            self.lut = np.full(int(self.uniq[-1]) + 1, -1, dtype=np.int64)
            self.lut[self.uniq] = np.arange(len(self.uniq), dtype=np.int64)
        # Hash-table lookup for large key spaces (pair keys); falls back to searchsorted
        self.pd_index = None
        if self.lut is None and len(self.uniq):
            try:
                import pandas as pd
                self.pd_index = pd.Index(self.uniq)
            except Exception:
                self.pd_index = None

    def map_gid(self, qkeys):
        """Raw group keys -> dense gid (or -1 when the group does not exist)."""
        if len(self.uniq) == 0:
            return np.full(len(qkeys), -1, dtype=np.int64)
        if self.lut is not None:
            in_range = (qkeys >= 0) & (qkeys < len(self.lut))
            out = np.full(len(qkeys), -1, dtype=np.int64)
            out[in_range] = self.lut[qkeys[in_range]]
            return out
        if self.pd_index is not None:
            return self.pd_index.get_indexer(qkeys).astype(np.int64)
        pos = np.searchsorted(self.uniq, qkeys)
        pos = np.minimum(pos, len(self.uniq) - 1)
        found = self.uniq[pos] == qkeys
        return np.where(found, pos, np.int64(-1))

    def segment_ranks(self, gid, qsec):
        """For each query: absolute lower_bound index of qsec within its segment.

        rank = first position p in [start, end) with times[p] >= qsec
        (== end when all segment events precede qsec). gid<0 -> rank 0.
        Vectorized: edge probes + compacted binary search, no Python per-row loops.
        """
        m = len(qsec)
        rank = np.zeros(m, dtype=np.int64)
        has = gid >= 0
        if not has.any():
            return rank
        vidx = np.nonzero(has)[0]
        lo = self.starts[gid[vidx]]
        hi = self.ends[gid[vidx]]
        key = qsec[vidx].astype(np.int32, copy=False)
        times = self.times

        nonempty = hi > lo
        vidx, lo, hi, key = vidx[nonempty], lo[nonempty], hi[nonempty], key[nonempty]

        # Edge probe 1: last event of the segment precedes the key -> rank = end
        last = times[hi - 1]
        after = last < key
        rank[vidx[after]] = hi[after]
        keep = ~after
        vidx, lo, hi, key = vidx[keep], lo[keep], hi[keep], key[keep]

        # Edge probe 2: first event at/after the key -> rank = start
        if len(vidx):
            first = times[lo]
            before = first >= key
            rank[vidx[before]] = lo[before]
            keep = ~before
            vidx, lo, hi, key = vidx[keep], lo[keep], hi[keep], key[keep]

        # Compacted binary search on the unresolved queries only
        while len(vidx):
            mid = (lo + hi) >> 1
            less = times[mid] < key
            lo = np.where(less, mid + 1, lo)
            hi = np.where(less, hi, mid)
            cont = hi > lo
            rank[vidx[~cont]] = lo[~cont]
            vidx, lo, hi, key = vidx[cont], lo[cont], hi[cont], key[cont]
        return rank

    def last_gap(self, gid, qsec, rank=None):
        """Seconds (qsec - last_time < qsec) per query; NaN when never / group missing.

        rank: optional precomputed segment_ranks(gid, qsec) to avoid a repeat pass.
        """
        m = len(qsec)
        gap = np.full(m, np.nan, dtype=np.float64)
        has = gid >= 0
        if not has.any():
            return gap
        r = self.segment_ranks(gid, qsec) if rank is None else rank
        li = r - 1
        ok = has & (li >= self.starts[np.maximum(gid, 0)])
        gap[ok] = qsec[ok].astype(np.float64) - self.times[li[ok]].astype(np.float64)
        return gap


class RecencyStats:
    """Leak-free recency/popularity-trend features. See module docstring for the contract."""

    def __init__(self, src: np.ndarray, dst: np.ndarray, time: np.ndarray):
        src = np.asarray(src, dtype=np.int64).ravel()
        dst = np.asarray(dst, dtype=np.int64).ravel()
        tim = np.asarray(time, dtype=np.float64).ravel()
        if not (len(src) == len(dst) == len(tim)):
            raise ValueError('src/dst/time length mismatch')

        m = tim > 0.0  # time=0 edges carry no timestamp: excluded from every time feature
        s, d = src[m], dst[m]
        t = tim[m].astype(np.int64)
        if len(t) and (int(t.max()) >= (1 << 31) or int(t.min()) < 0):
            raise ValueError('timestamps must fit int32 seconds (0 <= t < 2**31)')
        self.num_edges = int(len(t))

        self.dst_index = _GroupedTimes(d, t)   # cand popularity windows + cand last-seen
        self.src_index = _GroupedTimes(s, t)   # src last-active
        # pair key: src * 2**bits + dst (unique; no int64 overflow at ds3 scale)
        max_node = int(max(s.max(initial=0), d.max(initial=0)))
        self.pair_shift = max(max_node.bit_length(), 1)
        self.pair_index = _GroupedTimes(s * (1 << self.pair_shift) + d, t)

    # ------------------------------------------------------------------

    def _window_counts(self, gid, qt):
        """(cnt30, cnt7, r): dst-grouped events in [t-30d, t) / [t-7d, t), plus rank(t).

        Counts are 0 when the group is missing; r is reused by last_gap so the
        dst index is searched exactly 3 times per batch (r, l30, l7).
        """
        idx = self.dst_index
        r = idx.segment_ranks(gid, qt)
        l30 = idx.segment_ranks(gid, np.maximum(qt - _SEC_30D, np.int64(0)))
        l7 = idx.segment_ranks(gid, np.maximum(qt - _SEC_7D, np.int64(0)))
        has = gid >= 0
        cnt30 = np.where(has, r - l30, 0).astype(np.float64)
        cnt7 = np.where(has, r - l7, 0).astype(np.float64)
        return cnt30, cnt7, r

    @staticmethod
    def _gap_feat(gap):
        """log1p(gap_seconds + 1); NaN (never) -> log1p(1e10)."""
        return np.where(np.isnan(gap), NEVER, np.log1p(gap + 1.0)).astype(np.float32)

    # ------------------------------------------------------------------

    def batch_features(self, src_arr: np.ndarray, time_arr: np.ndarray,
                       cand_mat: np.ndarray) -> np.ndarray:
        src_arr = np.asarray(src_arr, dtype=np.int64).ravel()
        time_arr = np.asarray(time_arr, dtype=np.float64).ravel()
        cand_mat = np.asarray(cand_mat, dtype=np.int64)
        n, k = cand_mat.shape
        if len(src_arr) != n or len(time_arr) != n:
            raise ValueError('src_arr/time_arr must have shape [N] matching cand_mat [N, K]')

        out = np.full((n, k, 6), NEVER, dtype=np.float32)  # covers t<=0 rows by construction
        valid = time_arr > 0.0
        if not valid.any():
            return out

        rows = np.nonzero(valid)[0]
        m = len(rows)
        qs_row = src_arr[rows]
        qt_row = time_arr[rows].astype(np.int64)
        qc = cand_mat[rows].ravel()                    # (m*K,)
        qt = np.repeat(qt_row, k)                      # (m*K,)

        # dims 0-3: candidate-dst grouped stats (per candidate)
        gid_dst = self.dst_index.map_gid(qc)
        cnt30, cnt7, r_dst = self._window_counts(gid_dst, qt)
        gap_dst = self.dst_index.last_gap(gid_dst, qt, rank=r_dst)
        # dim 4: (src, dst) pair last interaction (per candidate)
        qs = np.repeat(qs_row, k)
        gid_pair = self.pair_index.map_gid(qs * (1 << self.pair_shift) + qc)
        gap_pair = self.pair_index.last_gap(gid_pair, qt)
        # dim 5: src last active (per row, broadcast over K)
        gid_src_row = self.src_index.map_gid(qs_row)
        gap_src_row = self.src_index.last_gap(gid_src_row, qt_row)

        feats = np.empty((m, k, 6), dtype=np.float32)
        feats[:, :, 0] = np.log1p(cnt30).reshape(m, k).astype(np.float32)
        feats[:, :, 1] = np.log1p(cnt7).reshape(m, k).astype(np.float32)
        feats[:, :, 2] = (np.log1p(cnt7 + 0.5) - np.log1p(cnt30 + 0.5)).reshape(m, k).astype(np.float32)
        feats[:, :, 3] = self._gap_feat(gap_dst).reshape(m, k)
        feats[:, :, 4] = self._gap_feat(gap_pair).reshape(m, k)
        feats[:, :, 5] = self._gap_feat(gap_src_row)[:, None]
        out[rows] = feats
        return out


if __name__ == '__main__':
    # Tiny self-check: 5 edges, verify counts/windows/last-seen by hand.
    s = np.array([1, 2, 1, 3, 1])
    d = np.array([10, 10, 20, 10, 10])
    t = np.array([1000.0, 2000.0, 3000.0, 4000.0, 0.0])  # last edge has no timestamp
    st = RecencyStats(s, d, t)
    q_src = np.array([1])
    q_t = np.array([1000.0 + 8 * 86400.0])  # everything is >7d and <30d old
    q_c = np.array([[10, 20, 99]])
    f = st.batch_features(q_src, q_t, q_c)[0]
    close = lambda a, b: np.isclose(a, b, rtol=1e-6, atol=1e-6)
    assert close(f[0, 0], np.log1p(3)) and close(f[1, 0], np.log1p(1))  # cnt30: dst 10 x3, 20 x1
    assert f[0, 1] == 0.0 and f[1, 1] == 0.0                  # nothing within 7 days
    assert f[2, 0] == 0.0                                     # dst 99 never appeared -> cnt 0
    assert close(f[0, 3], np.log1p((q_t[0] - 4000.0) + 1.0))
    assert close(f[1, 4], np.log1p((q_t[0] - 3000.0) + 1.0))  # pair (1,20) last at 3000
    assert f[2, 4] == NEVER                                   # pair (1,99) never interacted
    assert close(f[0, 5], np.log1p((q_t[0] - 3000.0) + 1.0))  # src 1 last active at 3000
    # t<=0 rows: all NEVER
    f0 = st.batch_features(np.array([1]), np.array([0.0]), np.array([[10]]))
    assert np.all(f0 == NEVER)
    print('recency_features self-check OK\n', f)
