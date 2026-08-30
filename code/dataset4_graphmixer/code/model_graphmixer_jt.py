"""
model_graphmixer_jt.py
======================
Full GraphMixer for Jittor temporal link prediction (SVD-initialized).

Architecture (full / "满血版"):
  - Node embedding with SVD init (shared or bipartite)
  - Sinusoidal positional encoding (precomputed, numpy-only)
  - MLP-Mixer blocks: token-mix + channel-mix (both with residual)
  - Mean pooling over valid history positions
  - Query = src_emb + history_proj(pooled_memory)
  - MLP scorer: concat(query, candidate) -> 2-layer MLP -> score

Loss: listwise cross_entropy + BPR (same as TemporalBPR v3).
"""
import math
import numpy as np
import jittor as jt
from jittor import nn


# =============================================================================
# 1. MLP-Mixer Block (token-mix + channel-mix, full GraphMixer style)
# =============================================================================

class MixerBlock(nn.Module):
    """Full MLP-Mixer block with token-mixing and channel-mixing."""

    def __init__(self, hidden_dim, max_history_length, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        # Token-mixing: operates on the sequence (time) dimension
        token_hidden = max(int(max_history_length * mlp_ratio), 4)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(max_history_length, token_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden, max_history_length),
            nn.Dropout(dropout),
        )

        # Channel-mixing: operates on the feature dimension
        channel_hidden = max(int(hidden_dim * mlp_ratio), 4)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def execute(self, x):
        """x: (B, L, D)"""
        # Token-mix
        h = self.token_norm(x)               # (B, L, D)
        h = h.transpose(0, 2, 1)             # (B, D, L)
        h = self.token_mlp(h)                # (B, D, L)
        h = h.transpose(0, 2, 1)             # (B, L, D)
        x = x + h

        # Channel-mix
        h = self.channel_norm(x)             # (B, L, D)
        h = self.channel_mlp(h)              # (B, L, D)
        x = x + h
        return x


# =============================================================================
# 2. MLP Scorer (full version: concat -> MLP)
# =============================================================================

class MLPScorer(nn.Module):
    """Concatenate query, candidate embeddings, and optional heuristic features, then score with MLP."""

    def __init__(self, hidden_dim, mlp_ratio=2.0, dropout=0.1, heuristic_dim=0):
        super().__init__()
        self.heuristic_dim = heuristic_dim
        input_dim = hidden_dim * 2 + heuristic_dim
        hidden = max(int(input_dim * mlp_ratio), 4)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def execute(self, query, candidate, heuristic=None):
        """
        query:     (B, D)
        candidate: (B, N, D)
        heuristic: (B, N, H) or None
        Returns:   scores (B, N)
        """
        B, N, D = candidate.shape
        query_r = query.unsqueeze(1).repeat(1, N, 1)  # (B, N, D)
        if self.heuristic_dim > 0 and heuristic is not None:
            concat = jt.concat([query_r, candidate, heuristic], dim=2)  # (B, N, 2*D+H)
        else:
            concat = jt.concat([query_r, candidate], dim=2)  # (B, N, 2*D)
        scores = self.mlp(concat).squeeze(2)          # (B, N)
        return scores


# =============================================================================
# 3. GraphMixer Model (full, compatible with TemporalBPR v3 training interface)
# =============================================================================

class GraphMixerModel(nn.Module):
    """
    Full GraphMixer with MLP-Mixer blocks and MLP scorer.
    Optionally includes heuristic features (degree, popularity, edge_count) for scoring.

    Compatible interface:
      - calculate_loss(src_np, hist_np, cand_np) -> loss, loss_dict
      - execute(src_np, hist_np, cand_np) -> scores (B, N)
    """

    def __init__(self, src_count, dst_count, hidden_dim, initial_features,
                 shared_nodes=False, num_layers=2, max_history_length=10,
                 mlp_ratio=2.0, dropout=0.1, temperature=0.1, time_decay=0.5,
                 heuristic_degree=None, heuristic_popularity=None,
                 heuristic_edge_count=None, edge_count_max=1.0,
                 use_known_flag=False, heuristic_dst_known=None,
                 extra_feat_dim=0,
                 initial_src_features=None, initial_dst_features=None):
        super().__init__()
        self.shared_nodes = shared_nodes
        self.hidden_dim = hidden_dim
        self.max_history_length = max_history_length
        self.temperature = temperature
        self.time_decay = time_decay

        # Heuristic features
        self.heuristic_degree = heuristic_degree
        self.heuristic_popularity = heuristic_popularity
        self.heuristic_edge_count = heuristic_edge_count
        self.edge_count_max = edge_count_max
        self.use_heuristics = (heuristic_degree is not None)
        # Optional 4th heuristic dim: dst_known flag (candidate seen as dst in train)
        self.use_known_flag = bool(use_known_flag)
        self.heuristic_dst_known = heuristic_dst_known  # bool numpy array indexed by node id
        # Scorer heuristic input width: 3 base dims (degree/popularity/edge_count)
        # when heuristics are on, plus 1 optional dst_known dim. Not hardcoded.
        self.heuristic_dim = (3 if self.use_heuristics else 0) + (1 if self.use_known_flag else 0)
        # Optional caller-supplied extra features, concatenated AFTER the
        # heuristic block inside the scorer input. Layout convention when both
        # feature groups are enabled (callers concatenate in this fixed order):
        #   [RecencyStats 6-dim block | CFStats 8-dim block]  -> extra_feat_dim=14
        # either group alone uses 6 (v2 recency behavior) or 8 (CF only).
        self.extra_feat_dim = int(extra_feat_dim)

        # ------------------------------------------------------------------
        # Feature projection (if node features dim != hidden_dim)
        # ------------------------------------------------------------------
        # ds4: non-shared mode may pass SEPARATE per-tower tables
        # (initial_src_features / initial_dst_features, both already aligned to
        # the model's id space). When they are given they take precedence over
        # initial_features for the embedding init; default None reproduces the
        # original single-table slicing behavior exactly.
        dual_features = (initial_src_features is not None) and (initial_dst_features is not None)
        if dual_features:
            if initial_src_features.shape[1] != initial_dst_features.shape[1]:
                raise ValueError(
                    f'initial_src_features dim {initial_src_features.shape[1]} != '
                    f'initial_dst_features dim {initial_dst_features.shape[1]}')
            feature_dim = initial_src_features.shape[1]
        else:
            feature_dim = initial_features.shape[1]
        self.feature_proj = nn.Linear(feature_dim, hidden_dim) if feature_dim != hidden_dim else None

        # ------------------------------------------------------------------
        # Embeddings (SVD-initialized, same pattern as TemporalBPR v3)
        # ------------------------------------------------------------------
        if shared_nodes:
            self.src_emb = nn.Embedding(src_count, feature_dim, padding_idx=0)
            self.dst_emb = self.src_emb
            with jt.no_grad():
                self.src_emb.weight.update(jt.array(initial_features.astype(np.float32)))
            # NOTE: weight[0].update() was removed: it is a silent no-op on CUDA
            # and crashes NPU ACL (scalar-index getitem, ERROR 561103). Row 0
            # zero-invariant is enforced by the caller (train script line ~600).
        else:
            if dual_features:
                src_init = initial_src_features
                dst_init = initial_dst_features
            else:
                src_init = initial_features[:src_count] if initial_features.shape[0] >= src_count else initial_features
                dst_init = initial_features[:dst_count] if initial_features.shape[0] >= dst_count else initial_features

            if src_init.shape[0] < src_count:
                src_init = np.concatenate([
                    src_init,
                    np.zeros((src_count - src_init.shape[0], feature_dim), dtype=np.float32)
                ], axis=0)
            if dst_init.shape[0] < dst_count:
                dst_init = np.concatenate([
                    dst_init,
                    np.zeros((dst_count - dst_init.shape[0], feature_dim), dtype=np.float32)
                ], axis=0)

            self.src_emb = nn.Embedding(src_count, feature_dim, padding_idx=0)
            self.dst_emb = nn.Embedding(dst_count, feature_dim, padding_idx=0)
            with jt.no_grad():
                self.src_emb.weight.update(jt.array(src_init.astype(np.float32)))
                self.dst_emb.weight.update(jt.array(dst_init.astype(np.float32)))
            # NOTE: weight[0].update() removed (silent no-op on CUDA, crashes
            # NPU ACL scalar-index getitem). Caller zeroes row 0 of both tables.

        # ------------------------------------------------------------------
        # Positional encoding (precomputed with numpy, no jt.exp in graph)
        # ------------------------------------------------------------------
        pe = np.zeros((max_history_length, hidden_dim), dtype=np.float32)
        position = np.arange(max_history_length, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, hidden_dim, 2, dtype=np.float32) * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        if hidden_dim % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        self.pos_embedding = jt.array(pe)  # (L, D), non-trainable buffer

        # ------------------------------------------------------------------
        # Full MLP-Mixer blocks
        # ------------------------------------------------------------------
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(hidden_dim, max_history_length, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # Query projector + MLP scorer
        # ------------------------------------------------------------------
        self.history_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.scorer = MLPScorer(hidden_dim, mlp_ratio, dropout,
                                self.heuristic_dim + self.extra_feat_dim)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _encode_history(self, hist_np, time_gap_np=None):
        """
        Encode history sequence through full GraphMixer.
        hist_np: (B, L) int64 numpy array, 0 = padding
        time_gap_np: (B, L) float32 numpy array, time gaps for each history position
        Returns: pooled memory (B, D)
        """
        B, L = hist_np.shape

        # Node embeddings
        hist_jt = jt.array(hist_np)
        x = self.dst_emb(hist_jt)           # (B, L, feature_dim)
        if self.feature_proj is not None:
            x = self.feature_proj(x)         # (B, L, hidden_dim)

        # Positional encoding (up to L, precomputed, no grad)
        pos_emb = self.pos_embedding[:L].stop_grad().unsqueeze(0)  # (1, L, hidden_dim)
        x = x + pos_emb

        # Zero out padding positions
        mask = (hist_np == 0)                # (B, L) bool numpy
        if mask.any():
            mask_jt = jt.array(mask).unsqueeze(2).float()  # (B, L, 1)
            x = x * (1.0 - mask_jt)

        # Full MLP-Mixer blocks
        for block in self.mixer_blocks:
            x = block(x)
        x = self.norm(x)

        # Mean pooling over valid positions with time-aware weights
        valid_mask = (~mask).astype(np.float32)[:, :, None]  # (B, L, 1) numpy
        valid_mask_jt = jt.array(valid_mask)
        
        # Base weights: exponential decay from recent to old positions
        positions = jt.arange(L, dtype='float32').unsqueeze(0).unsqueeze(2)  # (1, L, 1)
        time_weights = jt.exp(-self.time_decay * (L - 1 - positions))  # (1, L, 1), recent higher
        
        # If time_gap provided, modulate weights by recency (smaller gap = higher weight)
        if time_gap_np is not None:
            time_gap_jt = jt.array(time_gap_np).unsqueeze(2)  # (B, L, 1)
            recency = jt.exp(-time_gap_jt)  # (B, L, 1), clamped by tanh in data loader
            time_weights = time_weights * recency  # combine position decay + recency
        
        time_weights = time_weights * valid_mask_jt
        time_weights = time_weights / (time_weights.sum(dim=1, keepdims=True) + 1e-8)
        memory = (x * time_weights).sum(dim=1)  # (B, D)
        return memory

    def _get_heuristic_features(self, src_np, cand_np):
        """
        Compute heuristic features for (src, candidate) pairs.
        Returns (B, N, H) tensor, H = self.heuristic_dim:
          base dims (when use_heuristics): [degree, popularity, edge_count]
          optional extra dim (when use_known_flag): [dst_known]
        Unseen candidates get all-zero feature values (no NaN possible).
        Returns None when H == 0.
        """
        if self.heuristic_dim == 0:
            return None
        B, N = cand_np.shape
        h = np.zeros((B, N, self.heuristic_dim), dtype=np.float32)
        col = 0
        if self.use_heuristics:
            num_nodes = self.heuristic_degree.shape[0]
            # clip candidate indices to valid range [0, num_nodes-1]
            cand_clipped = np.clip(cand_np, 0, num_nodes - 1)
            h[:, :, col + 0] = self.heuristic_degree[cand_clipped]
            h[:, :, col + 1] = self.heuristic_popularity[cand_clipped]
            ec = self.heuristic_edge_count
            if isinstance(ec, tuple):
                # ds4 vectorized form: (sorted int64 pair keys, float32 counts,
                # shift) with key = (src << shift) | dst. One searchsorted per
                # batch replaces the per-pair Python dict loop (232M lookups
                # at ds4 test scale would otherwise dominate inference).
                keys_sorted, vals_sorted, shift = ec
                if len(keys_sorted) == 0:
                    pass  # column stays 0.0
                else:
                    qk = ((src_np.astype(np.int64)[:, None]) << np.int64(shift)) | cand_np
                    pos = np.searchsorted(keys_sorted, qk.ravel())
                    pos = np.minimum(pos, len(keys_sorted) - 1)
                    hit = keys_sorted[pos] == qk.ravel()
                    h[:, :, col + 2] = np.where(hit, vals_sorted[pos], 0.0).reshape(B, N)
            else:
                # edge_count: dict lookup per pair (unseen nodes get 0)
                for i in range(B):
                    for j in range(N):
                        h[i, j, col + 2] = ec.get((int(src_np[i]), int(cand_np[i, j])), 0.0)
            if self.edge_count_max > 1.0:
                h[:, :, col + 2] = h[:, :, col + 2] / self.edge_count_max
            col += 3
        if self.use_known_flag:
            if self.heuristic_dst_known is not None:
                known_clipped = np.clip(cand_np, 0, self.heuristic_dst_known.shape[0] - 1)
                h[:, :, col] = self.heuristic_dst_known[known_clipped].astype(np.float32)
            # else: column stays 0.0 (conservative: nothing known)
        return jt.array(h)

    def _score(self, src_np, hist_np, cand_np, time_gap_np=None, extra_feats=None):
        """
        Compute scores for candidates via MLP scorer.
        src_np: (B,) int64
        hist_np: (B, L) int64
        cand_np: (B, N) int64
        time_gap_np: (B, L) float32, optional time gaps
        extra_feats: (B, N, E) float32, optional caller-supplied feature block
            (e.g. RecencyStats.batch_features output); concatenated after the
            heuristic block. Zeros are substituted when the model was built
            with extra_feat_dim > 0 but no features are passed.
        Returns: scores (B, N)
        """
        B = src_np.shape[0]
        N = cand_np.shape[1]

        # Encode history
        memory = self._encode_history(hist_np, time_gap_np)  # (B, D)

        # Query vector
        src_jt = jt.array(src_np)
        src_emb = self.src_emb(src_jt)          # (B, feature_dim)
        if self.feature_proj is not None:
            src_emb = self.feature_proj(src_emb)  # (B, hidden_dim)
        query = src_emb + self.history_proj(memory)
        query = self.query_norm(query)          # (B, hidden_dim)

        # Candidate embeddings
        cand_jt = jt.array(cand_np)
        cand_emb = self.dst_emb(cand_jt)        # (B, N, feature_dim)
        if self.feature_proj is not None:
            cand_emb = self.feature_proj(cand_emb)  # (B, N, hidden_dim)

        # Heuristic features (+ optional extra block appended after them)
        heuristic = self._get_heuristic_features(src_np, cand_np)
        if self.extra_feat_dim > 0:
            if extra_feats is None:
                extra_feats = np.zeros((B, N, self.extra_feat_dim), dtype=np.float32)
            extra_jt = jt.array(np.ascontiguousarray(extra_feats, dtype=np.float32))
            heuristic = extra_jt if heuristic is None else jt.concat([heuristic, extra_jt], dim=2)

        # MLP scorer
        scores = self.scorer(query, cand_emb, heuristic)   # (B, N)
        return scores

    # ------------------------------------------------------------------
    # Public API (compatible with TemporalBPR v3 training loop)
    # ------------------------------------------------------------------

    def execute(self, src_np, hist_np, cand_np, time_gap_np=None, extra_feats=None):
        """Inference: return raw scores (B, N)."""
        return self._score(src_np, hist_np, cand_np, time_gap_np, extra_feats=extra_feats)

    forward = execute

    def calculate_loss(self, src_np, hist_np, cand_np, time_gap_np=None, extra_feats=None):
        """
        Training loss: listwise cross_entropy + BPR.
        cand_np: (B, 1+num_neg) = [pos, neg1, neg2, ...]
        """
        scores = self._score(src_np, hist_np, cand_np, time_gap_np,
                             extra_feats=extra_feats)       # (B, N)
        B, N = scores.shape

        # Listwise loss
        labels = jt.zeros((B,), dtype='int64')
        logits = scores / self.temperature
        listwise_loss = nn.cross_entropy_loss(logits, labels)

        # BPR loss: ReLU margin (NPU-safe, no softplus/exp)
        pos_scores = scores[:, 0:1]            # (B, 1)
        neg_scores = scores[:, 1:]             # (B, N-1)
        margin = 1.0
        # [910B NPU] nn.relu is jt.ternary(cond, x, 0.0) (nn.py:182); ACL's
        # ternary broadcasts the scalar via aclnnExpand, whose ExpandAiCore
        # kernel init fails (ERROR 561103) and segfaults inside libnnopbase.
        # Arithmetic relu uses only compare + cast + mul; identical math/grads.
        bpr_margin = neg_scores - pos_scores + margin
        bpr_loss = ((bpr_margin > 0).float32() * bpr_margin).mean()

        loss = listwise_loss + 0.2 * bpr_loss
        return loss, {
            'loss': loss.item(),
            'listwise': listwise_loss.item(),
            'bpr': bpr_loss.item()
        }


# =============================================================================
# 4. Simple sanity check
# =============================================================================

if __name__ == '__main__':
    jt.flags.use_cuda = 0  # CPU test

    # Shared nodes (dataset1 style)
    num_nodes = 100
    hidden_dim = 32
    init_feat = np.random.randn(num_nodes, hidden_dim).astype(np.float32)
    model = GraphMixerModel(
        src_count=num_nodes, dst_count=num_nodes, hidden_dim=hidden_dim,
        initial_features=init_feat, shared_nodes=True,
        num_layers=2, max_history_length=10, mlp_ratio=2.0, dropout=0.1
    )

    src = np.random.randint(0, num_nodes, size=(4,))
    hist = np.random.randint(1, num_nodes, size=(4, 10))
    cand = np.random.randint(1, num_nodes, size=(4, 32))

    scores = model.execute(src, hist, cand)
    print('Sanity check scores shape:', scores.shape)

    loss, loss_dict = model.calculate_loss(src, hist, cand)
    print('Sanity check loss:', loss.item(), loss_dict)

    # Bipartite (dataset2 style)
    src_count = 20
    dst_count = 80
    init_feat2 = np.random.randn(max(src_count, dst_count), hidden_dim).astype(np.float32)
    model2 = GraphMixerModel(
        src_count=src_count, dst_count=dst_count, hidden_dim=hidden_dim,
        initial_features=init_feat2, shared_nodes=False,
        num_layers=2, max_history_length=10
    )
    scores2 = model2.execute(src % src_count, hist % dst_count, cand % dst_count)
    print('Bipartite scores shape:', scores2.shape)

    # Dual per-tower features + vectorized edge_count (ds4 style).
    # NOTE: jittor's weight[0].update() slice-update is a silent no-op, so the
    # "row 0 = padding = 0" invariant must come from the caller zeroing row 0
    # of the feature tables (gen_features_ds4.py does; train script re-enforces).
    src_feat = np.random.randn(src_count, hidden_dim).astype(np.float32)
    dst_feat = np.random.randn(dst_count, hidden_dim).astype(np.float32)
    src_feat[0] = 0.0
    dst_feat[0] = 0.0
    shift = int(dst_count).bit_length()
    pairs = np.array([1, 2, 3], dtype=np.int64)
    pdst = np.array([5, 6, 7], dtype=np.int64)
    ec_keys = np.sort(pairs << np.int64(shift) | pdst)
    ec_vals = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    model3 = GraphMixerModel(
        src_count=src_count, dst_count=dst_count, hidden_dim=hidden_dim,
        initial_features=None, initial_src_features=src_feat, initial_dst_features=dst_feat,
        shared_nodes=False, num_layers=2, max_history_length=10,
        heuristic_degree=np.ones(dst_count, dtype=np.float32),
        heuristic_popularity=np.ones(dst_count, dtype=np.float32),
        heuristic_edge_count=(ec_keys, ec_vals, shift), edge_count_max=6.0,
    )
    w_src = model3.src_emb.weight.numpy()
    w_dst = model3.dst_emb.weight.numpy()
    assert np.allclose(w_src, src_feat) and np.allclose(w_dst, dst_feat)
    assert np.all(w_src[0] == 0) and np.all(w_dst[0] == 0)
    h = model3._get_heuristic_features(np.array([1, 3, 9]), np.array([[5, 6, 7], [6, 7, 5], [7, 8, 5]])).numpy()
    assert np.isclose(h[0, 0, 2], 4.0 / 6.0)                 # pair (1,5) exists
    assert h[0, 1, 2] == 0.0 and h[0, 2, 2] == 0.0           # (1,6) / (1,7) absent
    assert h[1, 0, 2] == 0.0 and np.isclose(h[1, 1, 2], 1.0) # (3,7) = max -> 1.0
    assert np.all(h[2, :, 2] == 0.0)                         # src 9 has no pairs
    scores3 = model3.execute(np.array([1, 2]), np.ones((2, 10), dtype=np.int64),
                             np.ones((2, 32), dtype=np.int64))
    assert scores3.shape == (2, 32) and np.isfinite(scores3.numpy()).all()
    print('Dual-feature + vectorized edge_count check OK.')

    print('Full GraphMixer model OK.')
