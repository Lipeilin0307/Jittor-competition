import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .io_data import (
    cleanup_path,
    dataset_dir,
    dump_json,
    ensure_dir,
    read_test,
    read_train,
    row_rank_score,
    row_zscore,
    softmax,
    split_edges,
    TestRow,
    tie_aware_mrr,
    top1_change,
    write_scores_csv,
)
from .temporal_graph import FEATURE_NAMES, GraphFeatureModel
from .parallel_features import feature_tensor_parallel
from .evaluation import score_feature_tensor, top1_stats
from .candidate_ranker import (
    BASELINE_MLP_WEIGHT,
    _as_path,
    build_hard_feature_parts,
    _candidate_pool,
    _baseline_logits,
    _load_feature_mlp_predictor,
    _load_baseline_test_parts,
    _load_baseline_weights,
    _read_json,
    score_linear_anchor,
    train_linear_anchor,
    _train_feature_mlp,
)


CONTEXT_FEATURE_NAMES = FEATURE_NAMES + [
    "seq_mean",
    "seq_max",
    "seq_last",
    "audience_dot",
    "audience_count",
    "recent_source_hit",
    "rank_seq_max",
    "rank_audience_dot",
    "source_activity",
    "source_recent_coverage",
    "source_history_fill",
    "pop_x_source_activity",
    "recent_pop_x_source_activity",
    "profile_x_source_activity",
]

# Fixed before seeing split=1. Used only to mine diverse hard negatives for
# split=0 training rows; learned scoring weights are fitted later on split=0.
HONEST_MINING_WEIGHTS = {name: 0.0 for name in FEATURE_NAMES}
HONEST_MINING_WEIGHTS.update({
    "rank_pop": 1.0,
    "rank_recent_pop": 1.0,
    "rank_pair": 1.0,
    "rank_recency": 0.5,
    "rank_transition": 0.5,
    "candidate_seen_in_test": 0.25,
})

_SEQ_BASE = None
_SEQ_SRC = None
_SEQ_DST = None
_SEQ_CONTEXT = None
_SEQ_LEN = None


def _append_sequence_bounds_worker(payload: tuple) -> str:
    shard, start, end, out_dir, prefix = payload
    if _SEQ_BASE is None or _SEQ_SRC is None or _SEQ_DST is None or _SEQ_CONTEXT is None or _SEQ_LEN is None:
        raise RuntimeError("fork worker did not inherit sequence feature context")
    augmented = _append_sequence_features(
        _SEQ_BASE[start:end], _SEQ_SRC[start:end], _SEQ_DST[start:end], _SEQ_CONTEXT, src_seq_len=_SEQ_LEN
    ).astype(np.float16)
    path = Path(out_dir) / f"{prefix}_{int(shard):03d}.npy"
    np.save(path, augmented)
    return str(path)


def _fit_model(edges: Sequence[Tuple[int, int, int]], test_rows, out_path: Path, seed: int, svd_dim: int) -> GraphFeatureModel:
    model = GraphFeatureModel("dataset2", svd_dim=svd_dim, seed=seed).fit(edges, test_rows)
    model.save(out_path)
    return model


class SequenceContext:
    def __init__(self, model: GraphFeatureModel, edges: Sequence[Tuple[int, int, int]], dst_seq_len: int = 64):
        self.model = model
        self.dst_seq_len = int(dst_seq_len)
        self.dst_recent_src: Dict[int, List[int]] = {}
        temp: Dict[int, List[int]] = {}
        for src, dst, _time in sorted(edges, key=lambda x: (x[2], x[0], x[1])):
            arr = temp.setdefault(int(dst), [])
            arr.append(int(src))
            if len(arr) > self.dst_seq_len:
                del arr[0: len(arr) - self.dst_seq_len]
        self.dst_recent_src = temp
        self.audience_mean: Dict[int, np.ndarray] = {}
        self.audience_count: Dict[int, float] = {}
        if model.src_emb is None:
            return
        dim = model.src_emb.shape[1]
        for dst, srcs in self.dst_recent_src.items():
            vecs = []
            for src in srcs:
                idx = model.src_to_id.get(int(src))
                if idx is not None:
                    vecs.append(model.src_emb[idx])
            if vecs:
                arr = np.asarray(vecs, dtype=np.float32)
                mean = arr.mean(axis=0)
                norm = max(float(np.linalg.norm(mean)), 1e-6)
                self.audience_mean[int(dst)] = (mean / norm).astype(np.float32)
                self.audience_count[int(dst)] = float(np.log1p(len(vecs)))
        self.max_audience_count = max(self.audience_count.values(), default=1.0)


def _append_sequence_features(base: np.ndarray, src_ids: np.ndarray, dst_ids: np.ndarray, ctx: SequenceContext, src_seq_len: int = 64) -> np.ndarray:
    model = ctx.model
    if model.src_emb is None or model.dst_emb is None:
        return base.astype(np.float32, copy=False)
    dim = min(model.src_emb.shape[1], model.dst_emb.shape[1])
    n, c, _ = base.shape
    extra = np.zeros((n, c, 14), dtype=np.float32)
    zero_src = np.zeros(dim, dtype=np.float32)
    zero_dst = np.zeros(dim, dtype=np.float32)
    for i in range(n):
        src = int(src_ids[i])
        src_idx = model.src_to_id.get(src)
        src_vec = model.src_emb[src_idx] if src_idx is not None else zero_src
        recent = list(model.src_recent.get(src, ())) [-int(src_seq_len):]
        source_activity = math.log1p(float(model.src_count.get(src, 0))) / max(float(model.max_log_src), 1.0)
        source_recent_coverage = float(len(set(recent))) / max(len(recent), 1)
        source_history_fill = min(len(recent), int(src_seq_len)) / max(int(src_seq_len), 1)
        hist_vecs = []
        for hist_dst in recent:
            idx = model.dst_to_id.get(int(hist_dst))
            if idx is not None:
                hist_vecs.append(model.dst_emb[idx])
        if hist_vecs:
            hist = np.asarray(hist_vecs, dtype=np.float32)
            hist_mean = hist.mean(axis=0)
            hist_mean = hist_mean / max(float(np.linalg.norm(hist_mean)), 1e-6)
            hist_last = hist[-1]
            hist_last = hist_last / max(float(np.linalg.norm(hist_last)), 1e-6)
        else:
            hist = None
            hist_mean = zero_dst
            hist_last = zero_dst
        dst_vecs = np.zeros((c, dim), dtype=np.float32)
        aud_vecs = np.zeros((c, dim), dtype=np.float32)
        aud_count = np.zeros(c, dtype=np.float32)
        recent_src_hit = np.zeros(c, dtype=np.float32)
        for j, dst in enumerate(dst_ids[i]):
            dst = int(dst)
            didx = model.dst_to_id.get(dst)
            if didx is not None:
                v = model.dst_emb[didx]
                dst_vecs[j] = v / max(float(np.linalg.norm(v)), 1e-6)
            av = ctx.audience_mean.get(dst)
            if av is not None:
                aud_vecs[j] = av
                aud_count[j] = ctx.audience_count.get(dst, 0.0) / max(ctx.max_audience_count, 1.0)
            if src in ctx.dst_recent_src.get(dst, ()):
                recent_src_hit[j] = 1.0
        seq_mean = dst_vecs @ hist_mean
        seq_last = dst_vecs @ hist_last
        if hist is not None and len(hist):
            hist_norm = hist / np.maximum(np.linalg.norm(hist, axis=1, keepdims=True), 1e-6)
            seq_max = (dst_vecs @ hist_norm.T).max(axis=1)
        else:
            seq_max = np.zeros(c, dtype=np.float32)
        aud_dot = aud_vecs @ (src_vec / max(float(np.linalg.norm(src_vec)), 1e-6))
        extra[i, :, 0] = seq_mean
        extra[i, :, 1] = seq_max
        extra[i, :, 2] = seq_last
        extra[i, :, 3] = aud_dot
        extra[i, :, 4] = aud_count
        extra[i, :, 5] = recent_src_hit
        extra[i, :, 6] = row_rank_score(seq_max)[0]
        extra[i, :, 7] = row_rank_score(aud_dot)[0]
        extra[i, :, 8] = source_activity
        extra[i, :, 9] = source_recent_coverage
        extra[i, :, 10] = source_history_fill
        extra[i, :, 11] = base[i, :, FEATURE_NAMES.index("pop")] * source_activity
        extra[i, :, 12] = base[i, :, FEATURE_NAMES.index("recent_pop")] * source_activity
        extra[i, :, 13] = base[i, :, FEATURE_NAMES.index("profile")] * source_activity
    return np.concatenate([base.astype(np.float32, copy=False), extra], axis=2).astype(np.float32, copy=False)


def append_sequence_features_parallel(
    base: np.ndarray,
    src_ids: np.ndarray,
    dst_ids: np.ndarray,
    ctx: SequenceContext,
    src_seq_len: int,
    out_dir: Path,
    prefix: str,
    workers: int,
) -> np.ndarray:
    """Append 14 sequence features in parallel without copying the graph model."""
    workers = max(1, min(int(workers), len(base), max(1, (os.cpu_count() or 1) - 8)))
    if workers == 1 or os.name != "posix":
        return _append_sequence_features(base, src_ids, dst_ids, ctx, src_seq_len=src_seq_len).astype(np.float16)
    out_dir = ensure_dir(out_dir)
    for old in out_dir.glob(f"{prefix}_*.npy"):
        old.unlink()
    global _SEQ_BASE, _SEQ_SRC, _SEQ_DST, _SEQ_CONTEXT, _SEQ_LEN
    _SEQ_BASE = base
    _SEQ_SRC = src_ids
    _SEQ_DST = dst_ids
    _SEQ_CONTEXT = ctx
    _SEQ_LEN = int(src_seq_len)
    bounds = np.linspace(0, len(base), workers + 1, dtype=np.int64)
    tasks = [
        (shard, int(bounds[shard]), int(bounds[shard + 1]), str(out_dir), prefix)
        for shard in range(workers)
        if int(bounds[shard + 1]) > int(bounds[shard])
    ]
    # 优化：去掉maxtasksperchild=1，提高进程复用率
    with mp.get_context("fork").Pool(processes=len(tasks)) as pool:
        paths = pool.map(_append_sequence_bounds_worker, tasks)
    return np.concatenate([np.load(path, mmap_mode="r") for path in paths], axis=0).astype(np.float16, copy=False)


def _hard_lists_from_edges(
    model_path: Path,
    model: GraphFeatureModel,
    edges: Sequence[Tuple[int, int, int]],
    weights: Dict[str, float],
    out_dir: Path,
    prefix: str,
    workers: int,
    max_pool: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    hot = [dst for dst, _ in model.dst_count.most_common(8000)]
    recent_hot = [dst for dst, _ in model.dst_recent_count.most_common(8000)]
    counts = np.asarray(list(model.dst_count.values()), dtype=np.float64)
    low_cut = float(np.percentile(counts, 40)) if len(counts) else 0.0
    low_pop = [dst for dst, count in model.dst_count.items() if count <= low_cut]
    known_dst = list(model.dst_count.keys())

    ensure_dir(out_dir)
    for old in out_dir.glob(f"{prefix}_part_*.npz"):
        old.unlink()
    workers = max(1, int(workers))
    part_reports = build_hard_feature_parts(
        model,
        edges,
        weights,
        hot,
        recent_hot,
        low_pop,
        known_dst,
        int(max_pool),
        out_dir,
        prefix,
        workers,
        int(seed),
    )
    part_files = sorted(out_dir.glob(f"{prefix}_part_*.npz"))
    features = []
    src_ids = []
    dst_ids = []
    labels = []
    for path in part_files:
        data = np.load(path)
        features.append(data["features"])
        src_ids.append(data["src_ids"])
        dst_ids.append(data["dst_ids"])
        labels.append(data["labels"])
    return (
        np.concatenate(features, axis=0).astype(np.float16),
        np.concatenate(src_ids, axis=0),
        np.concatenate(dst_ids, axis=0),
        np.concatenate(labels, axis=0),
        part_reports,
    )


def _template_lists_from_edges(
    model: GraphFeatureModel,
    edges: Sequence[Tuple[int, int, int]],
    test_rows: Sequence[TestRow],
    out_dir: Path,
    prefix: str,
    workers: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Insert a known future edge into real test candidate templates.

    This makes the supervised candidate set match online candidate frequency,
    source repetition, and known/unknown destination proportions.
    """
    rng = np.random.default_rng(int(seed))
    by_src: Dict[int, List[TestRow]] = {}
    for row in test_rows:
        by_src.setdefault(int(row.src), []).append(row)
    rows: List[TestRow] = []
    labels = np.zeros(len(edges), dtype=np.int64)
    template_hits = 0
    positive_already_present = 0
    for i, (src, pos_dst, time) in enumerate(edges):
        templates = by_src.get(int(src))
        if templates:
            template = templates[i % len(templates)]
            template_hits += 1
        else:
            template = test_rows[int(rng.integers(0, len(test_rows)))]
        candidates = list(template.candidates)
        try:
            label = candidates.index(int(pos_dst))
            positive_already_present += 1
        except ValueError:
            label = int(rng.integers(0, len(candidates)))
            candidates[label] = int(pos_dst)
        rows.append(TestRow(int(src), int(time), tuple(candidates)))
        labels[i] = label
    features = feature_tensor_parallel(model, rows, out_dir, f"{prefix}_features", int(workers))
    src_ids = np.asarray([row.src for row in rows], dtype=np.int64)
    dst_ids = np.asarray([row.candidates for row in rows], dtype=np.int64)
    return features, src_ids, dst_ids, labels, {
        "rows": len(rows),
        "same_source_template_fraction": template_hits / max(len(rows), 1),
        "positive_already_present_fraction": positive_already_present / max(len(rows), 1),
    }


def build_context_features(args) -> dict:
    data_dir = _as_path(args.data_dir)
    baseline_root = _as_path(args.baseline_root)
    artifacts = ensure_dir(_as_path(args.artifacts))
    reports = ensure_dir(_as_path(args.reports))
    ds_dir = dataset_dir(data_dir, "dataset2")
    test_rows = read_test(ds_dir / "test.csv")
    split0, split1, split_meta = split_edges(ds_dir, all_train=False, prefer_official=True)
    split0 = sorted(split0, key=lambda x: (x[2], x[0], x[1]))
    split1 = sorted(split1, key=lambda x: (x[2], x[0], x[1]))
    if int(args.fit_edge_limit) > 0:
        keep = int(args.fit_edge_limit)
        split0 = split0[: min(len(split0), keep)]
        split1 = split1[: min(len(split1), max(1000, keep // 5))]
        test_rows = test_rows[: min(len(test_rows), 3000)]

    cut = max(1, min(len(split0) - 1, int(len(split0) * float(args.history_frac))))
    history_edges = split0[:cut]
    train_pool = split0[cut:]
    rng = np.random.default_rng(int(args.seed))
    train_limit = min(max(int(args.train_rows), int(args.template_train_rows)), len(train_pool))
    valid_limit = min(max(int(args.valid_rows), int(args.template_valid_rows)), len(split1))
    train_order = rng.permutation(len(train_pool))[:train_limit]
    valid_order = rng.permutation(len(split1))[:valid_limit]
    train_edges = [train_pool[int(i)] for i in train_order]
    valid_edges = [split1[int(i)] for i in valid_order]
    hard_train_edges = train_edges[: min(int(args.train_rows), len(train_edges))]
    template_train_edges = train_edges[: min(int(args.template_train_rows), len(train_edges))]
    hard_valid_edges = valid_edges[: min(int(args.valid_rows), len(valid_edges))]
    template_valid_edges = valid_edges[: min(int(args.template_valid_rows), len(valid_edges))]

    model_dir = ensure_dir(artifacts / "models")
    block_model_path = model_dir / "history_graph.pkl"
    valid_model_path = model_dir / "validation_graph.pkl"
    inference_model_path = model_dir / "inference_graph.pkl"
    stable_artifacts = baseline_root / "artifacts"
    stable_valid_path = stable_artifacts / "dataset2_feature_model_val.pkl"
    stable_inference_path = stable_artifacts / "dataset2_feature_model_final.pkl"
    reuse_stable_graphs = str(args.reuse_stable_graphs) == "1"
    model_sources = {"history": "fitted"}
    if reuse_stable_graphs and block_model_path.exists():
        print(f"[ranker] reusing history graph {block_model_path}", flush=True)
        block_model = GraphFeatureModel.load(block_model_path)
        model_sources["history"] = "existing"
    else:
        print(f"[ranker] fitting history graph edges={len(history_edges)}", flush=True)
        block_model = _fit_model(history_edges, test_rows, block_model_path, int(args.seed), int(args.svd_dim))
    if reuse_stable_graphs and stable_valid_path.exists():
        print(f"[ranker] reusing stable validation graph {stable_valid_path}", flush=True)
        valid_model = GraphFeatureModel.load(stable_valid_path)
        valid_model_path = stable_valid_path
        model_sources["validation"] = "stable_baseline"
    else:
        print(f"[ranker] fitting validation graph edges={len(split0)}", flush=True)
        valid_model = _fit_model(split0, test_rows, valid_model_path, int(args.seed) + 1, int(args.svd_dim))
        model_sources["validation"] = "fitted"

    weights = dict(HONEST_MINING_WEIGHTS)
    hard_dir = ensure_dir(args.scratch_root / "hard_candidates")
    print(f"[ranker] building train hard candidates rows={len(hard_train_edges)}", flush=True)
    train_x, train_src, train_dst, train_y, train_parts = _hard_lists_from_edges(
        block_model_path,
        block_model,
        hard_train_edges,
        weights,
        hard_dir,
        "hard_train",
        int(args.workers),
        int(args.max_pool),
        int(args.seed),
    )
    print("[ranker] appending train sequence and audience features", flush=True)
    train_ctx = SequenceContext(block_model, history_edges, dst_seq_len=int(args.dst_seq_len))
    train_x = append_sequence_features_parallel(
        train_x, train_src, train_dst, train_ctx, int(args.src_seq_len), args.scratch_root / "context_features", "hard_train", int(args.workers)
    )
    print(f"[ranker] building train test-template candidates rows={len(template_train_edges)}", flush=True)
    template_train_x, template_train_src, template_train_dst, template_train_y, template_train_meta = _template_lists_from_edges(
        block_model,
        template_train_edges,
        test_rows,
        args.scratch_root / "template_candidates",
        "train",
        int(args.workers),
        int(args.seed) + 7,
    )
    template_train_x = append_sequence_features_parallel(
        template_train_x,
        template_train_src,
        template_train_dst,
        train_ctx,
        int(args.src_seq_len),
        args.scratch_root / "context_features",
        "template_train",
        int(args.workers),
    )
    train_x = np.concatenate([train_x, template_train_x], axis=0)
    train_src = np.concatenate([train_src, template_train_src], axis=0)
    train_dst = np.concatenate([train_dst, template_train_dst], axis=0)
    train_y = np.concatenate([train_y, template_train_y], axis=0)

    print(f"[ranker] building validation hard candidates rows={len(hard_valid_edges)}", flush=True)
    valid_x, valid_src, valid_dst, valid_y, valid_parts = _hard_lists_from_edges(
        valid_model_path,
        valid_model,
        hard_valid_edges,
        weights,
        hard_dir,
        "hard_valid",
        int(args.workers),
        int(args.max_pool),
        int(args.seed) + 17,
    )
    print("[ranker] appending validation sequence and audience features", flush=True)
    valid_ctx = SequenceContext(valid_model, split0, dst_seq_len=int(args.dst_seq_len))
    valid_x = append_sequence_features_parallel(
        valid_x, valid_src, valid_dst, valid_ctx, int(args.src_seq_len), args.scratch_root / "context_features", "hard_valid", int(args.workers)
    )
    print(f"[ranker] building validation test-template candidates rows={len(template_valid_edges)}", flush=True)
    template_valid_x, template_valid_src, template_valid_dst, template_valid_y, template_valid_meta = _template_lists_from_edges(
        valid_model,
        template_valid_edges,
        test_rows,
        args.scratch_root / "template_candidates",
        "valid",
        int(args.workers),
        int(args.seed) + 23,
    )
    template_valid_x = append_sequence_features_parallel(
        template_valid_x,
        template_valid_src,
        template_valid_dst,
        valid_ctx,
        int(args.src_seq_len),
        args.scratch_root / "context_features",
        "template_valid",
        int(args.workers),
    )
    valid_x = np.concatenate([valid_x, template_valid_x], axis=0)
    valid_src = np.concatenate([valid_src, template_valid_src], axis=0)
    valid_dst = np.concatenate([valid_dst, template_valid_dst], axis=0)
    valid_y = np.concatenate([valid_y, template_valid_y], axis=0)
    train_path = args.scratch_root / "ranker_train.npz"
    valid_path = args.scratch_root / "ranker_valid.npz"
    # Anchor the neural context branch to the already-validated base scorer.
    # The MLP learns a bounded residual instead of relearning popularity/pair
    # ordering from scratch, which is important for the mostly-new-pair ds4 set.
    train_anchor = score_feature_tensor(
        train_x[:, :, : len(FEATURE_NAMES)].astype(np.float32), weights)
    valid_anchor = score_feature_tensor(
        valid_x[:, :, : len(FEATURE_NAMES)].astype(np.float32), weights)
    train_x = np.concatenate([train_x, train_anchor[:, :, None]], axis=2)
    valid_x = np.concatenate([valid_x, valid_anchor[:, :, None]], axis=2)
    np.savez(train_path, features=train_x, src_ids=train_src, dst_ids=train_dst, labels=train_y)
    np.savez(valid_path, features=valid_x, src_ids=valid_src, dst_ids=valid_dst, labels=valid_y)

    if reuse_stable_graphs and stable_inference_path.exists():
        print(f"[ranker] reusing stable inference graph {stable_inference_path}", flush=True)
        inference_model_path = stable_inference_path
        model_sources["inference"] = "stable_baseline"
    else:
        print("[ranker] fitting all-train inference graph", flush=True)
        all_rows = read_train(ds_dir / "train.csv")
        all_edges = [(r.src, r.dst, r.time) for r in all_rows]
        if int(args.fit_edge_limit) > 0:
            all_edges = all_edges[: min(len(all_edges), int(args.fit_edge_limit))]
        _fit_model(all_edges, test_rows, inference_model_path, int(args.seed) + 2, int(args.svd_dim))
        model_sources["inference"] = "fitted"

    valid_feature_logits = score_feature_tensor(valid_x[:, :, : len(FEATURE_NAMES)].astype(np.float32), weights)
    report = {
        "split": split_meta,
        "history_frac": float(args.history_frac),
        "history_edges": len(history_edges),
        "train_pool_edges": len(train_pool),
        "hard_train_rows": int(len(hard_train_edges)),
        "template_train_rows": int(len(template_train_edges)),
        "hard_valid_rows": int(len(hard_valid_edges)),
        "template_valid_rows": int(len(template_valid_edges)),
        "train_rows": int(len(train_y)),
        "valid_rows": int(len(valid_y)),
        "svd_dim": int(args.svd_dim),
        "src_seq_len": int(args.src_seq_len),
        "dst_seq_len": int(args.dst_seq_len),
        "feature_dim": int(train_x.shape[-1] - 1),
        "context_input_dim": int(train_x.shape[-1]),
        "context_anchor": "base feature score appended as final input column",
        "hard_negative_mining": "fixed_prior_no_split1_tuning",
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "block_model": str(block_model_path),
        "valid_model": str(valid_model_path),
        "inference_model": str(inference_model_path),
        "model_sources": model_sources,
        "feature_valid_mrr": tie_aware_mrr(valid_feature_logits, valid_y),
        "label_check": {
            "train_min": int(train_y.min()),
            "train_max": int(train_y.max()),
            "valid_min": int(valid_y.min()),
            "valid_max": int(valid_y.max()),
        },
        "parts": {"hard_train": train_parts, "hard_valid": valid_parts},
        "template_sets": {"train": template_train_meta, "valid": template_valid_meta},
    }
    dump_json(reports / "build_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:6000], flush=True)
    # hard_candidates / template_candidates / context_features 三类分片已被消费到 ranker_{train,valid}.npz
    # 立即清理，避免 /dev/shm 累积占用
    cleanup_path(args.scratch_root / "hard_candidates")
    cleanup_path(args.scratch_root / "template_candidates")
    cleanup_path(args.scratch_root / "context_features")
    return report


def train_context_ranker(args) -> dict:
    artifacts = ensure_dir(_as_path(args.artifacts))
    reports = ensure_dir(_as_path(args.reports))
    train = np.load(args.scratch_root / "ranker_train.npz")
    valid = np.load(args.scratch_root / "ranker_valid.npz")
    train_x = train["features"].astype(np.float32)
    valid_x = valid["features"].astype(np.float32)
    train_y = train["labels"].astype(np.int64)
    valid_y = valid["labels"].astype(np.int64)
    anchor_path = Path(str(args.anchor_weights)) if str(args.anchor_weights).strip() else artifacts / "honest_anchor_weights.npz"
    if anchor_path.exists():
        anchor_weights = np.load(anchor_path)["weights"].astype(np.float32)
        print(f"[ranker] loading split0-only anchor {anchor_path}", flush=True)
    else:
        anchor_weights = train_linear_anchor(
            train_x, train_y, anchor_path, seed=int(args.anchor_seed),
            epochs=int(args.anchor_epochs), batch_size=int(args.anchor_batch_size),
            lr=float(args.anchor_lr),
        )
    # Keep prepared caches from the pre-anchor pipeline usable. The residual
    # predictor expects the base feature score as its final input column.
    expected_core_dim = len(FEATURE_NAMES)
    if train_x.shape[-1] == len(CONTEXT_FEATURE_NAMES):
        train_anchor = score_linear_anchor(train_x, anchor_weights)
        valid_anchor = score_linear_anchor(valid_x, anchor_weights)
        train_x = np.concatenate([train_x, train_anchor[:, :, None]], axis=2)
        valid_x = np.concatenate([valid_x, valid_anchor[:, :, None]], axis=2)
        print(f"[ranker] appended anchor to legacy cache: {len(CONTEXT_FEATURE_NAMES)} -> {train_x.shape[-1]} features", flush=True)
    elif train_x.shape[-1] == len(CONTEXT_FEATURE_NAMES) + 1:
        train_x[:, :, -1] = score_linear_anchor(train_x, anchor_weights)
        valid_x[:, :, -1] = score_linear_anchor(valid_x, anchor_weights)
        print("[ranker] replaced prepared anchor with split0-only learned anchor", flush=True)
    else:
        raise ValueError(f"unexpected context cache width {train_x.shape[-1]}; expected {len(CONTEXT_FEATURE_NAMES)} or {len(CONTEXT_FEATURE_NAMES) + 1}")
    out = []
    for seed in [int(x) for x in str(args.seeds).split(",") if x.strip()]:
        try:
            item = _train_feature_mlp(
                train_x,
                train_y,
                valid_x,
                valid_y,
                out_dir=artifacts / f"candidate_mlp_seed{seed}",
                seed=seed,
                hidden=int(args.hidden),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                margin_weight=float(args.margin_weight),
                margin=float(args.margin),
                profile_calibration_weight=float(args.profile_calibration_weight),
            )
        except Exception as exc:
            item = {"status": "failed", "seed": seed, "error": repr(exc)}
        out.append(item)
    report = {"models": out, "anchor_weights": str(anchor_path), "anchor_source": "split0_train_cache_only"}
    dump_json(reports / "model_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:6000], flush=True)
    if not any(item.get("status") == "trained" for item in out):
        raise RuntimeError("all candidate MLP training runs failed; preserving caches for diagnosis")
    # ranker_{train,valid}.npz 已被消费用于训练 3 个 seed MLP，删掉腾出 /dev/shm
    cleanup_path(args.scratch_root / "ranker_train.npz")
    cleanup_path(args.scratch_root / "ranker_valid.npz")
    return report


def _feature_tensor_sharded(model_path: Path, rows, out_dir: Path, workers: int) -> np.ndarray:
    model = GraphFeatureModel.load(model_path)
    return feature_tensor_parallel(model, rows, out_dir, "features_part", int(workers))


def predict_context_ranker(args) -> dict:
    data_dir = _as_path(args.data_dir)
    baseline_root = _as_path(args.baseline_root)
    artifacts = ensure_dir(_as_path(args.artifacts))
    reports = ensure_dir(_as_path(args.reports))
    build = _read_json(reports / "build_report.json")
    test_rows = read_test(dataset_dir(data_dir, "dataset2") / "test.csv")
    inference_model_path = Path(build["inference_model"])
    if str(args.reuse_baseline_features) == "1":
        features, feature_logits, mlp_logits = _load_baseline_test_parts(baseline_root)
        # 若 features 为空（predict_stable_dataset2 已清理 shards 且只存了 logits），需重算 features
        if features.shape[-1] == 0:
            print("[ranker] reuse_baseline_features=1 但 features 已清理，回退到重算", flush=True)
            features = _feature_tensor_sharded(inference_model_path, test_rows, args.scratch_root / "test_features", int(args.workers))
    else:
        features = _feature_tensor_sharded(inference_model_path, test_rows, args.scratch_root / "test_features", int(args.workers))
        weights = _load_baseline_weights(baseline_root)
        feature_logits = score_feature_tensor(features, weights).astype(np.float32)
        _, _, mlp_logits = _load_baseline_test_parts(baseline_root)
    inference_model = GraphFeatureModel.load(inference_model_path)
    test_src = np.asarray([r.src for r in test_rows], dtype=np.int64)
    test_dst = np.asarray([r.candidates for r in test_rows], dtype=np.int64)
    print("[ranker] appending test sequence and audience features", flush=True)
    inference_ctx = SequenceContext(inference_model, [(r.src, r.dst, r.time) for r in read_train(dataset_dir(data_dir, "dataset2") / "train.csv")], dst_seq_len=int(args.dst_seq_len))
    features = append_sequence_features_parallel(
        features,
        test_src,
        test_dst,
        inference_ctx,
        int(args.src_seq_len),
        args.scratch_root / "context_features",
        "test",
        int(args.workers),
    )
    anchor_path = Path(str(args.anchor_weights)) if str(args.anchor_weights).strip() else artifacts / "honest_anchor_weights.npz"
    anchor_weights = np.load(anchor_path)["weights"].astype(np.float32)
    context_anchor = score_linear_anchor(features, anchor_weights)
    features = np.concatenate([features, context_anchor[:, :, None]], axis=2)
    baseline = _baseline_logits(feature_logits, mlp_logits, BASELINE_MLP_WEIGHT)

    model_report = _read_json(reports / "model_report.json")
    logits = []
    for item in model_report["models"]:
        if item.get("status") != "trained":
            continue
        predictor = _load_feature_mlp_predictor(Path(item["checkpoint"]), Path(item["norm"]), hidden=int(args.hidden))
        logits.append(predictor(features, batch_size=int(args.predict_batch_size)))
    if not logits:
        raise RuntimeError("no trained context ranking models")
    context_logits = np.mean([row_zscore(x) for x in logits], axis=0).astype(np.float32)
    np.save(artifacts / "context_logits.npy", context_logits)
    np.save(artifacts / "baseline_logits.npy", baseline)
    report = {
        "models": len(logits),
        "shape": list(context_logits.shape),
        "reuse_baseline_features": str(args.reuse_baseline_features),
        "top1_change_vs_baseline": top1_change(baseline, context_logits),
        "baseline_top1": top1_stats(baseline, test_rows, inference_model),
        "context_top1": top1_stats(context_logits, test_rows, inference_model),
    }
    dump_json(reports / "predict_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:5000], flush=True)
    # test_features / test context_features 已消费完，context_logits + baseline_logits 已落盘
    cleanup_path(args.scratch_root / "test_features")
    cleanup_path(args.scratch_root / "context_features")
    return report


def package_submission(args) -> dict:
    """dataset2-only：仅输出 dataset2.csv 提交文件。

    本流水线只处理 dataset2；dataset1.csv 由独立的 GraphMixer 流水线
    （code/dataset1_graphmixer/）生成，二者最终在提交阶段合并。
    因此这里不再尝试复制 dataset1.csv，也不再打包合并 zip，
    仅把融合后的 dataset2 概率写成 CSV。
    """
    baseline_root = _as_path(args.baseline_root)
    artifacts = _as_path(args.artifacts)
    reports = ensure_dir(_as_path(args.reports))
    submission = ensure_dir(_as_path(args.submission))
    baseline = np.load(artifacts / "baseline_logits.npy")
    context_logits = np.load(artifacts / "context_logits.npy")
    blend_weight = float(args.blend_weight)
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError(f"blend_weight must be between 0 and 1, got {blend_weight}")
    name = str(args.output_name or "temporal_ranker_blend_0p10")
    logits = row_zscore(baseline) * (1.0 - blend_weight) + row_zscore(context_logits) * blend_weight
    out_dir = ensure_dir(submission / name)
    d2 = out_dir / "dataset2.csv"
    d2_check = write_scores_csv(softmax(logits), d2)
    manifest = {
        name: {
            "csv": str(d2),
            "blend_weight": blend_weight,
            "dataset2": d2_check,
            "top1_change_vs_baseline": top1_change(baseline, logits),
        }
    }
    print(f"packed {name} {manifest[name]}", flush=True)
    dump_json(reports / "submission_manifest.json", manifest)
    return manifest
