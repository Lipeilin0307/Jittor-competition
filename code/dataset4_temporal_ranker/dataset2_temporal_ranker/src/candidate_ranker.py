import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# On ARM64, initialize Jittor's OpenMP runtime before scikit-learn when the
# neural stage runs in its own process. Feature-only stages leave this off.
if os.environ.get("JITTOR_EARLY_IMPORT", "0") == "1":
    import jittor as _jittor_early

from .io_data import dataset_dir, ensure_dir, read_test, row_zscore, tie_aware_mrr
from .temporal_graph import FEATURE_NAMES, GraphFeatureModel
from .evaluation import score_feature_tensor


BASELINE_MLP_WEIGHT = 5.5
CONTEXT_RESIDUAL_SCALE = 0.25
_HARD_MODEL = None
_HARD_EDGES = None
_HARD_CONFIG = None


def score_linear_anchor(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    core = features[:, :, :len(FEATURE_NAMES)].astype(np.float32)
    vector = np.asarray(weights, dtype=np.float32).reshape(-1)
    if vector.shape[0] != len(FEATURE_NAMES):
        raise ValueError(f"anchor has {vector.shape[0]} weights, expected {len(FEATURE_NAMES)}")
    return np.einsum("bcf,f->bc", core, vector, optimize=True).astype(np.float32)


def train_linear_anchor(train_x: np.ndarray, train_y: np.ndarray, out_path: Path,
                        seed: int, epochs: int = 5, batch_size: int = 2048,
                        lr: float = 2e-2) -> np.ndarray:
    """Fit the anchor on split=0-derived rows only; no validation data is read."""
    import jittor as jt
    from jittor import nn

    _configure_jittor_backend(jt)
    try:
        jt.set_global_seed(seed)
    except Exception:
        pass
    np.random.seed(seed)

    class LinearAnchor(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, 1)

        def execute(self, x):
            b, c, f = x.shape
            return self.linear(x.reshape((b * c, f))).reshape((b, c))

    net = LinearAnchor(len(FEATURE_NAMES))
    opt = nn.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    core = train_x[:, :, :len(FEATURE_NAMES)]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_y))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            logits = net(jt.array(core[idx].astype(np.float32)))
            loss = nn.cross_entropy_loss(logits, jt.array(train_y[idx]))
            opt.step(loss)
            losses.append(float(loss.data))
        print(f"honest_anchor epoch={epoch} loss={float(np.mean(losses)):.6f}", flush=True)
    weights = np.asarray(net.linear.weight.data, dtype=np.float32).reshape(-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, weights=weights, feature_names=np.asarray(FEATURE_NAMES),
             source=np.asarray(["split0_train_cache_only"]))
    return weights


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_path(value) -> Path:
    return Path(value).expanduser().resolve()


def _load_baseline_weights(baseline_root: Path) -> Dict[str, float]:
    report = _read_json(baseline_root / "reports" / "dataset2_train_report.json")
    return {k: float(v) for k, v in report["weights"].items()}


def _load_test_src_dst(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    test_rows = read_test(dataset_dir(data_dir, "dataset2") / "test.csv")
    src = np.asarray([r.src for r in test_rows], dtype=np.int64)
    dst = np.asarray([r.candidates for r in test_rows], dtype=np.int64)
    return src, dst


def _load_baseline_test_parts(baseline_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载 dataset2 测试集的 feature_logits / mlp_logits / features。

    优先从 predict_stable_dataset2 落盘的合并文件读取（不再依赖 shards，shards 已被清理）；
    shards 路径作为 legacy fallback 保留。
    """
    stable_artifacts = baseline_root / "artifacts"
    mlp_logits_path = stable_artifacts / "dataset2_mlp_logits.npy"
    feature_logits_path = stable_artifacts / "dataset2_feature_logits.npy"

    if mlp_logits_path.exists() and feature_logits_path.exists():
        mlp_logits = np.load(mlp_logits_path).astype(np.float32)
        feature_logits = np.load(feature_logits_path).astype(np.float32)
        # features 默认不落盘（体积大 1.29GB）；若调用方需要 features，应通过 reuse_baseline_features=0 重算
        # 这里返回空 features 兼容签名；实际复用路径只读 logits
        features = np.zeros((mlp_logits.shape[0], mlp_logits.shape[1], 0), dtype=np.float32)
        return features, feature_logits, mlp_logits

    # legacy fallback：从 shards 读取
    shard_dir = stable_artifacts / "dataset2_predict_shards"
    feature_logits = np.concatenate([np.load(p) for p in sorted(shard_dir.glob("feature_logits_part_*.npy"))], axis=0).astype(np.float32)
    mlp_logits = np.concatenate([np.load(p) for p in sorted(shard_dir.glob("mlp_logits_part_*.npy"))], axis=0).astype(np.float32)
    features = np.concatenate([np.load(p, mmap_mode="r") for p in sorted(shard_dir.glob("features_part_*.npy"))], axis=0).astype(np.float32)
    return features, feature_logits, mlp_logits


def _baseline_logits(feature_logits: np.ndarray, mlp_logits: np.ndarray, weight: float = BASELINE_MLP_WEIGHT) -> np.ndarray:
    return (row_zscore(feature_logits) + float(weight) * row_zscore(mlp_logits)).astype(np.float32)


def _candidate_pool(
    model: GraphFeatureModel,
    src: int,
    pos_dst: int,
    rng: np.random.Generator,
    hot: Sequence[int],
    recent_hot: Sequence[int],
    low_pop: Sequence[int],
    known_dst: Sequence[int],
    max_pool: int,
) -> List[int]:
    pool = [int(pos_dst)]
    seen = {int(pos_dst)}

    def add_many(values, limit=None):
        count = 0
        for value in values:
            value = int(value)
            if value not in seen:
                seen.add(value)
                pool.append(value)
                count += 1
                if len(pool) >= max_pool or (limit is not None and count >= limit):
                    break

    recent = list(model.src_recent.get(src, ()))
    add_many(reversed(recent), 120)
    for hist_dst in reversed(recent[-24:]):
        table = model.transition.get(hist_dst)
        if table:
            add_many((dst for dst, _ in sorted(table.items(), key=lambda x: -x[1])[:40]), 40)
    add_many(recent_hot, 160)
    add_many(hot, 160)
    if low_pop:
        add_many(rng.choice(np.asarray(low_pop, dtype=np.int64), size=min(120, len(low_pop)), replace=False), 120)
    while len(pool) < max_pool and known_dst:
        add_many([known_dst[int(rng.integers(0, len(known_dst)))]], 1)
    return pool[:max_pool]


def _build_hard_rows(
    shard_id: int,
    model: GraphFeatureModel,
    edges,
    weights,
    hot,
    recent_hot,
    low_pop,
    known_dst,
    max_pool: int,
    out_dir: str,
    seed: int,
    prefix: str = "hard",
) -> dict:
    rng = np.random.default_rng(int(seed) + int(shard_id) * 1009)
    features = np.zeros((len(edges), 100, len(FEATURE_NAMES)), dtype=np.float32)
    src_ids = np.zeros(len(edges), dtype=np.int64)
    dst_ids = np.zeros((len(edges), 100), dtype=np.int64)
    labels = np.zeros(len(edges), dtype=np.int64)
    meta_bad = 0
    for i, (src, pos_dst, time) in enumerate(edges):
        pool = _candidate_pool(model, src, pos_dst, rng, hot, recent_hot, low_pop, known_dst, max_pool)
        if len(pool) < 100:
            meta_bad += 1
            continue
        pool_feats = model.feature_row(src, time, pool)
        pool_score = score_feature_tensor(pool_feats.reshape(1, len(pool), len(FEATURE_NAMES)), weights)[0]
        order = np.argsort(-pool_score)
        negs = []
        for idx in order:
            cand = int(pool[int(idx)])
            if cand != int(pos_dst) and cand not in negs:
                negs.append(cand)
            if len(negs) >= 99:
                break
        while len(negs) < 99:
            cand = int(known_dst[int(rng.integers(0, len(known_dst)))])
            if cand != int(pos_dst) and cand not in negs:
                negs.append(cand)
        cands = [int(pos_dst)] + negs[:99]
        rng.shuffle(cands)
        label = cands.index(int(pos_dst))
        pool_pos = {candidate: pos for pos, candidate in enumerate(pool)}
        selected = [pool_pos[candidate] for candidate in cands]
        features[i] = pool_feats[np.asarray(selected, dtype=np.int64)]
        src_ids[i] = int(src)
        dst_ids[i] = np.asarray(cands, dtype=np.int64)
        labels[i] = int(label)
        if (i + 1) % 2000 == 0:
            print(f"hard_worker shard={shard_id} rows={i + 1}/{len(edges)}", flush=True)
    out_dir = ensure_dir(Path(out_dir))
    part = out_dir / f"{prefix}_part_{int(shard_id):03d}.npz"
    np.savez(part, features=features.astype(np.float16), src_ids=src_ids, dst_ids=dst_ids, labels=labels)
    return {"shard": int(shard_id), "rows": len(edges), "bad": meta_bad, "path": str(part)}


def _build_hard_worker(payload: tuple) -> dict:
    (
        shard_id,
        model_path,
        edges,
        weights,
        hot,
        recent_hot,
        low_pop,
        known_dst,
        max_pool,
        _rows_per_shard,
        out_dir,
        seed,
    ) = payload
    return _build_hard_rows(
        shard_id,
        GraphFeatureModel.load(Path(model_path)),
        edges,
        weights,
        hot,
        recent_hot,
        low_pop,
        known_dst,
        max_pool,
        out_dir,
        seed,
    )


def _build_hard_bounds_worker(payload: tuple) -> dict:
    shard_id, start, end, out_dir, prefix, seed = payload
    if _HARD_MODEL is None or _HARD_EDGES is None or _HARD_CONFIG is None:
        raise RuntimeError("fork worker did not inherit hard-negative context")
    weights, hot, recent_hot, low_pop, known_dst, max_pool = _HARD_CONFIG
    return _build_hard_rows(
        shard_id,
        _HARD_MODEL,
        _HARD_EDGES[start:end],
        weights,
        hot,
        recent_hot,
        low_pop,
        known_dst,
        max_pool,
        out_dir,
        seed,
        prefix,
    )


def build_hard_feature_parts(
    model: GraphFeatureModel,
    edges,
    weights,
    hot,
    recent_hot,
    low_pop,
    known_dst,
    max_pool: int,
    out_dir: Path,
    prefix: str,
    workers: int,
    seed: int,
) -> List[dict]:
    """Build candidate lists with a shared read-only graph model on Linux."""
    workers = max(1, min(int(workers), len(edges), max(1, (os.cpu_count() or 1) - 8)))
    if workers == 1 or os.name != "posix":
        return [_build_hard_rows(0, model, edges, weights, hot, recent_hot, low_pop, known_dst, max_pool, str(out_dir), seed, prefix)]
    global _HARD_MODEL, _HARD_EDGES, _HARD_CONFIG
    _HARD_MODEL = model
    _HARD_EDGES = edges
    _HARD_CONFIG = (weights, hot, recent_hot, low_pop, known_dst, max_pool)
    bounds = np.linspace(0, len(edges), workers + 1, dtype=np.int64)
    tasks = [
        (shard, int(bounds[shard]), int(bounds[shard + 1]), str(out_dir), prefix, int(seed))
        for shard in range(workers)
        if int(bounds[shard + 1]) > int(bounds[shard])
    ]
    # 优化：去掉maxtasksperchild=1，提高进程复用率
    with mp.get_context("fork").Pool(processes=len(tasks)) as pool:
        return pool.map(_build_hard_bounds_worker, tasks)


def _loss_with_hard_terms(logits, labels, ce_weight=1.0, margin_weight=0.0, bpr_weight=0.0, margin=0.5):
    import jittor as jt
    from jittor import nn

    loss = nn.cross_entropy_loss(logits, labels) * float(ce_weight)
    if float(margin_weight) > 0.0:
        candidate_count = int(logits.shape[1])
        positive_mask = (
            jt.arange(candidate_count).reshape((1, candidate_count))
            == labels.reshape((-1, 1))
        ).float32()
        positive = (logits * positive_mask).sum(dim=1)
        hardest_negative = (logits - positive_mask * 1e6).max(dim=1)
        hard_margin = nn.relu(hardest_negative - positive + float(margin)).mean()
        loss = loss + float(margin_weight) * hard_margin
    return loss


def _configure_jittor_backend(jt) -> str:
    backend = os.environ.get("JITTOR_BACKEND", "cuda").strip().lower()
    if backend == "acl":
        if not getattr(jt.compiler, "has_acl", False):
            raise RuntimeError("JITTOR_BACKEND=acl requested, but Jittor ACL is unavailable")
        jt.flags.use_acl = 1
    elif backend == "cuda":
        jt.flags.use_cuda = 1
    elif backend == "cpu":
        jt.flags.use_cuda = 0
    else:
        raise ValueError(f"unsupported JITTOR_BACKEND={backend!r}")
    return backend


def _train_feature_mlp(
    train_x,
    train_y,
    valid_x,
    valid_y,
    out_dir: Path,
    seed: int,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    margin_weight: float = 0.0,
    margin: float = 0.25,
    profile_calibration_weight: float = 0.0,
) -> dict:
    ensure_dir(out_dir)
    import jittor as jt
    from jittor import nn

    backend = _configure_jittor_backend(jt)
    try:
        jt.set_global_seed(seed)
    except Exception:
        pass
    np.random.seed(seed)

    class CandidateMLP(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, 1),
            )
            self.skip = nn.Linear(dim, 1)

        def execute(self, x, anchor):
            b, c, f = x.shape
            flat = x.reshape((b * c, f))
            residual = (self.net(flat) + 0.15 * self.skip(flat)).reshape((b, c))
            return anchor + CONTEXT_RESIDUAL_SCALE * residual

    if train_x.shape[-1] < 2:
        raise ValueError('anchored context MLP requires an anchor feature')
    train_core, valid_core = train_x[:, :, :-1], valid_x[:, :, :-1]
    train_anchor = train_x[:, :, -1].astype(np.float32)
    valid_anchor = valid_x[:, :, -1].astype(np.float32)
    mean = train_core.reshape(-1, train_core.shape[-1]).mean(axis=0, keepdims=True).astype(np.float32)
    std = train_core.reshape(-1, train_core.shape[-1]).std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    train_xn = ((train_core - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    valid_xn = ((valid_core - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    valid_profile = valid_core[:, :, FEATURE_NAMES.index("profile")].astype(np.float32)
    net = CandidateMLP(train_core.shape[-1], hidden)
    opt = nn.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    history = []
    ckpt = out_dir / "feature_mlp.pkl"
    best_mrr = -float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        order = np.arange(len(train_y))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            logits = net(jt.array(train_xn[idx]), jt.array(train_anchor[idx]))
            loss = _loss_with_hard_terms(
                logits,
                jt.array(train_y[idx]),
                margin_weight=float(margin_weight),
                margin=float(margin),
            )
            opt.step(loss)
            losses.append(float(loss.data))
        pred = _predict_feature_mlp_net(net, valid_xn, valid_anchor, batch_size=max(batch_size, 1024))
        selection_pred = pred
        if float(profile_calibration_weight) != 0.0:
            selection_pred = (
                row_zscore(pred)
                + float(profile_calibration_weight) * row_zscore(valid_profile)
            ).astype(np.float32)
        item = {"epoch": epoch, "loss": float(np.mean(losses)), "valid_mrr": tie_aware_mrr(selection_pred, valid_y)}
        history.append(item)
        if item["valid_mrr"] > best_mrr:
            best_mrr = float(item["valid_mrr"])
            best_epoch = int(epoch)
            net.save(str(ckpt))
        print(f"context_mlp seed={seed} epoch={epoch} loss={item['loss']:.6f} valid_mrr={item['valid_mrr']:.6f}", flush=True)
    norm = out_dir / "feature_norm.npz"
    np.savez(norm, mean=mean, std=std, feature_names=np.asarray(FEATURE_NAMES),
             residual_scale=np.asarray([CONTEXT_RESIDUAL_SCALE], dtype=np.float32))
    return {
        "status": "trained",
        "backend": backend,
        "seed": seed,
        "checkpoint": str(ckpt),
        "norm": str(norm),
        "architecture": "anchored_residual_mlp_3xhidden",
        "anchor": "last feature column",
        "residual_scale": CONTEXT_RESIDUAL_SCALE,
        "checkpoint_selection": "best_full_candidate_validation_mrr",
        "margin_weight": float(margin_weight),
        "margin": float(margin),
        "profile_calibration_weight": float(profile_calibration_weight),
        "best_epoch": best_epoch,
        "best_valid_mrr": best_mrr,
        "history": history,
    }


def _predict_feature_mlp_net(net, features: np.ndarray, anchor: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    import jittor as jt

    out = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        pred = net(jt.array(features[start:start + batch_size].astype(np.float32)),
                   jt.array(anchor[start:start + batch_size].astype(np.float32)))
        out[start:start + batch_size] = np.asarray(pred.data, dtype=np.float32)
    return out


def _load_feature_mlp_predictor(ckpt: Path, norm_path: Path, hidden: int):
    import jittor as jt
    from jittor import nn

    class CandidateMLP(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, 1),
            )
            self.skip = nn.Linear(dim, 1)

        def execute(self, x, anchor):
            b, c, f = x.shape
            flat = x.reshape((b * c, f))
            residual = (self.net(flat) + 0.15 * self.skip(flat)).reshape((b, c))
            return anchor + CONTEXT_RESIDUAL_SCALE * residual

    norm = np.load(norm_path)
    mean = norm["mean"].reshape(1, 1, -1)
    std = norm["std"].reshape(1, 1, -1)
    _configure_jittor_backend(jt)
    net = CandidateMLP(len(mean.reshape(-1)), hidden)
    net.load(str(ckpt))

    def predict(x: np.ndarray, batch_size=2048):
        if x.shape[-1] != mean.shape[-1] + 1:
            raise ValueError(f'anchored predictor expects {mean.shape[-1] + 1} features, got {x.shape[-1]}')
        anchor = x[:, :, -1].astype(np.float32)
        xn = ((x[:, :, :-1] - mean) / std).astype(np.float32)
        return _predict_feature_mlp_net(net, xn, anchor, batch_size=batch_size)

    return predict
