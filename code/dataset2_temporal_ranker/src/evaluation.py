from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .io_data import Edge, TestRow, row_zscore, sample_evenly, tie_aware_mrr
from .temporal_graph import FEATURE_NAMES, SEARCH_COMPONENTS, GraphFeatureModel
from .parallel_features import feature_tensor_parallel


@dataclass
class ValidationSet:
    name: str
    rows: List[TestRow]
    labels: np.ndarray
    weight: float
    meta: dict
    features: Optional[np.ndarray] = None
    normalized_features: Optional[np.ndarray] = None


def _history_stats(edges: Sequence[Edge]) -> Tuple[Counter, Counter, Counter, List[int], List[int]]:
    src_hist = defaultdict(list)
    dst_count = Counter()
    pair_count = Counter()
    for src, dst, time in edges:
        src_hist[src].append((time, dst))
        dst_count[dst] += 1
        pair_count[(src, dst)] += 1
    known_dst = list(dst_count.keys())
    hot_dst = [dst for dst, _ in dst_count.most_common(max(1000, min(50000, len(dst_count))))]
    return src_hist, dst_count, pair_count, known_dst, hot_dst


def _fill_candidates(
    src: int,
    pos_dst: int,
    src_hist: dict,
    dst_count: Counter,
    known_dst: Sequence[int],
    hot_dst: Sequence[int],
    rng: np.random.Generator,
    low_pop: bool = False,
) -> Tuple[Tuple[int, ...], int]:
    candidates = [pos_dst]
    seen = {pos_dst}

    hist = sorted(src_hist.get(src, ()), reverse=True)
    for _, dst in hist[:80]:
        if dst not in seen:
            candidates.append(dst)
            seen.add(dst)
        if len(candidates) >= 34:
            break

    hot_iter = list(hot_dst)
    if low_pop:
        hot_iter = list(reversed(hot_iter))
    for dst in hot_iter[:2000]:
        if dst not in seen:
            candidates.append(dst)
            seen.add(dst)
        if len(candidates) >= 70:
            break

    while len(candidates) < 100 and known_dst:
        dst = int(known_dst[int(rng.integers(0, len(known_dst)))])
        if dst not in seen:
            candidates.append(dst)
            seen.add(dst)

    while len(candidates) < 100:
        # This branch should be rare; keep IDs deterministic enough for debugging.
        dst = int(rng.integers(1, 2_000_000_000))
        if dst not in seen:
            candidates.append(dst)
            seen.add(dst)

    rng.shuffle(candidates)
    return tuple(candidates[:100]), candidates.index(pos_dst)


def make_hard_negative_validation(
    valid_edges: Sequence[Edge],
    history_edges: Sequence[Edge],
    max_events: int,
    seed: int,
    name: str = "hard_negative",
    weight: float = 1.0,
) -> ValidationSet:
    rng = np.random.default_rng(seed)
    src_hist, dst_count, _pair_count, known_dst, hot_dst = _history_stats(history_edges)
    indices = sample_evenly(len(valid_edges), max_events)
    rows: List[TestRow] = []
    labels: List[int] = []
    for idx in indices:
        src, dst, time = valid_edges[int(idx)]
        candidates, label = _fill_candidates(src, dst, src_hist, dst_count, known_dst, hot_dst, rng)
        rows.append(TestRow(src, time, candidates))
        labels.append(label)
    return ValidationSet(
        name=name,
        rows=rows,
        labels=np.asarray(labels, dtype=np.int64),
        weight=float(weight),
        meta={"events": len(rows), "kind": "src_history_hot_random_negatives"},
    )


def make_test_injection_validation(
    valid_edges: Sequence[Edge],
    test_rows: Sequence[TestRow],
    history_edges: Sequence[Edge],
    max_events: int,
    seed: int,
    name: str = "test_candidate_injection",
    weight: float = 1.0,
    low_pop_only: bool = False,
) -> ValidationSet:
    rng = np.random.default_rng(seed)
    by_src = defaultdict(list)
    for row in test_rows:
        by_src[row.src].append(row)
    _src_hist, dst_count, _pair_count, known_dst, _hot_dst = _history_stats(history_edges)
    if low_pop_only and valid_edges:
        counts = np.asarray([dst_count.get(dst, 0) for _, dst, _ in valid_edges], dtype=np.float64)
        threshold = float(np.percentile(counts, 35))
        pool_edges = [edge for edge in valid_edges if dst_count.get(edge[1], 0) <= threshold]
    else:
        threshold = None
        pool_edges = list(valid_edges)

    indices = sample_evenly(len(pool_edges), max_events)
    rows: List[TestRow] = []
    labels: List[int] = []
    src_template_hits = 0
    replaced_known = 0
    for k, idx in enumerate(indices):
        src, dst, time = pool_edges[int(idx)]
        templates = by_src.get(src)
        if templates:
            template = templates[k % len(templates)]
            src_template_hits += 1
        else:
            template = test_rows[int(rng.integers(0, len(test_rows)))]
        candidates = list(template.candidates)
        if dst in candidates:
            label = candidates.index(dst)
        else:
            known_positions = [i for i, cand in enumerate(candidates) if cand in known_dst and cand != dst]
            if known_positions:
                pos = int(known_positions[int(rng.integers(0, len(known_positions)))])
                replaced_known += 1
            else:
                pos = int(rng.integers(0, len(candidates)))
            candidates[pos] = dst
            label = pos
        rows.append(TestRow(src, time, tuple(candidates)))
        labels.append(label)
    return ValidationSet(
        name=name,
        rows=rows,
        labels=np.asarray(labels, dtype=np.int64),
        weight=float(weight),
        meta={
            "events": len(rows),
            "kind": "valid_positive_inserted_into_real_test_candidate_rows",
            "same_src_template_frac": src_template_hits / max(len(rows), 1),
            "replaced_known_frac": replaced_known / max(len(rows), 1),
            "low_pop_threshold": threshold,
        },
    )


def make_teacher_pseudo_validation(
    test_rows: Sequence[TestRow],
    teacher_scores: np.ndarray,
    max_events: int,
    seed: int,
    name: str = "teacher_top1_pseudo",
    weight: float = 0.15,
) -> ValidationSet:
    rng = np.random.default_rng(seed)
    if len(test_rows) != teacher_scores.shape[0]:
        raise ValueError("teacher scores and test rows have different row counts")
    entropy = -np.sum(np.clip(teacher_scores, 1e-12, 1.0) * np.log(np.clip(teacher_scores, 1e-12, 1.0)), axis=1)
    confident = np.argsort(entropy)[: min(len(test_rows), max(int(max_events) * 3, int(max_events)))]
    if len(confident) > max_events:
        confident = rng.choice(confident, size=int(max_events), replace=False)
    confident = np.sort(confident)
    rows = [test_rows[int(i)] for i in confident]
    labels = np.argmax(teacher_scores[confident], axis=1).astype(np.int64)
    return ValidationSet(
        name=name,
        rows=rows,
        labels=labels,
        weight=float(weight),
        meta={
            "events": len(rows),
            "kind": "teacher_top1_stability_not_ground_truth",
            "entropy_mean": float(np.mean(entropy[confident])) if len(confident) else None,
        },
    )


def build_validation_sets(
    dataset: str,
    train_edges: Sequence[Edge],
    valid_edges: Sequence[Edge],
    test_rows: Sequence[TestRow],
    max_events: int,
    seed: int,
    teacher_scores: Optional[np.ndarray] = None,
) -> List[ValidationSet]:
    sets: List[ValidationSet] = []
    if not valid_edges:
        return sets
    if dataset == "dataset2":
        per = max(1000, int(max_events))
        sets.append(make_hard_negative_validation(valid_edges, train_edges, per, seed + 11, "official_hard", 0.35))
        sets.append(make_test_injection_validation(valid_edges, test_rows, train_edges, per, seed + 23, "test_injection", 1.00))
        sets.append(make_test_injection_validation(valid_edges, test_rows, train_edges, per, seed + 37, "lowpop_injection", 0.85, low_pop_only=True))
        if teacher_scores is not None:
            sets.append(make_teacher_pseudo_validation(test_rows, teacher_scores, min(per, 20000), seed + 41, weight=0.15))
    else:
        per = max(1000, int(max_events))
        sets.append(make_hard_negative_validation(valid_edges, train_edges, per, seed + 13, "time_tail_hard", 0.70))
        sets.append(make_test_injection_validation(valid_edges, test_rows, train_edges, per, seed + 29, "test_injection", 0.85))
        if teacher_scores is not None:
            sets.append(make_teacher_pseudo_validation(test_rows, teacher_scores, min(per, 15000), seed + 43, weight=0.10))
    return sets


def normalize_feature_tensor(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return ((values - mean) / np.where(std < 1e-6, 1.0, std)).astype(np.float32)


def score_feature_tensor(features: np.ndarray, weights: Dict[str, float], normalized: Optional[np.ndarray] = None) -> np.ndarray:
    zfeatures = normalize_feature_tensor(features) if normalized is None else normalized
    coeff = np.zeros(zfeatures.shape[-1], dtype=np.float32)
    for name, weight in weights.items():
        if abs(float(weight)) >= 1e-12:
            coeff[FEATURE_NAMES.index(name)] = float(weight)
    return np.tensordot(zfeatures, coeff, axes=([2], [0])).astype(np.float32)


def evaluate_weights_on_set(vset: ValidationSet, weights: Dict[str, float]) -> Tuple[float, np.ndarray]:
    if vset.features is None:
        raise ValueError(f"validation set {vset.name} has no features")
    scores = score_feature_tensor(vset.features, weights, normalized=vset.normalized_features)
    return tie_aware_mrr(scores, vset.labels), scores


def aggregate_mrr(vsets: Sequence[ValidationSet], weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    total_weight = 0.0
    total = 0.0
    detail: Dict[str, float] = {}
    for vset in vsets:
        mrr, _ = evaluate_weights_on_set(vset, weights)
        detail[vset.name] = mrr
        total += float(vset.weight) * mrr
        total_weight += float(vset.weight)
    return total / max(total_weight, 1e-12), detail


def attach_features(
    model: GraphFeatureModel,
    vsets: Sequence[ValidationSet],
    workers: int = 1,
    cache_dir: Optional[Path] = None,
) -> None:
    for vset in vsets:
        print(f"building validation features name={vset.name} rows={len(vset.rows)}", flush=True)
        if int(workers) > 1 and cache_dir is not None:
            vset.features = feature_tensor_parallel(model, vset.rows, cache_dir, f"valid_{vset.name}", int(workers))
        else:
            vset.features = model.feature_tensor(vset.rows)
        vset.normalized_features = normalize_feature_tensor(vset.features)


def evaluate_components(vsets: Sequence[ValidationSet]) -> Dict[str, dict]:
    report = {}
    for name in FEATURE_NAMES:
        weights = {name: 1.0}
        aggregate, detail = aggregate_mrr(vsets, weights)
        report[name] = {"aggregate_mrr": aggregate, "by_set": detail}
    return report


def search_weights_multi(vsets: Sequence[ValidationSet], rounds: int = 4) -> Tuple[Dict[str, float], List[dict]]:
    weights: Dict[str, float] = {"rule": 1.0}
    for vset in vsets:
        if vset.features is None:
            raise ValueError(f"validation set {vset.name} has no features")
        if vset.normalized_features is None:
            vset.normalized_features = normalize_feature_tensor(vset.features)
    current_scores = [vset.normalized_features[:, :, FEATURE_NAMES.index("rule")].copy() for vset in vsets]

    def aggregate_scores(scores_list: Sequence[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        total_weight = 0.0
        total = 0.0
        detail: Dict[str, float] = {}
        for vset, scores in zip(vsets, scores_list):
            mrr = tie_aware_mrr(scores, vset.labels)
            detail[vset.name] = mrr
            total += float(vset.weight) * mrr
            total_weight += float(vset.weight)
        return total / max(total_weight, 1e-12), detail

    best, detail = aggregate_scores(current_scores)
    history = [{"aggregate_mrr": best, "by_set": detail, "weights": dict(weights)}]
    grid = [-2.0, -1.25, -0.75, -0.35, -0.15, 0.0, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.55, 0.85, 1.25, 1.75, 2.5]
    for round_idx in range(1, int(rounds) + 1):
        improved = False
        for component in SEARCH_COMPONENTS:
            current = weights.get(component, 0.0)
            local_best = best
            local_value = current
            local_detail = detail
            local_scores = current_scores
            component_idx = FEATURE_NAMES.index(component)
            for value in grid:
                delta = float(value) - float(current)
                trial_scores = [scores + delta * vset.normalized_features[:, :, component_idx] for scores, vset in zip(current_scores, vsets)]
                score, score_detail = aggregate_scores(trial_scores)
                if score > local_best + 1e-8:
                    local_best = score
                    local_value = float(value)
                    local_detail = score_detail
                    local_scores = trial_scores
            if local_best > best + 1e-8:
                if abs(local_value) < 1e-12:
                    weights.pop(component, None)
                else:
                    weights[component] = local_value
                best = local_best
                detail = local_detail
                current_scores = local_scores
                improved = True
                item = {
                    "round": round_idx,
                    "component": component,
                    "aggregate_mrr": best,
                    "by_set": detail,
                    "weights": dict(weights),
                }
                history.append(item)
                print(
                    f"weight_search round={round_idx} component={component} aggregate_mrr={best:.6f} value={local_value}",
                    flush=True,
                )
        if not improved:
            break
    return weights, history


def top1_stats(scores: np.ndarray, rows: Sequence[TestRow], model: GraphFeatureModel) -> dict:
    top_idx = np.argmax(scores, axis=1)
    top_dst = [rows[i].candidates[int(top_idx[i])] for i in range(len(rows))]
    counts = np.asarray([model.dst_count.get(dst, 0) for dst in top_dst], dtype=np.float64)
    known = counts > 0
    return {
        "rows": len(rows),
        "top1_unique_dst": int(len(set(top_dst))),
        "top1_known_frac": float(np.mean(known)) if len(known) else 0.0,
        "top1_count_median": float(np.median(counts)) if len(counts) else 0.0,
        "top1_count_mean": float(np.mean(counts)) if len(counts) else 0.0,
        "top1_count_p95": float(np.percentile(counts, 95)) if len(counts) else 0.0,
    }
