#!/usr/bin/env python3
"""Leakage-controlled dataset4 evaluation for the Temporal Ranker.

Candidate construction follows honest_eval_kit/eval_testlike.py: each row has
one positive, 77 popularity-weighted split=0 destinations, and 22 destinations
never observed in split=0. The graph and all ranker features are built from the
official split=0 only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _topk_without_replacement(keys, k):
    pool = keys.shape[1]
    return np.argpartition(keys, pool - k, axis=1)[:, pool - k:]


def construct_candidates(positives, seen_ids, seen_weights, unseen_ids,
                         num_seen, num_unseen, rng, chunk_rows=512):
    """Exact sampling algorithm from the supplied honest evaluator."""
    n = len(positives)
    chunk_seen = max(1, min(chunk_rows, (1 << 28) // len(seen_ids)))
    chunk_unseen = max(1, min(chunk_rows, (1 << 28) // len(unseen_ids)))
    seen_index = {int(v): i for i, v in enumerate(seen_ids)}
    unseen_index = {int(v): i for i, v in enumerate(unseen_ids)}
    pos_seen_idx = np.array([seen_index.get(int(p), -1) for p in positives], dtype=np.int64)
    pos_unseen_idx = np.array([unseen_index.get(int(p), -1) for p in positives], dtype=np.int64)

    weights = seen_weights.astype(np.float32)
    seen_neg = np.empty((n, num_seen), dtype=np.int64)
    for start in range(0, n, chunk_seen):
        stop = min(start + chunk_seen, n)
        u = np.maximum(rng.random((stop - start, len(seen_ids)), dtype=np.float32), np.float32(1e-37))
        keys = np.log(u) / weights[None, :]
        rows = pos_seen_idx[start:stop]
        hit = rows >= 0
        keys[np.nonzero(hit)[0], rows[hit]] = -np.inf
        seen_neg[start:stop] = seen_ids[_topk_without_replacement(keys, num_seen)]

    unseen_neg = np.empty((n, num_unseen), dtype=np.int64)
    for start in range(0, n, chunk_unseen):
        stop = min(start + chunk_unseen, n)
        keys = rng.random((stop - start, len(unseen_ids)), dtype=np.float32)
        rows = pos_unseen_idx[start:stop]
        hit = rows >= 0
        keys[np.nonzero(hit)[0], rows[hit]] = -np.inf
        unseen_neg[start:stop] = unseen_ids[_topk_without_replacement(keys, num_unseen)]

    candidates = np.concatenate([positives[:, None], seen_neg, unseen_neg], axis=1)
    order = np.argsort(rng.random(candidates.shape), axis=1)
    candidates = np.take_along_axis(candidates, order, axis=1)
    labels = (candidates == positives[:, None]).argmax(axis=1).astype(np.int64)
    if (np.diff(np.sort(candidates, axis=1), axis=1) == 0).any():
        raise RuntimeError("candidate construction produced duplicates")
    return candidates, labels


def metrics(scores, labels):
    if not np.isfinite(scores).all():
        raise FloatingPointError("honest evaluation received NaN or Inf scores")
    pos = scores[np.arange(len(labels)), labels]
    ranks = 1 + (scores > pos[:, None]).sum(axis=1)
    rr = 1.0 / ranks
    return ranks, {
        "mrr": float(rr.mean()),
        "hit_at_1": float((ranks <= 1).mean()),
        "hit_at_10": float((ranks <= 10).mean()),
        "median_rank": float(np.median(ranks)),
    }


def grouped(ranks, mask):
    if not np.any(mask):
        return {"rows": 0, "mrr": None, "hit_at_10": None}
    selected = ranks[mask]
    return {
        "rows": int(len(selected)),
        "mrr": float((1.0 / selected).mean()),
        "hit_at_10": float((selected <= 10).mean()),
    }


def evaluate_score_set(scores, labels, pair_seen):
    ranks, overall = metrics(scores, labels)
    return {
        "overall": overall,
        "pair_seen": grouped(ranks, pair_seen),
        "pair_unseen": grouped(ranks, ~pair_seen),
    }


def grouped_by_quantile(ranks, values, bins=5):
    values = np.asarray(values, dtype=np.float64)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    groups = []
    for index in range(max(0, len(edges) - 1)):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (values >= lower) & (values <= upper if index == len(edges) - 2 else values < upper)
        item = grouped(ranks, mask)
        item.update({"lower": lower, "upper": upper})
        groups.append(item)
    return groups


def row_zscore(values):
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm", required=True)
    parser.add_argument("--anchor-weights", required=True,
                        help="split=0-only anchor weights produced by honest training")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=1024)
    parser.add_argument("--src-seq-len", type=int, default=64)
    parser.add_argument("--dst-seq-len", type=int, default=64)
    parser.add_argument("--residual-weights", default="0,1.0",
                        help="pre-registered residual fractions; 0 is a diagnostic baseline")
    parser.add_argument("--primary-residual-weight", type=float, default=1.0,
                        help="pre-registered primary model; never selected from evaluation results")
    parser.add_argument("--analysis-output", default="")
    parser.add_argument("--dump-output", default="",
                        help="optional deterministic full-candidate diagnostic NPZ")
    parser.add_argument("--profile-calibration-weights", default="",
                        help="pre-registered row-zscore profile residual weights")
    parser.add_argument("--primary-profile-weight", type=float, default=None)
    args = parser.parse_args()
    started = time.time()

    sys.path.insert(0, str(Path(args.code_root).resolve()))
    from src.candidate_ranker import _load_feature_mlp_predictor, score_linear_anchor
    from src.context_stage import CONTEXT_FEATURE_NAMES, SequenceContext, _append_sequence_features
    from src.io_data import TestRow
    from src.temporal_graph import GraphFeatureModel

    train_path = Path(args.data_dir) / "dataset2" / "train.csv"
    test_path = Path(args.data_dir) / "dataset2" / "test.csv"
    frame = pd.read_csv(train_path, usecols=["src", "dst", "time", "split"])
    split0 = frame[frame["split"] == 0].sort_values(["time", "src", "dst"], kind="mergesort")
    split1 = frame[frame["split"] == 1].sort_values("time", kind="mergesort")
    take = np.linspace(0, len(split1) - 1, min(args.num_rows, len(split1)), dtype=np.int64)
    validation = split1.iloc[take].reset_index(drop=True)

    model = GraphFeatureModel.load(Path(args.graph))
    split0_min = int(split0["time"].min())
    split0_max = int(split0["time"].max())
    split1_min = int(validation["time"].min())
    split1_max = int(validation["time"].max())
    if split0_max >= split1_min:
        raise RuntimeError(
            "official split ranges overlap; the cached split=0 graph is not strictly time-safe "
            f"(split0_max={split0_max}, validation_min={split1_min})"
        )
    graph_edges = int(sum(model.src_count.values()))
    graph_pairs = int(sum(model.pair_count.values()))
    if graph_edges != len(split0) or graph_pairs != len(split0):
        raise RuntimeError(
            "evaluation graph is not the official split=0 graph: "
            f"graph_edges={graph_edges}, graph_pairs={graph_pairs}, split0_edges={len(split0)}"
        )
    if int(model.time_max) != split0_max or int(model.time_min) != split0_min:
        raise RuntimeError(
            "evaluation graph time range does not match official split=0: "
            f"graph=[{int(model.time_min)}, {int(model.time_max)}], "
            f"split0=[{split0_min}, {split0_max}]"
        )

    seen_ids = np.asarray(sorted(model.dst_count), dtype=np.int64)
    seen_weights = np.asarray([model.dst_count[int(dst)] for dst in seen_ids], dtype=np.float64)
    space_max = max(max(model.test_candidate_count, default=0), int(frame["dst"].max()))
    unseen_ids = np.setdiff1d(np.arange(1, space_max + 1, dtype=np.int64), seen_ids)
    positives = validation["dst"].to_numpy(np.int64)
    candidates, labels = construct_candidates(
        positives, seen_ids, seen_weights, unseen_ids, 77, 22, np.random.default_rng(args.seed)
    )
    srcs = validation["src"].to_numpy(np.int64)
    times = validation["time"].to_numpy(np.int64)
    rows = [TestRow(int(src), int(ts), tuple(map(int, cands)))
            for src, ts, cands in zip(srcs, times, candidates)]

    print(f"[honest-temporal] rows={len(rows)} candidates=1+77+22 graph={args.graph}", flush=True)
    base = model.feature_tensor(rows, progress_every=1000)
    edges0 = list(split0[["src", "dst", "time"]].itertuples(index=False, name=None))
    context = SequenceContext(model, edges0, dst_seq_len=args.dst_seq_len)
    features = _append_sequence_features(base, srcs, candidates, context, src_seq_len=args.src_seq_len)
    anchor_weights = np.load(Path(args.anchor_weights))["weights"].astype(np.float32)
    anchor = score_linear_anchor(features, anchor_weights)
    features = np.concatenate([features, anchor[:, :, None]], axis=2)
    predictor = _load_feature_mlp_predictor(Path(args.checkpoint), Path(args.norm), hidden=args.hidden)
    scores = predictor(features, batch_size=args.predict_batch_size)

    pair_seen = np.fromiter(
        (model.pair_count.get((int(src), int(dst)), 0) > 0 for src, dst in zip(srcs, positives)),
        dtype=bool,
        count=len(srcs),
    )
    if args.dump_output:
        dump_path = Path(args.dump_output)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dump_path,
            candidates=candidates.astype(np.int32),
            labels=labels.astype(np.int16),
            srcs=srcs.astype(np.int64),
            times=times.astype(np.int64),
            pair_seen=pair_seen,
            scores=scores.astype(np.float32),
            anchor=anchor.astype(np.float32),
            features=features[:, :, :-1].astype(np.float16),
            feature_names=np.asarray(CONTEXT_FEATURE_NAMES),
            seed=np.asarray([args.seed], dtype=np.int64),
        )
        print(f"[honest-temporal] diagnostic dump={dump_path}", flush=True)
    residual_weights = [float(x) for x in args.residual_weights.split(",") if x.strip()]
    score_sets = {}
    for weight in residual_weights:
        blended = anchor + weight * (scores - anchor)
        score_sets[f"residual_weight_{weight:g}"] = evaluate_score_set(blended, labels, pair_seen)
    profile_weights = [float(x) for x in args.profile_calibration_weights.split(",") if x.strip()]
    if profile_weights:
        profile = features[:, :, CONTEXT_FEATURE_NAMES.index("profile")]
        model_z = row_zscore(scores)
        profile_z = row_zscore(profile)
        for weight in profile_weights:
            calibrated = model_z + weight * profile_z
            score_sets[f"profile_calibration_{weight:g}"] = evaluate_score_set(calibrated, labels, pair_seen)
    if args.primary_profile_weight is None:
        primary_name = f"residual_weight_{args.primary_residual_weight:g}"
        primary_scores = anchor + args.primary_residual_weight * (scores - anchor)
    else:
        primary_name = f"profile_calibration_{args.primary_profile_weight:g}"
        profile = features[:, :, CONTEXT_FEATURE_NAMES.index("profile")]
        primary_scores = row_zscore(scores) + args.primary_profile_weight * row_zscore(profile)
    if primary_name not in score_sets:
        raise ValueError(f"primary score set {primary_name} was not requested")
    primary = score_sets[primary_name]
    primary_ranks, _ = metrics(primary_scores, labels)
    analysis = None
    if args.analysis_output:
        component_metrics = {}
        for index, name in enumerate(CONTEXT_FEATURE_NAMES):
            component_metrics[name] = evaluate_score_set(features[:, :, index], labels, pair_seen)
        positive_known = np.fromiter(
            (model.dst_count.get(int(dst), 0) > 0 for dst in positives), dtype=bool, count=len(positives)
        )
        source_counts = np.fromiter(
            (model.src_count.get(int(src), 0) for src in srcs), dtype=np.float64, count=len(srcs)
        )
        positive_pop = np.fromiter(
            (model.dst_count.get(int(dst), 0) for dst in positives), dtype=np.float64, count=len(positives)
        )
        candidate_known_counts = np.fromiter(
            (sum(model.dst_count.get(int(dst), 0) > 0 for dst in row) for row in candidates),
            dtype=np.int64,
            count=len(candidates),
        )
        analysis = {
            "positive_dst_known": grouped(primary_ranks, positive_known),
            "positive_dst_unseen": grouped(primary_ranks, ~positive_known),
            "source_activity_quantiles": grouped_by_quantile(primary_ranks, source_counts),
            "positive_popularity_quantiles": grouped_by_quantile(primary_ranks, positive_pop),
            "time_quantiles": grouped_by_quantile(primary_ranks, times),
            "candidate_known_count": {
                "min": int(candidate_known_counts.min()),
                "max": int(candidate_known_counts.max()),
                "mean": float(candidate_known_counts.mean()),
                "histogram": {str(int(v)): int((candidate_known_counts == v).sum()) for v in np.unique(candidate_known_counts)},
            },
            "component_metrics": component_metrics,
        }
        analysis_path = Path(args.analysis_output)
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    result = {
        "dataset": "dataset4",
        "model": "temporal_ranker",
        "rows": int(len(rows)),
        "candidate_recipe": {"positive": 1, "seen": 77, "unseen": 22},
        "seed": int(args.seed),
        "strict_time_check": {
            "split0_min": split0_min,
            "split0_max": split0_max,
            "validation_min": split1_min,
            "validation_max": split1_max,
            "passed": True,
            "graph_edges": graph_edges,
            "graph_pairs": graph_pairs,
            "graph_time_min": int(model.time_min),
            "graph_time_max": int(model.time_max),
        },
        "primary_score_set": primary_name,
        "selection_policy": "pre_registered_before_evaluation",
        "overall": primary["overall"],
        "pair_seen": primary["pair_seen"],
        "pair_unseen": primary["pair_unseen"],
        "score_sets": score_sets,
        "analysis_output": str(Path(args.analysis_output).resolve()) if args.analysis_output else None,
        "dump_output": str(Path(args.dump_output).resolve()) if args.dump_output else None,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
