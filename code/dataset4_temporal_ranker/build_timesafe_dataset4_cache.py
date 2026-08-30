#!/usr/bin/env python3
"""Build leakage-controlled dataset4 train/validation caches for Temporal Ranker v2."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _topk_without_replacement(keys, k):
    pool = keys.shape[1]
    return np.argpartition(keys, pool - k, axis=1)[:, pool - k:]


def construct_honest_candidates(positives, seen_ids, seen_weights, unseen_ids, rng, chunk_rows=512):
    """Match honest_eval_kit: 1 positive + 77 weighted seen + 22 uniform unseen."""
    if len(seen_ids) < 78:
        raise ValueError(f"honest evaluation needs at least 78 seen destinations, got {len(seen_ids)}")
    if len(unseen_ids) < 23:
        raise ValueError(f"honest evaluation needs at least 23 unseen destinations, got {len(unseen_ids)}")
    n = len(positives)
    chunk_seen = max(1, min(chunk_rows, (1 << 28) // len(seen_ids)))
    chunk_unseen = max(1, min(chunk_rows, (1 << 28) // len(unseen_ids)))
    seen_index = {int(value): index for index, value in enumerate(seen_ids)}
    unseen_index = {int(value): index for index, value in enumerate(unseen_ids)}
    pos_seen_idx = np.asarray([seen_index.get(int(value), -1) for value in positives], dtype=np.int64)
    pos_unseen_idx = np.asarray([unseen_index.get(int(value), -1) for value in positives], dtype=np.int64)

    weights = seen_weights.astype(np.float32)
    seen_neg = np.empty((n, 77), dtype=np.int64)
    for start in range(0, n, chunk_seen):
        stop = min(start + chunk_seen, n)
        uniform = np.maximum(
            rng.random((stop - start, len(seen_ids)), dtype=np.float32),
            np.float32(1e-37),
        )
        keys = np.log(uniform) / weights[None, :]
        rows = pos_seen_idx[start:stop]
        hit = rows >= 0
        keys[np.nonzero(hit)[0], rows[hit]] = -np.inf
        seen_neg[start:stop] = seen_ids[_topk_without_replacement(keys, 77)]

    unseen_neg = np.empty((n, 22), dtype=np.int64)
    for start in range(0, n, chunk_unseen):
        stop = min(start + chunk_unseen, n)
        keys = rng.random((stop - start, len(unseen_ids)), dtype=np.float32)
        rows = pos_unseen_idx[start:stop]
        hit = rows >= 0
        keys[np.nonzero(hit)[0], rows[hit]] = -np.inf
        unseen_neg[start:stop] = unseen_ids[_topk_without_replacement(keys, 22)]

    candidates = np.concatenate([positives[:, None], seen_neg, unseen_neg], axis=1)
    order = np.argsort(rng.random(candidates.shape), axis=1)
    candidates = np.take_along_axis(candidates, order, axis=1)
    labels = (candidates == positives[:, None]).argmax(axis=1).astype(np.int64)
    if (np.diff(np.sort(candidates, axis=1), axis=1) == 0).any():
        raise RuntimeError("honest candidate construction produced duplicates")
    return candidates, labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--train-rows", type=int, default=160000)
    parser.add_argument("--valid-rows", type=int, default=30000)
    parser.add_argument("--history-frac", type=float, default=0.70)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--cos-hard-count", type=int, default=24)
    parser.add_argument("--cos-pool-size", type=int, default=320)
    parser.add_argument("--pair-decoy-count", type=int, default=6)
    parser.add_argument("--final-eval-rows", type=int, default=5000)
    parser.add_argument("--final-eval-seed", type=int, default=42)
    return parser.parse_args()


def weighted_candidate_rows(model, edges, count, rng, hard_count, hard_pool, decoy_count):
    from src.io_data import TestRow

    seen_ids = np.asarray(sorted(model.dst_count), dtype=np.int64)
    seen_set = set(int(x) for x in seen_ids)
    eligible = [edge for edge in edges if int(edge[1]) in seen_set]
    if len(eligible) < count:
        count = len(eligible)
    chosen = rng.permutation(len(eligible))[:count]
    selected_edges = [eligible[int(index)] for index in chosen]
    srcs = np.asarray([edge[0] for edge in selected_edges], dtype=np.int64)
    positives = np.asarray([edge[1] for edge in selected_edges], dtype=np.int64)
    times = np.asarray([edge[2] for edge in selected_edges], dtype=np.int64)

    max_id = max(
        max(model.test_candidate_count, default=0),
        int(seen_ids.max()),
        int(getattr(model, "candidate_space_max", 0)),
    )
    unseen_ids = np.setdiff1d(np.arange(1, max_id + 1, dtype=np.int64), seen_ids)
    seen_weights = np.asarray([model.dst_count[int(dst)] for dst in seen_ids], dtype=np.float64)
    seen_prob = seen_weights / seen_weights.sum()
    candidates = np.empty((count, 100), dtype=np.int64)

    draw_count = max(192, int(hard_pool), 77 + int(decoy_count) + int(hard_count) + 32)
    for start in range(0, count, 1024):
        stop = min(start + 1024, count)
        draws = rng.choice(seen_ids, size=(stop - start, draw_count), replace=True, p=seen_prob)
        cold_draws = rng.choice(unseen_ids, size=(stop - start, 32), replace=True)
        for offset in range(stop - start):
            row_index = start + offset
            src = int(srcs[row_index])
            positive = int(positives[row_index])
            used = {positive}
            decoys = []
            if decoy_count:
                for dst in reversed(model.src_recent.get(src, ())):
                    dst = int(dst)
                    if dst not in used:
                        used.add(dst)
                        decoys.append(dst)
                    if len(decoys) >= int(decoy_count):
                        break

            pool = []
            for dst in draws[offset]:
                dst = int(dst)
                if dst not in used:
                    used.add(dst)
                    pool.append(dst)

            hard_values = []
            if hard_count and pool:
                short = model._profile_variant_scores(src, pool, model.src_profile_short)
                long = model._profile_variant_scores(src, pool, model.src_profile_long)
                similarity = 0.65 * short + 0.35 * long
                order = np.argsort(-similarity, kind="stable")[:int(hard_count)]
                hard_values = [pool[int(index)] for index in order]
            hard_set = set(hard_values)
            random_values = [dst for dst in pool if dst not in hard_set]
            seen = decoys + hard_values
            seen.extend(random_values[:77 - len(seen)])
            while len(seen) < 77:
                dst = int(rng.choice(seen_ids, p=seen_prob))
                if dst not in used:
                    used.add(dst)
                    seen.append(dst)

            cold = []
            cold_used = set()
            for dst in cold_draws[offset]:
                dst = int(dst)
                if dst not in cold_used and dst != positive:
                    cold_used.add(dst)
                    cold.append(dst)
                if len(cold) == 22:
                    break
            while len(cold) < 22:
                dst = int(rng.choice(unseen_ids))
                if dst not in cold_used and dst != positive:
                    cold_used.add(dst)
                    cold.append(dst)

            row = np.asarray([positive] + seen[:77] + cold[:22], dtype=np.int64)
            row = row[rng.permutation(100)]
            candidates[row_index] = row
        print(f"sampled rows={stop}/{count} hard={hard_count} decoys={decoy_count}", flush=True)

    labels = (candidates == positives[:, None]).argmax(axis=1).astype(np.int64)
    rows = [
        TestRow(int(src), int(time), tuple(map(int, row)))
        for src, time, row in zip(srcs, times, candidates)
    ]
    return rows, srcs, candidates, labels, selected_edges


def save_cache(name, model, context_edges, edges, count, out_root, workers, rng, hard_count, hard_pool, decoy_count):
    from src.context_stage import SequenceContext, append_sequence_features_parallel
    from src.parallel_features import feature_tensor_parallel

    rows, srcs, candidates, labels, selected_edges = weighted_candidate_rows(
        model, edges, count, rng, hard_count, hard_pool, decoy_count
    )
    scratch = out_root / "scratch"
    base = feature_tensor_parallel(
        model, rows, scratch / f"base_{name}", f"base_{name}", int(workers)
    )
    context = SequenceContext(model, context_edges, dst_seq_len=64)
    features = append_sequence_features_parallel(
        base,
        srcs,
        candidates,
        context,
        64,
        scratch / f"context_{name}",
        name,
        int(workers),
    ).astype(np.float16)
    path = scratch / f"ranker_{name}.npz"
    np.savez(path, features=features, src_ids=srcs, dst_ids=candidates, labels=labels)
    pair_seen = np.fromiter(
        (model.pair_count.get((int(src), int(dst)), 0) > 0 for src, dst in zip(srcs, [e[1] for e in selected_edges])),
        dtype=bool,
        count=len(selected_edges),
    )
    return {
        "rows": int(len(rows)),
        "path": str(path),
        "pair_seen_fraction": float(pair_seen.mean()),
        "hard_count": int(hard_count),
        "pair_decoy_count": int(decoy_count),
    }


def save_honest_validation_cache(model, context_edges, edges, count, out_root, workers, rng):
    from src.context_stage import SequenceContext, append_sequence_features_parallel
    from src.io_data import TestRow
    from src.parallel_features import feature_tensor_parallel

    count = min(int(count), len(edges))
    chosen = rng.permutation(len(edges))[:count]
    selected = [edges[int(index)] for index in chosen]
    srcs = np.asarray([edge[0] for edge in selected], dtype=np.int64)
    positives = np.asarray([edge[1] for edge in selected], dtype=np.int64)
    times = np.asarray([edge[2] for edge in selected], dtype=np.int64)
    seen_ids = np.asarray(sorted(model.dst_count), dtype=np.int64)
    seen_weights = np.asarray([model.dst_count[int(dst)] for dst in seen_ids], dtype=np.float64)
    space_max = max(
        max(model.test_candidate_count, default=0),
        int(getattr(model, "candidate_space_max", 0)),
    )
    unseen_ids = np.setdiff1d(np.arange(1, space_max + 1, dtype=np.int64), seen_ids)
    candidates, labels = construct_honest_candidates(
        positives, seen_ids, seen_weights, unseen_ids, rng
    )
    rows = [
        TestRow(int(src), int(time), tuple(map(int, row)))
        for src, time, row in zip(srcs, times, candidates)
    ]
    scratch = out_root / "scratch"
    base = feature_tensor_parallel(
        model, rows, scratch / "base_valid", "base_valid", int(workers)
    )
    context = SequenceContext(model, context_edges, dst_seq_len=64)
    features = append_sequence_features_parallel(
        base,
        srcs,
        candidates,
        context,
        64,
        scratch / "context_valid",
        "valid",
        int(workers),
    ).astype(np.float16)
    path = scratch / "ranker_valid.npz"
    np.savez(path, features=features, src_ids=srcs, dst_ids=candidates, labels=labels)
    pair_seen = np.fromiter(
        (model.pair_count.get((int(src), int(dst)), 0) > 0 for src, dst in zip(srcs, positives)),
        dtype=bool,
        count=len(selected),
    )
    return {
        "rows": int(len(rows)),
        "path": str(path),
        "pair_seen_fraction": float(pair_seen.mean()),
        "candidate_recipe": {"positive": 1, "seen": 77, "unseen": 22},
        "sampling": "exact_honest_eval_kit_weighted_without_replacement",
    }


def main():
    args = parse_args()
    code_root = Path(args.code_root).resolve()
    sys.path.insert(0, str(code_root))
    from src.io_data import read_test
    from src.temporal_graph import GraphFeatureModel

    out_root = Path(args.out_root).resolve()
    artifacts = out_root / "artifacts"
    scratch = out_root / "scratch"
    reports = out_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.train_csv, usecols=["src", "dst", "time", "split"])
    split0_frame = frame[frame["split"] == 0].sort_values(["time", "src", "dst"], kind="mergesort")
    split1_frame = frame[frame["split"] == 1].sort_values("time", kind="mergesort").reset_index(drop=True)
    split0 = list(split0_frame[["src", "dst", "time"]].itertuples(index=False, name=None))
    split1 = list(split1_frame[["src", "dst", "time"]].itertuples(index=False, name=None))
    cut = max(1, min(len(split0) - 1, int(len(split0) * float(args.history_frac))))
    history = split0[:cut]
    train_pool = split0[cut:]
    test_rows = read_test(Path(args.test_csv))

    train_graph_path = artifacts / "train_graph.pkl"
    valid_graph_path = artifacts / "valid_graph.pkl"
    print(f"fitting train graph edges={len(history)}", flush=True)
    train_model = GraphFeatureModel("dataset2", svd_dim=int(args.svd_dim), seed=args.seed).fit(history, test_rows)
    train_model.candidate_space_max = int(frame["dst"].max())
    train_model.save(train_graph_path)
    print(f"fitting validation graph edges={len(split0)}", flush=True)
    valid_model = GraphFeatureModel("dataset2", svd_dim=int(args.svd_dim), seed=args.seed + 1).fit(split0, test_rows)
    valid_model.candidate_space_max = int(frame["dst"].max())
    valid_model.save(valid_graph_path)

    final_indices = set(
        int(index)
        for index in np.linspace(
            0,
            len(split1) - 1,
            min(int(args.final_eval_rows), len(split1)),
            dtype=np.int64,
        )
    )
    valid_pool = [edge for index, edge in enumerate(split1) if index not in final_indices]
    train_rng = np.random.default_rng(int(args.seed) + 10)
    valid_rng = np.random.default_rng(int(args.seed) + 20)
    report = {
        "train_graph": str(train_graph_path),
        "valid_graph": str(valid_graph_path),
        "history_edges": len(history),
        "split0_edges": len(split0),
        "split1_edges": len(split1),
        "final_eval_rows_reserved": len(final_indices),
    }
    report["train"] = save_cache(
        "train",
        train_model,
        history,
        train_pool,
        int(args.train_rows),
        out_root,
        int(args.workers),
        train_rng,
        int(args.cos_hard_count),
        int(args.cos_pool_size),
        int(args.pair_decoy_count),
    )
    report["valid"] = save_honest_validation_cache(
        valid_model,
        split0,
        valid_pool,
        int(args.valid_rows),
        out_root,
        int(args.workers),
        valid_rng,
    )
    (reports / "cache_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
