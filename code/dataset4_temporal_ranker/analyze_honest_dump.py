#!/usr/bin/env python3
"""Full-100-candidate ablations for a deterministic honest-evaluation dump."""
import argparse
import json
from pathlib import Path

import numpy as np


def row_zscore(values):
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def metrics(scores, labels, mask=None):
    if mask is not None:
        scores = scores[mask]
        labels = labels[mask]
    pos = scores[np.arange(len(labels)), labels]
    ranks = 1 + (scores > pos[:, None]).sum(axis=1)
    return {
        "rows": int(len(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "hit_at_1": float(np.mean(ranks == 1)),
        "hit_at_10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
    }


def evaluate(scores, labels, pair_seen, tune_mask, confirm_mask):
    return {
        "overall": metrics(scores, labels),
        "pair_seen": metrics(scores, labels, pair_seen),
        "pair_unseen": metrics(scores, labels, ~pair_seen),
        "tune_half": metrics(scores, labels, tune_mask),
        "confirm_half": metrics(scores, labels, confirm_mask),
        "confirm_pair_unseen": metrics(scores, labels, confirm_mask & ~pair_seen),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.dump)
    labels = data["labels"].astype(np.int64)
    pair_seen = data["pair_seen"].astype(bool)
    scores = data["scores"].astype(np.float32)
    features = data["features"].astype(np.float32)
    names = [str(x) for x in data["feature_names"]]
    by_name = {name: features[:, :, index] for index, name in enumerate(names)}
    times = data["times"].astype(np.int64)
    time_order = np.argsort(times, kind="stable")
    tune_mask = np.zeros(len(labels), dtype=bool)
    tune_mask[time_order[:len(time_order) // 2]] = True
    confirm_mask = ~tune_mask

    base = row_zscore(scores)
    pair_unseen_candidate = (by_name["pair_log"] <= 0).astype(np.float32)
    profile = row_zscore(by_name["profile"])
    seq_mean = row_zscore(by_name["seq_mean"])
    seq_max = row_zscore(by_name["seq_max"])
    audience = row_zscore(by_name["audience_dot"])
    combined = 0.50 * profile + 0.20 * seq_mean + 0.15 * seq_max + 0.15 * audience
    signals = {
        "profile": profile,
        "rank_profile": row_zscore(by_name["rank_profile"]),
        "seq_mean": seq_mean,
        "seq_max": seq_max,
        "audience_dot": audience,
        "profile_seq_audience": combined,
        "profile_pair_unseen_gate": row_zscore(by_name["profile"] * pair_unseen_candidate),
        "combined_pair_unseen_gate": row_zscore(combined * pair_unseen_candidate),
        "popularity": 0.5 * row_zscore(by_name["pop"]) + 0.5 * row_zscore(by_name["recent_pop"]),
    }
    weights = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    results = {
        "baseline": evaluate(base, labels, pair_seen, tune_mask, confirm_mask),
        "ablations": {},
        "joint_ablations": {},
    }
    for name, signal in signals.items():
        directions = [-1.0] if name == "popularity" else [1.0]
        rows = []
        for direction in directions:
            for weight in weights:
                blended = base + direction * weight * signal
                item = {"weight": direction * weight}
                item.update(evaluate(blended, labels, pair_seen, tune_mask, confirm_mask))
                rows.append(item)
        results["ablations"][name] = rows

    # Select on the earlier half only, then report the untouched later half.
    selected = {}
    for name, rows in results["ablations"].items():
        selected[name] = max(rows, key=lambda item: item["tune_half"]["mrr"])
    joint_specs = {
        "profile_popularity_penalty": (profile, signals["popularity"]),
        "profile_seq_mean": (profile, seq_mean),
    }
    for name, (first, second) in joint_specs.items():
        rows = []
        for first_weight in [0.025, 0.05, 0.075, 0.10, 0.15]:
            for second_weight in [0.0, 0.025, 0.05, 0.075, 0.10]:
                direction = -1.0 if name == "profile_popularity_penalty" else 1.0
                blended = base + first_weight * first + direction * second_weight * second
                item = {"first_weight": first_weight, "second_weight": direction * second_weight}
                item.update(evaluate(blended, labels, pair_seen, tune_mask, confirm_mask))
                rows.append(item)
        results["joint_ablations"][name] = rows
        selected[name] = max(rows, key=lambda item: item["tune_half"]["mrr"])
    results["selected_on_tune_half"] = selected
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"baseline": results["baseline"], "selected_on_tune_half": selected}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
