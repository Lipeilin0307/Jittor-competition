import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .candidate_ranker import _as_path, _load_feature_mlp_predictor, _train_feature_mlp
from .io_data import (
    cleanup_path,
    dataset_dir,
    dump_json,
    ensure_dir,
    read_test,
    read_train,
    softmax,
    split_edges,
    validate_csv,
    write_scores_csv,
)
from .temporal_graph import FEATURE_NAMES, GraphFeatureModel
from .parallel_features import feature_tensor_parallel, score_rows_to_shards_parallel
from .evaluation import (
    aggregate_mrr,
    attach_features,
    build_validation_sets,
    evaluate_components,
    score_feature_tensor,
    search_weights_multi,
    top1_stats,
)


def _make_training_feature_block(vsets, max_rows: int, seed: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    rows = []
    labels = []
    rng = np.random.default_rng(seed)
    usable = [v for v in vsets if not v.name.startswith("teacher_") and v.features is not None]
    for vset in usable:
        idx = np.arange(len(vset.labels))
        rng.shuffle(idx)
        take = min(len(idx), max(1, int(max_rows / max(len(usable), 1))))
        keep = np.sort(idx[:take])
        rows.append(vset.features[keep])
        labels.append(vset.labels[keep])
    if not rows:
        raise ValueError("no validation features available to train stable baseline MLP")
    x = np.concatenate(rows, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    if len(y) > max_rows:
        idx = rng.choice(np.arange(len(y)), size=int(max_rows), replace=False)
        x = x[idx]
        y = y[idx]
    return x, y, {"train_rows": int(len(y)), "feature_dim": int(x.shape[-1])}


def train_stable_dataset(args, dataset: str) -> dict:
    data_dir = _as_path(args.data_dir)
    baseline_root = ensure_dir(_as_path(args.baseline_root))
    artifacts = ensure_dir(baseline_root / "artifacts")
    reports = ensure_dir(baseline_root / "reports")
    ds_dir = dataset_dir(data_dir, dataset)
    print(f"[stable] reading train/test rows dataset={dataset}", flush=True)
    train_edges, valid_edges, split_meta = split_edges(ds_dir, all_train=False, prefer_official=(dataset == "dataset2"))
    test_rows = read_test(ds_dir / "test.csv")
    model_path = artifacts / f"{dataset}_feature_model_val.pkl"
    if str(args.reuse_stable_graphs) == "1" and model_path.exists():
        print(f"[stable] reusing validation graph model path={model_path}", flush=True)
        model = GraphFeatureModel.load(model_path)
    else:
        print(f"[stable] fitting validation graph model dataset={dataset}", flush=True)
        model = GraphFeatureModel(
            dataset=dataset,
            svd_dim=int(args.stable_svd_dim),
            recent_limit=int(args.stable_recent_limit),
            transition_window=int(args.stable_transition_window),
            transition_topk=int(args.stable_transition_topk),
            seed=int(args.stable_seed),
        ).fit(train_edges, test_rows)
        model.save(model_path)

    print(f"[stable] building validation rows max_events={int(args.stable_max_valid_events)}", flush=True)
    vsets = build_validation_sets(dataset, train_edges, valid_edges, test_rows, int(args.stable_max_valid_events), int(args.stable_seed))
    vf_cache = args.scratch_root / f"validation_features_{dataset}"
    print(f"[stable] attaching validation features sets={len(vsets)} workers={int(args.stable_feature_workers)}", flush=True)
    attach_features(model, vsets, workers=int(args.stable_feature_workers), cache_dir=vf_cache)
    print("[stable] searching feature weights", flush=True)
    component_report = evaluate_components(vsets)
    weights, history = search_weights_multi(vsets, rounds=int(args.stable_search_rounds))
    aggregate, by_set = aggregate_mrr(vsets, weights)

    mlp_report = {"status": "disabled"}
    if dataset == "dataset2":
        try:
            train_x, train_y, train_meta = _make_training_feature_block(vsets, int(args.stable_mlp_train_rows), int(args.stable_seed))
            valid_x = vsets[0].features[: min(len(vsets[0].labels), 5000)] if vsets and vsets[0].features is not None else None
            valid_y = vsets[0].labels[: min(len(vsets[0].labels), 5000)] if vsets else None
            data_path = artifacts / "dataset2_stable_mlp_data.npz"
            np.savez(data_path, train_x=train_x, train_y=train_y, valid_x=valid_x, valid_y=valid_y)
            mlp_report = {
                "status": "deferred",
                "hidden": int(args.stable_mlp_hidden),
                "training_block": train_meta,
                "data": str(data_path),
            }
            if str(args.train_stable_mlp) == "1":
                mlp_report = _train_feature_mlp(
                    train_x,
                    train_y,
                    valid_x,
                    valid_y,
                    out_dir=artifacts / "dataset2_stable_mlp",
                    seed=int(args.stable_seed),
                    hidden=int(args.stable_mlp_hidden),
                    epochs=int(args.stable_mlp_epochs),
                    batch_size=int(args.stable_mlp_batch_size),
                    lr=float(args.stable_mlp_lr),
                )
                mlp_report["hidden"] = int(args.stable_mlp_hidden)
                mlp_report["training_block"] = train_meta
                mlp_report["data"] = str(data_path)
        except Exception as exc:
            mlp_report = {"status": "failed", "error": repr(exc), "hidden": int(args.stable_mlp_hidden)}

    report = {
        "dataset": dataset,
        "split": split_meta,
        "train_edges_used": len(train_edges),
        "valid_edges_used": len(valid_edges),
        "test_rows": len(test_rows),
        "feature_names": FEATURE_NAMES,
        "artifact": str(model_path),
        "svd_dim": int(args.stable_svd_dim),
        "validation_sets": [{"name": v.name, "rows": len(v.rows), "weight": v.weight, "meta": v.meta} for v in vsets],
        "component_report": component_report,
        "weights": weights,
        "weight_history": history,
        "aggregate_mrr": aggregate,
        "by_set_mrr": by_set,
        "jittor": mlp_report,
    }
    dump_json(reports / f"{dataset}_train_report.json", report)
    print(json.dumps({k: report[k] for k in ["dataset", "aggregate_mrr", "by_set_mrr", "weights", "jittor"]}, indent=2, ensure_ascii=False), flush=True)
    cleanup_path(vf_cache)  # validation_features 已 attach 到 vsets 内存，缓存可清
    return report


def predict_stable_dataset1(args) -> dict:
    data_dir = _as_path(args.data_dir)
    baseline_root = ensure_dir(_as_path(args.baseline_root))
    artifacts = ensure_dir(baseline_root / "artifacts")
    reports = ensure_dir(baseline_root / "reports")
    ds_dir = dataset_dir(data_dir, "dataset1")
    train_edges, _valid_edges, split_meta = split_edges(ds_dir, all_train=True)
    test_rows = read_test(ds_dir / "test.csv")
    report = json.loads((reports / "dataset1_train_report.json").read_text(encoding="utf-8"))
    weights: Dict[str, float] = {k: float(v) for k, v in report["weights"].items()}
    model = GraphFeatureModel(
        dataset="dataset1",
        svd_dim=int(report.get("svd_dim", args.stable_svd_dim)),
        recent_limit=int(args.stable_recent_limit),
        transition_window=int(args.stable_transition_window),
        transition_topk=int(args.stable_transition_topk),
        seed=int(args.stable_seed) + 100,
    ).fit(train_edges, test_rows)
    model_path = artifacts / "dataset1_feature_model_final.pkl"
    model.save(model_path)
    logits = model.score_rows(test_rows, weights, batch_size=int(args.stable_predict_batch_size)).astype(np.float32)
    np.save(artifacts / "dataset1_model_logits.npy", logits)
    out_dir = ensure_dir(baseline_root / "submission_mlp_peak" / "result_rebuild_mlpw_5p5")
    csv_path = out_dir / "dataset1.csv"
    check = write_scores_csv(softmax(logits), csv_path)
    pred_report = {
        "dataset": "dataset1",
        "split": split_meta,
        "weights": weights,
        "model": str(model_path),
        "logits": str(artifacts / "dataset1_model_logits.npy"),
        "csv": str(csv_path),
        "test_rows": len(test_rows),
        "top1_stats": top1_stats(logits, test_rows, model),
        "validation": check,
    }
    dump_json(reports / "dataset1_predict_report.json", pred_report)
    return pred_report


def _load_stable_mlp_predictor(report: dict, args):
    meta = report.get("jittor", {})
    if meta.get("status") != "trained":
        return None
    ckpt = Path(meta.get("checkpoint", ""))
    norm = Path(meta.get("norm", ""))
    if not ckpt.exists() or not norm.exists():
        return None
    return _load_feature_mlp_predictor(ckpt, norm, hidden=int(meta.get("hidden", args.stable_mlp_hidden)))


def predict_stable_dataset2(args) -> dict:
    data_dir = _as_path(args.data_dir)
    baseline_root = ensure_dir(_as_path(args.baseline_root))
    artifacts = ensure_dir(baseline_root / "artifacts")
    reports = ensure_dir(baseline_root / "reports")
    ds_dir = dataset_dir(data_dir, "dataset2")
    test_rows = read_test(ds_dir / "test.csv")
    report = json.loads((reports / "dataset2_train_report.json").read_text(encoding="utf-8"))
    weights: Dict[str, float] = {k: float(v) for k, v in report["weights"].items()}
    model_path = artifacts / "dataset2_feature_model_final.pkl"
    if str(args.reuse_stable_graphs) == "1" and model_path.exists():
        print(f"[stable] reusing final graph model path={model_path}", flush=True)
        model = GraphFeatureModel.load(model_path)
        split_meta = {"strategy": "all_train_reused", "model": str(model_path)}
    else:
        train_edges, _valid_edges, split_meta = split_edges(ds_dir, all_train=True)
        model = GraphFeatureModel(
            dataset="dataset2",
            svd_dim=int(report.get("svd_dim", args.stable_svd_dim)),
            recent_limit=int(args.stable_recent_limit),
            transition_window=int(args.stable_transition_window),
            transition_topk=int(args.stable_transition_topk),
            seed=int(args.stable_seed) + 100,
        ).fit(train_edges, test_rows)
        model.save(model_path)

    shard_dir = ensure_dir(args.scratch_root / "dataset2_predict_shards")
    predictor = _load_stable_mlp_predictor(report, args)
    for pattern in ("features_part_*.npy", "feature_logits_part_*.npy", "mlp_logits_part_*.npy"):
        for old in shard_dir.glob(pattern):
            old.unlink()
    shard_reports = []
    if predictor is None:
        print(
            f"[stable] scoring test rows directly rows={len(test_rows)} workers={int(args.stable_predict_workers)} ",
            f"batch_size={int(args.stable_predict_batch_size)}",
            flush=True,
        )
        logits_paths = score_rows_to_shards_parallel(
            model,
            test_rows,
            weights,
            shard_dir,
            "feature_logits_part",
            int(args.stable_predict_workers),
            int(args.stable_predict_batch_size),
        )
        for shard_id, logits_path_str in enumerate(logits_paths):
            logits_path = Path(logits_path_str)
            logits = np.load(logits_path, mmap_mode="r")
            shard_reports.append(
                {
                    "shard_id": shard_id,
                    "rows": int(logits.shape[0]),
                    "feature_path": None,
                    "logits_path": str(logits_path),
                }
            )
    else:
        feature_tensor_parallel(model, test_rows, shard_dir, "features_part", int(args.stable_predict_workers))
        feature_paths = sorted(shard_dir.glob("features_part_*.npy"))
        for shard_id, feature_path in enumerate(feature_paths):
            features = np.load(feature_path, mmap_mode="r")
            feature_logits = score_feature_tensor(features, weights).astype(np.float32)
            logits_path = shard_dir / f"feature_logits_part_{shard_id:02d}.npy"
            np.save(logits_path, feature_logits)
            shard_reports.append(
                {
                    "shard_id": shard_id,
                    "rows": int(features.shape[0]),
                    "feature_path": str(feature_path),
                    "logits_path": str(logits_path),
                }
            )

    feature_logits_all = []
    mlp_logits_all = []
    for item in shard_reports:
        feature_logits = np.load(item["logits_path"], mmap_mode="r")
        if predictor is None:
            mlp_logits = np.zeros_like(feature_logits, dtype=np.float32)
        else:
            features = np.load(item["feature_path"], mmap_mode="r")
            mlp_logits = predictor(features, batch_size=int(args.stable_predict_batch_size)).astype(np.float32)
        mlp_path = shard_dir / f"mlp_logits_part_{int(item['shard_id']):02d}.npy"
        np.save(mlp_path, mlp_logits)
        item["mlp_logits_path"] = str(mlp_path)
        feature_logits_all.append(feature_logits)
        mlp_logits_all.append(mlp_logits)
    feature_logits_arr = np.concatenate(feature_logits_all, axis=0)
    mlp_logits_arr = np.concatenate(mlp_logits_all, axis=0)
    combined = feature_logits_arr if predictor is None else feature_logits_arr + float(args.stable_mlp_output_weight) * mlp_logits_arr
    np.save(artifacts / "dataset2_model_logits.npy", combined.astype(np.float32))
    # 单独保存 mlp_logits / feature_logits，供 predict_context_ranker 复用（避免依赖已清理的 shards）
    np.save(artifacts / "dataset2_mlp_logits.npy", mlp_logits_arr.astype(np.float32))
    np.save(artifacts / "dataset2_feature_logits.npy", feature_logits_arr.astype(np.float32))
    pred_report = {
        "dataset": "dataset2",
        "split": split_meta,
        "weights": weights,
        "model": str(model_path),
        "logits": str(artifacts / "dataset2_model_logits.npy"),
        "test_rows": len(test_rows),
        "top1_stats": top1_stats(combined, test_rows, model),
        "jittor_used_in_prediction": predictor is not None,
        "shards": shard_reports,
    }
    dump_json(reports / "dataset2_predict_report.json", pred_report)
    cleanup_path(shard_dir)  # features/feature_logits/mlp_logits 分片已合并到 dataset2_model_logits.npy，可清
    return pred_report


def build_stable_baseline(args) -> dict:
    """dataset2-only：构建稳定基线（图特征模型 + 权重搜索 + 稳定 MLP）。

    本流水线仅处理 dataset2；dataset1 由独立的 GraphMixer 流水线
    （code/dataset1_graphmixer/）负责，二者互不依赖。
    原始双数据集实现中的 dataset1 训练 / 预测分支已在此处移除，
    train_stable_dataset1 / predict_stable_dataset1 的函数定义保留备查但不再调用。
    """
    baseline_root = _as_path(args.baseline_root)
    report_dir = ensure_dir(baseline_root / "reports")
    artifact_dir = ensure_dir(baseline_root / "artifacts")
    stable_report_path = report_dir / "stable_baseline_report.json"
    required_outputs = (
        artifact_dir / "dataset2_model_logits.npy",
        artifact_dir / "dataset2_feature_logits.npy",
        artifact_dir / "dataset2_mlp_logits.npy",
        artifact_dir / "dataset2_feature_model_final.pkl",
    )
    if (
        str(args.reuse_stable_graphs) == "1"
        and stable_report_path.exists()
        and all(path.exists() for path in required_outputs)
    ):
        print(f"[stable] reusing completed baseline path={stable_report_path}", flush=True)
        return json.loads(stable_report_path.read_text(encoding="utf-8"))

    report_path = report_dir / "dataset2_train_report.json"
    if str(args.reuse_stable_graphs) == "1" and report_path.exists():
        print(f"[stable] reusing completed training report path={report_path}", flush=True)
        train2 = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        train2 = train_stable_dataset(args, "dataset2")
    pred2 = predict_stable_dataset2(args)
    payload = {"dataset2_train": train2, "dataset2_predict": pred2}
    dump_json(stable_report_path, payload)
    return payload


def train_cached_stable_mlp(args) -> dict:
    """Train the deferred stable MLP without rebuilding any CPU graph artifact."""
    baseline_root = ensure_dir(_as_path(args.baseline_root))
    artifacts = ensure_dir(baseline_root / "artifacts")
    reports = ensure_dir(baseline_root / "reports")
    report_path = reports / "dataset2_train_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    data_path = Path(report.get("jittor", {}).get("data", artifacts / "dataset2_stable_mlp_data.npz"))
    if not data_path.exists():
        raise FileNotFoundError(f"missing deferred stable MLP data: {data_path}")
    data = np.load(data_path)
    mlp_report = _train_feature_mlp(
        data["train_x"].astype(np.float32),
        data["train_y"].astype(np.int64),
        data["valid_x"].astype(np.float32),
        data["valid_y"].astype(np.int64),
        out_dir=artifacts / "dataset2_stable_mlp",
        seed=int(args.stable_seed),
        hidden=int(args.stable_mlp_hidden),
        epochs=int(args.stable_mlp_epochs),
        batch_size=int(args.stable_mlp_batch_size),
        lr=float(args.stable_mlp_lr),
    )
    mlp_report["hidden"] = int(args.stable_mlp_hidden)
    mlp_report["data"] = str(data_path)
    report["jittor"] = mlp_report
    dump_json(report_path, report)
    return mlp_report


def refresh_stable_mlp_logits(args) -> dict:
    """Apply a newly trained stable MLP to already materialized test features."""
    baseline_root = ensure_dir(_as_path(args.baseline_root))
    artifacts = ensure_dir(baseline_root / "artifacts")
    reports = ensure_dir(baseline_root / "reports")
    report = json.loads((reports / "dataset2_train_report.json").read_text(encoding="utf-8"))
    predictor = _load_stable_mlp_predictor(report, args)
    if predictor is None:
        raise RuntimeError("stable MLP checkpoint is unavailable")
    shard_dir = artifacts / "dataset2_predict_shards"
    feature_paths = sorted(shard_dir.glob("features_part_*.npy"))
    feature_logit_paths = sorted(shard_dir.glob("feature_logits_part_*.npy"))
    if not feature_paths or len(feature_paths) != len(feature_logit_paths):
        raise FileNotFoundError("missing stable feature prediction shards")
    feature_logits_all = []
    mlp_logits_all = []
    for shard_id, (feature_path, feature_logit_path) in enumerate(zip(feature_paths, feature_logit_paths)):
        features = np.load(feature_path, mmap_mode="r")
        feature_logits = np.load(feature_logit_path).astype(np.float32)
        mlp_logits = predictor(features, batch_size=int(args.stable_predict_batch_size)).astype(np.float32)
        np.save(shard_dir / f"mlp_logits_part_{shard_id:02d}.npy", mlp_logits)
        feature_logits_all.append(feature_logits)
        mlp_logits_all.append(mlp_logits)
    combined = np.concatenate(feature_logits_all, axis=0) + float(args.stable_mlp_output_weight) * np.concatenate(mlp_logits_all, axis=0)
    np.save(artifacts / "dataset2_model_logits.npy", combined.astype(np.float32))
    payload = {"shards": len(feature_paths), "rows": int(combined.shape[0]), "mlp_output_weight": float(args.stable_mlp_output_weight)}
    dump_json(reports / "dataset2_mlp_refresh_report.json", payload)
    return payload
