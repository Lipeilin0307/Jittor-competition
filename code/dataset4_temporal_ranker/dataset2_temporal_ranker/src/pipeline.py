import argparse
from argparse import Namespace
from copy import copy
from typing import Iterable, List, Optional

from .io_data import cleanup_path, dump_json, ensure_dir, make_scratch
from .candidate_ranker import _as_path
from .context_stage import build_context_features, package_submission, predict_context_ranker, train_context_ranker
from .stable_stage import build_stable_baseline, refresh_stable_mlp_logits, train_cached_stable_mlp


def _parse_blends(value: str) -> List[float]:
    blends = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            blends.append(float(item))
    if not blends:
        raise ValueError("at least one blend weight is required")
    return blends


def _blend_name(weight: float) -> str:
    if abs(float(weight) - 1.0) < 1e-12:
        return "temporal_ranker_pure"
    label = f"{weight:.2f}".replace(".", "p")
    return f"temporal_ranker_blend_{label}"


def _pack_one(args: Namespace, weight: float, name: Optional[str] = None) -> dict:
    item_args = copy(args)
    item_args.blend_weight = float(weight)
    item_args.output_name = name or _blend_name(float(weight))
    return package_submission(item_args)


def package_sweep(args: Namespace, weights: Iterable[float]) -> dict:
    manifest = {}
    for weight in weights:
        manifest.update(_pack_one(args, float(weight)))
    dump_json(ensure_dir(_as_path(args.reports)) / "submission_manifest.json", manifest)
    return manifest


def run_all(args: Namespace) -> None:
    # scratch_root 由 main() 统一注入；run_all 直接使用，无需重复创建
    if not hasattr(args, "scratch_root") or args.scratch_root is None:
        args.scratch_root = make_scratch(getattr(args, "scratch_dir", "") or None)
        print(f"[temporal-ranker] scratch_root={args.scratch_root}", flush=True)
    if str(args.build_baseline) == "1":
        build_stable_baseline(args)
    build_context_features(args)
    train_context_ranker(args)
    predict_context_ranker(args)
    _pack_one(args, float(args.blend_weight), args.output_name)


def run_prepare(args: Namespace) -> None:
    """Build only CPU artifacts while an existing run occupies the NPU."""
    prepare_args = copy(args)
    prepare_args.train_stable_mlp = "0"
    build_stable_baseline(prepare_args)
    build_context_features(prepare_args)


def run_neural(args: Namespace) -> None:
    """Finish a previously prepared run after the Ascend device is free."""
    train_cached_stable_mlp(args)
    refresh_stable_mlp_logits(args)
    train_context_ranker(args)
    predict_context_ranker(args)
    package_sweep(args, _parse_blends(args.sweep_blends))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Temporal graph link-ranking pipeline")
    p.add_argument("--data-dir", default="data_A")
    p.add_argument("--baseline-root", default="baseline_artifacts")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--reports", default="reports")
    p.add_argument("--submission", default="submission")
    p.add_argument("--scratch-dir", default="",
                   help="scratch 根目录（默认 /dev/shm/temporal_ranker_<pid>）；所有大张量分片写这里，不占磁盘")
    p.add_argument("--build-baseline", default="1")
    p.add_argument("--stable-seed", type=int, default=2026)
    p.add_argument("--stable-svd-dim", type=int, default=128)
    p.add_argument("--stable-recent-limit", type=int, default=160)
    p.add_argument("--stable-transition-window", type=int, default=16)
    p.add_argument("--stable-transition-topk", type=int, default=384)
    p.add_argument("--stable-max-valid-events", type=int, default=30000)
    p.add_argument("--stable-search-rounds", type=int, default=5)
    p.add_argument("--stable-feature-workers", type=int, default=64)
    p.add_argument("--stable-predict-workers", type=int, default=64)
    p.add_argument("--stable-predict-batch-size", type=int, default=16384)
    p.add_argument("--train-stable-mlp", default="1")
    p.add_argument("--stable-mlp-train-rows", type=int, default=80000)
    p.add_argument("--stable-mlp-hidden", type=int, default=192)
    p.add_argument("--stable-mlp-epochs", type=int, default=8)
    p.add_argument("--stable-mlp-batch-size", type=int, default=2048)
    p.add_argument("--stable-mlp-lr", type=float, default=8e-4)
    p.add_argument("--stable-mlp-output-weight", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=3026)
    p.add_argument("--workers", type=int, default=96)
    p.add_argument("--history-frac", type=float, default=0.70)
    p.add_argument("--train-rows", type=int, default=400000)
    p.add_argument("--valid-rows", type=int, default=60000)
    p.add_argument("--template-train-rows", type=int, default=400000)
    p.add_argument("--template-valid-rows", type=int, default=100000)
    p.add_argument("--max-pool", type=int, default=700)
    p.add_argument("--svd-dim", type=int, default=128)
    p.add_argument("--fit-edge-limit", type=int, default=0)
    p.add_argument("--src-seq-len", type=int, default=64)
    p.add_argument("--dst-seq-len", type=int, default=64)
    p.add_argument("--seeds", default="3407,42,256")
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--predict-batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--margin-weight", type=float, default=0.0)
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--profile-calibration-weight", type=float, default=0.0)
    p.add_argument("--anchor-weights", default="")
    p.add_argument("--anchor-seed", type=int, default=20260813)
    p.add_argument("--anchor-epochs", type=int, default=5)
    p.add_argument("--anchor-batch-size", type=int, default=2048)
    p.add_argument("--anchor-lr", type=float, default=2e-2)
    p.add_argument("--reuse-baseline-features", default="0")
    p.add_argument("--reuse-stable-graphs", default="1")
    p.add_argument("--blend-weight", type=float, default=0.10)
    p.add_argument("--output-name", default="temporal_ranker_blend_0p10")
    p.add_argument("--sweep-blends", default="0.02,0.05,0.10,0.20,0.35,1.00")

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("baseline")
    sub.add_parser("prepare")
    sub.add_parser("neural")
    sub.add_parser("build")
    sub.add_parser("train")
    sub.add_parser("predict")
    sub.add_parser("package")
    sub.add_parser("package-sweep")
    sub.add_parser("all")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command not in ("package", "package-sweep"):
        scratch = make_scratch(getattr(args, "scratch_dir", "") or None)
        args.scratch_root = scratch
        print(f"[temporal-ranker] scratch_root={scratch}", flush=True)
        succeeded = False
        try:
            _dispatch(args)
            succeeded = True
        finally:
            if args.command in ("prepare", "build") or not succeeded:
                print(f"[temporal-ranker] preserving prepared scratch_root={scratch}", flush=True)
            else:
                cleanup_path(scratch)
    else:
        _dispatch(args)


def _dispatch(args: Namespace) -> None:
    if args.command == "baseline":
        build_stable_baseline(args)
    elif args.command == "prepare":
        run_prepare(args)
    elif args.command == "neural":
        run_neural(args)
    elif args.command == "build":
        build_context_features(args)
    elif args.command == "train":
        train_context_ranker(args)
    elif args.command == "predict":
        predict_context_ranker(args)
    elif args.command == "package":
        _pack_one(args, float(args.blend_weight), args.output_name)
    elif args.command == "package-sweep":
        package_sweep(args, _parse_blends(args.sweep_blends))
    elif args.command == "all":
        run_all(args)
    else:
        raise SystemExit(args.command)


if __name__ == "__main__":
    main()
