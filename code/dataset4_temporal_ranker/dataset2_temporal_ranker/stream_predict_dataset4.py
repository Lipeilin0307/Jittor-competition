import argparse
import csv
import gc
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.candidate_ranker import (
    _load_feature_mlp_predictor,
    score_linear_anchor,
)
from src.context_stage import CONTEXT_FEATURE_NAMES, SequenceContext, _append_sequence_features
from src.io_data import TestRow, row_zscore, softmax
from src.temporal_graph import GraphFeatureModel


def build_sequence_context_streaming(
    model: GraphFeatureModel,
    train_csv: Path,
    dst_seq_len: int,
) -> SequenceContext:
    """Build the inference context without loading and sorting all train edges."""
    ctx = SequenceContext.__new__(SequenceContext)
    ctx.model = model
    ctx.dst_seq_len = int(dst_seq_len)

    recent_by_dst = {}
    last_time = None
    edges = 0
    started = time.time()
    with train_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"src", "dst", "time"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{train_csv}: expected src,dst,time columns")
        for row in reader:
            timestamp = int(row["time"])
            if last_time is not None and timestamp < last_time:
                raise ValueError(
                    "train.csv is not time sorted; the streaming context would be incorrect"
                )
            last_time = timestamp
            dst = int(row["dst"])
            values = recent_by_dst.get(dst)
            if values is None:
                values = deque(maxlen=ctx.dst_seq_len)
                recent_by_dst[dst] = values
            values.append(int(row["src"]))
            edges += 1
            if edges % 1_000_000 == 0:
                print(
                    f"context edges={edges} destinations={len(recent_by_dst)} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    ctx.audience_mean = {}
    ctx.audience_count = {}
    if model.src_emb is not None:
        for index, (dst, srcs) in enumerate(recent_by_dst.items(), start=1):
            vectors = []
            for src in srcs:
                src_index = model.src_to_id.get(int(src))
                if src_index is not None:
                    vectors.append(model.src_emb[src_index])
            if vectors:
                array = np.asarray(vectors, dtype=np.float32)
                mean = array.mean(axis=0)
                norm = max(float(np.linalg.norm(mean)), 1e-6)
                ctx.audience_mean[int(dst)] = (mean / norm).astype(np.float32)
                ctx.audience_count[int(dst)] = float(np.log1p(len(vectors)))
            if index % 100_000 == 0:
                print(
                    f"context audiences={index}/{len(recent_by_dst)}",
                    flush=True,
                )

    ctx.max_audience_count = max(ctx.audience_count.values(), default=1.0)
    # The feature function only performs membership tests after audience vectors
    # have been built. Sets avoid repeated scans through up to 64 source IDs.
    for dst in list(recent_by_dst):
        recent_by_dst[dst] = frozenset(int(src) for src in recent_by_dst[dst])
    ctx.dst_recent_src = recent_by_dst
    print(
        f"context ready edges={edges} destinations={len(ctx.dst_recent_src)} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return ctx


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def iter_test_chunks(path: Path, chunk_rows: int, skip_rows: int):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) != 102:
            raise ValueError(f"{path}: expected a 102-column test CSV")

        skipped = 0
        while skipped < skip_rows:
            try:
                row = next(reader)
            except StopIteration as exc:
                raise ValueError(
                    f"partial output has {skip_rows} rows but test.csv ended at {skipped}"
                ) from exc
            if len(row) != 102:
                raise ValueError(f"{path}: malformed row while resuming")
            skipped += 1

        chunk = []
        for line_number, row in enumerate(reader, start=skip_rows + 2):
            if len(row) != 102:
                raise ValueError(
                    f"{path}:{line_number}: expected 102 columns, got {len(row)}"
                )
            chunk.append(
                TestRow(
                    int(row[0]),
                    int(row[1]),
                    tuple(int(value) for value in row[2:]),
                )
            )
            if len(chunk) == chunk_rows:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def write_probability_chunk(handle, logits: np.ndarray) -> tuple:
    probabilities = softmax(row_zscore(logits))
    rounded = np.round(probabilities, 8)
    correction = 1.0 - rounded.sum(axis=1)
    best = np.argmax(rounded, axis=1)
    rounded[np.arange(len(rounded)), best] = np.clip(
        rounded[np.arange(len(rounded)), best] + correction,
        0.0,
        1.0,
    )
    np.savetxt(handle, rounded, fmt="%.8f", delimiter=",")
    return float(rounded.sum(axis=1).min()), float(rounded.sum(axis=1).max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-bounded dataset4 inference")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm", required=True)
    parser.add_argument("--anchor-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--chunk-rows", type=int, default=2000)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--src-seq-len", type=int, default=64)
    parser.add_argument("--dst-seq-len", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--profile-calibration-weight", type=float, default=0.0)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    if args.chunk_rows <= 0 or args.predict_batch_size <= 0:
        raise ValueError("chunk and prediction batch sizes must be positive")

    data_dir = Path(args.data_dir).expanduser().resolve()
    dataset_dir = data_dir / "dataset2"
    train_csv = dataset_dir / "train.csv"
    test_csv = dataset_dir / "test.csv"
    required_paths = [
        train_csv,
        test_csv,
        Path(args.graph),
        Path(args.checkpoint),
        Path(args.norm),
        Path(args.anchor_weights),
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    if args.restart:
        partial.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
    elif output.exists():
        print(f"output already exists: {output}", flush=True)
        return

    completed_rows = count_lines(partial)
    print(f"resume rows={completed_rows} partial={partial}", flush=True)

    started = time.time()
    model = GraphFeatureModel.load(Path(args.graph))
    context = build_sequence_context_streaming(
        model,
        train_csv,
        dst_seq_len=int(args.dst_seq_len),
    )
    anchor_weights = np.load(args.anchor_weights)["weights"].astype(np.float32)
    predictor = _load_feature_mlp_predictor(
        Path(args.checkpoint),
        Path(args.norm),
        hidden=int(args.hidden),
    )

    mode = "a" if completed_rows else "w"
    sum_min = math.inf
    sum_max = -math.inf
    with partial.open(mode, encoding="utf-8", newline="") as output_handle:
        for rows in iter_test_chunks(test_csv, int(args.chunk_rows), completed_rows):
            src_ids = np.fromiter((row.src for row in rows), dtype=np.int64)
            dst_ids = np.asarray([row.candidates for row in rows], dtype=np.int64)
            base = model.feature_tensor(rows, progress_every=0)
            features = _append_sequence_features(
                base,
                src_ids,
                dst_ids,
                context,
                src_seq_len=int(args.src_seq_len),
            )
            anchor = score_linear_anchor(features, anchor_weights)
            features = np.concatenate([features, anchor[:, :, None]], axis=2)
            logits = predictor(features, batch_size=int(args.predict_batch_size))
            if float(args.profile_calibration_weight) != 0.0:
                profile = features[:, :, CONTEXT_FEATURE_NAMES.index("profile")]
                logits = (
                    row_zscore(logits)
                    + float(args.profile_calibration_weight) * row_zscore(profile)
                ).astype(np.float32)
            chunk_min, chunk_max = write_probability_chunk(output_handle, logits)
            output_handle.flush()
            completed_rows += len(rows)
            sum_min = min(sum_min, chunk_min)
            sum_max = max(sum_max, chunk_max)
            print(
                f"predicted rows={completed_rows} chunk={len(rows)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
            del rows, src_ids, dst_ids, base, features, anchor, logits
            gc.collect()

    os.replace(partial, output)
    report = {
        "output": str(output),
        "rows": completed_rows,
        "columns": 100,
        "chunk_rows": int(args.chunk_rows),
        "predict_batch_size": int(args.predict_batch_size),
        "profile_calibration_weight": float(args.profile_calibration_weight),
        "probability_sum_min": None if math.isinf(sum_min) else sum_min,
        "probability_sum_max": None if math.isinf(sum_max) else sum_max,
        "elapsed_seconds": time.time() - started,
    }
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output.with_suffix(".report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
