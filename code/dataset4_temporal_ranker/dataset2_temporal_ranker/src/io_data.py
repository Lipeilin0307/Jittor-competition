import csv
import json
import math
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Edge = Tuple[int, int, int]


@dataclass(frozen=True)
class TrainRow:
    src: int
    dst: int
    time: int
    split: Optional[str] = None


@dataclass(frozen=True)
class TestRow:
    src: int
    time: int
    candidates: Tuple[int, ...]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_scratch(scratch_dir: Optional[str]) -> Path:
    """返回本次运行的 scratch 根目录。优先用户传入；否则 /dev/shm；最后 fallback 到系统 tmp。"""
    candidates: List[Path] = []
    if scratch_dir:
        candidates.append(Path(scratch_dir).expanduser())
    candidates.append(Path("/dev/shm") / f"temporal_ranker_{os.getpid()}")
    candidates.append(Path(tempfile.gettempdir()) / f"temporal_ranker_{os.getpid()}")
    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".write_test"
            probe.touch()
            probe.unlink()
            return base
        except OSError:
            continue
    raise RuntimeError("no usable scratch directory found (tried: %s)" % ", ".join(str(c) for c in candidates))


def cleanup_path(path) -> None:
    """递归删除一个文件或目录（含内容），不抛错。"""
    try:
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    except OSError:
        pass


def dump_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_train(path: Path) -> List[TrainRow]:
    rows: List[TrainRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if not {"src", "dst", "time"}.issubset(set(fields)):
            raise ValueError(f"{path}: expected src,dst,time columns, got {fields}")
        has_split = "split" in fields
        for row in reader:
            split = row.get("split") if has_split else None
            rows.append(TrainRow(int(row["src"]), int(row["dst"]), int(row["time"]), split))
    rows.sort(key=lambda x: (x.time, x.src, x.dst))
    return rows


def train_edges(rows: Sequence[TrainRow]) -> List[Edge]:
    return [(r.src, r.dst, r.time) for r in rows]


def split_edges(
    dataset_dir: Path,
    all_train: bool = False,
    prefer_official: bool = True,
    time_valid_frac: float = 0.15,
) -> Tuple[List[Edge], List[Edge], dict]:
    rows = read_train(dataset_dir / "train.csv")
    if all_train:
        return train_edges(rows), [], {
            "strategy": "all_train",
            "total_edges": len(rows),
            "train_edges": len(rows),
            "valid_edges": 0,
        }

    split_values = [r.split for r in rows if r.split not in (None, "")]
    if prefer_official and split_values:
        train = [(r.src, r.dst, r.time) for r in rows if str(r.split) == "0"]
        valid = [(r.src, r.dst, r.time) for r in rows if str(r.split) != "0"]
        if train and valid:
            return train, valid, {
                "strategy": "official_split",
                "total_edges": len(rows),
                "train_edges": len(train),
                "valid_edges": len(valid),
                "split_counts": {str(v): split_values.count(v) for v in sorted(set(split_values))},
            }

    cut = max(1, min(len(rows) - 1, int(len(rows) * (1.0 - time_valid_frac))))
    train_rows = rows[:cut]
    valid_rows = rows[cut:]
    return train_edges(train_rows), train_edges(valid_rows), {
        "strategy": f"time_tail_{time_valid_frac:.2f}",
        "total_edges": len(rows),
        "train_edges": len(train_rows),
        "valid_edges": len(valid_rows),
    }


def read_test(path: Path) -> List[TestRow]:
    rows: List[TestRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path}: empty test file")
        if len(header) != 102:
            raise ValueError(f"{path}: expected 102 columns, got {len(header)}")
        for line_no, row in enumerate(reader, start=2):
            if len(row) != 102:
                raise ValueError(f"{path}:{line_no}: expected 102 columns, got {len(row)}")
            rows.append(TestRow(int(row[0]), int(row[1]), tuple(int(x) for x in row[2:])))
    return rows


def dataset_dir(data_dir: Path, dataset: str) -> Path:
    path = data_dir / dataset
    if not (path / "train.csv").exists() or not (path / "test.csv").exists():
        raise FileNotFoundError(f"missing train/test under {path}")
    return path


def row_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.where(std < 1e-6, 1.0, std)


def row_rank_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    order = np.argsort(-values, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    for i in range(values.shape[0]):
        sorted_values = values[i, order[i]]
        starts = np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
        competition = np.maximum.accumulate(
            np.where(starts, np.arange(1, values.shape[1] + 1), 0)
        ).astype(np.float32)
        ranks[i, order[i]] = competition
    return 1.0 / ranks


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64) / max(float(temperature), 1e-6)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)


def tie_aware_ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or len(labels) != scores.shape[0]:
        raise ValueError(f"invalid score/label shapes: scores={scores.shape}, labels={labels.shape}")
    if not np.isfinite(scores).all():
        raise FloatingPointError("MRR received NaN or Inf scores")
    if ((labels < 0) | (labels >= scores.shape[1])).any():
        raise ValueError("MRR labels are outside the candidate range")
    pos = scores[np.arange(len(labels)), labels]
    greater = (scores > pos[:, None]).sum(axis=1)
    equal_other = (np.abs(scores - pos[:, None]) <= 1e-12).sum(axis=1) - 1
    return 1.0 + greater + 0.5 * equal_other


def tie_aware_mrr(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    return float(np.mean(1.0 / tie_aware_ranks(scores, labels)))


def labels_from_candidates(rows: Sequence[TestRow], positives: Sequence[int]) -> np.ndarray:
    labels = []
    for row, dst in zip(rows, positives):
        try:
            labels.append(row.candidates.index(dst))
        except ValueError:
            labels.append(-1)
    return np.asarray(labels, dtype=np.int64)


def read_zip_scores(zip_path: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            reader = csv.reader((line.decode("utf-8") for line in f))
            rows = [[float(x) for x in row if x != ""] for row in reader]
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 100:
        raise ValueError(f"{zip_path}:{member}: expected Nx100 probabilities, got {arr.shape}")
    return arr


def copy_member_from_zip(zip_path: Path, member: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def _format_prob_row(row: np.ndarray) -> List[str]:
    row = np.asarray(row, dtype=np.float64)
    row = np.clip(row, 0.0, 1.0)
    total = float(row.sum())
    if not math.isfinite(total) or total <= 0:
        row = np.full(row.shape, 1.0 / len(row), dtype=np.float64)
    else:
        row = row / total
    rounded = np.round(row, 8)
    diff = 1.0 - float(rounded.sum())
    idx = int(np.argmax(rounded))
    rounded[idx] = min(1.0, max(0.0, rounded[idx] + diff))
    return [f"{float(v):.8f}" for v in rounded]


def write_scores_csv(scores_or_probs: np.ndarray, out_path: Path, is_logits: bool = False, temperature: float = 1.0) -> dict:
    ensure_dir(out_path.parent)
    arr = softmax(scores_or_probs, temperature=temperature) if is_logits else np.asarray(scores_or_probs, dtype=np.float64)
    rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in arr:
            writer.writerow(_format_prob_row(row))
            rows += 1
    return validate_csv(out_path, expected_rows=rows)


def make_result_zip(dataset1_csv: Path, dataset2_csv: Path, zip_path: Path) -> None:
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(dataset1_csv, arcname="dataset1.csv")
        zf.write(dataset2_csv, arcname="dataset2.csv")


def validate_csv(path: Path, expected_rows: Optional[int] = None) -> dict:
    rows = 0
    bad = 0
    sum_min = float("inf")
    sum_max = float("-inf")
    min_val = float("inf")
    max_val = float("-inf")
    first_bad: Optional[int] = None
    with path.open(newline="", encoding="utf-8") as f:
        for rows, row in enumerate(csv.reader(f), start=1):
            try:
                vals = [float(x) for x in row]
            except ValueError:
                vals = []
            if len(vals) != 100:
                bad += 1
                first_bad = first_bad or rows
                continue
            total = float(sum(vals))
            sum_min = min(sum_min, total)
            sum_max = max(sum_max, total)
            min_val = min(min_val, min(vals))
            max_val = max(max_val, max(vals))
            if min(vals) < -1e-12 or max(vals) > 1.0 + 1e-12 or abs(total - 1.0) > 1e-5:
                bad += 1
                first_bad = first_bad or rows
    if expected_rows is not None and rows != expected_rows:
        bad += abs(rows - expected_rows)
    return {
        "path": str(path),
        "rows": rows,
        "expected_rows": expected_rows,
        "bad": bad,
        "first_bad": first_bad,
        "sum_min": None if rows == 0 else sum_min,
        "sum_max": None if rows == 0 else sum_max,
        "min_val": None if rows == 0 else min_val,
        "max_val": None if rows == 0 else max_val,
    }


def top1_change(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return 0.0
    return float(np.mean(np.argmax(a, axis=1) != np.argmax(b, axis=1)))


def score_summary(scores_or_probs: np.ndarray, candidates: Optional[Sequence[TestRow]] = None) -> dict:
    arr = np.asarray(scores_or_probs)
    top_idx = np.argmax(arr, axis=1)
    top_vals = arr[np.arange(len(arr)), top_idx]
    out = {
        "rows": int(arr.shape[0]),
        "cols": int(arr.shape[1]) if arr.ndim == 2 else None,
        "top_prob_mean": float(np.mean(top_vals)),
        "top_prob_p95": float(np.percentile(top_vals, 95)),
        "entropy_mean": float(np.mean(-np.sum(np.clip(arr, 1e-12, 1.0) * np.log(np.clip(arr, 1e-12, 1.0)), axis=1))),
    }
    if candidates is not None:
        top_dst = [row.candidates[int(i)] for row, i in zip(candidates, top_idx)]
        out["top_dst_unique"] = int(len(set(top_dst)))
    return out


def sample_evenly(n: int, limit: int) -> np.ndarray:
    limit = min(int(limit), int(n))
    if limit <= 0:
        return np.asarray([], dtype=np.int64)
    if limit >= n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, limit, dtype=np.int64)
