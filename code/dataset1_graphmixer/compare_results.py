"""
compare_results.py
==================
Compare two submission files (dataset1.csv format: Q rows x 100 cols, no header).
Used to verify that a re-run reproduces the A-list submission.

Usage:
  python compare_results.py --a path/to/old_dataset1.csv --b path/to/new_dataset1.csv

Reported metrics:
  - shape check
  - top-1 candidate agreement rate (per-row argmax)
  - top-3 candidate set agreement (Jaccard, averaged)
  - Pearson correlation of scores (sampled rows)
  - mean absolute difference
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=str, required=True, help='reference csv (A-list submission)')
    parser.add_argument('--b', type=str, required=True, help='reproduced csv')
    parser.add_argument('--sample_rows', type=int, default=20000)
    args = parser.parse_args()

    print(f'Loading {args.a} ...')
    a = np.loadtxt(args.a, delimiter=',')
    print(f'Loading {args.b} ...')
    b = np.loadtxt(args.b, delimiter=',')

    print(f'shape a={a.shape}, b={b.shape}')
    assert a.shape == b.shape, 'Shape mismatch!'

    # top-1 agreement
    top1_a = a.argmax(axis=1)
    top1_b = b.argmax(axis=1)
    top1_agree = (top1_a == top1_b).mean()

    # top-3 set Jaccard
    top3_a = np.argsort(-a, axis=1)[:, :3]
    top3_b = np.argsort(-b, axis=1)[:, :3]
    jac = []
    for ra, rb in zip(top3_a, top3_b):
        sa, sb = set(ra.tolist()), set(rb.tolist())
        jac.append(len(sa & sb) / len(sa | sb))
    top3_jaccard = float(np.mean(jac))

    # Pearson correlation on sampled rows
    rng = np.random.default_rng(0)
    rows = rng.choice(len(a), size=min(args.sample_rows, len(a)), replace=False)
    corrs = []
    for i in rows:
        ra, rb = a[i], b[i]
        if ra.std() > 0 and rb.std() > 0:
            corrs.append(np.corrcoef(ra, rb)[0, 1])
    pearson = float(np.mean(corrs))

    mad = float(np.abs(a - b).mean())

    print('\n===== comparison report =====')
    print(f'top-1 agreement      : {top1_agree * 100:.2f}%')
    print(f'top-3 Jaccard        : {top3_jaccard:.4f}')
    print(f'row Pearson corr     : {pearson:.4f}')
    print(f'mean abs diff        : {mad:.6f}')
    ok = top1_agree > 0.99 and pearson > 0.99
    print(f'\nverdict: {"PASS (reproduction consistent)" if ok else "CHECK (investigate differences)"}')


if __name__ == '__main__':
    main()
