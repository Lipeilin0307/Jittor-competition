# -*- coding: utf-8 -*-
"""rank_prep.py -- convert a 2322538x100 score csv to row-wise ordinal percentile ranks (.npy).
Usage: python rank_prep.py --src <csv> --out <npy>
Ranks are per-row: highest score -> 1.0, lowest -> ~0.01 (rank/100). Ties broken by column order (stable).
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.src.lower().endswith('.npy'):
        v = np.load(args.src).astype(np.float32, copy=False)
        print(f'loaded {args.src}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]')
    else:
        df = pd.read_csv(args.src, header=None, dtype=np.float32)
        v = df.to_numpy()
        print(f'loaded {args.src}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]')
    n, k = v.shape

    # ordinal ranks: order = argsort ascending; rank position 0..k-1 -> (pos+1)/k
    order = np.argsort(v, axis=1, kind='stable')
    ranks = np.empty((n, k), dtype=np.float32)
    rows = np.arange(n, dtype=np.int64)[:, None]
    ranks[rows, order] = (np.arange(1, k + 1, dtype=np.float32) / k)[None, :]
    np.save(args.out, ranks)
    print(f'saved {args.out}: shape={ranks.shape}, range=[{ranks.min():.4f}, {ranks.max():.4f}]')


if __name__ == '__main__':
    main()
