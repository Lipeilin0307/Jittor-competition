# -*- coding: utf-8 -*-
"""blend_emit.py -- emit blended submission csv from two rank matrices (fast byte writer).

Blend values live on an exact 1e-4 grid (ranks are i/100; weights are exact multiples of 0.05),
so %.8f text is exact. Byte-level vectorized writer: ~1 min for 2.32M x 100.

Usage:
  weighted: python blend_emit.py --a v3_rank.npy --b craft_rank.npy --w 0.65 --out dataset4_ens_w65.csv
  gated:    python blend_emit.py --a v3_rank.npy --b craft_rank.npy --gate_srcs test_srcs.npy ^
              --gate_cands test_cands.npy --gate_keys pair_keys.npy --out dataset4_ens_gate.csv
"""
import argparse
import time

import numpy as np


def fast_write(out_f32: np.ndarray, path: str):
    """out_f32 in [0.01, 1.0] on a 1e-4 grid. Writes %.8f rows, comma-joined."""
    n, k = out_f32.shape
    v = np.rint(out_f32.astype(np.float64) * 1e4).astype(np.int32)  # [1..10000]
    row_bytes = k * 10 + (k - 1) + 1  # "0.dddddddd"*100 + commas + \n
    buf = np.empty((n, row_bytes), dtype=np.uint8)
    c0 = (v >= 10000)
    d1 = (v % 10000) // 1000
    d2 = (v % 1000) // 100
    d3 = (v % 100) // 10
    d4 = v % 10
    for j in range(k):
        b = j * 11  # 10 chars + comma; last element: 10 chars, then \n at row end
        buf[:, b + 0] = np.where(c0[:, j], 49, 48)
        buf[:, b + 1] = 46
        buf[:, b + 2] = d1[:, j] + 48
        buf[:, b + 3] = d2[:, j] + 48
        buf[:, b + 4] = d3[:, j] + 48
        buf[:, b + 5] = d4[:, j] + 48
        buf[:, b + 6:b + 10] = 48
        if j < k - 1:
            buf[:, b + 10] = 44  # comma
    buf[:, -1] = 10  # newline
    with open(path, 'wb') as f:
        f.write(buf.tobytes())
    return row_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--c', default=None, help='第三方秩矩阵（可选）')
    ap.add_argument('--w', type=float, default=None)
    ap.add_argument('--wb', type=float, default=None, help='三方时 b 的权重')
    ap.add_argument('--wc', type=float, default=None, help='三方时 c 的权重')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gate_srcs', default=None)
    ap.add_argument('--gate_cands', default=None)
    ap.add_argument('--gate_keys', default=None)
    args = ap.parse_args()

    t0 = time.time()
    ra = np.load(args.a, mmap_mode='r')
    rb = np.load(args.b, mmap_mode='r')
    n, k = ra.shape
    print(f'ranks loaded: {n}x{k} ({time.time()-t0:.0f}s)')

    if args.gate_keys is not None:
        srcs = np.load(args.gate_srcs)
        cands = np.load(args.gate_cands)
        keys = np.load(args.gate_keys)
        cand_keys = (srcs[:, None] << 23) | cands
        seen = np.isin(cand_keys, keys)
        print(f'gate: seen {seen.sum()}/{seen.size} ({seen.mean()*100:.2f}%) ({time.time()-t0:.0f}s)')
        blend = np.where(seen, ra, rb).astype(np.float32)
        out = blend.copy()
    else:
        assert args.w is not None
        w = np.float32(args.w)
        if args.c is not None:
            assert args.wb is not None and args.wc is not None, '三方混合需 --wb 和 --wc'
            wb, wc = np.float32(args.wb), np.float32(args.wc)
            assert abs(float(w + wb + wc) - 1.0) < 1e-6, f'权重和须为1: {w}+{wb}+{wc}'
            rc = np.load(args.c, mmap_mode='r')
            blend = (w * ra + wb * rb + wc * rc).astype(np.float32)
            print(f'3-way blend: {w}*a + {wb}*b + {wc}*c')
        else:
            blend = (w * ra + (np.float32(1.0) - w) * rb).astype(np.float32)
        out = blend  # already in [0.01, 1.0] on the 1e-4 grid

    print(f'blend done ({time.time()-t0:.0f}s), range=[{out.min():.4f}, {out.max():.4f}]')
    rb_ = fast_write(out, args.out)
    print(f'saved {args.out} (row_bytes={rb_}), total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
