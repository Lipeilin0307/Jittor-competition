# -*- coding: utf-8 -*-
"""emit_nway.py -- N-way weighted rank blend -> submission csv.
Reuses fast_write from blend_emit.py (values on 1e-4 grid, %.8f exact text).
Usage: python emit_nway.py --members a.npy,b.npy,c.npy --weights 0.45,0.15,0.40 --out out.csv
"""
import argparse
import time

import numpy as np

from blend_emit import fast_write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', required=True, help='comma-separated rank .npy paths')
    ap.add_argument('--weights', required=True, help='comma-separated weights, must sum to 1')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    paths = [p.strip() for p in args.members.split(',')]
    ws = [np.float32(float(x)) for x in args.weights.split(',')]
    assert len(paths) == len(ws) >= 2, 'members/weights count mismatch'
    s = float(sum(ws))
    assert abs(s - 1.0) < 1e-6, f'weights must sum to 1, got {s}'

    t0 = time.time()
    blend = None
    for p, w in zip(paths, ws):
        r = np.load(p, mmap_mode='r')
        term = (w * r).astype(np.float32)
        blend = term if blend is None else (blend + term)
        print(f'  + {float(w):.4f} * {p} ({time.time()-t0:.0f}s)', flush=True)
    blend = blend.astype(np.float32)
    print(f'blend range=[{blend.min():.4f}, {blend.max():.4f}] ({time.time()-t0:.0f}s)')
    rb = fast_write(blend, args.out)
    print(f'saved {args.out} (row_bytes={rb}), total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
