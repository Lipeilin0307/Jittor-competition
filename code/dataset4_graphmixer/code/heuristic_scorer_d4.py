# -*- coding: utf-8 -*-
"""ds4 启发式打分器（零训练，第 4 集成成员）——行区间切片 + 分块流式

信号（全部按 dst 全局统计，行内排序用）：
  pop      : train.csv dst 频次 × exp 时间衰减（openjittor 配方；ds4 train 跨度仅
             ~2 天，衰减近似退化为纯频次）
  cand_freq: dst 在 test.csv c1..c100 的出现次数（测试期热度，对 UNK 候选也有信号）
  cf_tau   : cand_freq 的时间衰减版（τ=60 天）

分片用法（单次运行被 290s 限时，切成两片跑）：
  python heuristic_scorer_d4.py --stage scan --row_start 0       --row_end 1300000
  python heuristic_scorer_d4.py --stage scan --row_start 1300000 --row_end 2322538
  python heuristic_scorer_d4.py --stage score [--no_csv]
小数据调试：--stage all
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

DAY = 86400.0
CHUNK = 200_000


def zscore(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std())
    return (x - mu) / (sd if sd > 1e-12 else 1.0)


def cache_dir_of(out: str) -> str:
    return os.path.splitext(os.path.abspath(out))[0] + '_cache'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=r'F:\download\data_B')
    ap.add_argument('--dataset', default='dataset4')
    ap.add_argument('--out', default=r'D:\work_d4\dataset4_heuristic.csv')
    ap.add_argument('--stage', choices=['scan', 'score', 'all'], default='all')
    ap.add_argument('--row_start', type=int, default=0)
    ap.add_argument('--row_end', type=int, default=-1)
    ap.add_argument('--pop_window_days', type=float, default=90.0)
    ap.add_argument('--pop_tau_days', type=float, default=30.0)
    ap.add_argument('--cf_tau_days', type=float, default=60.0)
    ap.add_argument('--w_pop', type=float, default=1.0)
    ap.add_argument('--w_cand', type=float, default=1.0)
    ap.add_argument('--w_cf', type=float, default=1.0)
    ap.add_argument('--no_csv', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    cache = cache_dir_of(args.out)
    os.makedirs(cache, exist_ok=True)

    if args.stage in ('scan', 'all'):
        train_path = os.path.join(args.data_dir, args.dataset, 'train.csv')
        test_path = os.path.join(args.data_dir, args.dataset, 'test.csv')

        # ---------- 1) train 侧 pop ----------
        tr = pd.read_csv(train_path, usecols=['dst', 'time'])
        tr_time = tr['time'].to_numpy(np.float64)
        t_max = float(tr_time.max())
        span_days = (t_max - float(tr_time.min())) / DAY
        m = tr_time >= (t_max - args.pop_window_days * DAY)
        dst_in_win = tr['dst'].to_numpy(np.int64)[m].astype(np.int32)
        w_pop = np.exp(-(t_max - tr_time[m]) / (args.pop_tau_days * DAY)).astype(np.float32)
        del tr, tr_time
        print(f'[load] train rows_in_window={len(w_pop)} span={span_days:.2f}d ({time.time()-t0:.0f}s)', flush=True)

        # ---------- 2) test 候选矩阵切片（int32） ----------
        te = pd.read_csv(test_path)
        cand_cols = [c for c in te.columns if c.startswith('c')]
        assert len(cand_cols) == 100, f'expect 100 candidate cols, got {len(cand_cols)}'
        n_total = len(te)
        row_start = max(0, args.row_start)
        row_end = n_total if args.row_end < 0 else min(args.row_end, n_total)
        assert row_start < row_end, f'empty slice [{row_start},{row_end})'
        cand_mat = te[cand_cols].to_numpy(np.int32, copy=True)   # 全量 (Q,100)，词典必须全量构建
        row_time_full = te['time'].to_numpy(np.float64)
        t_te_max = float(row_time_full.max())
        t_te_min = float(row_time_full.min())
        row_time = row_time_full[row_start:row_end]
        del te, row_time_full
        n_rows = row_end - row_start
        print(f'[load] test rows=[{row_start},{row_end}) of {n_total} '
              f'test_span={(t_te_max-t_te_min)/DAY:.2f}d ({time.time()-t0:.0f}s)', flush=True)

        # ---------- 3) dst 词典（全量候选，保证各分片一致） ----------
        head_check = cand_mat[row_start:row_start + 64].copy()
        uniq = np.unique(np.concatenate([dst_in_win, cand_mat.ravel()]))
        print(f'[dict] unique dsts={len(uniq)} ({time.time()-t0:.0f}s)', flush=True)

        U = len(uniq)
        # id->稠密位 直查表：候选 id 是有界小整数（<21 亿），查表比 searchsorted 快一个量级
        max_id = int(uniq[-1])
        assert max_id < 100_000_000, f'id too large for direct map: {max_id}'
        id2pos = np.empty(max_id + 1, dtype=np.int32)
        id2pos[uniq] = np.arange(U, dtype=np.int32)
        pop = np.bincount(id2pos[dst_in_win], weights=w_pop, minlength=U).astype(np.float32)
        np.save(os.path.join(cache, 'pop.npy'), pop)   # 各分片 pop 相同，重复存覆盖即可
        del dst_in_win, w_pop

        # ---------- 4) 分块统计（只扫本切片行） ----------
        cand_pos = np.empty((n_rows, 100), dtype=np.int32)
        cand_freq = np.zeros(U, dtype=np.float64)
        cf_tau = np.zeros(U, dtype=np.float64)
        for s in range(0, n_rows, CHUNK):
            e = min(s + CHUNK, n_rows)
            ci = id2pos[cand_mat[row_start + s:row_start + e].ravel()]
            cand_pos[s:e] = ci.reshape(e - s, 100)
            cand_freq += np.bincount(ci, minlength=U)
            rw = np.exp(-(t_te_max - row_time[s:e]) / (args.cf_tau_days * DAY)).astype(np.float32)
            cf_tau += np.bincount(ci, weights=np.repeat(rw, 100), minlength=U)
            print(f'[scan] {row_start+e}/{row_end} ({time.time()-t0:.0f}s)', flush=True)
        del cand_mat
        assert (uniq[cand_pos[:64]].astype(np.int32) == head_check).all(), 'cand_pos mapping mismatch'
        del head_check

        np.save(os.path.join(cache, f'cand_pos_{row_start}_{row_end}.npy'), cand_pos)
        np.save(os.path.join(cache, f'cand_freq_{row_start}_{row_end}.npy'), cand_freq.astype(np.float32))
        np.save(os.path.join(cache, f'cf_tau_{row_start}_{row_end}.npy'), cf_tau.astype(np.float32))
        print(f'[cache] slice [{row_start},{row_end}) saved ({time.time()-t0:.0f}s)', flush=True)
        if args.stage == 'scan':
            print('[done] stage=scan', flush=True)
            return

    # ---------- 5) 合并分片 + 出分 ----------
    pos_parts = sorted(glob.glob(os.path.join(cache, 'cand_pos_*_*.npy')))
    assert pos_parts, f'no cand_pos parts in {cache}'
    cand_pos = np.concatenate([np.load(p) for p in pos_parts], axis=0)
    n_rows = cand_pos.shape[0]
    cand_freq32 = np.zeros_like(np.load(pos_parts[0].replace('cand_pos', 'cand_freq'), mmap_mode='r'))
    cf_tau32 = np.zeros_like(cand_freq32)
    for p in pos_parts:
        cand_freq32 += np.load(p.replace('cand_pos', 'cand_freq'))
        cf_tau32 += np.load(p.replace('cand_pos', 'cf_tau'))
    pop = np.load(os.path.join(cache, 'pop.npy'))
    print(f'[merge] parts={len(pos_parts)} rows={n_rows} ({time.time()-t0:.0f}s)', flush=True)

    score_universe = (args.w_pop * zscore(np.log1p(pop))
                      + args.w_cand * zscore(np.log1p(cand_freq32))
                      + args.w_cf * zscore(np.log1p(cf_tau32))).astype(np.float32)

    S = np.empty((n_rows, 100), dtype=np.float32)
    for s in range(0, n_rows, CHUNK):
        e = min(s + CHUNK, n_rows)
        B = score_universe[cand_pos[s:e]]
        rmin = B.min(axis=1, keepdims=True)
        rmax = B.max(axis=1, keepdims=True)
        const = (rmax - rmin) <= 1e-12
        span = np.where(const, 1.0, rmax - rmin)
        S[s:e] = np.where(const, 0.5, 0.01 + 0.99 * (B - rmin) / span)
        print(f'[score] {e}/{n_rows} ({time.time()-t0:.0f}s)', flush=True)
    S = np.round(S, 4).astype(np.float32)

    cov_pop = float((pop[cand_pos] > 0).mean())
    cov_cand = float((cand_freq32[cand_pos] > 0).mean())
    const_rows = int((S == 0.5).all(axis=1).sum())
    print(f'[cov] pop>0: {cov_pop:.4f}  cand_freq>0: {cov_cand:.4f}  const_rows: {const_rows}', flush=True)
    print(f'[stats] shape={S.shape} range=[{S.min():.4f},{S.max():.4f}] ({time.time()-t0:.0f}s)', flush=True)

    # ---------- 6) 保存 ----------
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.splitext(args.out)[0] + '.npy'
    np.save(npy_path, S)
    print(f'[save] {npy_path} ({os.path.getsize(npy_path)/1e9:.2f} GB)', flush=True)
    if args.no_csv:
        print(f'[done] --no_csv, total {time.time()-t0:.0f}s', flush=True)
        return
    t1 = time.time()
    for i, s in enumerate(range(0, n_rows, CHUNK)):
        e = min(s + CHUNK, n_rows)
        block = np.char.mod('%.4f', S[s:e])
        pd.DataFrame(block).to_csv(args.out, header=False, index=False, mode='w' if i == 0 else 'a')
        print(f'[csv] {e}/{n_rows} ({time.time()-t1:.0f}s)', flush=True)
    print(f'[save] {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB, total {time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
