"""第三步：将军（席位）与关键环节供需数据的关系（预登记见项目文档《候选假设-信息时效与持续性第三步》）。

  python tools/keylink05.py --data hist/all --out site/bt/keylink05 --start 2005-01-01 --basis-start 2012-06-01

供需数据：research/keylink_x.txt（供需失衡分支 v5/keylink.py 生成的关键环节失衡度 x，正 = 偏紧）。
品种：P、Y、OI ← MPOB；A、B、M ← 美国大豆；C、CS ← 美国玉米；CF ← 中国棉花；RM ← 加拿大菜籽。
T1 谁领先：公布前后 20 个交易日将军净持仓变化方向，与公布值变化方向（Δx）是否一致；用价格同期变化作对照。
T2 增量：v2.7（上线口径）实际成交按“最新供需状态与方向的关系”分组看每笔 R。
T3 持有：同一批信号，60 日上限 vs 120 日的每笔 R 差，按供需同向且偏离 ≥15% 与否分组。
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tools'))
import v27_repro as VR  # noqa: E402
import gen05 as G5  # noqa: E402
import persist05 as P5  # noqa: E402
import vstats as VS  # noqa: E402
from ftrade import kline, tracker  # noqa: E402
import backtest as bt  # noqa: E402
import lagtest as LG  # noqa: E402
import optimize as op  # noqa: E402

log = VR.log
SRC = {'P': 'MPOB', 'Y': 'MPOB', 'OI': 'MPOB', 'A': 'US_SOY', 'B': 'US_SOY', 'M': 'US_SOY', 'C': 'US_CORN', 'CS': 'US_CORN', 'CF': 'COTTON', 'RM': 'MM'}
GRP = {'P': '油脂', 'Y': '油脂', 'OI': '油脂', 'A': '大豆', 'B': '大豆', 'M': '大豆', 'C': '玉米', 'CS': '玉米', 'CF': '棉花', 'RM': '菜粕'}


def load_x(fn):
    X = {}
    for part in open(fn, encoding='utf-8').read().strip().split('|'):
        k, body = part.split(':', 1)
        X[k] = [(f'20{d[:2]}-{d[2:4]}-{d[4:6]}', float(x)) for d, x in (z.split(',') for z in body.split(';'))]
    return X


def wilson(k, n):
    if not n:
        return None
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(p * 100, 1), round((c - h) * 100, 1), round((c + h) * 100, 1), n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/keylink05')
    ap.add_argument('--start', default='2005-01-01')
    ap.add_argument('--end', default='2026-10-31')
    ap.add_argument('--basis-start', default='2012-06-01')
    ap.add_argument('--cachedir', default='hist')
    ap.add_argument('--x', default=os.path.join(HERE, 'research', 'keylink_x.txt'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    np.random.seed(5)
    t00 = time.time()
    tracker.MULT.update(bt.EXTRA_MULT)
    tracker.MULT.update({k: v for k, v in dict(WH=20, PM=50, RI=20, OI=10, ZC=100, MA=10).items() if k not in tracker.MULT})
    bars, H = bt.load([a.data])
    H = {d: x for d, x in H.items() if a.start <= d <= a.end}
    fp = VR.hashlib.sha256('|'.join(['long05', VR.sha(os.path.join(HERE, 'ftrade/v46.py')), VR.sha(os.path.join(HERE, 'tools/v27_repro.py')),
                                     a.start, a.end, str(len(H)), str(len(bars))]).encode()).hexdigest()[:12]
    R = VR.build_rows(bars, H, os.path.join(a.cachedir, f'rows_long_{fp}.json.gz'), a.end)
    days = sorted(H)
    RAW, ROLLS = bt.build_series(bars, days)
    ADJ = {v: kline.adjust(RAW[v], ROLLS[v]) for v in RAW}
    d0 = min(R)
    data = {'days': days[days.index(d0) - 1:], 'RAW': RAW, 'ADJ': ADJ, 'ROLLS': ROLLS, 'R': R}
    carry = VR.carry_fn(RAW, bars)
    blow = VR.basis_fetch_fn(a.basis_start, a.end, os.path.join(a.cachedir, 'basis_fetch_long.json.gz'))
    seat = VR.strong_fn(H, ADJ, 'clean', years=range(int(min(H)[:4]) + 2, int(a.end[:4]) + 1))
    S, _ = VR.make_S(data, carry, blow, seat, True)
    X0 = dict(LG.V25, **LG.REAL, noWeak=True)
    X = load_x(a.x)
    DD = data['days']
    DI = {D: i for i, D in enumerate(DD)}
    rowmap = {D: {r['v']: r for r in data['R'].get(D, [])} for D in DD}
    idx = {v: {x['d']: i for i, x in enumerate(k)} for v, k in ADJ.items()}
    log(f'准备完成（{time.time() - t00:.0f}s），供需序列 {[(k, len(v)) for k, v in X.items()]}')

    def g_at(v, D):
        r = rowmap.get(D, {}).get(v)
        if not r or r.get('gNet') is None:
            return None
        oi = r.get('oiRaw') or r.get('oi') or 0
        return r['gNet'] / oi if oi else None

    import bisect

    def eff_day(t):
        j = bisect.bisect_right(DD, t)
        return j if j < len(DD) else None
    EFF = {k: [(DD[j], x) for t, x in xs if (j := eff_day(t)) is not None] for k, xs in X.items()}

    # ---------- T1 谁领先 ----------
    rec1 = []
    for v, src in SRC.items():
        xs = X.get(src) or []
        k = ADJ.get(v)
        if k is None:
            continue
        for (t0, x0), (t, x) in zip(xs, xs[1:]):
            dx = x - x0
            if abs(dx) < 0.05:
                continue
            j = eff_day(t)
            if j is None or j < 21 or j + 19 >= len(DD):
                continue
            s = 1 if dx > 0 else -1
            ga, gb, gc = g_at(v, DD[j - 21]), g_at(v, DD[j - 1]), g_at(v, DD[j + 19])
            ia, ib, ic = idx.get(v, {}).get(DD[j - 21]), idx.get(v, {}).get(DD[j - 1]), idx.get(v, {}).get(DD[j + 19])
            r1 = rowmap.get(DD[j - 1], {}).get(v)
            e = dict(v=v, t=t, s=s, x=x)
            if ga is not None and gb is not None and gb != ga:
                e['g_pre'] = (gb - ga > 0) == (s > 0)
            if gb is not None and gc is not None and gc != gb:
                e['g_post'] = (gc - gb > 0) == (s > 0)
            if ia is not None and ib is not None and k[ib]['c'] != k[ia]['c']:
                e['p_pre'] = (k[ib]['c'] > k[ia]['c']) == (s > 0)
            if ib is not None and ic is not None and k[ic]['c'] != k[ib]['c']:
                e['p_post'] = (k[ic]['c'] > k[ib]['c']) == (s > 0)
            if r1 and (r1.get('ngen') or 0) >= 1 and abs(x) >= 0.15:
                e['lvl'] = (r1['dir'] == '多') == (x > 0)
            rec1.append(e)
    T1 = {}
    for nm, f in (('全部', lambda e: True), ('2019前(郑商所为主)', lambda e: e['t'] < '2019-01-01'), ('2019起', lambda e: e['t'] >= '2019-01-01'),
                  *[(g, (lambda g: lambda e: GRP[e['v']] == g)(g)) for g in sorted(set(GRP.values()))]):
        L = [e for e in rec1 if f(e)]
        T1[nm] = {key: wilson(sum(1 for e in L if e.get(key) is True), sum(1 for e in L if key in e))
                  for key in ('g_pre', 'g_post', 'p_pre', 'p_post', 'lvl')}
    out = {'T1_谁领先(一致率%,95%下限,上限,样本)': T1,
           'T1_说明': 'g_pre/g_post：公布前/后 20 个交易日将军净持仓变化方向与 Δx 一致；p_pre/p_post：同期价格；lvl：公布前一天将军站的方向与供需状态（|x|≥15%）一致。'}

    # ---------- T2 / T3 ----------
    st, E = P5.run_hold(data, S, X0, maxhold=60)
    real = [t for t in st['closed'] if not t.get('observe') and t['v'] in SRC and t.get('risk')]

    def xlast(v, D):
        xs = EFF.get(SRC[v]) or []
        j = bisect.bisect_right([d for d, _ in xs], D)
        return xs[j - 1][1] if j else None
    rec2 = []
    for t in real:
        x = xlast(t['v'], t['base'])
        if x is None:
            continue
        sg = 1 if t['dir'] == '多' else -1
        grp = '供需同向且偏离≥15%' if abs(x) >= 0.15 and x * sg > 0 else ('供需反向且偏离≥15%' if abs(x) >= 0.15 else '供需偏离<15%')
        k, i = ADJ.get(t['v']), idx.get(t['v'], {}).get(t['base'])
        r60 = VS.trade(k, i, t['dir'], 3.0, 60) if k is not None and i is not None else None
        r120 = VS.trade(k, i, t['dir'], 3.0, 120) if k is not None and i is not None else None
        rec2.append(dict(d=t['base'], v=t['v'], g=grp, R=(t['pnl'] - op.tcost(t)) / t['risk'],
                         dR=(r120 - r60) if r60 is not None and r120 is not None else None))
    T2, T3 = {}, {}
    for nm, s0, s1 in (('全部', '0000', '9999'), ('2019前', '0000', '2019-01-01'), ('2019起', '2019-01-01', '9999')):
        L = [x for x in rec2 if s0 <= x['d'] < s1]
        G = {g: [dict(d=x['d'], R=x['R']) for x in L if x['g'] == g] for g in ('供需同向且偏离≥15%', '供需偏离<15%', '供需反向且偏离≥15%')}
        T2[nm] = {g: G5.cstats(v) for g, v in G.items()} | {'差 反向−偏离<15%': G5.cdiff(G['供需反向且偏离≥15%'], G['供需偏离<15%']),
                                                          '差 同向−偏离<15%': G5.cdiff(G['供需同向且偏离≥15%'], G['供需偏离<15%'])}
        T3[nm] = {g: G5.cstats([dict(d=x['d'], R=x['dR']) for x in L if x['g'] == g and x['dR'] is not None])
                  for g in ('供需同向且偏离≥15%', '供需偏离<15%', '供需反向且偏离≥15%')}
    out['T2_v2.7成交按供需分组(每笔净R)'] = T2
    out['T3_120日减60日(每笔R差，只靠3ATR移动止损)'] = T3
    out['样本'] = dict(T1=len(rec1), T2=len(rec2), 品种成交={v: sum(1 for x in rec2 if x['v'] == v) for v in SRC})
    json.dump(out, open(os.path.join(a.out, 'keylink05.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('# 将军与关键环节供需\n\n```\n' + json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n```\n')
    log(f'全部完成（{time.time() - t00:.0f}s）')


if __name__ == '__main__':
    main()
