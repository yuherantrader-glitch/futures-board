"""基差过滤全期对照：有基差 vs 无基差（2005-05 ~ 2026-10，同 long05 数据与引擎，v2.7 冻结 clean）。

  python tools/basis05.py --data hist/all --out site/bt/basis05 --start 2005-01-01 --basis-start 2012-06-01

A 引擎：V27 / ECB × 有基差 / 无基差 × 固定 2% / 2%15% 复利 / 盈利升档；分段 2005–2012（无基差数据，应一致）、2013–2016（基差规则样本外）、
  2017 起（基差规则样本内），另给分年。
B 每信号：顺期限结构、强席位不反向的合格候选（每日前 5 名），按“基差不利”分组，T2 + 3ATR 移动止损每笔 R（与 vstats 同口径，扣 0.03R），
  分 2013–2016 / 2017 起 / 分年 / 多空；只计新信号（前一日不同向在列）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tools'))
import v27_repro as VR  # noqa: E402
import robust05 as R5  # noqa: E402
import astra05 as A5  # noqa: E402
import vstats as VS  # noqa: E402
from ftrade import kline, tracker  # noqa: E402
import backtest as bt  # noqa: E402
import lagtest as LG  # noqa: E402
import optimize as op  # noqa: E402

log = VR.log
SEGS = [('2005-2012', '0000', '2013-01-01'), ('2013-2016', '2013-01-01', '2017-01-01'), ('2017起', '2017-01-01', '9999')]


def segstats(E, comp):
    out = {}
    for nm, a, b in SEGS:
        xs = [e for e in E if a <= e['date'] < b]
        if len(xs) < 2:
            continue
        prev = [e for e in E if e['date'] < a]
        e0 = prev[-1]['equity'] if prev else 1e6
        eq = np.array([e0] + [e['equity'] for e in xs])
        pk = np.maximum.accumulate(eq)
        if comp:
            out[nm] = dict(年化=round(((eq[-1] / e0) ** (244 / len(xs)) - 1) * 100, 1), 回撤=round(float((1 - eq / pk).max()) * 100, 1))
        else:
            out[nm] = dict(万每年=round((eq[-1] - e0) / 1e4 / (len(xs) / 244), 1), 回撤占本金=round(float((pk - eq).max()) / 1e4, 1))
    return out


def rstats(xs):
    if not xs:
        return {'n': 0}
    r = np.array(xs)
    bs = [np.mean(np.random.choice(r, len(r))) for _ in range(1000)] if len(r) > 5 else [0, 0]
    return dict(n=len(r), 每笔R=round(float(r.mean()), 3), 胜率=round(float((r > 0).mean()) * 100, 1), 合计=round(float(r.sum()), 1),
                区间95=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/basis05')
    ap.add_argument('--start', default='2005-01-01')
    ap.add_argument('--end', default='2026-10-31')
    ap.add_argument('--basis-start', default='2012-06-01')
    ap.add_argument('--cachedir', default='hist')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    np.random.seed(7)
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
    log(f'准备完成（{time.time() - t00:.0f}s），基差覆盖 {blow.cover}')
    X0 = dict(LG.V25, **LG.REAL, noWeak=True)
    nb = A5.no_basis()
    out = {'基差覆盖': blow.cover}

    # ---------- A 引擎 ----------
    eng = {}
    for lab, use in (('V27', True), ('ECB', False)):
        for bl_nm, bl in (('有基差', blow), ('无基差', nb)):
            S, _ = VR.make_S(data, carry, bl, seat, use)
            for sl, sz, comp in (('固定2%', ('fixed', 0.02, 0.10, False), False), ('2%15%复利', ('fixed', 0.02, 0.15, True), True),
                                 ('盈利升档', ('ladder',), True)):
                st, E = A5.run3(data, S, sz, X0)
                m = A5.summ(E, st, comp)
                eng[f'{lab}·{bl_nm}·{sl}'] = {k: m[k] for k in ('终值万', '年化', '最大回撤', '笔', 'R净', '合计R')} | {'分段': segstats(E, comp), '分年': m['分年']}
                log(f'{lab}·{bl_nm}·{sl}：{ {k: v for k, v in eng[f"{lab}·{bl_nm}·{sl}"].items() if k != "分年"} }')
    out['A_引擎'] = eng

    # ---------- B 每信号 ----------
    idx = {v: {x['d']: i for i, x in enumerate(k)} for v, k in ADJ.items()}
    prev, rec = set(), []
    for D in data['days']:
        rows = data['R'].get(D, [])
        c = sorted([r for r in rows if VR.elig(r)], key=lambda r: (-r['nbig'], -VR.conf(r)))[:5]
        now = set()
        for r in c:
            v, dr = r['v'], r['dir']
            ca = carry(v, D)
            if ca is not None and ca != 0 and (ca > 0) != (dr == '多'):
                continue
            now.add((v, dr))
            if (v, dr) in prev:
                continue
            p = blow.pct(v, D)
            if p is None:
                continue
            k, i = ADJ.get(v), idx.get(v, {}).get(D)
            if k is None or i is None or not VS.ma5_ok(k, i, dr):
                continue
            x = VS.trade(k, i, dr)
            if x is None:
                continue
            rec.append(dict(d=D, v=v, dir=dr, R=x, low=bool(blow(v, D, dr)), pct=round(p, 3), seatAg=bool(seat(v, D, dr))))
        prev = now
    sig = {}
    for nm, a_, b_ in SEGS[1:] + [('全部', '0000', '9999')]:
        L = [x for x in rec if a_ <= x['d'] < b_]
        lo, ok = [x['R'] for x in L if x['low']], [x['R'] for x in L if not x['low']]
        g = dict(基差不利_被挡=rstats(lo), 基差不挡=rstats(ok))
        if lo and ok:
            d = np.mean(ok) - np.mean(lo)
            bs = [np.mean(np.random.choice(ok, len(ok))) - np.mean(np.random.choice(lo, len(lo))) for _ in range(1000)]
            g['差_不挡减被挡'] = dict(差=round(float(d), 3), 区间95=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])
        for dr in ('多', '空'):
            g[f'{dr}·被挡'] = rstats([x['R'] for x in L if x['low'] and x['dir'] == dr])
            g[f'{dr}·不挡'] = rstats([x['R'] for x in L if not x['low'] and x['dir'] == dr])
        g['强席位不反向·被挡'] = rstats([x['R'] for x in L if x['low'] and not x['seatAg']])
        g['强席位不反向·不挡'] = rstats([x['R'] for x in L if not x['low'] and not x['seatAg']])
        sig[nm] = g
    yr = {}
    for y in sorted({x['d'][:4] for x in rec}):
        L = [x for x in rec if x['d'][:4] == y]
        lo, ok = [x['R'] for x in L if x['low']], [x['R'] for x in L if not x['low']]
        yr[y] = dict(被挡=f"{len(lo)} 笔 {np.mean(lo):+.3f}" if lo else '0', 不挡=f"{len(ok)} 笔 {np.mean(ok):+.3f}" if ok else '0',
                     差=round(float(np.mean(ok) - np.mean(lo)), 3) if lo and ok else None)
    sig['分年'] = yr
    # 分位分组（不利方向分位：多看 pct，空看 1-pct）
    qb = {}
    for lo_, hi_ in ((0, 0.2), (0.2, 1 / 3), (1 / 3, 0.5), (0.5, 0.67), (0.67, 0.8), (0.8, 1.01)):
        L = [x['R'] for x in rec if lo_ <= (x['pct'] if x['dir'] == '多' else 1 - x['pct']) < hi_]
        qb[f'{lo_:.2f}-{hi_:.2f}'] = rstats(L)
    sig['按顺向基差分位'] = qb
    sig['说明'] = '新信号 = 合格候选（每日前 5、顺期限结构）当天在列、前一日不同向在列；需有基差分位、顺 5 日线；T2 + 只靠 3ATR 移动止损，最长 60 根，扣 0.03R。被挡 = 顺向基差处在过去 250 张表的低三分之一。'
    out['B_每信号'] = sig
    json.dump(out, open(os.path.join(a.out, 'basis05.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('# 基差过滤 全期对照\n\n```\n' + json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n```\n')
    log(f'全部完成（{time.time() - t00:.0f}s）')


if __name__ == '__main__':
    main()
