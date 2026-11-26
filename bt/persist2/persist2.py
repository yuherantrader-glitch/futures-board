"""信息时效与持续性 · 第二步（预登记见项目文档《候选假设-信息时效与持续性第二步》）。

  python tools/persist2.py --data hist/all --out site/bt/persist2 --start 2005-01-01 --basis-start 2012-06-01

E 前期放宽：开仓后前 N 个持有日不移动止损（初始止损 K×ATR，按 K×ATR 定手数，每笔风险仍 2%），之后回到 3ATR 移动止损。
P 浮盈加仓：持有满 10 日、浮盈 ≥1.5×ATR、当天仍同向且 ≥2 个将军 → 次日开盘加一份（风险 1% 或 2%，止损同原仓），总风险仍 ≤15%。
引擎 = 上线口径 B（tools/live27.run_live / persist05.run_hold），只改上述一处。
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
import astra05 as A5  # noqa: E402
import persist05 as P5  # noqa: E402
from ftrade import kline, tracker  # noqa: E402
import backtest as bt  # noqa: E402
import lagtest as LG  # noqa: E402
import optimize as op  # noqa: E402

log = VR.log
STEPS = P5.STEPS


def run_var(data, S, X0, early=0, k_init=3.0, add=None, rowmap=None):
    """early>0：前 early 个持有日不移动止损；k_init：初始止损倍数（也用于定手数）。
    add=(浮盈门槛ATR, 加仓风险)：持有满 10 日加一份。"""
    op.COMPOUND = True

    class LT4(LG.LT):
        def settle(self, D, st, touched):
            snap = {}
            if early:
                for p in st['open']:
                    if (p.get('age') or 0) < early and not p.get('isAdd'):
                        snap[id(p)] = (p['stopPx'], p['entry'])
            out = super().settle(D, st, touched)
            if snap:
                for p in st['open']:
                    s = snap.get(id(p))
                    if s:
                        p['stopPx'] = round(s[0] + (p['entry'] - s[1]), 2)
            return out

        def step(self, D, st, sigs, touched, today=False, weak=None):
            out = super().step(D, st, sigs, touched, today, weak)
            ex = {p['v']: p['exitNext'] for p in st['open'] if p.get('exitNext')}
            for p in st['open']:
                if p['v'] in ex and not p.get('exitNext'):
                    p['exitNext'] = ex[p['v']]
                if not p.get('pending'):
                    p['held'] = (p.get('age') or 0) + (p.get('ageOffset') or 0)
            if add:
                new = []
                heat = sum(p.get('budgetPct') or 0 for p in st['open'])
                for p in st['open']:
                    if p.get('pending') or p.get('isAdd') or p.get('added') or p.get('exitNext') or (p.get('age') or 0) != 10:
                        continue
                    p['added'] = True
                    sg = 1 if p['dir'] == '多' else -1
                    c, a = p.get('lastClose'), p.get('atr') or 0
                    r = (rowmap or {}).get(D, {}).get(p['v'])
                    if not c or not a or sg * (c - p['entry']) < add[0] * a or not r or r['dir'] != p['dir'] or (r.get('ngen') or 0) < 2:
                        continue
                    cap = self.cfg['capital']
                    pct = min(add[1], self.cfg['maxHeat'] - heat)
                    per = abs(c - p['stopPx']) * p['mult']
                    if pct < 0.003 or per <= 0:
                        continue
                    lots = math.floor(cap * pct / per)
                    if lots < 1:
                        continue
                    q = dict(p, id=p['id'] + '_add', base=D, entry=c, lots=lots, risk=round(lots * per), budget=round(cap * pct),
                             budgetPct=round(lots * per / cap, 4), pending=True, fillDate=None, age=0, held=10, ageOffset=10, peak=None,
                             pnl=0, R=0, isAdd=True, added=True, signals=[D + '加仓'], rolls=list(p.get('rolls') or []),
                             hardPx=p['stopPx'], lastDate=D, notional=round(c * lots * p['mult']))
                    new.append(q)
                    heat += q['budgetPct']
                st['open'] += new
            return out
    cfg = {'trail': 0, 'ma5': False, **X0, 'maxHold': 60, 'fixed': 0.02, 'stopATR': k_init}
    T = LT4(data['RAW'], data['ADJ'], data['ROLLS'], cfg)
    st = {'open': [], 'closed': [], 'from': data['days'][0]}
    E, last_eq, pk, lvl = [], 1e6, 1e6, 0
    for D in data['days'][1:]:
        sigs = [s for s in S[D]['sigs'] if op.ma5_ok(data['ADJ'], s['v'], D, s['dir'])]
        op.compound(T, st)
        T.cfg['capital'] = last_eq
        T.cfg['fixed'], T.cfg['maxHeat'] = 0.02, 0.15
        dd = 1 - last_eq / pk
        if dd >= STEPS[0][0]:
            lvl = 2
        elif dd >= STEPS[1][0]:
            lvl = max(lvl, 1)
        elif dd <= 0.05:
            lvl = 0
        if lvl:
            _, r, h = STEPS[0] if lvl == 2 else STEPS[1]
            T.cfg['fixed'], T.cfg['maxHeat'] = r, h
        T.step(D, st, [dict(s) for s in sigs], set(), today=True, weak={})
        real = [t for t in st['closed'] if not t.get('observe')]
        live = [p for p in st['open'] if not p.get('observe') and not p.get('pending')]
        eq = 1e6 + sum(t['pnl'] or 0 for t in real) + sum(p['pnl'] or 0 for p in live) - sum(op.tcost(t) for t in real)
        last_eq, pk = eq, max(pk, eq)
        E.append(dict(date=D, equity=eq, nOpen=len(live), notional=0, heat=0))
    pk = 1e6
    for e in E:
        pk = max(pk, e['equity'])
        e['peak'], e['dd'] = pk, 1 - e['equity'] / pk
    return st, E


def describe(st, E):
    m = A5.summ(E, st, True)
    real = [t for t in st['closed'] if not t.get('observe')]
    net = lambda t: t['pnl'] - op.tcost(t)
    los = [t for t in real if net(t) <= 0]
    adds = [t for t in real if t.get('isAdd')]
    y0, d0 = P5.seg_ann(E, '0000', '2017-01-01')
    y1, d1 = P5.seg_ann(E, '2017-01-01', '9999')
    return {k: m[k] for k in ('终值万', '年化', '最大回撤', '笔', 'R净')} | {
        '收益÷回撤': round(m['年化'] / m['最大回撤'], 3) if m['最大回撤'] else None,
        '2005-2016 年化/回撤': [y0, d0], '2017起 年化/回撤': [y1, d1],
        '胜率': round(100 - len(los) / len(real) * 100, 1) if real else None,
        '亏损单持仓中位天': float(np.median([t.get('age') or 0 for t in los])) if los else None,
        '10日内止损 笔/净万': [sum(1 for t in los if (t.get('age') or 0) <= 10), round(sum(net(t) for t in los if (t.get('age') or 0) <= 10) / 1e4, 1)],
        '加仓 笔/净万': [len(adds), round(sum(net(t) for t in adds) / 1e4, 1)], '分年': m['分年']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/persist2')
    ap.add_argument('--start', default='2005-01-01')
    ap.add_argument('--end', default='2026-10-31')
    ap.add_argument('--basis-start', default='2012-06-01')
    ap.add_argument('--cachedir', default='hist')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
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
    rowmap = {D: {r['v']: r for r in data['R'].get(D, [])} for D in data['days']}
    log(f'准备完成（{time.time() - t00:.0f}s）')
    V = [('B0 现行', {}),
         ('E4-10 前10日4ATR不移动', dict(early=10, k_init=4.0)),
         ('E3-10 前10日3ATR不移动', dict(early=10, k_init=3.0)),
         ('E4-5 前5日4ATR不移动', dict(early=5, k_init=4.0)),
         ('P1 第10日浮盈≥1.5ATR加1%', dict(add=(1.5, 0.01))),
         ('P2 第10日浮盈≥1.5ATR加2%', dict(add=(1.5, 0.02)))]
    out, res = {}, {}
    for nm, kw in V:
        st, E = run_var(data, S, X0, rowmap=rowmap, **kw)
        res[nm] = describe(st, E)
        log(f"{nm}：{ {k: v for k, v in res[nm].items() if k != '分年'} }")
    base = res['B0 现行']

    def judge(x):
        ok = [x['2005-2016 年化/回撤'][0] >= base['2005-2016 年化/回撤'][0], (x['收益÷回撤'] or 0) >= (base['收益÷回撤'] or 0) + 0.03,
              x['2017起 年化/回撤'][0] >= base['2017起 年化/回撤'][0] - 1, x['最大回撤'] <= base['最大回撤'] + 2]
        return ok, all(ok)
    for nm, x in res.items():
        if nm.startswith('B0'):
            continue
        x['通过标准①②③④'], x['通过'] = judge(x)
    passE = [n for n, x in res.items() if n.startswith('E') and x.get('通过')]
    passP = [n for n, x in res.items() if n.startswith('P') and x.get('通过')]
    if passE and passP:
        be = max(passE, key=lambda n: res[n]['收益÷回撤'])
        bp = max(passP, key=lambda n: res[n]['收益÷回撤'])
        kw = dict(dict(V)[be], **dict(V)[bp])
        st, E = run_var(data, S, X0, rowmap=rowmap, **kw)
        x = describe(st, E)
        x['通过标准①②③④'], x['通过'] = judge(x)
        res[f'组合 {be.split()[0]}+{bp.split()[0]}（仅供参考）'] = x
    out['结果'] = res
    out['说明'] = '通过标准：①2005–2016 年化 ≥ 现行；②全期收益÷回撤 ≥ 现行 + 0.03；③2017 起年化不低于现行 1 个百分点以上；④最大回撤 ≤ 现行 + 2 个百分点。'
    json.dump(out, open(os.path.join(a.out, 'persist2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('# 信息时效与持续性 第二步\n\n```\n' + json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n```\n')
    log(f'全部完成（{time.time() - t00:.0f}s）')


if __name__ == '__main__':
    main()
