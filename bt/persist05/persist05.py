"""信息时效与持续性 · 第一步（预登记见项目文档《候选假设-信息时效与持续性第一步》）。

  python tools/persist05.py --data hist/all --out site/bt/persist05 --start 2005-01-01 --basis-start 2012-06-01

A 信息衰减曲线：新信号（大将军≥3 / <3）与无信号对照，信号后第 h 天的顺向涨跌 ÷ ATR。
B 信息年龄：上线口径 B 的实际成交，按“将军已同向多少天”“价格已走多远”分档看每笔 R。
C 60 日上限：C0 / C90 / C120 / C∞ / Ccond（满 60 日仍同向且顺结构就继续拿）。
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
import astra05 as A5  # noqa: E402
import basis05 as B5  # noqa: E402
import gen05 as G5  # noqa: E402
import vstats as VS  # noqa: E402
from ftrade import kline, tracker  # noqa: E402
import backtest as bt  # noqa: E402
import lagtest as LG  # noqa: E402
import optimize as op  # noqa: E402

log = VR.log
HS = [1, 3, 5, 10, 20, 40, 60, 90, 120]
STEPS = [(0.15, 0.01, 0.10), (0.10, 0.015, 0.12)]


def fwd(k, i, dr):
    if i + 1 >= len(k):
        return None
    a = tracker.atr(k, i)
    if not a:
        return None
    sg = 1 if dr == '多' else -1
    e = k[i + 1]['o']
    return {h: sg * (k[i + h]['c'] - e) / a for h in HS if i + h < len(k)}


def run_hold(data, S, X0, maxhold=60, cond=None, cap=250):
    """= tools/live27.run_live（上线口径 B），只改持有上限。"""
    op.COMPOUND = True

    class LT3(LG.LT):
        def step(self, D, st, sigs, touched, today=False, weak=None):
            out = super().step(D, st, sigs, touched, today, weak)
            for p in st['open']:
                if not p.get('pending'):
                    p['held'] = p.get('age', 0)
                    if cond and maxhold - 1 <= p['held'] < cap - 1 and cond(p, D):
                        p['held'] = maxhold - 2
            return out
    T = LT3(data['RAW'], data['ADJ'], data['ROLLS'], {'trail': 0, 'ma5': False, **X0, 'maxHold': maxhold, 'fixed': 0.02})
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


def seg_ann(E, a, b):
    xs = [e for e in E if a <= e['date'] < b]
    prev = [e for e in E if e['date'] < a]
    e0 = prev[-1]['equity'] if prev else 1e6
    eq = np.array([e0] + [e['equity'] for e in xs])
    pk = np.maximum.accumulate(eq)
    return round(((eq[-1] / e0) ** (244 / len(xs)) - 1) * 100, 2), round(float((1 - eq / pk).max()) * 100, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/persist05')
    ap.add_argument('--start', default='2005-01-01')
    ap.add_argument('--end', default='2026-10-31')
    ap.add_argument('--basis-start', default='2012-06-01')
    ap.add_argument('--cachedir', default='hist')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    np.random.seed(26)
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
    idx = {v: {x['d']: i for i, x in enumerate(k)} for v, k in ADJ.items()}
    rowmap = {D: {r['v']: r for r in data['R'].get(D, [])} for D in data['days']}
    log(f'准备完成（{time.time() - t00:.0f}s）')
    out = {}

    # ---------- A 衰减曲线 ----------
    prev, rec = set(), []
    for n, D in enumerate(data['days']):
        now = set()
        for r in data['R'].get(D, []):
            if not r.get('big'):
                continue
            v = r['v']
            k, i = ADJ.get(v), idx.get(v, {}).get(D)
            if k is None or i is None or i < 20:
                continue
            if VR.elig(r):
                dr, grp = r['dir'], ('大将军≥3' if r['nbig'] >= 3 else '大将军<3')
                now.add((v, dr))
                if (v, dr) in prev:
                    continue
            else:
                if n % 2:
                    continue
                m5 = sum(x['c'] for x in k[i - 4:i + 1]) / 5
                dr, grp = ('多' if k[i]['c'] > m5 else '空'), '无信号'
            if not VS.ma5_ok(k, i, dr):
                continue
            ca = carry(v, D)
            if ca is None or ca == 0 or (ca > 0) != (dr == '多'):
                continue
            f = fwd(k, i, dr)
            if f:
                rec.append(dict(d=D, v=v, dir=dr, g=grp, f=f))
        prev = now
    log(f'A 样本 {len(rec)}')
    A = {}
    for nm, s0, s1 in (('2005-2016', '0000', '2017-01-01'), ('2017起', '2017-01-01', '9999'), ('全部', '0000', '9999')):
        L = [x for x in rec if s0 <= x['d'] < s1]
        tab = {}
        for h in HS:
            G = {g: [dict(d=x['d'], R=x['f'][h]) for x in L if x['g'] == g and h in x['f']] for g in ('无信号', '大将军<3', '大将军≥3')}
            tab[h] = {g: (round(float(np.mean([y['R'] for y in v])), 3) if v else None) for g, v in G.items()} | {
                'n≥3': len(G['大将军≥3']), '差≥3−无': G5.cdiff(G['大将军≥3'], G['无信号']), '差<3−无': G5.cdiff(G['大将军<3'], G['无信号'])}
        A[nm] = tab
    out['A_衰减曲线(ATR)'] = A

    # ---------- C 持有上限（先跑，B 用 C0 的成交）----------
    def cond(p, D):
        r = rowmap.get(D, {}).get(p['v'])
        if not r or r['dir'] != p['dir'] or (r.get('ngen') or 0) < 2:
            return False
        ca = carry(p['v'], D)
        return not (ca is not None and ca != 0 and (ca > 0) != (p['dir'] == '多'))
    C, runs = {}, {}
    for nm, kw in (('C0 现行60日', dict(maxhold=60)), ('C90', dict(maxhold=90)), ('C120', dict(maxhold=120)),
                   ('C∞(250)', dict(maxhold=250)), ('Ccond 满60日仍同向顺结构续拿', dict(maxhold=60, cond=cond))):
        st, E = run_hold(data, S, X0, **kw)
        runs[nm] = (st, E)
        m = A5.summ(E, st, True)
        real = [t for t in st['closed'] if not t.get('observe')]
        long_ = [t for t in real if (t.get('age') or 0) > 60]
        hit = [t for t in real if '上限' in (t.get('why') or '')]
        y0, d0_ = seg_ann(E, '0000', '2017-01-01')
        y1, d1 = seg_ann(E, '2017-01-01', '9999')
        C[nm] = {k: m[k] for k in ('终值万', '年化', '最大回撤', '笔', 'R净')} | {
            '收益÷回撤': round(m['年化'] / m['最大回撤'], 3) if m['最大回撤'] else None,
            '2005-2016 年化/回撤': [y0, d0_], '2017起 年化/回撤': [y1, d1],
            '持有超60日 笔/净万': [len(long_), round(sum(t['pnl'] - op.tcost(t) for t in long_) / 1e4, 1)],
            '因上限平仓 笔/净万': [len(hit), round(sum(t['pnl'] - op.tcost(t) for t in hit) / 1e4, 1)],
            '最长持有天': max([t.get('age') or 0 for t in real] or [0]), '分年': m['分年']}
        log(f"{nm}：{ {k: v for k, v in C[nm].items() if k != '分年'} }")
    base = C['C0 现行60日']
    for nm, x in C.items():
        if nm.startswith('C0'):
            continue
        ok = [x['2005-2016 年化/回撤'][0] >= base['2005-2016 年化/回撤'][0], (x['收益÷回撤'] or 0) >= (base['收益÷回撤'] or 0),
              x['2017起 年化/回撤'][0] >= base['2017起 年化/回撤'][0] - 1, x['最大回撤'] <= base['最大回撤'] + 2]
        x['通过标准①②③④'] = ok
        x['通过'] = all(ok)
    out['C_持有上限'] = C

    # ---------- B 信息年龄 ----------
    st, E = runs['C0 现行60日']
    real = [t for t in st['closed'] if not t.get('observe')]
    DI = {D: i for i, D in enumerate(data['days'])}
    recB = []
    for t in real:
        D, v, dr = t['base'], t['v'], t['dir']
        j = DI.get(D)
        if j is None or not t.get('risk'):
            continue
        age, s0 = 0, D
        while j - age - 1 >= 0:
            Dp = data['days'][j - age - 1]
            r = rowmap.get(Dp, {}).get(v)
            if not r or not VR.elig(r) or r['dir'] != dr:
                break
            age += 1
            s0 = Dp
        k = ADJ.get(v)
        i, i0 = idx.get(v, {}).get(D), idx.get(v, {}).get(s0)
        a_ = tracker.atr(k, i) if k is not None and i is not None else 0
        sg = 1 if dr == '多' else -1
        mv = sg * (k[i]['c'] - k[i0]['c']) / a_ if a_ and i0 is not None else 0
        recB.append(dict(d=D, age=age, mv=mv, R=(t['pnl'] - op.tcost(t)) / t['risk']))
    Bt = {}
    for nm, s0, s1 in (('2005-2016', '0000', '2017-01-01'), ('2017起', '2017-01-01', '9999'), ('全部', '0000', '9999')):
        L = [x for x in recB if s0 <= x['d'] < s1]
        g = {}
        for lab, f in (('年龄0天(当天新出)', lambda x: x['age'] == 0), ('1-4天', lambda x: 1 <= x['age'] <= 4),
                       ('5-19天', lambda x: 5 <= x['age'] <= 19), ('≥20天', lambda x: x['age'] >= 20),
                       ('已走<1ATR', lambda x: x['mv'] < 1), ('已走1-3ATR', lambda x: 1 <= x['mv'] < 3), ('已走≥3ATR', lambda x: x['mv'] >= 3)):
            g[lab] = G5.cstats([dict(d=x['d'], R=x['R']) for x in L if f(x)])
        Bt[nm] = g
    out['B_信息年龄(每笔净R)'] = Bt
    out['说明'] = 'A：h 天顺向涨跌 ÷ 信号日 ATR（未扣成本）；B：上线口径 B 实际成交的净 R；C：上线口径 B 只改持有上限。预登记见项目文档。'
    json.dump(out, open(os.path.join(a.out, 'persist05.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('# 信息时效与持续性 第一步\n\n```\n' + json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n```\n')
    log(f'全部完成（{time.time() - t00:.0f}s）')


if __name__ == '__main__':
    main()
