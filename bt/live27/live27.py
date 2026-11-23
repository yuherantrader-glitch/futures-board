"""v2.7 上线口径复算（2005-05 ~ 2026-10，同 long05 数据，V27 冻结 clean）。

  python tools/live27.py --data hist/all --out site/bt/live27 --start 2005-01-01 --basis-start 2012-06-01

上线口径：起始 100 万；按前一日盯市净权益定仓；每笔 2%、总 15%、单笔硬上限 3%、同板块 6%；按手向下取整（不足 1 手跳过）；
60 日上限从首次成交算、不因同向信号重置；回撤降档 ≥10% → 1.5%/12%，≥15% → 1%/10%，回到峰值 5% 以内恢复。
对照：A 回测口径（同向重置、按已实现权益定仓、不降档）/ B 上线口径 / C 上线但同向重置 / D 上线但不降档。
另给：分段、分年、出场原因、因不足 1 手被跳过的信号、最近 6 个月逐笔（给上线前对账用）、期末持仓。
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
import basis05 as B5  # noqa: E402
from ftrade import kline, tracker  # noqa: E402
import backtest as bt  # noqa: E402
import lagtest as LG  # noqa: E402
import optimize as op  # noqa: E402

log = VR.log
STEPS = [(0.15, 0.01, 0.10), (0.10, 0.015, 0.12)]   # 由深到浅
RECOVER = 0.05


class LT2(LG.LT):
    """noReset=True：60 日计数用实际持有天数（age），同向信号不清零。"""

    def step(self, D, st, sigs, touched, today=False, weak=None):
        out = super().step(D, st, sigs, touched, today, weak)
        if self.cfg.get('noReset'):
            for p in st['open']:
                if not p.get('pending'):
                    p['held'] = p.get('age', 0)
        return out


def run_live(data, S, xcfg, no_reset=True, mtm=True, steps=True):
    op.COMPOUND = True
    T = LT2(data['RAW'], data['ADJ'], data['ROLLS'], {'trail': 0, 'ma5': False, **xcfg, 'maxHold': 60, 'fixed': 0.02,
                                                      'noReset': no_reset})
    T.cfg['maxHeat'] = 0.15
    skips = []
    orig = T.open

    def open2(D, s, st):
        has = any(p['v'] == s['v'] and p['dir'] == s['dir'] for p in st['open'])
        r = orig(D, s, st)
        if not has and r.get('skip') and '1手风险' in str(r['skip']):
            skips.append(dict(d=D, v=s['v'], dir=s['dir'], why=r['skip']))
        return r
    T.open = open2
    st = {'open': [], 'closed': [], 'from': data['days'][0]}
    E, last_eq, pk, lvl, lvdays = [], 1e6, 1e6, 0, [0, 0, 0]
    for D in data['days'][1:]:
        sigs = [s for s in S[D]['sigs'] if op.ma5_ok(data['ADJ'], s['v'], D, s['dir'])]
        op.compound(T, st)
        if mtm:
            T.cfg['capital'] = last_eq
        T.cfg['fixed'], T.cfg['maxHeat'] = 0.02, 0.15
        if steps:
            dd = 1 - last_eq / pk
            if dd >= STEPS[0][0]:
                lvl = 2
            elif dd >= STEPS[1][0]:
                lvl = max(lvl, 1)
            elif dd <= RECOVER:
                lvl = 0
            if lvl:
                _, r, h = STEPS[0] if lvl == 2 else STEPS[1]
                T.cfg['fixed'], T.cfg['maxHeat'] = r, h
        lvdays[lvl] += 1
        T.step(D, st, [dict(s) for s in sigs], set(), today=True, weak={})
        real = [t for t in st['closed'] if not t.get('observe')]
        live = [p for p in st['open'] if not p.get('observe') and not p.get('pending')]
        eq = 1e6 + sum(t['pnl'] or 0 for t in real) + sum(p['pnl'] or 0 for p in live) - sum(op.tcost(t) for t in real)
        margin = sum(abs(p.get('lastClose') or p['entry']) * p['lots'] * p['mult'] for p in live) * 0.12
        last_eq, pk = eq, max(pk, eq)
        E.append(dict(date=D, equity=eq, nOpen=len(live), notional=0, heat=0, margin=margin / eq if eq > 0 else 0))
    pk = 1e6
    for e in E:
        pk = max(pk, e['equity'])
        e['peak'], e['dd'] = pk, 1 - e['equity'] / pk
    st['skips'], st['lvdays'] = skips, lvdays
    return st, E


def report(st, E):
    m = A5.summ(E, st, True)
    real = [t for t in st['closed'] if not t.get('observe')]
    why = {}
    for t in real:
        w = why.setdefault(t['why'], [0, 0.0])
        w[0] += 1
        w[1] += (t['pnl'] - op.tcost(t)) / 1e4
    hold = np.array([t['age'] for t in real if t.get('age')])
    mg = np.array([e['margin'] for e in E])
    sk = st['skips']
    return {k: m[k] for k in ('终值万', '年化', '最大回撤', '笔', 'R净', '跌破本金最深')} | {
        '分段': B5.segstats(E, True), '分年': m['分年'], '出场原因(笔,净万)': {k: [v[0], round(v[1], 1)] for k, v in why.items()},
        '持仓天数中位': float(np.median(hold)) if len(hold) else None,
        '保证金占用p95/最大(12%估)': [round(float(np.percentile(mg, 95)) * 100, 1), round(float(mg.max()) * 100, 1)],
        '降档天数(正常/1档/2档)': st['lvdays'],
        '不足1手跳过': dict(笔=len(sk), 按品种={v: sum(1 for x in sk if x['v'] == v) for v in sorted({x['v'] for x in sk})},
                       按年={y: sum(1 for x in sk if x['d'][:4] == y) for y in sorted({x['d'][:4] for x in sk})})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/live27')
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
    log(f'准备完成（{time.time() - t00:.0f}s）')
    X0 = dict(LG.V25, **LG.REAL, noWeak=True)
    S, _ = VR.make_S(data, carry, blow, seat, True)
    out = {'口径': __doc__}
    st, E = A5.run3(data, S, ('fixed', 0.02, 0.15, True), X0)
    out['A_回测口径'] = {k: v for k, v in A5.summ(E, st, True).items() if k in ('终值万', '年化', '最大回撤', '笔', 'R净')} | {'分段': B5.segstats(E, True)}
    log(f"A {out['A_回测口径']}")
    runs = {}
    for nm, kw in (('B_上线口径', {}), ('C_上线但同向重置', dict(no_reset=False)), ('D_上线但不降档', dict(steps=False))):
        st, E = run_live(data, S, X0, **kw)
        runs[nm] = (st, E)
        out[nm] = report(st, E)
        log(f"{nm} {({k: v for k, v in out[nm].items() if k not in ('分年',)})}")
    st, E = runs['B_上线口径']
    real = sorted([t for t in st['closed'] if not t.get('observe')], key=lambda t: t.get('fillDate') or t['base'])
    cut = '2026-05-01'
    out['B_最近6个月逐笔'] = [dict(信号日=t['base'], 成交=t.get('fillDate'), 品种=t['v'], 方向=t['dir'], 手=t['lots'], 开=t['entry'],
                              平日=t['exitDate'], 平=t['exit'], 原因=t['why'], 净元=round(t['pnl'] - op.tcost(t)))
                         for t in real if (t['exitDate'] or '') >= cut]
    out['B_期末持仓'] = [dict(信号日=p['base'], 成交=p.get('fillDate'), 品种=p['v'], 方向=p['dir'], 手=p['lots'], 开=p['entry'],
                           止损=p['stopPx'], 持有天=p.get('age'), 浮盈元=round(p.get('pnl') or 0)) for p in st['open']]
    out['B_不足1手跳过明细(近50)'] = st['skips'][-50:]
    with open(os.path.join(a.out, 'trades_B.csv'), 'w', encoding='utf-8-sig') as f:
        f.write('信号日,成交日,品种,方向,手,开,平日,平,原因,净元\n')
        for t in real:
            f.write(f"{t['base']},{t.get('fillDate')},{t['v']},{t['dir']},{t['lots']},{t['entry']},{t['exitDate']},{t['exit']},{t['why']},{round(t['pnl'] - op.tcost(t))}\n")
    with open(os.path.join(a.out, 'equity_B.csv'), 'w', encoding='utf-8') as f:
        f.write('date,equity,dd,margin\n')
        for e in E:
            f.write(f"{e['date']},{round(e['equity'])},{round(e['dd'], 4)},{round(e['margin'], 4)}\n")
    json.dump(out, open(os.path.join(a.out, 'live27.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('# v2.7 上线口径复算\n\n```\n' + json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n```\n')
    log(f'全部完成（{time.time() - t00:.0f}s）')


if __name__ == '__main__':
    main()
