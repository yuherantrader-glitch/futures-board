"""回应 Astra 2026-11-21 审计的三项补测（同 long05 / robust05 的数据与引擎，v2.7 冻结 clean）。

  python tools/astra05.py --data hist/all --out site/bt/astra05 --start 2005-01-01 --basis-start 2012-06-01

1 联合剔除主要盈利品种：在选股之前剔除（会改选下一个候选），V27 与 ECB，固定 2% 与 2%15% 复利。
2 只用上期所、2007-01-04 ~ 2016-11-30、不用基差（对齐 Astra 早期子样本），V27 vs ECB，盈利升档 / 固定 2% / 复利；
  另附全交易所同区间、及上期所带基差的对照。
3 风控：盯市净权益作定仓基数、回撤降档（Astra 版与温和版）、固定 1.5%，与 2%15% 复利、现行盈利升档并列。
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
import long05 as L5  # noqa: E402
import robust05 as R5  # noqa: E402
from ftrade import kline, tracker  # noqa: E402
import backtest as bt  # noqa: E402
import lagtest as LG  # noqa: E402
import optimize as op  # noqa: E402

log = VR.log
SHFE = {'CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'AU', 'AG', 'RB', 'WR', 'HC', 'SS', 'FU', 'BU', 'RU', 'SP', 'AO', 'BR', 'AD', 'OP'}
NOBASIS = type('nb', (), {})


def no_basis():
    f = lambda v, D, dr: False   # noqa: E731
    f.pct = lambda v, D: None
    return f


def run3(data, S, sizing, xcfg, base='realized', steps=None, ladder=None):
    """sizing 同 v27_repro.run。base='mtm' 用前一日盯市净权益定仓；steps=[(回撤, 每笔, 总上限)] 从净值峰值回撤到该档即用该档
    （取比当前档更低者），每笔为 0 表示暂停新开仓。"""
    op.COMPOUND = sizing[0] == 'ladder' or sizing[3]
    LAD = ladder or R5.LADDER0
    T = LG.LT(data['RAW'], data['ADJ'], data['ROLLS'], {'trail': 0, 'ma5': False, **xcfg, 'maxHold': 60,
                                                        'fixed': 0.02 if sizing[0] == 'ladder' else sizing[1]})
    if sizing[0] != 'ladder':
        T.cfg['maxHeat'] = sizing[2]
    st = {'open': [], 'closed': [], 'from': data['days'][0]}
    E, last_eq, pk = [], 1e6, 1e6
    paused = 0
    for D in data['days'][1:]:
        x = S[D]
        sigs = [s for s in x['sigs'] if op.ma5_ok(data['ADJ'], s['v'], D, s['dir'])]
        op.compound(T, st)
        if base == 'mtm' and op.COMPOUND:
            T.cfg['capital'] = last_eq
        if sizing[0] == 'ladder':
            cap = T.cfg['capital']
            i = max(k for k, (m, _, _) in enumerate(LAD) if cap >= m * 1e6) if cap >= 1e6 else 0
            T.cfg['fixed'], T.cfg['maxHeat'] = LAD[i][1:]
        elif sizing[0] == 'fixed':
            T.cfg['fixed'], T.cfg['maxHeat'] = sizing[1], sizing[2]
        if steps:
            dd = 1 - last_eq / pk
            for th, r, h in steps:
                if dd >= th:
                    if r == 0:
                        sigs = []
                        paused += 1
                    else:
                        T.cfg['fixed'], T.cfg['maxHeat'] = min(T.cfg['fixed'], r), min(T.cfg['maxHeat'], h)
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
    st['paused_days'] = paused
    return st, E


def summ(E, st, comp=True):
    m = R5.metrics(E, st, comp)
    e = np.array([x['equity'] for x in E])
    m['年化'] = round(((e[-1] / 1e6) ** (244 / len(e)) - 1) * 100, 1)
    m['跌破本金最深'] = round(max(0, 1 - e.min() / 1e6) * 100, 1)
    m['暂停开仓天'] = st.get('paused_days', 0)
    ys = {}
    for x in E:
        ys[x['date'][:4]] = x['equity']
    prev, yr = 1e6, {}
    for y, v in sorted(ys.items()):
        yr[y] = round((v / prev - 1) * 100, 1)
        prev = v
    m['分年'] = yr
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/astra05')
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
    y0 = int(min(H)[:4])
    seat = VR.strong_fn(H, ADJ, 'clean', years=range(y0 + 2, int(a.end[:4]) + 1))
    log(f'准备完成（{time.time() - t00:.0f}s）')
    X0 = dict(LG.V25, **LG.REAL, noWeak=True)
    FIX, COMP, LAD = ('fixed', 0.02, 0.10, False), ('fixed', 0.02, 0.15, True), ('ladder',)
    out = {}

    def sub(R_filter=None, days=None):
        d = dict(data)
        if R_filter:
            d['R'] = {D: [r for r in rows if R_filter(r['v'])] for D, rows in data['R'].items()}
        if days:
            d['days'] = [x for x in data['days'] if days[0] <= x <= days[1]]
        return d

    # ---------- 1 联合剔除 ----------
    S0, _ = VR.make_S(data, carry, blow, seat, True)
    st0, E0 = R5.run2(data, S0, FIX, X0)
    contrib = {}
    for t in st0['closed']:
        if t.get('risk') and not t.get('observe'):
            contrib[t['v']] = contrib.get(t['v'], 0) + ((t['pnl'] or 0) - op.tcost(t)) / t['risk']
    top = [v for v, _ in sorted(contrib.items(), key=lambda x: -x[1])]
    log(f'贡献前 10（固定 2%，R）：{[(v, round(contrib[v], 1)) for v in top[:10]]}')
    SETS = {'不剔除': set(), '剔除 AL': {'AL'}, '剔除 RB': {'RB'}, '剔除 AL+RB': {'AL', 'RB'}, '剔除 AL+RB+CU': {'AL', 'RB', 'CU'},
            f'剔除贡献前3 {"+".join(top[:3])}': set(top[:3]), f'剔除贡献前5 {"+".join(top[:5])}': set(top[:5]),
            f'剔除贡献前10': set(top[:10])}
    ex = {}
    for nm, sset in SETS.items():
        d = sub(lambda v, s=sset: v not in s)
        r = {}
        for lab, use in (('V27', True), ('ECB', False)):
            S, _ = VR.make_S(d, carry, blow, seat, use)
            for sl, sz, comp in (('固定2%', FIX, False), ('2%15%复利', COMP, True)):
                st, E = R5.run2(d, S, sz, X0)
                m = R5.metrics(E, st, comp)
                r[f'{lab}·{sl}'] = {k: m[k] for k in ('终值万', '最大回撤', '笔', 'R净', '合计R', 'R净_样本外', 'R净_样本内') if k in m} | (
                    {'年化': m.get('年化'), '样本外年化': m['样本外'].get('年化'), '样本内年化': m['样本内'].get('年化')} if comp else
                    {'样本外万每年': m['样本外'].get('年均万'), '样本内万每年': m['样本内'].get('年均万')})
        ex[nm] = r
        log(f'{nm}：{r}')
    out['1_联合剔除'] = dict(贡献前10=[(v, round(contrib[v], 1)) for v in top[:10]], 结果=ex)

    # ---------- 2 上期所 2007-2016 不用基差 ----------
    WIN = ('2007-01-04', '2016-11-30')
    nb = no_basis()
    sh = {}
    for nm, filt, bl in (('上期所·不用基差', lambda v: v in SHFE, nb), ('上期所·用基差(2013起有)', lambda v: v in SHFE, blow),
                         ('全交易所·不用基差', None, nb), ('全交易所·用基差', None, blow)):
        d = sub(filt, WIN)
        r = {}
        for lab, use in (('V27', True), ('ECB', False)):
            S, _ = VR.make_S(d, carry, bl, seat, use)
            for sl, sz in (('盈利升档', LAD), ('2%15%复利', COMP), ('固定2%', FIX)):
                st, E = run3(d, S, sz, X0)
                m = summ(E, st, sl != '固定2%')
                r[f'{lab}·{sl}'] = {k: m[k] for k in ('终值万', '年化', '最大回撤', '笔', 'R净', '合计R', '分年')}
        sh[nm] = r
        log(f'{nm}：{ {k: {kk: vv for kk, vv in v.items() if kk != "分年"} for k, v in r.items()} }')
    # 强席位名单在该区间每年的规模
    sh['强席位名单数'] = {y: len(x) for y, x in seat.lists.items() if int(y) <= 2016}
    sh['上期所强席位名单数'] = {y: sum(1 for k in x if k.split('|', 1)[1] in SHFE) for y, x in seat.lists.items() if int(y) <= 2016}
    out['2_上期所早期'] = sh

    # ---------- 3 风控 ----------
    ASTRA = [(0.05, 0.015, 0.12), (0.10, 0.015, 0.10), (0.20, 0.01, 0.08), (0.25, 0, 0)]
    SOFT = [(0.10, 0.015, 0.12), (0.15, 0.01, 0.10)]
    LAD_A = [(1.0, 0.02, 0.12), (1.3, 0.0225, 0.15), (1.6, 0.025, 0.18)]   # Astra 建议：不恢复 3% 档
    CASES = [('2%15%复利（已实现基数）', COMP, {}), ('2%15%复利 盯市基数', COMP, dict(base='mtm')),
             ('2%15%复利 盯市 + Astra 降档', COMP, dict(base='mtm', steps=ASTRA)),
             ('2%15%复利 盯市 + 温和降档', COMP, dict(base='mtm', steps=SOFT)),
             ('1.5%15%复利 盯市', ('fixed', 0.015, 0.15, True), dict(base='mtm')),
             ('盈利升档 现行', LAD, {}), ('盈利升档 盯市基数', LAD, dict(base='mtm')),
             ('盈利升档 盯市 + Astra 降档', LAD, dict(base='mtm', steps=ASTRA)),
             ('盈利升档 盯市 + 温和降档', LAD, dict(base='mtm', steps=SOFT)),
             ('Astra 阶梯 2/2.25/2.5% 盯市 + 降档', LAD, dict(base='mtm', steps=ASTRA, ladder=LAD_A))]
    rk = {}
    for nm, sz, kw in CASES:
        r = {}
        for lab, use in (('V27', True), ('ECB', False)):
            S, _ = VR.make_S(data, carry, blow, seat, use)
            st, E = run3(data, S, sz, X0, **kw)
            m = summ(E, st)
            r[lab] = {k: m[k] for k in ('终值万', '年化', '最大回撤', '跌破本金最深', '笔', '暂停开仓天')} | {
                '样本外年化': m['样本外'].get('年化'), '样本外回撤': m['样本外'].get('最大回撤'),
                '样本内年化': m['样本内'].get('年化'), '样本内回撤': m['样本内'].get('最大回撤'),
                '收益回撤比': round(m['年化'] / max(0.1, m['最大回撤']), 2)}
        rk[nm] = r
        log(f'{nm}：{r}')
    out['3_风控'] = rk
    json.dump(out, open(os.path.join(a.out, 'astra05.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('# Astra 三项补测\n\n```\n' + json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n```\n')
    log(f'全部完成（{time.time() - t00:.0f}s）')


if __name__ == '__main__':
    main()
