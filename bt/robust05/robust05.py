"""v2.7 稳健性与压力测试（2005 起长回测的数据与引擎）。

  python tools/robust05.py --data hist/all --out site/bt/robust05 --start 2005-01-01 --basis-start 2012-06-01

一次跑完：
  A 参数邻域：止损/移动止损 ATR、持仓上限（40 / 60 / 不设 / 60 按首次开仓不清零）、强席位门槛、强席位名单比例、基差分位、升档档位
  B 成本 1/2/3 倍；定仓基数 = min(已实现, 盯市)
  C 盈利升档 6 个起点
  D 压力：回撤持续、最差单日、同日多笔止损、跳空止损 R 分布、保证金占用、连续两个跌停、极端行情窗口
  E 分环境：商品指数高/中/低波动、牛/熊（指数在 250 日均线上/下）、美联储加息/降息/零利率区间
每个参数变体都分样本外（2017 前）/ 样本内两段报告。口径与 tools/long05.py 相同（v2.7 冻结 clean，无前视）。
"""
import argparse
import inspect
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tools'))
import v27_repro as VR  # noqa: E402
import long05 as L5  # noqa: E402
from ftrade import kline, tracker  # noqa: E402
import backtest as bt  # noqa: E402
import lagtest as LG  # noqa: E402
import optimize as op  # noqa: E402

log = VR.log
SPLIT = L5.SPLIT
LADDER0 = list(VR.LADDER)
TCOST0 = op.tcost
MARGIN = 0.12          # 假设保证金率（交易所 + 期货公司加收，多数品种 10%~15%）
LIMIT = 0.06           # 假设单日涨跌停幅度（多数品种 4%~8%）

# 美联储利率周期（近似，按月）
FED = [('2004-06', '2006-06', '加息'), ('2006-07', '2007-08', '高位不动'), ('2007-09', '2008-12', '降息'), ('2009-01', '2015-11', '零利率'),
       ('2015-12', '2018-12', '加息'), ('2019-01', '2019-06', '高位不动'), ('2019-07', '2020-03', '降息'), ('2020-04', '2022-02', '零利率'),
       ('2022-03', '2023-07', '加息'), ('2023-08', '2024-08', '高位不动'), ('2024-09', '2026-12', '降息')]


def run2(data, S, sizing, xcfg, maxhold=60, hold_fill=None, cap_min=False, ladder=None):
    """= v27_repro.run，另加：maxhold 可变；hold_fill=N 按首次成交起算 N 个交易日强制平（不被续持清零）；
    cap_min 定仓基数取 min(已实现, 盯市)；ladder 自定义升档表。另记每日名义市值。"""
    op.COMPOUND = sizing[0] == 'ladder' or sizing[3]
    LAD = ladder or LADDER0
    T = LG.LT(data['RAW'], data['ADJ'], data['ROLLS'], {'trail': 0, 'ma5': False, **xcfg, 'maxHold': maxhold,
                                                        'fixed': 0.02 if sizing[0] == 'ladder' else sizing[1]})
    if sizing[0] != 'ladder':
        T.cfg['maxHeat'] = sizing[2]
    st = {'open': [], 'closed': [], 'from': data['days'][0]}
    E, idx = [], {d: i for i, d in enumerate(data['days'])}
    last_eq = 1e6
    for D in data['days'][1:]:
        x = S[D]
        sigs = [s for s in x['sigs'] if op.ma5_ok(data['ADJ'], s['v'], D, s['dir'])]
        op.compound(T, st)
        if cap_min and op.COMPOUND:
            T.cfg['capital'] = min(T.cfg['capital'], last_eq)
        if sizing[0] == 'ladder':
            cap = T.cfg['capital']
            i = max(k for k, (m, _, _) in enumerate(LAD) if cap >= m * 1e6) if cap >= 1e6 else 0
            T.cfg['fixed'], T.cfg['maxHeat'] = LAD[i][1:]
        T.step(D, st, [dict(s) for s in sigs], set(), today=True, weak={})
        if hold_fill:
            out = []
            for p in st['open']:
                if p.get('pending') or not p.get('fillDate') or p.get('lastDate') != D:
                    continue
                if idx[D] - idx[p['fillDate']] + 1 >= hold_fill:
                    T.close(p, D, p['lastClose'], f'满{hold_fill}日(首次起算)')
                    out.append(p)
            if out:
                st['closed'] += out
                st['open'] = [p for p in st['open'] if p not in out]
        real = [t for t in st['closed'] if not t.get('observe')]
        live = [p for p in st['open'] if not p.get('observe') and not p.get('pending')]
        fl = sum(p['pnl'] or 0 for p in live)
        cost = sum(op.tcost(t) for t in real)
        eq = 1e6 + sum(t['pnl'] or 0 for t in real) + fl - cost
        last_eq = eq
        E.append(dict(date=D, equity=eq, nOpen=len(live), notional=sum((p.get('lots') or 0) * (p.get('mult') or 0) * (p.get('lastClose') or 0) for p in live),
                      heat=sum(p.get('budgetPct') or 0 for p in live)))
    pk = 1e6
    for e in E:
        pk = max(pk, e['equity'])
        e['peak'], e['dd'] = pk, 1 - e['equity'] / pk
    return st, E


def metrics(E, st, comp):
    real = [t for t in st['closed'] if not t.get('observe')]
    rn = [((t['pnl'] or 0) - op.tcost(t)) / t['risk'] for t in real if t.get('risk')]
    e = np.array([x['equity'] for x in E])
    m = dict(终值万=round(e[-1] / 1e4, 1), 最大回撤=round(max(x['dd'] for x in E) * 100, 1), 笔=len(real),
             R净=round(float(np.mean(rn)), 3) if rn else None, 合计R=round(float(np.sum(rn)), 1) if rn else None)
    if comp:
        m['年化'] = round(((e[-1] / 1e6) ** (244 / len(e)) - 1) * 100, 1)
    m['样本外'] = L5.seg(E, '0000', SPLIT, comp)
    m['样本内'] = L5.seg(E, SPLIT, '9999', comp)
    for k in ('样本外', '样本内'):
        m[k] = {a: (float(b) if isinstance(b, (np.floating, float)) else b) for a, b in m[k].items()}
    tr_o = [r for t, r in zip([t for t in real if t.get('risk')], rn) if t['base'] < SPLIT]
    tr_i = [r for t, r in zip([t for t in real if t.get('risk')], rn) if t['base'] >= SPLIT]
    m['R净_样本外'] = round(float(np.mean(tr_o)), 3) if tr_o else None
    m['R净_样本内'] = round(float(np.mean(tr_i)), 3) if tr_i else None
    return m


def wrap_seat(seat, thr):
    def f(v, D, dr):
        s = seat.net(v, D)
        return s is not None and abs(s) >= thr and (s > 0) != (dr == '多')
    f.net = seat.net
    return f


def wrap_basis(blow, q):
    def f(v, D, dr):
        p = blow.pct(v, D)
        return p is not None and (p if dr == '多' else 1 - p) <= q
    f.pct = blow.pct
    return f


def strong_top(H, ADJ, years, topq):
    """强席位名单比例改为 topq（复制 v27_repro.strong_fn，只换比例常数）。"""
    src = inspect.getsource(VR.strong_fn).replace('len(g) * 0.1', f'len(g) * {topq}').replace('def strong_fn(', 'def _strong_q(')
    ns = dict(VR.__dict__)
    exec(src, ns)
    return ns['_strong_q'](H, ADJ, 'clean', years=years)


def dd_episodes(E, top=5):
    eps, start, pk_i = [], None, 0
    eq = [e['equity'] for e in E]
    pk = eq[0]
    for i, v in enumerate(eq):
        if v >= pk:
            if start is not None:
                seg = eq[start:i]
                j = start + int(np.argmin(seg))
                eps.append(dict(起=E[start - 1]['date'] if start else E[0]['date'], 谷=E[j]['date'], 恢复=E[i]['date'],
                                深度=round((1 - eq[j] / pk) * 100, 1), 下跌天=j - start + 1, 恢复天=i - j, 共=i - start + 1))
                start = None
            pk = v
        elif start is None:
            start = i
    if start is not None:
        seg = eq[start:]
        j = start + int(np.argmin(seg))
        eps.append(dict(起=E[start - 1]['date'], 谷=E[j]['date'], 恢复='(未恢复)', 深度=round((1 - eq[j] / pk) * 100, 1),
                        下跌天=j - start + 1, 恢复天=len(eq) - j - 1, 共=len(eq) - start))
    return sorted(eps, key=lambda x: -x['深度'])[:top], max((x['共'] for x in eps), default=0)


def stress(E, st):
    real = [t for t in st['closed'] if not t.get('observe')]
    eq = np.array([e['equity'] for e in E])
    ret = np.diff(eq) / eq[:-1]
    days = [e['date'] for e in E]
    worst = sorted(zip(ret, days[1:]))[:5]
    by = {}
    for t in real:
        if '止损' in (t.get('why') or ''):
            by.setdefault(t['exitDate'], []).append(t)
    multi = {d: ts for d, ts in by.items() if len(ts) >= 2}
    mloss = sorted(((sum((t['pnl'] or 0) - op.tcost(t) for t in ts), d, len(ts), '、'.join(t['v'] + t['dir'] for t in ts)) for d, ts in multi.items()))[:5]
    eqd = {e['date']: e['equity'] for e in E}
    gap = [((t['pnl'] or 0) - op.tcost(t)) / t['risk'] for t in real if t.get('why') == '跳空止损' and t.get('risk')]
    allr = [((t['pnl'] or 0) - op.tcost(t)) / t['risk'] for t in real if t.get('risk')]
    mg = np.array([e['notional'] * MARGIN / e['equity'] for e in E])
    lim2 = np.array([e['notional'] * LIMIT * 2 / e['equity'] for e in E])
    eps, longest = dd_episodes(E)
    under = 0
    run = best = 0
    for e in E:
        run = run + 1 if e['dd'] > 0 else 0
        best = max(best, run)
    streak = cur = 0
    for t in sorted(real, key=lambda t: t['exitDate']):
        cur = cur + 1 if (t['pnl'] or 0) - op.tcost(t) <= 0 else 0
        streak = max(streak, cur)
    return dict(
        最差单日=[f'{d} {r * 100:.1f}%' for r, d in worst], 单日1分位=round(float(np.percentile(ret, 1)) * 100, 2),
        同日止损2笔以上天数=len(multi), 同日止损3笔以上天数=sum(1 for ts in multi.values() if len(ts) >= 3),
        同日止损最大亏损=[f'{d} {n}笔 {v / 1e4:.1f}万（占当日权益 {v / eqd.get(d, 1e6) * 100:.1f}%） {s}' for v, d, n, s in mloss],
        跳空止损=dict(笔=len(gap), 平均R=round(float(np.mean(gap)), 2) if gap else None, 最差R=round(float(min(gap)), 2) if gap else None,
                  比R=round(float(np.percentile(gap, 5)), 2) if gap else None),
        全部最差R=round(float(min(allr)), 2), 亏损超过1点5R笔数=sum(1 for r in allr if r < -1.5),
        保证金占用=dict(假设保证金率=MARGIN, 平均=round(float(mg.mean()) * 100, 1), 分位95=round(float(np.percentile(mg, 95)) * 100, 1),
                    最高=round(float(mg.max()) * 100, 1), 最高日=days[int(mg.argmax())], 保证金率上调5成后最高=round(float(mg.max()) * 150, 1)),
        连续两个停板=dict(假设单板=LIMIT, 全部持仓同时反向两板最大亏损占权益=round(float(lim2.max()) * 100, 1), 分位95=round(float(np.percentile(lim2, 95)) * 100, 1)),
        回撤前五=eps, 最长回撤持续交易日=longest, 最长连续处于回撤交易日=best, 最长连亏笔=streak)


WINDOWS = [('2008 金融危机', '2008-07-01', '2008-12-31'), ('2015-16 商品熊市尾段', '2015-06-01', '2016-02-29'),
           ('2016 黑色暴涨', '2016-03-01', '2016-12-31'), ('2020-03 疫情冲击', '2020-01-15', '2020-05-31'),
           ('2022 俄乌/镍逼空', '2022-02-15', '2022-04-30'), ('2023 区间震荡', '2023-01-01', '2023-12-31'),
           ('2026 至今', '2026-01-01', '2026-10-31')]


def windows(E, st):
    real = [t for t in st['closed'] if not t.get('observe')]
    out = []
    for name, a, b in WINDOWS:
        xs = [e for e in E if a <= e['date'] <= b]
        if len(xs) < 2:
            continue
        prev = [e for e in E if e['date'] < a]
        e0 = prev[-1]['equity'] if prev else 1e6
        eq = np.array([e0] + [e['equity'] for e in xs])
        pk = np.maximum.accumulate(eq)
        tr = [t for t in real if a <= t['exitDate'] <= b]
        out.append(dict(区间=name, 起=a, 止=b, 收益=round((eq[-1] / e0 - 1) * 100, 1), 区间内回撤=round(float((1 - eq / pk).max()) * 100, 1),
                        平仓笔=len(tr), 区间内R=round(sum(((t['pnl'] or 0) - op.tcost(t)) / t['risk'] for t in tr if t.get('risk')), 1),
                        平均持仓=round(float(np.mean([e['nOpen'] for e in xs])), 1)))
    return out


def regimes(data, E, st):
    """商品等权指数（主连复权日收益等权）→ 60 日波动三分位、250 日均线牛熊；美联储周期。按交易所在区间分组看每笔 R 与日收益。"""
    rets = {}
    for v, k in data['ADJ'].items():
        for a, b in zip(k[:-1], k[1:]):
            if a['c'] > 0:
                r = b['c'] / a['c'] - 1
                if abs(r) < 0.2:
                    rets.setdefault(b['d'], []).append(r)
    days = sorted(d for d in rets if len(rets[d]) >= 3)
    r = pd.Series([np.mean(rets[d]) for d in days], index=days)
    idx = (1 + r).cumprod()
    vol = r.rolling(60, min_periods=40).std() * np.sqrt(244)
    ma = idx.rolling(250, min_periods=200).mean()
    q1, q2 = vol.quantile(1 / 3), vol.quantile(2 / 3)
    lab_v = {d: ('低波动' if v <= q1 else '中波动' if v <= q2 else '高波动') for d, v in vol.dropna().items()}
    lab_t = {d: ('牛（指数在年线上）' if idx[d] > m else '熊（指数在年线下）') for d, m in ma.dropna().items()}

    def fed(d):
        return next((c for a, b, c in FED if a <= d[:7] <= b), '?')
    real = [t for t in st['closed'] if not t.get('observe') and t.get('risk')]
    eqd = [(e['date'], e['equity']) for e in E]
    dret = {d2: (e2 - e1) / 1e6 for (d1, e1), (d2, e2) in zip(eqd[:-1], eqd[1:])}   # 固定 2% 不复利：日盈亏 ÷ 本金
    out = {}
    for name, lab in (('波动', lab_v), ('牛熊', lab_t), ('美联储', None)):
        g = {}
        for t in real:
            k = fed(t['base']) if lab is None else lab.get(t['base'])
            if k is None:
                continue
            x = g.setdefault(k, {'R': [], 'days': 0, 'ret': 0.0})
            x['R'].append(((t['pnl'] or 0) - op.tcost(t)) / t['risk'])
        for d, v in dret.items():
            k = fed(d) if lab is None else lab.get(d)
            if k is None:
                continue
            x = g.setdefault(k, {'R': [], 'days': 0, 'ret': 0.0})
            x['days'] += 1
            x['ret'] += v
        out[name] = {k: dict(笔=len(x['R']), 每笔R=round(float(np.mean(x['R'])), 3) if x['R'] else None,
                             胜率=round(sum(1 for z in x['R'] if z > 0) / len(x['R']) * 100, 1) if x['R'] else None,
                             交易日=x['days'], 年化收益占本金=round(x['ret'] / max(1, x['days']) * 244 * 100, 1)) for k, x in sorted(g.items())}
    out['说明'] = f'波动三分位界线 年化 {q1 * 100:.1f}% / {q2 * 100:.1f}%；分组按开仓日；年化收益用 v2.7 固定 2% 不复利的日盈亏 ÷ 100 万'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/robust05')
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
    YRS = range(y0 + 2, int(a.end[:4]) + 1)
    seat = VR.strong_fn(H, ADJ, 'clean', years=YRS)
    log(f'准备完成（{time.time() - t00:.0f}s）')
    X0 = dict(LG.V25, **LG.REAL, noWeak=True)
    FIX, COMP = ('fixed', 0.02, 0.10, False), ('fixed', 0.02, 0.15, True)
    S_base, _ = VR.make_S(data, carry, blow, seat, True)
    S_ecb, _ = VR.make_S(data, carry, blow, seat, False)
    res, keep = {}, {}

    def case(name, S, xcfg=X0, sizes=(('固定2%', FIX, False), ('2%15%复利', COMP, True)), **kw):
        for lab, sz, comp in sizes:
            t0 = time.time()
            st, E = run2(data, S, sz, xcfg, **kw)
            m = metrics(E, st, comp)
            res.setdefault(name, {})[lab] = m
            keep[(name, lab)] = (st, E)
            log(f'{name} · {lab}：{ {k: v for k, v in m.items() if k not in ("样本外", "样本内")} }（{time.time() - t0:.0f}s）')

    # A 参数邻域
    case('基准 v2.7', S_base)
    case('ECB（不用强席位）', S_ecb)
    for s in (2.5, 3.5):
        case(f'止损/移动 {s}ATR', S_base, xcfg=dict(X0, stopATR=s, trail=s))
    case('持仓上限 40 日', S_base, maxhold=40)
    case('持仓上限 不设', S_base, maxhold=99999)
    case('持仓上限 60 日按首次开仓', S_base, maxhold=99999, hold_fill=60)
    for thr in (0.0025, 0.01):
        case(f'强席位门槛 {thr * 100:g}%', VR.make_S(data, carry, blow, wrap_seat(seat, thr), True)[0])
    for topq in (0.05, 0.15):
        s2 = strong_top(H, ADJ, YRS, topq)
        case(f'强席位名单前 {topq * 100:g}%', VR.make_S(data, carry, blow, s2, True)[0])
    for q in (0.25, 0.4):
        case(f'基差分位 {q:g}', VR.make_S(data, carry, wrap_basis(blow, q), seat, True)[0])
    # B 成本 / 定仓基数
    for k in (2, 3):
        op.tcost = lambda t, k=k: k * TCOST0(t)
        case(f'成本 {k} 倍', S_base)
    op.tcost = TCOST0
    case('定仓基数 min(已实现,盯市)', S_base, sizes=(('2%15%复利', COMP, True),), cap_min=True)
    # 升档档位
    LADS = {'现行 1.3/1.6 倍 → 2.5%/3%': LADDER0,
            '1.2/1.4 倍': [(1.0, 0.02, 0.15), (1.2, 0.025, 0.20), (1.4, 0.03, 0.20)],
            '1.5/2.0 倍': [(1.0, 0.02, 0.15), (1.5, 0.025, 0.20), (2.0, 0.03, 0.20)],
            '只升到 2.5%': [(1.0, 0.02, 0.15), (1.3, 0.025, 0.20)]}
    for nm, L in LADS.items():
        case(f'盈利升档 {nm}', S_base, sizes=(('盈利升档', ('ladder',), True),), ladder=L)
    case('盈利升档 现行 + 定仓 min(已实现,盯市)', S_base, sizes=(('盈利升档', ('ladder',), True),), cap_min=True)
    # C 起点
    starts = {}
    for s0 in ('2005-05-20', '2008-01-02', '2011-01-04', '2014-01-02', '2017-03-01', '2020-01-02'):
        dd = dict(data, days=[d for d in data['days'] if d >= s0])
        out = {}
        for lab, sz, comp in (('盈利升档', ('ladder',), True), ('2%15%复利', COMP, True)):
            st, E = run2(dd, S_base, sz, X0)
            e = np.array([x['equity'] for x in E])
            out[lab] = dict(终值万=round(e[-1] / 1e4, 1), 年化=round(((e[-1] / 1e6) ** (244 / len(e)) - 1) * 100, 1),
                            最大回撤=round(max(x['dd'] for x in E) * 100, 1), 跌破本金最深=round(max(0, 1 - e.min() / 1e6) * 100, 1))
        starts[s0] = out
        log(f'起点 {s0}：{out}')
    # D 压力（基准，固定 2% 与 2%15% 复利）
    ST = {lab: stress(keep[('基准 v2.7', lab)][1], keep[('基准 v2.7', lab)][0]) for lab in ('固定2%', '2%15%复利')}
    stL, EL = run2(data, S_base, ('ladder',), X0)
    ST['盈利升档'] = stress(EL, stL)
    WIN = {lab: windows(E, st) for lab, (st, E) in (('2%15%复利', keep[('基准 v2.7', '2%15%复利')]), ('盈利升档', (stL, EL)))}
    # E 分环境
    st, E = keep[('基准 v2.7', '固定2%')]
    REG = regimes(data, E, st)
    out = dict(params=res, starts=starts, stress=ST, windows=WIN, regimes=REG, margin_assume=MARGIN, limit_assume=LIMIT,
               range=f"{data['days'][1]} ~ {data['days'][-1]}")
    json.dump(out, open(os.path.join(a.out, 'robust05.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    # 摘要
    md = [f"# v2.7 稳健性与压力测试（{out['range']}）", '', '## A/B 参数邻域（固定 2% 不复利；2%15% 复利）', '',
          '| 变体 | 固定2% 终值 | 回撤 | R净 外/内 | 样本外 万/年 | 样本内 万/年 | 复利 终值 | 年化 | 回撤 | 样本外 年化 | 样本内 年化 |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for k, v in res.items():
        f, c = v.get('固定2%'), v.get('2%15%复利') or v.get('盈利升档')
        ff = (f"{f['终值万']} | {f['最大回撤']}% | {f['R净_样本外']}/{f['R净_样本内']} | {f['样本外'].get('年均万')} | {f['样本内'].get('年均万')}"
              if f else '- | - | - | - | -')
        cc = (f"{c['终值万']} | {c.get('年化')}% | {c['最大回撤']}% | {c['样本外'].get('年化')}% | {c['样本内'].get('年化')}%" if c else '- | - | - | - | -')
        md.append(f'| {k} | {ff} | {cc} |')
    md += ['', '## C 盈利升档 / 2%15% 复利 不同起点', '', '| 起点 | 升档 终值/年化/回撤 | 复利 终值/年化/回撤 |', '|---|---|---|']
    for s0, o in starts.items():
        md.append(f"| {s0} | {o['盈利升档']['终值万']} / {o['盈利升档']['年化']}% / {o['盈利升档']['最大回撤']}% | "
                  f"{o['2%15%复利']['终值万']} / {o['2%15%复利']['年化']}% / {o['2%15%复利']['最大回撤']}% |")
    md += ['', '## D 压力', '', '```', json.dumps(ST, ensure_ascii=False, indent=1, default=float), '```', '', '## 极端区间', '', '```',
           json.dumps(WIN, ensure_ascii=False, indent=1), '```', '', '## E 分环境（v2.7 固定 2%）', '', '```', json.dumps(REG, ensure_ascii=False, indent=1), '```']
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))
    log(f'全部完成（{time.time() - t00:.0f}s）')


if __name__ == '__main__':
    main()
