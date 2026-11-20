"""长回测：用现行系统（实盘 v2.5 主表 + 研究版 v2.7 冻结 / ECB）跑 2005 年至今。

  python tools/long05.py --data hist/all --out site/bt/long05 --start 2005-01-01 --basis-start 2011-01-01

- 候选、强席位、基差、引擎全部复用 tools/v27_repro.py（无前视 clean 口径），只是把起点提前、强席位检验年从数据第 3 年开始。
- 基差：生意社现期表从 --basis-start 起抓；没有基差的日子不做基差过滤（照常开仓），报告里单列。
- v2.5：tracker.select_signals 主表（多空各前 5）+ 5 日线过滤 + 3ATR 移动止损 + 将军撤退/大将军转向出场；
  不含小品种池和凯利（凯利早年会停摆，见 数据与回测约定.md），用固定 2% 风险、组合上限 10%。
- 分段：2005–2016 为 v2.7 参数从未见过的样本外，2017 起为样本内。
- 数据缺口：大商所 2019 年前没有席位（不出信号）；郑商所 2010 年前没有；2005–2009 只有上期所。
"""
import argparse
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
from ftrade import kline, tracker  # noqa: E402
from ftrade.names import NAMES  # noqa: E402
import backtest as bt  # noqa: E402
import lagtest as LG  # noqa: E402
import optimize as op  # noqa: E402

log = VR.log
SPLIT = '2017-01-01'


def v25_S(data):
    """实盘 v2.5 主表信号（用同一份按日候选 R，避免与 v2.7 口径不一）。"""
    S = {}
    for D in data['days']:
        rows = [dict(r, name=r['v']) for r in data['R'].get(D, [])]
        sigs = tracker.select_signals(rows)
        S[D] = {'sigs': [{**{k: s[k] for k in ('v', 'name', 'dir', 'stars', 'score', 'gr', 'ngen', 'gChg')}, 'sig': D + '收盘', 'note': ''}
                         for s in sigs],
                'rows': {r['v']: {k: r[k] for k in ('dir', 'gr', 'gChg', 'oi', 'score', 'retreat', 'bigTurn', 'ngen')} for r in rows}}
    return S


def run_v25(data, S, compound):
    op.COMPOUND = compound
    xcfg = dict(LG.V25, **LG.REAL)
    T = LG.LT(data['RAW'], data['ADJ'], data['ROLLS'], {'trail': 0, 'ma5': False, **xcfg, 'maxHold': 60, 'fixed': 0.02})
    T.cfg['maxHeat'] = 0.10
    st = {'open': [], 'closed': [], 'from': data['days'][0]}
    E = []
    for D in data['days'][1:]:
        x = S[D]
        sigs = [s for s in x['sigs'] if op.ma5_ok(data['ADJ'], s['v'], D, s['dir'])]
        op.compound(T, st)
        T.step(D, st, [dict(s) for s in sigs], set(), today=True, weak=LG.weak_from(x['rows'], xcfg))
        real = [t for t in st['closed'] if not t.get('observe')]
        fl = sum(p['pnl'] or 0 for p in st['open'] if not p.get('observe') and not p.get('pending'))
        cost = sum(op.tcost(t) for t in real)
        E.append(dict(date=D, realized=round(sum(t['pnl'] or 0 for t in real)), floating=round(fl), cost=round(cost),
                      capBase=round(T.cfg['capital']), tier='2%/10%', nOpen=sum(1 for p in st['open'] if not p.get('observe')),
                      heat=round(sum(p.get('budgetPct') or 0 for p in st['open'] if not p.get('observe')), 4)))
    pk = 1e6
    for e in E:
        e['equity'] = 1e6 + e['realized'] + e['floating'] - e['cost']
        pk = max(pk, e['equity'])
        e['peak'] = round(pk)
        e['dd'] = round(1 - e['equity'] / pk, 5)
        e['equity'] = round(e['equity'])
    return st, E


def seg(E, d0, d1, compound):
    """区段：起止净值、区段内收益（复利=倍数；不复利=万元）、区段内最大回撤（从区段起点重新算峰值）。"""
    xs = [e for e in E if d0 <= e['date'] < d1]
    if len(xs) < 2:
        return {}
    prev = [e for e in E if e['date'] < d0]
    e0 = prev[-1]['equity'] if prev else 1e6
    eq = np.array([e0] + [e['equity'] for e in xs], float)
    pk = np.maximum.accumulate(eq)
    if compound:
        dd = (1 - eq / pk).max()
        yrs = len(xs) / 244
        return dict(起=xs[0]['date'], 止=xs[-1]['date'], 倍数=round(eq[-1] / e0, 2), 年化=round(((eq[-1] / e0) ** (1 / yrs) - 1) * 100, 1),
                    最大回撤=round(dd * 100, 1))
    dd = (pk - eq).max() / 1e6    # 不复利：回撤按初始本金的百分比
    return dict(起=xs[0]['date'], 止=xs[-1]['date'], 净盈亏万=round((eq[-1] - e0) / 1e4, 1), 年均万=round((eq[-1] - e0) / 1e4 / (len(xs) / 244), 1),
                最大回撤占本金=round(dd * 100, 1))


def yearly(E, st):
    real = [t for t in st['closed'] if not t.get('observe')]
    out = {}
    for y in sorted({e['date'][:4] for e in E}):
        xs = [e for e in E if e['date'][:4] == y]
        prev = [e for e in E if e['date'][:4] < y]
        e0 = prev[-1]['equity'] if prev else 1e6
        tr = [t for t in real if (t.get('exitDate') or '')[:4] == y]
        out[y] = dict(年末万=round(xs[-1]['equity'] / 1e4, 1), 收益=round((xs[-1]['equity'] / e0 - 1) * 100, 1), 笔=len(tr),
                      品种=len({t['v'] for t in tr}))
    return out


def plot(curves, out, title, fname, log_y=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in fm.fontManager.ttflist:
        if 'Noto Sans CJK' in f.name or 'WenQuanYi' in f.name:
            plt.rcParams['font.family'] = f.name
            break
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={'height_ratios': [3, 1.3]})
    for name, (E, color, w) in curves.items():
        x = pd.to_datetime([e['date'] for e in E])
        a1.plot(x, [e['equity'] / 1e4 for e in E], color=color, lw=w, label=f"{name}  {E[-1]['equity'] / 1e4:.0f}万")
        a2.plot(x, [-e['dd'] * 100 for e in E], color=color, lw=w * 0.6)
    for a in (a1, a2):
        a.axvline(pd.Timestamp(SPLIT), color='#888', ls='--', lw=0.8)
        a.grid(alpha=0.3)
    a1.text(pd.Timestamp(SPLIT), a1.get_ylim()[1], ' ← 样本外 | 样本内 →', va='top', ha='center', fontsize=9, color='#555')
    if log_y:
        a1.set_yscale('log')
    a1.set_ylabel('净值（万元）')
    a2.set_ylabel('回撤 %')
    a1.legend(loc='upper left', fontsize=9, frameon=False)
    a1.set_title(title, loc='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/long05')
    ap.add_argument('--start', default='2005-01-01')
    ap.add_argument('--end', default='2026-10-31')
    ap.add_argument('--basis-start', default='2011-01-01')
    ap.add_argument('--cachedir', default='hist')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tracker.MULT.update(bt.EXTRA_MULT)
    tracker.MULT.update({k: v for k, v in dict(WH=20, PM=50, RI=20, OI=10, ZC=100, MA=10).items() if k not in tracker.MULT})
    bars, H = bt.load([a.data])
    H = {d: x for d, x in H.items() if a.start <= d <= a.end}
    log(f'数据：{len(bars)} 个合约，席位 {len(H)} 天（{min(H)} ~ {max(H)}）')
    cover = {}
    for d, x in H.items():
        cover.setdefault(d[:4], set()).update(x)
    cover = {y: sorted(v) for y, v in sorted(cover.items())}
    for y, v in cover.items():
        log(f'  {y} 有席位品种 {len(v)}：{" ".join(v)}')
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
    log(f'基差表 {blow.cover}')
    y0 = int(min(H)[:4])
    seat = VR.strong_fn(H, ADJ, 'clean', years=range(y0 + 2, int(a.end[:4]) + 1))
    log(f'强席位训练截止：{seat.train}')
    log(f'强席位每年名单数：{ {y: len(x) for y, x in seat.lists.items()} }')
    xcfg = dict(LG.V25, **LG.REAL, noWeak=True)
    S_ecb, _ = VR.make_S(data, carry, blow, seat, False)
    S_v27, SIG = VR.make_S(data, carry, blow, seat, True)
    S_v25 = v25_S(data)
    pd.DataFrame(SIG).to_csv(os.path.join(a.out, 'signals_daily.csv'), index=False, encoding='utf-8-sig')
    CASES = [('V27_盈利升档', S_v27, ('ladder',), True), ('V27_2%15%复利', S_v27, ('fixed', 0.02, 0.15, True), True),
             ('V27_固定2%', S_v27, ('fixed', 0.02, 0.10, False), False),
             ('ECB_盈利升档', S_ecb, ('ladder',), True), ('ECB_2%15%复利', S_ecb, ('fixed', 0.02, 0.15, True), True),
             ('ECB_固定2%', S_ecb, ('fixed', 0.02, 0.10, False), False)]
    res, EQ = {}, {}
    for name, S, sz, comp in CASES:
        t0 = time.time()
        st, E = VR.run(data, S, sz, xcfg)
        m = VR.export(name, st, E, a.out)
        m['样本外2005-2016'] = seg(E, '0000', SPLIT, comp)
        m['样本内2017起'] = seg(E, SPLIT, '9999', comp)
        m['分年'] = yearly(E, st)
        res[name], EQ[name] = m, E
        log(f'{name}：{ {k: v for k, v in m.items() if k != "分年"} }（{time.time() - t0:.0f}s）')
    for name, comp in (('V25主表_固定2%', False), ('V25主表_2%10%复利', True)):
        t0 = time.time()
        st, E = run_v25(data, S_v25, comp)
        m = VR.export(name, st, E, a.out)
        m['样本外2005-2016'] = seg(E, '0000', SPLIT, comp)
        m['样本内2017起'] = seg(E, SPLIT, '9999', comp)
        m['分年'] = yearly(E, st)
        res[name], EQ[name] = m, E
        log(f'{name}：{ {k: v for k, v in m.items() if k != "分年"} }（{time.time() - t0:.0f}s）')
    rng = f"{data['days'][1]} ~ {data['days'][-1]}"
    plot({'v2.7 盈利升档': (EQ['V27_盈利升档'], '#c0392b', 2.0), 'ECB 盈利升档': (EQ['ECB_盈利升档'], '#2a6f97', 1.5),
          'v2.7 2%/15% 复利': (EQ['V27_2%15%复利'], '#e08e45', 1.2), 'v2.5 主表 2%/10% 复利': (EQ['V25主表_2%10%复利'], '#2e8b57', 1.2)},
         a.out, f'长回测 复利（{rng}，100 万起，对数坐标）', 'curves_compound.png')
    plot({'v2.7 固定2%': (EQ['V27_固定2%'], '#c0392b', 2.0), 'ECB 固定2%': (EQ['ECB_固定2%'], '#2a6f97', 1.5),
          'v2.5 主表 固定2%': (EQ['V25主表_固定2%'], '#2e8b57', 1.2)},
         a.out, f'长回测 不复利（{rng}，100 万，每笔风险 2%）', 'curves_fixed.png', log_y=False)
    json.dump(dict(range=rng, cover=cover, basis=blow.cover, strong_lists={y: len(x) for y, x in seat.lists.items()}, res=res),
              open(os.path.join(a.out, 'long05.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    md = [f'# 长回测（{rng}）', '', f'基差覆盖 {blow.cover}；强席位名单年份 {sorted(seat.lists)}', '',
          '| 方案 | 终值万 | 年化 | 最大回撤 | 笔 | 胜率 | R净 | 样本外 2005–2016 | 样本内 2017 起 |', '|---|---|---|---|---|---|---|---|---|']
    for k, m in res.items():
        md.append(f"| {k} | {m['终值万']} | {m['年化']}% | {m['最大回撤']}% | {m['笔']} | {m['胜率']}% | {m['平均R净']} | {m['样本外2005-2016']} | {m['样本内2017起']} |")
    md += ['', '## 分年（年末万 / 当年收益 / 平仓笔数 / 品种数）', '', '| 年 | ' + ' | '.join(res) + ' |', '|---' * (len(res) + 1) + '|']
    for y in sorted(res['V27_盈利升档']['分年']):
        md.append(f'| {y} | ' + ' | '.join(
            (lambda z: f"{z['年末万']} / {z['收益']:+.1f}% / {z['笔']} / {z['品种']}" if z else '-')(m['分年'].get(y)) for m in res.values()) + ' |')
    md += ['', '## 每年有席位数据的品种', ''] + [f'- {y}（{len(v)}）：{" ".join(v)}' for y, v in cover.items()]
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
