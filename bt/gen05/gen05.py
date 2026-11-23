"""大将军增量的样本外检验（2005–2016）+ 全期复核。同 long05 数据与引擎（实盘口径候选、无前视）。

  python tools/gen05.py --data hist/all --out site/bt/gen05 --start 2005-01-01 --basis-start 2012-06-01

甲1 每笔（复刻《候选假设-供需结构与将军增量》口径）：顺期限结构 + 顺 5 日线的大品种里，
    ① 无将军信号（当天未入选，按 5 日线方向，每 2 个交易日取样） ② 有信号·大将军<3 ③ 有信号·大将军≥3（新信号）。
    T2 + 只靠 3ATR 移动止损（最长 60 根），扣 0.03R；周聚类自助 95% 区间；分段 2005–2012 / 2013–2016 / 2017 起、分年、多空。
甲2 引擎：V27 / ECB 三种选股排序——现行（大将军数优先，再按分歧强度）/ 只按分歧强度（不看大将军）/ 只做大将军≥3；
    固定 2% 与 2%15% 复利，分段。
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
import vstats as VS  # noqa: E402
from ftrade import kline, tracker  # noqa: E402
import backtest as bt  # noqa: E402
import lagtest as LG  # noqa: E402

log = VR.log
SEGS = B5.SEGS + [('2005-2016', '0000', '2017-01-01'), ('全部', '0000', '9999')]


def wk(d):
    import datetime as dt
    y, w, _ = dt.date.fromisoformat(d).isocalendar()
    return f'{y}-{w:02d}'


def cstats(L):
    if not L:
        return {'n': 0}
    r = np.array([x['R'] for x in L])
    g = {}
    for x in L:
        g.setdefault(wk(x['d']), []).append(x['R'])
    ks = list(g)
    bs = []
    for _ in range(1000):
        s = [v for k in np.random.choice(len(ks), len(ks)) for v in g[ks[k]]]
        bs.append(np.mean(s))
    return dict(n=len(r), 每笔R=round(float(r.mean()), 3), 胜率=round(float((r > 0).mean()) * 100, 1),
                区间95=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])


def cdiff(A, B):
    """A − B 每笔差，周聚类自助（两组按同一组周重抽）。"""
    if not A or not B:
        return None
    ga, gb = {}, {}
    for x in A:
        ga.setdefault(wk(x['d']), []).append(x['R'])
    for x in B:
        gb.setdefault(wk(x['d']), []).append(x['R'])
    ks = sorted(set(ga) | set(gb))
    bs = []
    for _ in range(1000):
        pick = [ks[k] for k in np.random.choice(len(ks), len(ks))]
        a = [v for k in pick for v in ga.get(k, [])]
        b = [v for k in pick for v in gb.get(k, [])]
        if a and b:
            bs.append(np.mean(a) - np.mean(b))
    d = np.mean([x['R'] for x in A]) - np.mean([x['R'] for x in B])
    return dict(差=round(float(d), 3), 区间95=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])


def make_S2(data, carry, blow, seat, use_seat, key='cur', only3=False):
    """= v27_repro.make_S，可换排序：cur=(-大将军数, -分歧强度)；conf=只按分歧强度。only3=只取大将军≥3。"""
    S = {}
    for D in data['days']:
        rows = data['R'].get(D, [])
        c = [r for r in rows if VR.elig(r) and (not only3 or r['nbig'] >= 3)]
        c = sorted(c, key=(lambda r: (-r['nbig'], -VR.conf(r))) if key == 'cur' else (lambda r: -VR.conf(r)))
        chosen = None
        for r in c:
            ca = carry(r['v'], D)
            ag = ca is not None and ca != 0 and (ca > 0) != (r['dir'] == '多')
            if not ag and not blow(r['v'], D, r['dir']) and not (use_seat and seat(r['v'], D, r['dir'])):
                chosen = r
                break
        top = [chosen] if chosen else []
        S[D] = {'sigs': [{**{k: s[k] for k in ('v', 'dir', 'score', 'gr', 'ngen', 'gChg')}, 'stars': 3 if s['score'] >= 6 else 2,
                          'name': s['v'], 'sig': D + '收盘', 'note': ''} for s in top]}
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/gen05')
    ap.add_argument('--start', default='2005-01-01')
    ap.add_argument('--end', default='2026-10-31')
    ap.add_argument('--basis-start', default='2012-06-01')
    ap.add_argument('--cachedir', default='hist')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    np.random.seed(11)
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
    out = {}

    # ---------- 甲1 每笔 ----------
    idx = {v: {x['d']: i for i, x in enumerate(k)} for v, k in ADJ.items()}
    prev, rec = set(), []
    for n, D in enumerate(data['days']):
        rows = data['R'].get(D, [])
        now = set()
        for r in rows:
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
            x = VS.trade(k, i, dr)
            if x is None:
                continue
            rec.append(dict(d=D, v=v, dir=dr, g=grp, R=x))
        prev = now
    log(f'甲1 样本 {len(rec)}')
    g1 = {}
    for nm, s0, s1 in SEGS:
        L = [x for x in rec if s0 <= x['d'] < s1]
        G = {k: [x for x in L if x['g'] == k] for k in ('无信号', '大将军<3', '大将军≥3')}
        e = {k: cstats(v) for k, v in G.items()}
        e['有信号(全部)'] = cstats(G['大将军<3'] + G['大将军≥3'])
        e['差 大将军≥3−无信号'] = cdiff(G['大将军≥3'], G['无信号'])
        e['差 大将军≥3−大将军<3'] = cdiff(G['大将军≥3'], G['大将军<3'])
        e['差 大将军<3−无信号'] = cdiff(G['大将军<3'], G['无信号'])
        for dr in ('多', '空'):
            e[f'{dr}·差 大将军≥3−无信号'] = cdiff([x for x in G['大将军≥3'] if x['dir'] == dr], [x for x in G['无信号'] if x['dir'] == dr])
        g1[nm] = e
    yr = {}
    for y in sorted({x['d'][:4] for x in rec}):
        L = [x for x in rec if x['d'][:4] == y]
        m = {k: [x['R'] for x in L if x['g'] == k] for k in ('无信号', '大将军<3', '大将军≥3')}
        yr[y] = {k: (f'{len(v)} 笔 {np.mean(v):+.3f}' if v else '0') for k, v in m.items()}
        yr[y]['差≥3−无'] = round(float(np.mean(m['大将军≥3']) - np.mean(m['无信号'])), 3) if m['大将军≥3'] and m['无信号'] else None
    g1['分年'] = yr
    out['甲1_每笔'] = g1

    # ---------- 甲2 引擎 ----------
    X0 = dict(LG.V25, **LG.REAL, noWeak=True)
    eng = {}
    for lab, use in (('V27', True), ('ECB', False)):
        for nm, kw in (('现行排序', {}), ('不看大将军(只按分歧强度)', dict(key='conf')), ('只做大将军≥3', dict(only3=True))):
            S = make_S2(data, carry, blow, seat, use, **kw)
            for sl, sz, comp in (('固定2%', ('fixed', 0.02, 0.10, False), False), ('2%15%复利', ('fixed', 0.02, 0.15, True), True)):
                st, E = A5.run3(data, S, sz, X0)
                m = A5.summ(E, st, comp)
                eng[f'{lab}·{nm}·{sl}'] = {k: m[k] for k in ('终值万', '年化', '最大回撤', '笔', 'R净')} | {'分段': B5.segstats(E, comp)}
                log(f'{lab}·{nm}·{sl}：{eng[f"{lab}·{nm}·{sl}"]}')
    out['甲2_引擎'] = eng
    out['说明'] = '甲1 口径复刻 2017 起研究（顺期限结构 + 顺 5 日线、大品种）；2017 起一段用于和原研究对照。分段 2005–2012 只有上期所（2010 起含郑商所）。'
    json.dump(out, open(os.path.join(a.out, 'gen05.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('# 大将军增量 样本外检验\n\n```\n' + json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n```\n')
    log(f'全部完成（{time.time() - t00:.0f}s）')


if __name__ == '__main__':
    main()
