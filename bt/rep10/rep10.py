"""2010 起报告数据：账户 2010-01 起 100 万新开（席位名单、打分仍用 2005 起全部历史，口径不变）。

  python tools/rep10.py --data hist/all --out site/bt/rep10 --start 2005-01-01 --acct 2010-01-01 --basis-start 2012-06-01

输出 rep10.json：各口径摘要、周度净值/回撤/降档、月度收益、交易统计（板块、品种、方向、出场原因、持有天数、最好最差）。
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tools'))
import real05 as RL  # noqa: E402
import v27_repro as VR  # noqa: E402
import lagtest as LG  # noqa: E402
import optimize as op  # noqa: E402

log = VR.log


def lvl_series(E):
    pk, lvl, out = 1e6, 0, []
    for e in E:
        dd = 1 - e['equity'] / pk
        if dd >= 0.15:
            lvl = 2
        elif dd >= 0.10:
            lvl = max(lvl, 1)
        elif dd <= 0.05:
            lvl = 0
        out.append(lvl)
        pk = max(pk, e['equity'])
    return out


def weekly(E, lv=None):
    out, last = [], None
    for i, e in enumerate(E):
        wk = np.datetime64(e['date'], 'W')
        if last is not None and wk != last:
            p = E[i - 1]
            out.append([p['date'], round(p['equity'] / 1e4, 2), round(p['dd'] * 100, 2)] + ([lv[i - 1]] if lv else []))
        last = wk
    p = E[-1]
    out.append([p['date'], round(p['equity'] / 1e4, 2), round(p['dd'] * 100, 2)] + ([lv[-1]] if lv else []))
    return out


def monthly(E):
    ms, prev, out = {}, 1e6, {}
    for e in E:
        ms[e['date'][:7]] = e['equity']
    for m, v in sorted(ms.items()):
        out[m] = round((v / prev - 1) * 100, 2)
        prev = v
    return out


def dd_periods(E, n=5):
    res, pk, pkd, trough, trd, inn = [], 1e6, E[0]['date'], 1e6, None, False
    for e in E:
        q = e['equity']
        if q >= pk:
            if inn:
                res.append(dict(起=pkd, 谷=trd, 恢复=e['date'], 深=round((1 - trough / pk) * 100, 1)))
            pk, pkd, trough, inn = q, e['date'], q, False
        else:
            inn = True
            if q < trough:
                trough, trd = q, e['date']
    if inn:
        res.append(dict(起=pkd, 谷=trd, 恢复=None, 深=round((1 - trough / pk) * 100, 1)))
    days = {e['date']: i for i, e in enumerate(E)}
    for r in res:
        r['交易日'] = (days.get(r['恢复'], len(E) - 1) - days[r['起']])
    return sorted(res, key=lambda r: -r['深'])[:n]


def trade_stats(st):
    real = [t for t in st['closed'] if not t.get('observe') and t.get('risk')]
    for t in real:
        t['_net'] = (t['pnl'] or 0) - op.tcost(t)
        t['_R'] = t['_net'] / t['risk']

    def agg(L):
        if not L:
            return None
        w = [t for t in L if t['_net'] > 0]
        gw, gl = sum(t['_net'] for t in w), -sum(t['_net'] for t in L if t['_net'] <= 0)
        return dict(笔=len(L), 胜率=round(len(w) / len(L) * 100, 1), 每笔R=round(float(np.mean([t['_R'] for t in L])), 3),
                    净万=round(sum(t['_net'] for t in L) / 1e4, 1), 盈亏比=round(gw / gl, 2) if gl else None,
                    平均盈R=round(float(np.mean([t['_R'] for t in w])), 2) if w else None,
                    平均亏R=round(float(np.mean([t['_R'] for t in L if t['_net'] <= 0])), 2) if len(w) < len(L) else None)
    g = lambda key: {k: agg(v) for k, v in sorted(_grp(real, key).items(), key=lambda kv: -sum(t['_net'] for t in kv[1]))}
    held = np.array([t.get('held') or 0 for t in real])
    Rs = np.array([t['_R'] for t in real])
    top = sorted(real, key=lambda t: -t['_net'])
    row = lambda t: dict(品种=t.get('name') or t['v'], 方向=t['dir'], 进=t['base'], 出=t['exitDate'], 持有=t.get('held'), 净万=round(t['_net'] / 1e4, 1), R=round(t['_R'], 2), 出场=t.get('why'))
    tot = sum(t['_net'] for t in real if t['_net'] > 0)
    srt = sorted([t['_net'] for t in real if t['_net'] > 0], reverse=True)
    return dict(
        全部=agg(real), 方向=g(lambda t: t['dir']), 板块=g(lambda t: t.get('sector') or '其他'),
        品种=g(lambda t: t.get('name') or t['v']), 出场原因=g(lambda t: (t.get('why') or '其他').split('(')[0]),
        分年=g(lambda t: t['exitDate'][:4]),
        持有天数分布={k: int(((held >= a) & (held < b)).sum()) for k, a, b in (('1-5', 0, 6), ('6-20', 6, 21), ('21-40', 21, 41), ('41-60', 41, 61), ('60+', 61, 9999))},
        R分布={k: int(((Rs >= a) & (Rs < b)).sum()) for k, a, b in (('≤-1.5', -99, -1.5), ('-1.5~-0.5', -1.5, -0.5), ('-0.5~0', -0.5, 0), ('0~1', 0, 1), ('1~3', 1, 3), ('3~6', 3, 6), ('≥6', 6, 99))},
        最好10=[row(t) for t in top[:10]], 最差10=[row(t) for t in top[-10:][::-1]],
        前10笔占盈利=round(sum(srt[:10]) / tot * 100, 1) if tot else None, 前10pct笔占盈利=round(sum(srt[:max(1, len(srt) // 10)]) / tot * 100, 1) if tot else None,
        最长连亏=_streak(real), 平均持有=round(float(held.mean()), 1))


def _grp(L, f):
    G = defaultdict(list)
    for t in L:
        G[f(t)].append(t)
    return G


def _streak(real):
    m = c = 0
    for t in sorted(real, key=lambda t: t['exitDate']):
        c = c + 1 if t['_net'] <= 0 else 0
        m = max(m, c)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/rep10')
    ap.add_argument('--start', default='2005-01-01')
    ap.add_argument('--end', default='2026-10-31')
    ap.add_argument('--acct', default='2010-01-01')
    ap.add_argument('--basis-start', default='2012-06-01')
    ap.add_argument('--cachedir', default='hist')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t00 = time.time()
    data, carry, blow, seat = RL.prep(a)
    S, _ = VR.make_S(data, carry, blow, seat, True)
    X0 = dict(LG.V25, **LG.REAL, noWeak=True)
    days = data['days']
    i0 = next(i for i, d in enumerate(days) if d >= a.acct)
    d10 = dict(data, days=days[i0 - 1:])
    log(f'准备完成（{time.time() - t00:.0f}s），账户起点 {d10["days"][1]}')
    out = {'起点': d10['days'][1], '终点': days[-1], '口径': {}, '曲线': {}}
    V = (('现行回测口径', dict(), True), ('计入换月和封板', dict(roll=True, lock=True), True),
         ('1.5%/12%', dict(r=0.015, h=0.12, roll=True, lock=True), False), ('2.5%/15%', dict(r=0.025, h=0.15, roll=True, lock=True), False),
         ('3%/15%', dict(r=0.03, h=0.15, roll=True, lock=True), False))
    for nm, kw, full in V:
        st, E = RL.run_real(d10, S, X0, **kw)
        x = RL.describe(st, E)
        x.pop('2005-2016 年化/回撤', None)
        x['2010-2016 年化/回撤'] = list(RL.P5.seg_ann(E, '0000', '2017-01-01'))
        x['2021起 年化/回撤'] = list(RL.P5.seg_ann(E, '2021-01-01', '9999'))
        lv = lvl_series(E)
        x['降档天数 1.5%档/1%档'] = [lv.count(1), lv.count(2)]
        x['回撤段前5'] = dd_periods(E)
        out['口径'][nm] = x
        out['曲线'][nm] = weekly(E, lv if full else None)
        if nm == '计入换月和封板':
            out['月度'] = monthly(E)
            out['交易'] = trade_stats(st)
            out['持仓数分布'] = {k: int(sum(1 for e in E if a0 <= e['nOpen'] <= b0)) for k, a0, b0 in (('0', 0, 0), ('1-2', 1, 2), ('3-4', 3, 4), ('5-6', 5, 6), ('7+', 7, 99))}
        log(f"{nm}：{ {k: v for k, v in x.items() if k not in ('分年', '回撤段前5')} }")
    json.dump(out, open(os.path.join(a.out, 'rep10.json'), 'w', encoding='utf-8'), ensure_ascii=False, default=float)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('# rep10\n\n```\n' + json.dumps({k: v for k, v in out.items() if k != '曲线'}, ensure_ascii=False, indent=1, default=float) + '\n```\n')
    log(f'全部完成（{time.time() - t00:.0f}s）')


if __name__ == '__main__':
    main()
