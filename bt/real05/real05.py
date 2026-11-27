"""回测真实性 / 风险档位 / 品种池定义（预登记见项目文档《候选假设-回测真实性风险档位与品种池》）。

  python tools/real05.py --part real --data hist/all --out site/bt/real05 --start 2005-01-01 --basis-start 2012-06-01
  python tools/real05.py --part univ --data hist/all --out site/bt/univ05 --start 2005-01-01 --basis-start 2012-06-01

real：在上线口径 B 上加 ① 换月成本（每次换月按一次完整买卖计成本）② 一字板（当天最高 = 最低且涨跌 ≥2%）
      反向封板时止损、反向信号平仓都无法成交，顺延到下一个能成交的日子；开仓日被同向封板则无法成交（撤单）；
      再在“换月 + 一字板”口径上比较每笔风险档位。
univ：品种池（“大品种”）定义：现行（总持仓 ≥30 万手等）/ 现行 + 上市满 250 个交易日 / 按持仓金额排名前 20、25、30（上市满 250 日）。
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
import persist05 as P5  # noqa: E402
from ftrade import kline, tracker  # noqa: E402
import backtest as bt  # noqa: E402
import lagtest as LG  # noqa: E402
import optimize as op  # noqa: E402

log = VR.log


def run_real(data, S, X0, r=0.02, h=0.15, roll=False, lock=False):
    op.COMPOUND = True
    cnt = dict(锁死止损=0, 锁死平仓=0, 锁死不能开=0)

    class LT5(LG.LT):
        def settle(self, D, st, touched):
            saved = {}
            if lock:
                for p in st['open']:
                    if p.get('lastDate', '') >= D:
                        continue
                    k, i = self.bar(p['v'], D)
                    if k is None or not i:
                        continue
                    b, pc = k[i], k[i - 1]['c']
                    if not pc or b['h'] != b['l'] or abs(b['c'] / pc - 1) < 0.02:
                        continue
                    up = b['c'] > pc
                    sg = 1 if p['dir'] == '多' else -1
                    if p.get('pending'):
                        if (sg > 0) == up:
                            p['status'], p['lastDate'] = '撤单', D
                            cnt['锁死不能开'] += 1
                        continue
                    if (sg > 0) != up:
                        saved[id(p)] = (p['stopPx'], p.get('exitNext'))
                        p['stopPx'] = -1e18 if sg > 0 else 1e18
                        if p.get('exitNext'):
                            p['exitNext'] = None
                            cnt['锁死平仓'] += 1
                        else:
                            cnt['锁死止损'] += 1
            out = super().settle(D, st, touched)
            for p in st['open']:
                s = saved.get(id(p))
                if s:
                    p['stopPx'], p['exitNext'] = s[0], s[1]
            return out

        def step(self, D, st, sigs, touched, today=False, weak=None):
            out = super().step(D, st, sigs, touched, today, weak)
            for p in st['open']:
                if not p.get('pending'):
                    p['held'] = p.get('age', 0)
            return out
    T = LT5(data['RAW'], data['ADJ'], data['ROLLS'], {'trail': 0, 'ma5': False, **X0, 'maxHold': 60, 'fixed': r})
    st = {'open': [], 'closed': [], 'from': data['days'][0]}
    steps = [(0.15, r * 0.5, h * 0.67), (0.10, r * 0.75, h * 0.8)]
    E, last_eq, pk, lvl, rc = [], 1e6, 1e6, 0, 0.0
    for D in data['days'][1:]:
        sigs = [s for s in S[D]['sigs'] if op.ma5_ok(data['ADJ'], s['v'], D, s['dir'])]
        op.compound(T, st)
        T.cfg['capital'] = last_eq
        T.cfg['fixed'], T.cfg['maxHeat'] = r, h
        dd = 1 - last_eq / pk
        if dd >= steps[0][0]:
            lvl = 2
        elif dd >= steps[1][0]:
            lvl = max(lvl, 1)
        elif dd <= 0.05:
            lvl = 0
        if lvl:
            _, rr, hh = steps[0] if lvl == 2 else steps[1]
            T.cfg['fixed'], T.cfg['maxHeat'] = rr, hh
        T.step(D, st, [dict(s) for s in sigs], set(), today=True, weak={})
        real = [t for t in st['closed'] if not t.get('observe')]
        live = [p for p in st['open'] if not p.get('observe') and not p.get('pending')]
        rc = sum(len(t.get('rolls') or []) * op.tcost(t) for t in real + live) if roll else 0.0
        eq = 1e6 + sum(t['pnl'] or 0 for t in real) + sum(p['pnl'] or 0 for p in live) - sum(op.tcost(t) for t in real) - rc
        notional = sum(abs(p.get('lastClose') or p['entry']) * p['lots'] * p['mult'] for p in live)
        last_eq, pk = eq, max(pk, eq)
        E.append(dict(date=D, equity=eq, nOpen=len(live), notional=notional, heat=0))
    pk = 1e6
    for e in E:
        pk = max(pk, e['equity'])
        e['peak'], e['dd'] = pk, 1 - e['equity'] / pk
    st['cnt'], st['rollcost'] = cnt, rc
    return st, E


def describe(st, E):
    m = A5.summ(E, st, True)
    eq = np.array([e['equity'] for e in E])
    dr = eq[1:] / eq[:-1] - 1
    lev = np.array([e['notional'] / e['equity'] for e in E if e['equity'] > 0])
    y0, d0 = P5.seg_ann(E, '0000', '2017-01-01')
    y1, d1 = P5.seg_ann(E, '2017-01-01', '9999')
    real = [t for t in st['closed'] if not t.get('observe')]
    return {k: m[k] for k in ('终值万', '年化', '最大回撤', '笔', 'R净')} | {
        '收益÷回撤': round(m['年化'] / m['最大回撤'], 3) if m['最大回撤'] else None,
        '2005-2016 年化/回撤': [y0, d0], '2017起 年化/回撤': [y1, d1],
        '最差单日%': round(float(dr.min()) * 100, 2), '最差20日%': round(float((eq[20:] / eq[:-20] - 1).min()) * 100, 1),
        '名义杠杆 均/p95/最高': [round(float(lev.mean()), 2), round(float(np.percentile(lev, 95)), 2), round(float(lev.max()), 2)],
        '保证金占用% (按12%/20%保证金率) p95/最高': [round(float(np.percentile(lev, 95)) * 12, 1), round(float(lev.max()) * 12, 1),
                                           round(float(np.percentile(lev, 95)) * 20, 1), round(float(lev.max()) * 20, 1)],
        '换月 次数/成本万': [sum(len(t.get('rolls') or []) for t in real), round(st.get('rollcost', 0) / 1e4, 1)],
        '一字板事件': st.get('cnt'), '分年': m['分年']}


def prep(a):
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
    return data, carry, blow, seat


def judge(x, base):
    ok = [x['2005-2016 年化/回撤'][0] >= base['2005-2016 年化/回撤'][0], (x['收益÷回撤'] or 0) >= (base['收益÷回撤'] or 0) + 0.03,
          x['2017起 年化/回撤'][0] >= base['2017起 年化/回撤'][0] - 1, x['最大回撤'] <= base['最大回撤'] + 2]
    return ok, all(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='real')
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/real05')
    ap.add_argument('--start', default='2005-01-01')
    ap.add_argument('--end', default='2026-10-31')
    ap.add_argument('--basis-start', default='2012-06-01')
    ap.add_argument('--cachedir', default='hist')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t00 = time.time()
    data, carry, blow, seat = prep(a)
    X0 = dict(LG.V25, **LG.REAL, noWeak=True)
    log(f'准备完成（{time.time() - t00:.0f}s）')
    out = {}
    if a.part == 'real':
        S, _ = VR.make_S(data, carry, blow, seat, True)
        res = {}
        for nm, kw in (('R0 现行口径', {}), ('R1 +换月成本', dict(roll=True)), ('R2 +一字板', dict(lock=True)), ('R3 +换月+一字板', dict(roll=True, lock=True))):
            st, E = run_real(data, S, X0, **kw)
            res[nm] = describe(st, E)
            log(f"{nm}：{ {k: v for k, v in res[nm].items() if k != '分年'} }")
        out['真实性'] = res
        tiers = {}
        for r, h in ((0.015, 0.12), (0.02, 0.15), (0.025, 0.15), (0.03, 0.15), (0.03, 0.20), (0.04, 0.20)):
            st, E = run_real(data, S, X0, r=r, h=h, roll=True, lock=True)
            nm = f'每笔{r * 100:g}% 总{h * 100:g}%'
            tiers[nm] = describe(st, E)
            log(f"{nm}：{ {k: v for k, v in tiers[nm].items() if k != '分年'} }")
        out['风险档位(换月+一字板口径，降档按比例)'] = tiers
    else:
        R0 = data['R']
        idx = {v: {x['d']: i for i, x in enumerate(k)} for v, k in data['ADJ'].items()}

        def variant(kind, N=None):
            R2, nb = {}, []
            for D, rows in R0.items():
                if kind == 'U0':
                    big = {r['v'] for r in rows if r.get('big')}
                elif kind == 'U0L':
                    big = {r['v'] for r in rows if r.get('big') and (idx.get(r['v'], {}).get(D) or 0) >= 250}
                else:
                    cand = [(r.get('oi') or 0) * (r.get('close') or 0) * (tracker.MULT.get(r['v']) or 0) for r in rows]
                    order = sorted([(c, r['v']) for c, r in zip(cand, rows) if c > 0 and (idx.get(r['v'], {}).get(D) or 0) >= 250], reverse=True)
                    big = {v for _, v in order[:N]}
                R2[D] = [dict(r, big=r['v'] in big) for r in rows]
                nb.append((D, len(big)))
            return R2, nb
        res = {}
        for nm, kind, N in (('U0 现行门槛', 'U0', None), ('U0L 现行+上市满250日', 'U0L', None),
                            ('T20 持仓金额前20', 'T', 20), ('T25 持仓金额前25', 'T', 25), ('T30 持仓金额前30', 'T', 30)):
            R2, nb = variant(kind, N)
            d2 = dict(data, R=R2)
            S, _ = VR.make_S(d2, carry, blow, seat, True)
            st, E = run_real(d2, S, X0)
            x = describe(st, E)
            x['平均大品种数 2005-12/2013-16/2017起'] = [round(float(np.mean([n for d, n in nb if s0 <= d < s1])), 1) if any(s0 <= d < s1 for d, n in nb) else None
                                           for s0, s1 in (('0000', '2013-01-01'), ('2013-01-01', '2017-01-01'), ('2017-01-01', '9999'))]
            real = [t for t in st['closed'] if not t.get('observe')]
            x['成交品种数'] = len({t['v'] for t in real})
            res[nm] = x
            log(f"{nm}：{ {k: v for k, v in x.items() if k != '分年'} }")
        base = res['U0 现行门槛']
        for nm, x in res.items():
            if not nm.startswith('U0 '):
                x['通过标准①②③④'], x['通过'] = judge(x, base)
        out['品种池'] = res
    out['说明'] = '通过标准（品种池）：①2005–2016 年化 ≥ 现行；②全期收益÷回撤 ≥ 现行 + 0.03；③2017 起年化不低于现行 1 个百分点以上；④最大回撤 ≤ 现行 + 2 个百分点。'
    fn = 'real05' if a.part == 'real' else 'univ05'
    json.dump(out, open(os.path.join(a.out, f'{fn}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write(f'# {fn}\n\n```\n' + json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n```\n')
    log(f'全部完成（{time.time() - t00:.0f}s）')


if __name__ == '__main__':
    main()
