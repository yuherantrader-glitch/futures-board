"""上线前对账：用 ftrade/live27.py（线上实盘账户的引擎）逐日推进历史，和 tools/live27.py 的上线口径回测（B）逐笔比对。

  python tools/live_check.py --data hist/all --out site/bt/livecheck --start 2005-01-01 --basis-start 2012-06-01

信号同回测（v27_repro.make_S，V27），只检验线上引擎、定仓、降档、60 日不重置、存取状态是否与回测一致。
每天把状态写进 JSON 再读回（与线上每天一次收盘场相同），以发现序列化问题。
"""
import argparse
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tools'))
import v27_repro as VR  # noqa: E402
import live27 as BT  # noqa: E402
from ftrade import kline, tracker, store  # noqa: E402
from ftrade import live27 as LV  # noqa: E402
import backtest as bt  # noqa: E402
import lagtest as LG  # noqa: E402
import optimize as op  # noqa: E402

log = VR.log


def key(t):
    return (t['base'], t['v'], t['dir'], t['lots'], round(t['entry'], 2), t['exitDate'], round(t['exit'] or 0, 2), t['why'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/livecheck')
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
    log(f'准备完成（{time.time() - t00:.0f}s）')

    # 回测 B
    stB, EB = BT.run_live(data, S, X0)
    # 线上引擎逐日推进（每天存取一次 JSON）
    store.ROOT = tempfile.mkdtemp()
    op.COMPOUND = False
    st = {'open': [], 'closed': [], 'from': data['days'][0], 'risk': {'pk': LV.CAP0, 'lvl': 0}}
    E = []
    for D in data['days'][1:]:
        T = LV.engine(RAW, ADJ, ROLLS)
        sigs = [s for s in S[D]['sigs'] if op.ma5_ok(ADJ, s['v'], D, s['dir'])]
        LV.advance(T, st, [D], D, sigs)
        store.save(LV.FN, {'state': st})
        st = store.load(LV.FN, None)['state']
        E.append((D, st['risk']['eq']))
    A = sorted([key(t) for t in st['closed']])
    B = sorted([key(t) for t in stB['closed'] if not t.get('observe')])
    onlyA = [x for x in A if x not in set(B)]
    onlyB = [x for x in B if x not in set(A)]
    eqB = {e['date']: e['equity'] for e in EB}
    diff = max(abs(eq - eqB.get(d, eq)) for d, eq in E)
    out = {'线上引擎': dict(终值万=round(E[-1][1] / 1e4, 1), 笔=len(A), 期末持仓=len(st['open'])),
           '回测B': dict(终值万=round(EB[-1]['equity'] / 1e4, 1), 笔=len(B), 期末持仓=len(stB['open'])),
           '逐日净值最大差元': round(diff), '只在线上': onlyA[:30], '只在回测': onlyB[:30],
           '一致': not onlyA and not onlyB and diff < 1,
           '最近6个月逐笔(线上引擎)': [dict(信号日=t['base'], 成交=t.get('fillDate'), 品种=t['v'], 方向=t['dir'], 手=t['lots'], 开=t['entry'],
                                       止损初=t.get('stopPx'), 平日=t['exitDate'], 平=t['exit'], 原因=t['why'], 净元=round(t['pnl'] - op.tcost(t)))
                                  for t in sorted(st['closed'], key=lambda t: t['base']) if (t['exitDate'] or '') >= '2026-05-01'],
           '期末持仓(线上引擎)': [dict(信号日=p['base'], 品种=p['v'], 方向=p['dir'], 手=p['lots'], 开=p['entry'], 止损=p['stopPx'],
                                    持有天=p.get('age'), 浮盈元=round(p.get('pnl') or 0)) for p in st['open']]}
    json.dump(out, open(os.path.join(a.out, 'livecheck.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('# 线上引擎对账\n\n```\n' + json.dumps(out, ensure_ascii=False, indent=1, default=str) + '\n```\n')
    log(f"对账：一致={out['一致']} 线上 {out['线上引擎']} 回测 {out['回测B']} 最大差 {out['逐日净值最大差元']}")
    log(f'全部完成（{time.time() - t00:.0f}s）')


if __name__ == '__main__':
    main()
