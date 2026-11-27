"""每笔 4% / 总 20%，回撤 ≥10% 降到 2%/15%，≥15% 降到 1%/10%，回到离高点 5% 内恢复（2010 起，100 万，计入换月和封板）。

  python tools/tier4.py --data hist/all --out site/bt/tier4 --start 2005-01-01 --acct 2010-01-01 --basis-start 2012-06-01
"""
import argparse
import inspect
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tools'))
import real05 as RL  # noqa: E402
import rep10 as RP  # noqa: E402
import v27_repro as VR  # noqa: E402
import lagtest as LG  # noqa: E402

log = VR.log
src = inspect.getsource(RL.run_real)
for a, b in (('def run_real(data, S, X0, r=0.02, h=0.15, roll=False, lock=False):', 'def run_real2(data, S, X0, r=0.02, h=0.15, roll=False, lock=False, steps=None):'),
             ('    steps = [(0.15, r * 0.5, h * 0.67), (0.10, r * 0.75, h * 0.8)]', '    steps = steps or [(0.15, r * 0.5, h * 0.67), (0.10, r * 0.75, h * 0.8)]')):
    assert src.count(a) == 1, a
    src = src.replace(a, b)
exec(src, RL.__dict__)
run_real2 = RL.__dict__['run_real2']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--out', default='site/bt/tier4')
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
    out = {'起点': d10['days'][1], '终点': days[-1], '口径': {}, '曲线': {}}
    V = (('4%/20% 回撤10%降到2%/15%、15%降到1%/10%', dict(r=0.04, h=0.20, steps=[(0.15, 0.01, 0.10), (0.10, 0.02, 0.15)])),
         ('4%/20% 按比例降档(3%/16%、2%/13.4%)', dict(r=0.04, h=0.20)),
         ('2%/15% 现行', dict(r=0.02, h=0.15)))
    for nm, kw in V:
        st, E = run_real2(d10, S, X0, roll=True, lock=True, **kw)
        x = RL.describe(st, E)
        x.pop('2005-2016 年化/回撤', None)
        x['2010-2016 年化/回撤'] = list(RL.P5.seg_ann(E, '0000', '2017-01-01'))
        x['2021起 年化/回撤'] = list(RL.P5.seg_ann(E, '2021-01-01', '9999'))
        lv = RP.lvl_series(E)
        x['降档天数 一档/二档'] = [lv.count(1), lv.count(2)]
        x['回撤段前5'] = RP.dd_periods(E)
        out['口径'][nm] = x
        out['曲线'][nm] = RP.weekly(E, lv)
        log(f"{nm}：{ {k: v for k, v in x.items() if k not in ('分年', '回撤段前5')} }")
    json.dump(out, open(os.path.join(a.out, 'tier4.json'), 'w', encoding='utf-8'), ensure_ascii=False, default=float)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('# tier4\n\n```\n' + json.dumps({k: v for k, v in out.items() if k != '曲线'}, ensure_ascii=False, indent=1, default=float) + '\n```\n')
    log(f'全部完成（{time.time() - t00:.0f}s）')


if __name__ == '__main__':
    main()
