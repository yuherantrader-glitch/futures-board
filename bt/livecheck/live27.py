"""v2.7 实盘账户（2026-11-25 上线）：按上线说明的规则每天给出明早操作，并记一本 100 万起的账（data/shadow/LIVE.json）。

选股：与影子 V27 相同（shadow.pick 'V27'）——合格候选按 大将军数↓、分歧强度↓ 排序，依次排除逆期限结构、基差处低三分之一、
      长期强席位反向，取第 1 个；顺 5 日线才开；次日（夜盘）开盘成交。
出场：3ATR 初始止损 + 3ATR 移动止损（只收紧）；反向信号次日开盘平；满 60 个交易日平（从首次成交算，同向信号不重置）。
定仓：前一日盯市净值（100 万 + 已实现净盈亏 + 持仓浮盈）× 每笔风险；手数向下取整，不足 1 手不做；
      每笔 2%、总 15%、单笔硬上限 3%、同板块同向 6%；
      回撤降档：离高点 ≥10% → 1.5% / 12%，≥15% → 1% / 10%，回到离高点 5% 以内恢复；离高点 ≥25% 暂停开新仓（报警，复核后再定）。
数据缺失：期限结构或强席位缺失 → 当天不开新仓；基差缺失 → 不做基差过滤（与回测一致）。
与 tools/live27.py 的上线口径回测（B）同一套引擎和定仓逻辑；tools/live_check.py 用历史数据逐日复核两者一致。
"""
import copy
import os
import sys

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from ftrade import store  # noqa: E402

KEY = 'LIVE'
FN = 'shadow/LIVE.json'
NAME = 'v2.7 实盘'
CAP0 = 1e6
BASE = (0.02, 0.15)
STEPS = [(0.15, 0.01, 0.10), (0.10, 0.015, 0.12)]   # (离高点回撤 ≥, 每笔, 总上限)，由深到浅
RECOVER = 0.05
PAUSE = 0.25


def engine(RAW, ADJ, ROLLS):
    import lagtest as LG
    from ftrade import shadow

    class LT2(LG.LT):
        """60 日上限按实际持有天数（age）算，同向信号不清零。"""

        def step(self, D, st, sigs, touched, today=False, weak=None):
            out = super().step(D, st, sigs, touched, today, weak)
            for p in st['open']:
                if not p.get('pending'):
                    p['held'] = p.get('age', 0)
            return out
    return LT2(RAW, ADJ, ROLLS, dict(shadow.XCFG))


def equity(st):
    import optimize as op
    real = [t for t in st['closed'] if not t.get('observe')]
    live = [p for p in st['open'] if not p.get('observe') and not p.get('pending')]
    return CAP0 + sum((t.get('pnl') or 0) - op.tcost(t) for t in real) + sum(p.get('pnl') or 0 for p in live)


def sizing(st):
    """开盘前（= 前一日收盘后）定档。返回 (净值, 离高点回撤, 每笔, 总上限)。"""
    R = st.setdefault('risk', {'pk': CAP0, 'lvl': 0})
    eq = equity(st)
    dd = 1 - eq / R['pk'] if R['pk'] > 0 else 0
    if dd >= STEPS[0][0]:
        R['lvl'] = 2
    elif dd >= STEPS[1][0]:
        R['lvl'] = max(R['lvl'], 1)
    elif dd <= RECOVER:
        R['lvl'] = 0
    r, h = BASE if not R['lvl'] else (STEPS[0][1:] if R['lvl'] == 2 else STEPS[1][1:])
    return eq, dd, r, h


def advance(T, st, days, DT, sigs, log=print):
    """把账户从 st['from'] 推进到 DT；sigs 只在 DT 当天下。返回 (被暂停的信号, 当前档位)。"""
    paused = []
    for d in days:
        eq, dd, r, h = sizing(st)
        T.cfg['capital'], T.cfg['fixed'], T.cfg['maxHeat'] = eq, r, h
        today = [dict(s, sig=d + '收盘', note=s.get('note', '')) for s in sigs] if d == DT else []
        if today and dd >= PAUSE:
            paused = [dict(s, why=f'离高点回撤 {dd * 100:.1f}% ≥ {PAUSE * 100:.0f}%，暂停开新仓') for s in today]
            today = []
        T.step(d, st, today, set(), today=True, weak={})
        R = st['risk']
        R['eq'] = equity(st)
        R['pk'] = max(R['pk'], R['eq'])
        R['r'], R['h'], R['dd'] = r, h, 1 - R['eq'] / R['pk']
    st['from'] = DT
    return paused


def run(DT, rows, RAW, ADJ, ROLLS, carry=None, basis_low=None, seat=None, log=print):
    """收盘场调用。同一天重跑时从当天处理前的备份重算。"""
    import optimize as op
    from ftrade import shadow
    S = store.load(FN, None) or {'state': {'open': [], 'closed': [], 'from': None, 'risk': {'pk': CAP0, 'lvl': 0}},
                                 'prev': None, 'start': DT}
    st = copy.deepcopy(S['prev']['state']) if S.get('prev') and S['prev']['D'] == DT else S['state']
    backup = copy.deepcopy(st)
    T = engine(RAW, ADJ, ROLLS)
    notes, dropped = [], []
    if carry is None:
        picked = []
        notes.append('期限结构数据缺失，今天不开新仓')
    elif seat is None:
        picked = []
        notes.append('强席位名单或席位数据缺失，今天不开新仓')
    else:
        picked, dropped = shadow.pick(rows, 'V27', carry, basis_low, seat)
    if basis_low is None:
        notes.append('基差数据缺失，今天未做基差过滤')
    sigs = [s for s in picked if op.ma5_ok(ADJ, s['v'], DT, s['dir'])]
    skipped = [dict(s, why='逆 5 日线，观望') for s in picked if s not in sigs] + dropped
    frm = st.get('from')
    days = sorted({x['d'] for k in RAW.values() for x in k if (frm is None and x['d'] == DT) or (frm and frm < x['d'] <= DT)})
    skipped += advance(T, st, days, DT, sigs, log=log)
    for t in st['closed']:
        t.setdefault('cost', round(op.tcost(t)))
    S.update(state=st, prev={'D': DT, 'state': backup})
    store.save(FN, S)
    out = summary(st, DT, skipped, S.get('start'), notes)
    log(f"  v2.7 实盘：信号 {len(sigs)}，持仓 {len(st['open'])}，已平 {len(st['closed'])}，净值 {out['eq']:,.0f}")
    return out


def summary(st, D, skipped=(), start=None, notes=()):
    import optimize as op
    closed = [t for t in st['closed'] if not t.get('observe')]
    R = st.get('risk') or {}
    eq = R.get('eq', equity(st))
    return dict(date=D, start=start, eq=round(eq), pk=round(R.get('pk', CAP0)), dd=round(R.get('dd', 0), 4),
                lvl=R.get('lvl', 0), r=R.get('r', BASE[0]), h=R.get('h', BASE[1]),
                net=round(sum((t.get('pnl') or 0) - op.tcost(t) for t in closed)),
                float=round(sum(p.get('pnl') or 0 for p in st['open'] if not p.get('pending'))),
                n=len(closed), win=sum(1 for t in closed if (t.get('pnl') or 0) - op.tcost(t) > 0),
                heat=round(sum(p.get('budgetPct') or 0 for p in st['open']), 4),
                today_open=[p for p in st['open'] if p.get('base') == D],
                today_closed=[t for t in closed if t.get('exitDate') == D],
                exit_next=[p for p in st['open'] if p.get('exitNext')],
                holding=st['open'], skipped=list(skipped), notes=list(notes))


def state():
    S = store.load(FN, None)
    return S['state'] if S else None


def trades():
    """{id: 仓位}，给执行记录 / 偏离检查（execlog）用。"""
    st = state()
    return {t['id']: t for t in (st['closed'] + st['open'])} if st else {}


def symbols():
    st = state()
    return sorted({p['v'] for p in st['open']}) if st else []


def _nd(x):
    c = 'green' if x['dir'] == '空' else 'red'
    return f"<font color='{c}'>{x.get('name') or x['v']}{x['dir']}</font>"


def card_lines(x):
    """收盘卡片最上面的一段：明早操作 + 持仓止损 + 账户。"""
    if not x:
        return []
    ops = []
    for p in x['today_open']:
        ops.append(f"＋开 {_nd(p)} **{p['lots']} 手**｜初始止损 {p['stopPx']}｜风险 {(p.get('budgetPct') or 0) * 100:.1f}%")
    for p in x['exit_next']:
        ops.append(f"－平 {_nd(p)} {p['lots']} 手（{p['exitNext']}，开盘平）")
    for t in x['today_closed']:
        ops.append(f"✓ 今日已平 {_nd(t)} {t.get('entry')}→{t.get('exit')} {(t.get('pnl') or 0) - (t.get('cost') or 0):+,.0f} 元（{t.get('why')}）")
    L = ['**【v2.7 实盘 · 明早操作】**次日开盘成交']
    L += ops or ['无新开平仓，持仓按下面的止损跟踪']
    hold = [p for p in x['holding'] if not p.get('pending')]
    if hold:
        L.append('**持仓止损（今晚起按此价设止损单）**')
        for p in sorted(hold, key=lambda p: p.get('base') or ''):
            L.append(f"· {_nd(p)} {p['lots']} 手｜入 {p.get('entry')}｜止损 **{p['stopPx']}**｜浮 {p.get('pnl') or 0:+,.0f}｜第 {p.get('age', 0)} 天")
    tier = '正常' if not x['lvl'] else f"降档 {x['lvl']}"
    L.append(f"净值 **{x['eq']:,.0f}**｜离高点 {x['dd'] * 100:.1f}%｜{tier}：每笔 {x['r'] * 100:g}% / 总 {x['h'] * 100:g}%"
             f"｜风险占用 {x['heat'] * 100:.1f}%｜已平 {x['n']} 笔（{x.get('start') or ''} 起）")
    for s in x['skipped']:
        L.append(f"× {_nd(s)} {s.get('why', '逆 5 日线，观望')}")
    for nt in x['notes']:
        L.append(f"ⓘ {nt}")
    return L


def morning_lines(price):
    """早盘：price(v) → (夜盘最新价, 夜盘开盘价) 或 None。列待成交与越过 / 接近止损的持仓。"""
    st = state()
    if not st or not st['open']:
        return []
    L = []
    for p in sorted(st['open'], key=lambda p: p.get('base') or ''):
        q = price(p['v'])
        px, op = (q or (None, None))
        sg = 1 if p['dir'] == '多' else -1
        if p.get('pending'):
            L.append(f"· 待成交 {_nd(p)} {p['lots']} 手" + (f"，夜盘开 {op}" if op else '，无夜盘报价'))
            continue
        if px is None:
            continue
        gap = sg * (px - p['stopPx']) / p['atr'] if p.get('atr') else None
        if gap is not None and gap < 0.5:
            L.append(f"⚠️ {_nd(p)} 夜盘 {px}，" + ('已越过止损 ' if gap < 0 else f'离止损 {gap:.1f}ATR，止损 ') + str(p['stopPx']))
    return (['**【v2.7 实盘】**'] + L) if L else []
