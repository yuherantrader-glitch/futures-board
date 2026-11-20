"""v2.7 候选（ECB + 长期强席位过滤 + 盈利升档）十年回测的独立复现包：一条命令从原始数据重算，导出逐笔交易、每日盯市净值、每日候选。

把研究脚本（gen_rows.py / lt_focus.py / lt_carry.py / basis_tests.py + lt_b2.py / seat_panel.py + seat_l2.py / lt_seat.py / lt_ladder.py，
原在 /home/claude/sig 临时环境）的逻辑合并成一个文件，路径全部相对仓库，输入只有：
  ① hist/all       席位前 20 + 合约日 K（GitHub Release `data`，tools/datastore.py unpack）
  ② research/basis 生意社现期表 2017-01 ~ 2026-11（100ppi_basis_YYYY.csv.gz）
引擎用仓库现有 tools/lagtest.LT（与线上影子账户同一套）。

  python tools/v27_repro.py --data hist/all --basis research/basis --out site/bt/v27repro --end 2026-10-31
  （候选缓存按 代码+参数+数据 指纹命名，任一变化即重算；多进程依赖 Linux fork，Windows 需改成单进程。）

输出（--out）：
  summary.md / summary.json         各方案终值、年化、最大回撤（盯市）、笔数，与研究文档数字对照
  trades_<方案>.csv                 逐笔交易
  equity_<方案>.csv                 每日盯市净值（已实现 + 未平仓浮盈 − 已发生成本）
  signals_daily.csv                 每日 ECB/V27 候选前 5 名与各项过滤结果
  inputs_manifest.json              输入文件 sha256、覆盖日期、强席位每年训练截止日
已知口径（照研究原样，不在复现里修改，供审查）：
  - 定仓基数 = 100 万 + 已实现净盈亏（不含浮亏）；盈利升档也按它判断。
  - 成本只在平仓日计入净值（每边 1 跳 + 万 0.5）。
  - 强席位：同时跑 orig（研究原样，有前视）/ mature（只修训练标签）/ clean（无前视，正式口径）三版，并输出逐日入选差异。
  - 基差：信号日无表时用 3 个自然日内最近一张；分位 = 过去 250 张表（≥120 张）。
"""
import argparse
import bisect
import csv
import datetime as dt
import glob
import gzip
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tools'))
from ftrade import v46, kline, tracker  # noqa: E402
from ftrade.names import NAMES  # noqa: E402
import backtest as bt  # noqa: E402
import optimize as op  # noqa: E402
import lagtest as LG  # noqa: E402

W, WRG = 18, 0.65
KEEP = ('v', 'big', 'dir', 'score', 'conflict', 'retreat', 'bigTurn', 'oi', 'gr', 'gChg', 'lr', 'ngen', 'close', 'gen')
G = {}   # 子进程共享（fork）


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ───────── ① 每日候选（= gen_rows.py，GPIT=1 实盘口径）─────────
def kpit(D):
    out = {}
    for v, a in G['ADJ1'].items():
        p = bisect.bisect_right(G['AD'][v], D) - 1
        if p < 1:
            continue
        off = G['RAW1'][v][p]['c'] - a[p]['c']
        out[v] = [dict(x, o=x['o'] + off, h=x['h'] + off, l=x['l'] + off, c=x['c'] + off) for x in a[max(0, p - 110):p + 1]]
    return out


def rows_day(i):
    days, H = G['days'], G['H']
    D = days[i]
    if (dt.date.fromisoformat(D) - dt.date.fromisoformat(days[i - W])).days > 45 + 2 * W:
        return D, None
    cap = {}
    g0 = v46.generals

    def _cap(*a, **k):
        k.pop('wrG', None)
        g = g0(*a, **k, wrG=WRG)
        cap.clear()
        cap.update(g)
        return g
    v46.generals = _cap
    try:
        rows = v46.run(D, days[i - W:i + 1], [H[d] for d in days[i - W:i + 1]], kpit(D), names=NAMES, W=W)
    finally:
        v46.generals = g0
    res = []
    for r in rows:
        x = {k: r.get(k) for k in KEEP}
        x['nbig'] = (r.get('gen') or '').count('★')
        g = cap.get(r['v'], {})
        x['gNet'], x['lNet'], x['oiRaw'] = g.get('gNet'), g.get('lNet'), g.get('oi')
        res.append(x)
    return D, res


def build_rows(bars, H, cache, end):
    if os.path.exists(cache):
        return json.load(gzip.open(cache, 'rt'))
    days = sorted(H)
    bdays = sorted({b[0] for k in bars.values() for b in k})
    RAW1, ROLLS1 = bt.build_series(bars, bdays)
    ADJ1 = {v: kline.adjust(RAW1[v], ROLLS1[v]) for v in RAW1}
    G.update(days=days, H=H, RAW1=RAW1, ADJ1=ADJ1, AD={v: [x['d'] for x in ADJ1[v]] for v in ADJ1})
    todo = [i for i in range(max(W, 45), len(days)) if days[i] <= end]
    t0, R = time.time(), {}
    with Pool(os.cpu_count() or 2) as p:
        for n, (D, res) in enumerate(p.imap(rows_day, todo, chunksize=8)):
            if res is not None:
                R[D] = res
            if n % 200 == 0:
                log(f'  候选 {D}（{n}/{len(todo)}，{time.time() - t0:.0f}s）')
    json.dump(R, gzip.open(cache, 'wt'))
    return R


# ───────── ② 选股要素 ─────────
def elig(r):
    return r['big'] and r['score'] >= 4 and not r['conflict'] and not r['retreat'] and not r['bigTurn'] and r['ngen'] >= 2


def conf(r):
    oi = r.get('oiRaw') or r['oi'] or 1
    d = 1 if r['dir'] == '多' else -1
    lr = abs(r['lNet']) / oi if r.get('lNet') and (r['lNet'] > 0) - (r['lNet'] < 0) == -d else 0
    return abs(r['gNet'] or 0) / oi + lr


def carry_fn(RAW, bars):
    """= lt_carry.carry_fn：主力 vs 交割月晚于主力、持仓最大且 ≥ 主力 5% 的合约，年化价差。"""
    import re
    BY = {}
    for ct, k in bars.items():
        v = re.sub(r'\d+', '', ct).upper()
        ym = int(ct[-4:])
        m = (ym // 100) * 12 + ym % 100
        for b in k:
            BY.setdefault((v, b[0]), []).append((m, b[4], b[6]))
    MAIN = {v: {x['d']: x['ct'] for x in k} for v, k in RAW.items()}

    def carry(v, D):
        ct = MAIN.get(v, {}).get(D)
        xs = BY.get((v, D))
        if not ct or not xs:
            return None
        ym = int(ct[-4:])
        m0 = (ym // 100) * 12 + ym % 100
        near = next((x for x in xs if x[0] == m0), None)
        if not near or near[1] <= 0:
            return None
        far = [x for x in xs if x[0] > m0 and x[2] >= 0.05 * near[2] and x[1] > 0]
        if not far:
            return None
        f = max(far, key=lambda x: x[2])
        return (near[1] - f[1]) / f[1] * 12 / (f[0] - m0)
    return carry


BMAP = {'铜': 'CU', '螺纹钢': 'RB', '锌': 'ZN', '铝': 'AL', '黄金': 'AU', '天然橡胶': 'RU', '铅': 'PB', '白银': 'AG', '石油沥青': 'BU', '热轧卷板': 'HC', '镍': 'NI',
        '锡': 'SN', 'PTA': 'TA', '白糖': 'SR', '棉花': 'CF', '菜籽油OI': 'OI', '玻璃': 'FG', '菜籽粕': 'RM', '硅铁': 'SF', '锰硅': 'SM', '甲醇MA': 'MA', '棕榈油': 'P',
        '聚氯乙烯': 'V', '聚乙烯': 'L', '豆一': 'A', '豆粕': 'M', '豆油': 'Y', '玉米': 'C', '焦炭': 'J', '焦煤': 'JM', '铁矿石': 'I', '鸡蛋': 'JD', '聚丙烯': 'PP',
        '燃料油': 'FU', '线材': 'WR', '棉纱': 'CY', '纸浆': 'SP', '乙二醇': 'EG', '尿素': 'UR', '不锈钢': 'SS', '苯乙烯': 'EB', '液化石油气': 'PG', '纯碱': 'SA',
        '涤纶短纤': 'PF', '生猪': 'LH', '工业硅': 'SI', '碳酸锂': 'LC', '丁二烯橡胶': 'BR', 'PX': 'PX', '烧碱': 'SH', '瓶片': 'PR', '多晶硅': 'PS', '纯苯': 'BZ',
        '丙烯': 'PL', '强麦WH': 'WH', '普麦': 'PM'}


def basis_class_fn(path):
    """basis_class.txt.gz：由原始生意社表按 basis_fn 同口径预先算好的分类（每张表一行，每品种一个字符：
    L = 分位 ≤1/3（做多不利）、H = 分位 ≥2/3（做空不利）、M = 中间、. = 不足 120 张表）。结果与 basis_fn 完全一致，只是体积小。"""
    L = gzip.open(path, 'rt').read().split('\n')
    cols = L[0].split(',', 1)[1].split('|')
    ci = {v: i for i, v in enumerate(cols)}
    dates, codes = [], []
    for ln in L[1:]:
        if ln:
            d, c = ln.split(',', 1)
            dates.append(np.datetime64(d))
            codes.append(c)
    idx = np.array(dates)

    def code(v, D):
        if v not in ci:
            return None
        k = np.searchsorted(idx, np.datetime64(D), side='right') - 1
        if k < 0 or (np.datetime64(D) - idx[k]) > np.timedelta64(3, 'D'):
            return None
        c = codes[k][ci[v]]
        return None if c == '.' else c

    def low(v, D, dr):
        c = code(v, D)
        return bool(c) and ((dr == '多' and c == 'L') or (dr == '空' and c == 'H'))
    low.pct = code
    low.cover = (str(idx[0])[:10], str(idx[-1])[:10], len(idx))
    return low


def basis_fn(basis_dir):
    """= basis_tests.py + lt_b2.b2_against：顺向基差处在自身过去 250 张表的低三分之一 → 不做。"""
    if basis_dir.endswith('.txt.gz'):
        return basis_class_fn(basis_dir)
    fs = sorted(glob.glob(os.path.join(basis_dir, '*.csv.gz')))
    b = pd.concat([pd.read_csv(f, header=None, encoding='utf-8-sig', dtype=str) for f in fs])
    b['v'] = b[2].map(BMAP)
    b = b.dropna(subset=['v'])
    b['D'] = pd.to_datetime(b[0])
    b['spot'] = pd.to_numeric(b[3], errors='coerce')
    b['fut'] = pd.to_numeric(b[10], errors='coerce')
    b = b[(b.spot > 0) & (b.fut > 0)]
    b['br'] = b.spot / b.fut - 1
    B = b.pivot_table(index='D', columns='v', values='br', aggfunc='last').sort_index()
    PCT = B.apply(lambda s: s.dropna().rolling(250, min_periods=120).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True).reindex(s.index))
    idx = B.index.values
    C = {}

    def pct(v, D):
        if v not in PCT.columns:
            return None
        if (v, D) not in C:
            k = np.searchsorted(idx, np.datetime64(D), side='right') - 1
            C[(v, D)] = np.nan if k < 0 or (np.datetime64(D) - idx[k]) > np.timedelta64(3, 'D') else PCT[v].iloc[k]
        p = C[(v, D)]
        return None if p is None or np.isnan(p) else float(p)

    def low(v, D, dr):
        p = pct(v, D)
        if p is None:
            return False
        return (p if dr == '多' else 1 - p) <= 1 / 3
    low.pct = pct
    low.cover = (str(B.index.min().date()), str(B.index.max().date()), int(len(B)))
    return low


def strong_fn(H, ADJ, mode):
    """强席位过滤。每个检验年 Y，用 Y 之前的数据给（席位,品种）评分，取每品种前 10% 为强席位，Y 年内算其净持仓合计。
    mode：
      orig   = 研究原样（seat_l2.py）：训练标签含跨年未兑现的 5 日收益；检验年当天的净持仓也取自“按未来 5 日收益过滤过的面板”
               （缺未来收益或 |f|>10 的记录被删掉）→ 当天是否过滤取决于未来行情（前视，Astra 2026-11-18 复现）。
      mature = 只修训练标签（训练截止日回退 5 个交易日），当天净持仓仍取自过滤面板 → 仍有前视。
      clean  = 训练用已兑现标签；当天净持仓直接从当天原始席位（H[D]）计算，不用任何 D 之后的数据（与线上 shadow.seat_fn 同口径）。"""
    mature = mode in ('mature', 'clean')
    def rma(x, n):
        out = np.full(len(x), np.nan)
        if len(x) < n:
            return out
        out[n - 1] = np.mean(x[:n])
        for i in range(n, len(x)):
            out[i] = (out[i - 1] * (n - 1) + x[i]) / n
        return out
    FW = {}
    for v, k in ADJ.items():
        c = np.array([x['c'] for x in k], float)
        h = np.array([x['h'] for x in k], float)
        lo = np.array([x['l'] for x in k], float)
        pc = np.r_[np.nan, c[:-1]]
        tr = np.nanmax(np.vstack([h - lo, np.abs(h - pc), np.abs(lo - pc)]), axis=0)
        tr[0] = h[0] - lo[0]
        f = np.r_[c[5:] - c[:-5], np.full(5, np.nan)] / rma(tr, 14)
        FW[v] = dict(zip([x['d'] for x in k], f))
    rows = []
    for D in sorted(H):
        for v, x in H[D].items():
            fv = FW.get(v, {}).get(D)
            if fv is None or not np.isfinite(fv):
                continue
            oi = x.get('oi') or 0
            if oi <= 0:
                continue
            for s in set(x['L']) | set(x['S']):
                rows.append((D, v, s, (x['L'].get(s, 0) - x['S'].get(s, 0)) / oi, fv))
    P = pd.DataFrame(rows, columns=['D', 'v', 's', 'q', 'f'])
    P['D'] = pd.to_datetime(P.D)
    P['c'] = P.q * P.f
    P['y'] = P.D.dt.year
    P['wk'] = P.D.dt.strftime('%G-%V')
    log(f'  席位面板 {len(P):,} 行')
    P = P[(P.q.abs() <= 0.5) & (P.f.abs() <= 10)].copy()
    P['key'] = P.s + '|' + P.v

    def skill(tr):
        w = tr.groupby(['key', 'wk']).c.mean().reset_index()
        a = w.groupby('key').c.agg(['mean', 'std', 'count'])
        a['t'] = a['mean'] / (a['std'] / np.sqrt(a['count']))
        a['n'] = tr.groupby('key').size()
        a['v'] = a.index.str.split('|').str[1]
        return a[(a.n >= 250) & a['std'].gt(0)]
    NET, LISTS, TRAIN = [], {}, {}
    for Y in range(2019, 2027):
        tr = P[P.y < Y]
        if mature:   # 只用前瞻 5 日已兑现的标签：训练截止日往前退 5 个交易日
            dd = sorted(tr.D.unique())
            tr = tr[tr.D <= dd[-6]] if len(dd) > 6 else tr
        TRAIN[Y] = str(tr.D.max().date())
        sk = skill(tr)
        top = set()
        for v, g in sk.groupby('v'):
            if len(g) < 10:
                continue
            k = max(1, int(round(len(g) * 0.1)))
            top |= set(g.sort_values('t').index[-k:])
        LISTS[Y] = sorted(top)
        te = P[(P.y == Y) & P.key.isin(top)]
        NET.append(te.groupby(['v', 'D']).q.sum().reset_index())
    if mode == 'clean':
        SN, SS = {}, {}
        for Y, top in LISTS.items():
            by = {}
            for k in top:
                s, v = k.split('|', 1)
                by.setdefault(v, set()).add(s)
            SS[Y] = by
        for D in sorted(H):
            by = SS.get(int(D[:4]))
            if not by:
                continue
            for v, x in H[D].items():
                oi = x.get('oi') or 0
                if oi <= 0 or v not in by:
                    continue
                L, S_ = x.get('L') or {}, x.get('S') or {}
                hit = [s for s in by[v] if s in L or s in S_]
                if hit:
                    SN[(v, D)] = sum(L.get(s, 0) - S_.get(s, 0) for s in hit) / oi
    else:
        N = pd.concat(NET)
        SN = {(v, d.strftime('%Y-%m-%d')): s for v, d, s in zip(N.v, N.D, N.q)}

    def against(v, D, dr):
        s = SN.get((v, D))
        if s is None or abs(s) < 0.005:
            return False
        return (s > 0) != (dr == '多')
    against.net = lambda v, D: SN.get((v, D))
    against.train = TRAIN
    against.lists = LISTS
    return against


# ───────── ③ 引擎（= LG.run_one + lt_ladder.make_comp），逐日记录盯市净值 ─────────
LADDER = [(1.0, 0.02, 0.15), (1.3, 0.025, 0.20), (1.6, 0.03, 0.20)]


def run(data, S, sizing, xcfg):
    """sizing: ('fixed', r, heat, compound) 或 ('ladder',)。返回 (st, 每日记录)。"""
    op.COMPOUND = sizing[0] == 'ladder' or sizing[3]
    T = LG.LT(data['RAW'], data['ADJ'], data['ROLLS'], {'trail': 0, 'ma5': False, **xcfg, 'maxHold': 60,
                                                        'fixed': 0.02 if sizing[0] == 'ladder' else sizing[1]})
    if sizing[0] != 'ladder':
        T.cfg['maxHeat'] = sizing[2]
    st = {'open': [], 'closed': [], 'from': data['days'][0]}
    E = []
    for D in data['days'][1:]:
        x = S[D]
        sigs = [s for s in x['sigs'] if op.ma5_ok(data['ADJ'], s['v'], D, s['dir'])]
        op.compound(T, st)
        if sizing[0] == 'ladder':
            cap = T.cfg['capital']
            i = max(k for k, (m, _, _) in enumerate(LADDER) if cap >= m * 1e6) if cap >= 1e6 else 0
            T.cfg['fixed'], T.cfg['maxHeat'] = LADDER[i][1:]
        tier = (T.cfg['fixed'], T.cfg['maxHeat'], T.cfg['capital'])
        n0 = {p['id'] for p in st['open']}
        T.step(D, st, [dict(s) for s in sigs], set(), today=True, weak={})
        for p in st['open'] + st['closed']:
            if p['id'] not in n0 and 'tier' not in p:
                p['tier'] = f'{tier[0]*100:g}%/{tier[1]*100:g}%'
                p['capBase'] = round(tier[2])
            if not p.get('pending') and 'stop0' not in p and p.get('fillDate'):
                p['stop0'] = p['stopPx']
        real = [t for t in st['closed'] if not t.get('observe')]
        fl = sum(p['pnl'] or 0 for p in st['open'] if not p.get('observe') and not p.get('pending'))
        cost = sum(op.tcost(t) for t in real)
        E.append(dict(date=D, realized=round(sum(t['pnl'] or 0 for t in real)), floating=round(fl), cost=round(cost),
                      capBase=round(tier[2]), tier=f'{tier[0]*100:g}%/{tier[1]*100:g}%',
                      nOpen=sum(1 for p in st['open'] if not p.get('observe')),
                      heat=round(sum(p.get('budgetPct') or 0 for p in st['open'] if not p.get('observe')), 4)))
    pk = 1e6   # 峰值从初始本金起算（Astra：否则首日即亏时回撤漏算）
    for e in E:
        e['equity'] = 1e6 + e['realized'] + e['floating'] - e['cost']
        pk = max(pk, e['equity'])
        e['peak'] = round(pk)
        e['dd'] = round(1 - e['equity'] / pk, 5)
        e['equity'] = round(e['equity'])
    return st, E


def make_S(data, carry, blow, seat, use_seat):
    S, SIG = {}, []
    for D in data['days']:
        rows = data['R'].get(D, [])
        c = sorted([r for r in rows if elig(r)], key=lambda r: (-r['nbig'], -conf(r)))
        chosen = None
        for k, r in enumerate(c):
            ca = carry(r['v'], D)
            ag = ca is not None and ca != 0 and (ca > 0) != (r['dir'] == '多')
            bl = blow(r['v'], D, r['dir'])
            sa = seat(r['v'], D, r['dir'])
            ok = not ag and not bl and not (use_seat and sa)
            if ok and chosen is None:
                chosen = r
            if use_seat and k < 5:
                SIG.append(dict(date=D, rank=k + 1, v=r['v'], dir=r['dir'], score=r['score'], nbig=r['nbig'], conf=round(conf(r), 4),
                                carry=None if ca is None else round(ca, 4), carryAgainst=ag, basisPct=blow.pct(r['v'], D),
                                basisLow=bl, strongNet=seat.net(r['v'], D), strongAgainst=sa,
                                ECB=bool(not ag and not bl), pickedV27=r is chosen))
        top = [chosen] if chosen else []
        S[D] = {'sigs': [{**{k: s[k] for k in ('v', 'dir', 'score', 'gr', 'ngen', 'gChg')}, 'stars': 3 if s['score'] >= 6 else 2,
                          'name': s['v'], 'sig': D + '收盘', 'note': ''} for s in top]}
    return S, SIG


TCOLS = ['id', 'v', 'name', 'dir', 'base', 'fillDate', 'entry', 'stop0', 'atr', 'mult', 'lots', 'perLot', 'risk', 'budget', 'budgetPct', 'tier',
         'capBase', 'exitDate', 'exit', 'why', 'held', 'rolls', 'pnl', 'cost', 'net', 'R_gross', 'R_net', 'R_budget']


def export(name, st, E, out):
    real = [t for t in st['closed'] if not t.get('observe')]
    with open(os.path.join(out, f'trades_{name}.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, TCOLS, extrasaction='ignore')
        w.writeheader()
        for t in sorted(real + [dict(p, exitDate='(持有)', why='期末未平') for p in st['open'] if not p.get('observe')],
                        key=lambda t: (t['base'], t['id'])):
            c = op.tcost(t) if t['exitDate'] != '(持有)' else 0
            w.writerow(dict(t, cost=round(c), net=round((t['pnl'] or 0) - c), rolls='|'.join(t.get('rolls') or []),
                            name=NAMES.get(t['v'], t['v']), R_gross=round((t['pnl'] or 0) / t['risk'], 3) if t.get('risk') else '',
                            R_net=round(((t['pnl'] or 0) - c) / t['risk'], 3) if t.get('risk') else '',
                            R_budget=round((t['pnl'] or 0) / t['budget'], 3) if t.get('budget') else ''))
    pd.DataFrame(E).to_csv(os.path.join(out, f'equity_{name}.csv'), index=False, encoding='utf-8-sig')
    e = np.array([x['equity'] for x in E], float)
    yrs = len(e) / 244
    w = [t for t in real if (t['pnl'] or 0) - op.tcost(t) > 0]
    rn = [((t['pnl'] or 0) - op.tcost(t)) / t['risk'] for t in real if t.get('risk')]
    rg = [(t['pnl'] or 0) / t['risk'] for t in real if t.get('risk')]
    return dict(终值万=round(e[-1] / 1e4, 1), 年化=round(((e[-1] / 1e6) ** (1 / yrs) - 1) * 100, 1), 最大回撤=round(max(x['dd'] for x in E) * 100, 1),
                跌破本金最深=round(max(0, 1 - e.min() / 1e6) * 100, 1), 笔=len(real), 胜率=round(len(w) / max(1, len(real)) * 100, 1),
                平均R净=round(float(np.mean(rn)), 3) if rn else None, 平均R毛=round(float(np.mean(rg)), 3) if rg else None,
                起=E[0]['date'], 止=E[-1]['date'])


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='hist/all')
    ap.add_argument('--basis', default='research/basis')
    ap.add_argument('--out', default='site/bt/v27repro')
    ap.add_argument('--end', default='2026-10-31')
    ap.add_argument('--cachedir', default='hist')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tracker.MULT.update(bt.EXTRA_MULT)
    bars, H = bt.load([a.data])
    H = {d: x for d, x in H.items() if d <= a.end}
    log(f'数据：{len(bars)} 个合约，席位 {len(H)} 天（{min(H)} ~ {max(H)}）')
    # 候选缓存：键 = 代码（v46/v27_repro）+ 参数 + 数据指纹，任一变化即重算，不复用旧缓存
    fp = hashlib.sha256('|'.join([sha(os.path.join(HERE, 'ftrade/v46.py')), sha(os.path.abspath(__file__)), str(W), str(WRG), a.end,
                                  str(len(H)), min(H), max(H), str(len(bars)),
                                  *[sha(p) for p in sorted(glob.glob(os.path.join(a.data, 'bars*.json.gz')))]]).encode()).hexdigest()[:12]
    R = build_rows(bars, H, os.path.join(a.cachedir, f'rows_pit_{fp}.json.gz'), a.end)
    days = sorted(H)
    RAW, ROLLS = bt.build_series(bars, days)
    ADJ = {v: kline.adjust(RAW[v], ROLLS[v]) for v in RAW}
    d0 = min(R)
    data = {'days': days[days.index(d0) - 1:], 'RAW': RAW, 'ADJ': ADJ, 'ROLLS': ROLLS, 'R': R}
    carry = carry_fn(RAW, bars)
    blow = basis_fn(a.basis)
    log(f'基差表 {blow.cover}')
    SEAT = {m: strong_fn(H, ADJ, m) for m in ('orig', 'mature', 'clean')}
    for m, f in SEAT.items():
        log(f'强席位 {m} 训练截止：{f.train}')
    xcfg = dict(LG.V25, **LG.REAL, noWeak=True)
    S_ecb, _ = make_S(data, carry, blow, SEAT['clean'], False)
    SV, SIGS = {}, {}
    for m, f in SEAT.items():
        SV[m], SIGS[m] = make_S(data, carry, blow, f, True)
    pd.DataFrame(SIGS['clean']).to_csv(os.path.join(a.out, 'signals_daily.csv'), index=False, encoding='utf-8-sig')
    # 三个版本逐日入选差异
    diff = []
    for D in data['days']:
        pk = {m: tuple((s['v'], s['dir']) for s in SV[m][D]['sigs']) for m in SV}
        if len(set(pk.values())) > 1:
            diff.append(dict(date=D, **{m: ' '.join(v + d for v, d in pk[m]) or '-' for m in pk}))
    pd.DataFrame(diff).to_csv(os.path.join(a.out, 'pick_diff_orig_mature_clean.csv'), index=False, encoding='utf-8-sig')
    log(f'三版本入选不同的交易日：{len(diff)}')
    START = '2017-03-01'
    cut = lambda d: dict(d, days=[x for x in d['days'] if x >= START])
    TAG = {'orig': '研究原样（有前视）', 'mature': '只修训练标签（仍有前视）', 'clean': '无前视（正式）'}
    CASES = [('ECB_盈利升档_2017起', S_ecb, cut(data), ('ladder',), 'ECB 对照（不用强席位）'),
             ('ECB_2%15%复利_2017起', S_ecb, cut(data), ('fixed', 0.02, 0.15, True), 'ECB 对照'),
             ('ECB_固定2%_全段', S_ecb, data, ('fixed', 0.02, 0.10, False), '研究：净 161.1 万 / 回撤 14.9%')]
    for m in ('clean', 'orig', 'mature'):
        CASES += [(f'V27[{m}]_盈利升档_2017起', SV[m], cut(data), ('ladder',), TAG[m] + ('；研究：924 万 / 20.7%' if m == 'orig' else '')),
                  (f'V27[{m}]_2%15%复利_2017起', SV[m], cut(data), ('fixed', 0.02, 0.15, True), TAG[m] + ('；研究：538 万 / 15.3%' if m == 'orig' else '')),
                  (f'V27[{m}]_固定2%_全段', SV[m], data, ('fixed', 0.02, 0.10, False), TAG[m] + ('；研究：净 159.8 万 / 14.9%' if m == 'orig' else ''))]
    summ = {}
    for name, S, d, sz, doc in CASES:
        t0 = time.time()
        st, E = run(d, S, sz, xcfg)
        m = export(name, st, E, a.out)
        m['说明'] = doc
        summ[name] = m
        log(f'{name}：{m}（{time.time() - t0:.0f}s）')
    man = dict(end=a.end, seats_days=[min(H), max(H), len(H)], rows_days=[min(R), max(R), len(R)], rows_cache_key=fp, basis=blow.cover,
               strong_train_cutoff={m: f.train for m, f in SEAT.items()},
               strong_lists={m: {y: len(x) for y, x in f.lists.items()} for m, f in SEAT.items()},
               files={os.path.relpath(p, HERE): sha(p) for p in ([a.basis] if os.path.isfile(a.basis) else sorted(glob.glob(os.path.join(a.basis, '*.csv.gz'))))
                      + sorted(glob.glob(os.path.join(a.data, 'bars*.json.gz')))},
               engine={p: sha(os.path.join(HERE, p)) for p in ('tools/lagtest.py', 'tools/optimize.py', 'ftrade/tracker.py', 'ftrade/v46.py',
                                                              'tools/v27_repro.py')},
               commit=os.environ.get('GITHUB_SHA', ''))
    json.dump(man, open(os.path.join(a.out, 'inputs_manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump({m: f.lists for m, f in SEAT.items()}, open(os.path.join(a.out, 'strong_lists_by_year.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    json.dump(summ, open(os.path.join(a.out, 'summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    md = ['# v2.7 复现包（独立重算，含前视修复对照）', '',
          f'代码提交 {man["commit"][:10]}；数据截至 {a.end}；候选缓存键 {fp}；三版本入选不同的交易日 {len(diff)} 天。', '',
          '| 方案 | 终值（万） | 年化 | 最大回撤（盯市，含初始本金） | 跌破本金最深 | 笔数 | 胜率（净） | 平均 R 净 | 平均 R 毛 | 说明 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for k, m in summ.items():
        md.append(f"| {k} | {m['终值万']} | {m['年化']}% | {m['最大回撤']}% | {m['跌破本金最深']}% | {m['笔']} | {m['胜率']}% | {m['平均R净']} | {m['平均R毛']} | {m['说明']} |")
    md += ['', 'R 净 = (毛盈亏 − 成本) ÷ 实际初始风险；R 毛 = 毛盈亏 ÷ 实际初始风险。正式口径用 R 净。',
           'clean = 强席位训练只用已兑现标签、当天净持仓直接取当天原始席位（无前视，与线上影子账户同口径）；orig/mature 仅作对照。',
           '文件：trades_<方案>.csv、equity_<方案>.csv、signals_daily.csv（clean）、pick_diff_orig_mature_clean.csv、inputs_manifest.json、strong_lists_by_year.json。']
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
