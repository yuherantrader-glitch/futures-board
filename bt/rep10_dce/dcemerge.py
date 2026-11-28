"""把大商所 2016-04 ~ 2018-12 席位（用户从 Wind 导出、本地整理成 seats 格式）和日K 并入 hist/all。

  python tools/dcemerge.py --seats data_in/dce1618_seats.json.gz --bars data_in/bars-dce1618.json.gz --into hist/all

seats 文件：{日期: {品种: {oi, L:{会员:手}, S:{会员:手}, byC:{合约:多头合计}, main}}}，口径与 hist.py 相同（只取持仓 ≥ 品种 3% 的合约）。
已有的日文件只补进缺的品种，不覆盖。
"""
import argparse
import gzip
import json
import os
import re
import shutil


def gz(p):
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seats', required=True)
    ap.add_argument('--bars', required=True)
    ap.add_argument('--into', default='hist/all')
    a = ap.parse_args()
    S = gz(a.seats)
    sd = os.path.join(a.into, 'seats')
    os.makedirs(sd, exist_ok=True)
    n_new = n_upd = n_add = 0
    for d, V in sorted(S.items()):
        p = os.path.join(sd, d + '.json.gz')
        old = gz(p) if os.path.exists(p) else {}
        add = {v: x for v, x in V.items() if v not in old}
        if not add:
            continue
        n_add += len(add)
        if old:
            n_upd += 1
        else:
            n_new += 1
        old.update(add)
        with gzip.open(p, 'wt', encoding='utf-8') as f:
            json.dump(old, f, ensure_ascii=False, separators=(',', ':'))
    print(f'席位：{len(S)} 天，新建 {n_new} 个日文件，补充 {n_upd} 个，加入 {n_add} 个品种日', flush=True)
    B = gz(a.bars)
    have = {}
    for p in os.listdir(a.into):
        if p.startswith('bars') and p.endswith('.json.gz'):
            for c, k in gz(os.path.join(a.into, p)).items():
                v = re.sub(r'\d', '', c).upper()
                for b in k:
                    if '2016-01-01' <= b[0] <= '2018-12-31':
                        have.setdefault(v, set()).add(b[0])
    print('并入前 2016-2018 日K 覆盖（品种: 天数）', {v: len(have.get(v, ())) for v in sorted({re.sub(r'\d', '', c).upper() for c in B})}, flush=True)
    shutil.copyfile(a.bars, os.path.join(a.into, 'bars-dce1618.json.gz'))
    print(f'日K：{len(B)} 个合约 → {a.into}/bars-dce1618.json.gz', flush=True)


if __name__ == '__main__':
    main()
