"""F_0921 det 对齐质量诊断（正确 join 版）。
(scene,mmsi) 键 join index.csv(d_land_km) × point_corrections_det.csv(det_conf, shift_px)。
用法：python f0921_det_align_check.py
"""
import csv
from collections import defaultdict
import numpy as np
from pathlib import Path

IDX = Path('E:/Hermes/paper_tracking/f0921_chips/index.csv')
COR = Path('D:/F_0921_relabel_det/point_corrections_det.csv')


def load_cor():
    d = defaultdict(list)
    with COR.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            sp = r['shift_px'].strip()
            d[(r['scene'], r['mmsi'])].append(
                (float(r['det_conf']) if r['det_conf'].strip() else np.nan,
                 float(sp) if sp else np.nan))
    return d


def load_idx():
    d = defaultdict(list)
    with IDX.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            d[(r['scene'], r['mmsi'])].append(float(r['d_land_km']))
    return d


print('loading...', flush=True)
cor, idx = load_cor(), load_idx()
print(f'corrections keys: {len(cor)} ｜ index keys: {len(idx)}')
keys = set(cor) & set(idx)
print(f'共同键: {len(keys)} ｜ cor 独有 {len(set(cor)-set(idx))} ｜ idx 独有 {len(set(idx)-set(cor))}')
multi = sum(1 for k in keys if len(cor[k]) > 1 or len(idx[k]) > 1)
print(f'非唯一键: {multi}')

pairs = []
for k in keys:
    a, b = cor[k], idx[k]
    if len(a) == 1 and len(b) == 1:
        pairs.append((b[0], a[0][1], a[0][0]))
print(f'1对1 键: {len(pairs)}')
P = np.array(pairs)
dl, sh, cf = P[:, 0], P[:, 1], P[:, 2]
ok = np.isfinite(sh) & np.isfinite(dl)
dl, sh, cf = dl[ok], sh[ok], cf[ok]
print(f'有效: {len(dl)}')

print('\n=== 总体 ===')
print(f'shift_px: 中位 {np.median(sh):.1f} ｜ p90 {np.percentile(sh,90):.1f} ｜ mean {sh.mean():.1f}')
print(f'det_conf: 中位 {np.median(cf):.3f} ｜ p10 {np.percentile(cf,10):.3f} ｜ =0 占比 {100*(cf<=0).mean():.1f}%')

det = cf > 0
print(f'\n=== 检测成功率（det_conf>0 = 真被检测对齐）===')
print(f'总体: {100*det.mean():.1f}% ({int(det.sum())}/{len(det)})')
print('\n按离岸距离：')
for lo, hi, name in [(0, 1, '0-1km 近岸'), (1, 5, '1-5km'), (5, 20, '5-20km'), (20, 50, '20-50km'), (50, 1e9, '>50km 远海')]:
    m = (dl >= lo) & (dl < hi)
    if m.sum() == 0:
        continue
    print(f'  {name:12s} n={m.sum():7d}  检测到 {100*det[m].mean():5.1f}% ｜ 未检测(det_conf=0) {100*(~det[m]).mean():5.1f}%')

print('\n=== 仅看被检测到的点：shift_px 质量 ===')
for lo, hi, name in [(0, 1, '0-1km 近岸'), (1, 5, '1-5km'), (5, 20, '5-20km'), (20, 50, '20-50km'), (50, 1e9, '>50km 远海')]:
    m = (dl >= lo) & (dl < hi) & det
    if m.sum() == 0:
        continue
    s = sh[m]
    print(f'  {name:12s} n={m.sum():7d}  shift 中位 {np.median(s):5.1f} ｜ p90 {np.percentile(s,90):5.1f} ｜ ≤10px {100*(s<=10).mean():5.1f}% ｜ >60px {100*(s>60).mean():5.1f}%')

print('\n=== 远海 >50km 细节（被检测到的）===')
m = (dl >= 50) & det
for thr in [5, 10, 20, 30, 60, 100]:
    print(f'  ≤{thr:3d}px: {100*(sh[m]<=thr).mean():5.1f}%')
