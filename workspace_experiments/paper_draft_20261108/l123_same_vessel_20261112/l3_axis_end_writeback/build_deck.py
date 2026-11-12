"""Two editable slides from verified historical regional writeback tensors."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import json
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_drawing', ROOT.parents[1] / 'l123_editable_20261112/build_deck.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
d.prs = Presentation()
d.prs.slide_width, d.prs.slide_height = Inches(18), Inches(10)
data = json.loads((ROOT / 'recovery.json').read_text(encoding='utf-8'))
assert data['status'] == 'exact_region_writeback_recovered'
vmax = max(v['total_peak'] for v in data['summary'].values())
GOLD = 'D58C22'
GREEN = '42C790'


def outline(s, points, color, width=1.5):
    points = [(int(Inches(x)), int(Inches(y))) for x, y in points]
    b = s.shapes.build_freeform(*points[0])
    b.add_line_segments(points[1:], close=False)
    sh = b.convert_to_shape()
    sh.fill.background()
    sh.line.color.rgb = RGBColor.from_string(color)
    sh.line.width = Pt(width)
    return sh


def color(v):
    t = min(max(v / vmax, 0), 1)
    return ''.join(f'{round(255 * (1 - t) + c * t):02X}' for c in (20, 125, 121))


def grid(s, values, support, x, y, name):
    step = .43
    for iy in range(10):
        for ix in range(10):
            value = values[iy][ix]
            sh = d.rect(s, x + ix * step, y + iy * step, step, step,
                        fill=color(value), line='D8E2E5')
            sh.name = f'{name}_y{iy}_x{ix}_L2_{value:.9g}'
            if value > 1e-8:
                label = f'{value:.2f}' if value >= .01 else f'{value:.4f}'.lstrip('0')
                sh = d.txt(s, label, x + ix * step + .006, y + iy * step + .13,
                           step - .012, .20, 8 if value < .01 else 10,
                           'FFFFFF' if value / vmax > .55 else d.C['ink'], align=PP_ALIGN.CENTER)
                sh.name = f'{name}_value_y{iy}_x{ix}'
    for iy in range(10):
        for ix in range(10):
            if support[iy][ix] > 1e-8:
                x0, y0 = x + ix * step, y + iy * step
                outline(s, [(x0, y0), (x0 + step, y0), (x0 + step, y0 + step),
                            (x0, y0 + step), (x0, y0)], GOLD, 1.6)
    for i in range(10):
        d.txt(s, str(i), x + i * step, y + 4.32, step, .25, 10, d.C['muted'], align=PP_ALIGN.CENTER)
        d.txt(s, str(i), x - .28, y + i * step + .12, .23, .23, 10, d.C['muted'], align=PP_ALIGN.RIGHT)
    d.txt(s, 'y', x - .28, y - .3, .3, .25, 12, d.C['muted'])
    d.txt(s, 'x', x + 4.39, y + 4.31, .3, .25, 12, d.C['muted'])


def raw_panel(s, pol):
    view = data['query']['views'][pol]
    im = Image.open(view['raw'])
    raw_w, raw_h = im.size
    display = ROOT / 'images' / f'12457_{pol}_nearest8.png'
    im.resize((raw_w * 8, raw_h * 8), Image.Resampling.NEAREST).save(display)
    scale = 4.0 / 72
    x, y = .7 + (4.6 - raw_w * scale) / 2, 2.73 + (4.3 - raw_h * scale) / 2
    d.pic(s, display, x, y, raw_w * scale, raw_h * scale)
    placement = view['placement']
    for key, col in [('body', GREEN), ('ends', GOLD)]:
        for i, contour in enumerate(view['region_contours_native_xy_level_005'][key]):
            pts = [(x + (a - placement['pad_left'] + .5) * scale,
                    y + (b - placement['pad_top'] + .5) * scale) for a, b in contour]
            sh = outline(s, pts, col, 1.6)
            sh.name = f'{pol}_{key}_native_mask_contour_{i}_level005'
    d.line(s, 1.0, 7.30, 1.38, 7.30, GOLD, 2)
    d.txt(s, '轴端软掩膜 0.05 等值线', 1.47, 7.13, 3.9, .35, 14)
    d.line(s, 1.0, 7.76, 1.38, 7.76, GREEN, 2)
    d.txt(s, '框内范围', 1.47, 7.59, 3.6, .35, 14)
    d.txt(s, f'原始 {raw_w} × {raw_h} 像素；灰度固定 0–255', .7, 2.24, 4.8, .32, 13, d.C['muted'])


for p, pol in enumerate(('VV', 'VH')):
    stats = data['summary'][pol]
    historical = data['scores'][p + 1]['historical_margin']
    baseline = .13755786418914795
    delta = historical - baseline
    s = d.new(f'L3 轴端回写实例：12457 / {pol}',
              '散货船 · Jebel Ali · 历史 N_level_pattern / seed17 · 固定16个源参考 · M3频域关闭', p + 1)
    d.txt(s, '原始 SAR 与轴端几何范围', .7, 1.84, 5.1, .4, 21, bold=True)
    d.txt(s, '全部空间回写', 6.3, 1.84, 4.8, .4, 21, bold=True)
    d.txt(s, '仅轴端槽位的真实回写', 11.65, 1.84, 5.6, .4, 21, bold=True)
    d.txt(s, '每格：1024通道更新的 L2 范数', 6.3, 2.24, 4.8, .32, 13, d.C['muted'])
    d.txt(s, '固定 T、β；保留槽位16–19的搬运内容', 11.65, 2.24, 5.5, .32, 13, d.C['muted'])
    raw_panel(s, pol)
    grid(s, stats['total_l2_grid'], stats['axis_write_weight_grid'], 6.3, 2.73, pol + '_total')
    grid(s, stats['axis_l2_grid'], stats['axis_write_weight_grid'], 11.65, 2.73, pol + '_ends')
    for i in range(60):
        value = vmax * (59 - i) / 59
        d.rect(s, 16.35, 2.73 + i * 4.3 / 60, .21, 4.3 / 60 + .002, fill=color(value))
    d.txt(s, f'{vmax:.2f}', 16.64, 2.66, .7, .3, 12)
    d.txt(s, '0', 16.64, 6.83, .6, .3, 12)
    d.txt(s, '共同\nL2色标', 16.64, 4.45, .8, .75, 12, d.C['muted'])
    d.txt(s, f"总回写峰值 {stats['total_peak']:.4f}", 6.3, 7.31, 4.8, .33, 15)
    d.txt(s, f"轴端峰值 {stats['axis_peak']:.4f}，位于 (y,x) = (6,5)", 11.65, 7.31, 5.65, .33, 15)
    d.txt(s, '橙框：轴端回写非零覆盖的网格；两图及两页共用色标。', 6.3, 7.77, 10.9, .34, 14, d.C['muted'])
    if pol == 'VV':
        note = '主要落在(4,5)、(6,5)；(6,6)另有0.001553的微小更新。'
    else:
        note = '非零位置为(4,5)、(6,5)，幅度分别为3.3971、3.5712。'
    d.txt(s, note, 6.3, 8.13, 10.9, .34, 14, d.C['muted'])
    d.line(s, .7, 8.61, 17.3, 8.61)
    d.txt(s, '轴端移除分差', .7, 8.81, 2.55, .36, 18, bold=True)
    d.txt(s, f'完整M3  {baseline:+.5f}    →    撤除{pol}轴端  {historical:+.5f}', 3.35, 8.81, 8.9, .4, 20, bold=True)
    d.txt(s, f'Δm {delta:+.5f}；仍判对', 12.4, 8.83, 4.9, .36, 19, d.C['teal'], True)
    d.txt(s, '两端由检测轴定义，不标定船首/船尾；原图未改。范数表示更新强度，分类作用由固定路径移除检验。', .7, 9.36, 15.8, .32, 13, d.C['muted'])
    d.note(s, json.dumps(dict(source=str(ROOT / 'recovery.json'), polarisation=pol,
                             recovery=data['definition'], checks=data['checks'],
                             scores=data['scores'], grid=stats,
                             image_source=data['query']['views'][pol]['source'],
                             grid_alignment='10x10 grid centers at native160 coordinates 16*i, origin=0; original chip padded without resampling. Raw contour uses native coordinates minus pad offsets.',
                             interpretation='Historical axis removal reduces margin but preserves correct bulk prediction. Not necessary alone for this rescue; not a global causal claim. All 16 references retain their original T and beta.'), ensure_ascii=False))

path = ROOT / 'L3_12457_axis_end_editable.pptx'
d.prs.save(path)
print(str(path), flush=True)
