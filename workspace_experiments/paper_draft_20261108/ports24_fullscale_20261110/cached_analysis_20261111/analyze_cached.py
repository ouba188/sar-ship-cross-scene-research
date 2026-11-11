"""Describe completed predictions and original chips; never load a model."""
from pathlib import Path
import importlib.util
import json
import html
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import precision_recall_fscore_support, f1_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
STORE = Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')
LABELS = ['散货船', '渔船', '普通货船', '化学品/成品油轮', '集装箱船', '原油轮', '拖轮/拖带', '海工补给船']
OBSERVATIONS = {
    '198': '双极化均呈横向细长亮响应，周围显示较暗；同类和混淆类参考也都可呈细长形态。此例证明系统纠错，不足以定位到某一局部或某个模块。',
    '551': '查询近竖向，所示同类参考可为斜横向，混淆类也可近竖向。完整模型纠错不符合只按方向相似分类的简单规则，但个例不证明总体方向不变性。',
    '865': '双极化都有斜向拉长亮响应，轴向亮度不均匀；同类参考也为拉长形态。该例仍被误伤，细长或高亮不是救回的充分条件。',
    '967': 'VV可见线状延伸，VH更紧凑；所示同类参考较窄。参考外观差异可记录，但未经参考替换不能认定它导致误伤。',
    '35': '目标在较亮粗糙显示背景中占比较小，两极化均紧凑，两个模型仍正确。小目标或较亮背景本身不足以解释错误。',
    '41': '暗背景上有近竖向亮点链及横向线状延伸，与所示同类参考方向不同，两模型均正确。线状延伸不必然造成误判。',
    '403': 'VV紧凑，VH原图更大并有明显十字状线性响应和更长目标框；两模型都错。同类参考与查询有差异，但不能由图直接认定虚影或检测错误。',
    '410': '双极化均有紧凑亮响应，两个模型持续判错；与持续判对的小目标例35共同说明，粗略尺度和紧凑程度不能单独解释结果。',
    '457': '双极化呈相近的横向细长响应，完整模型相对SE救回。真实案例支持系统有效，但不直接证明某条参考的贡献。',
    '1070': '两极化均有倾斜细长响应，查询周围显示比所示同类参考更亮。整体港口收益不足与个别船三种子稳定救回可以同时存在。',
    '1025': '双极化都是横向亮响应，完整模型却将集装箱误判散货。与散货救回例的粗外观相似，不能把这种外观作为通用启用规则。',
    '1117': '两极化近竖向紧凑亮响应，完整模型相对SE误伤。方向和目标亮度不足以单独决定分类收益。',
}


def save(name, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')
    return frame


def main():
    # Reuse the existing paired analysis, including ship-cluster intervals.
    spec = importlib.util.spec_from_file_location('existing_analysis', ROOT / 'analyze.py')
    existing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(existing)
    existing.main()
    from sar_m123.input import prepare_view

    rows = json.loads((STORE / 'manifest.json').read_text(encoding='utf-8'))
    groups = json.loads((STORE / 'groups.json').read_text(encoding='utf-8'))
    protocol = json.loads((STORE / 'protocol.json').read_text(encoding='utf-8'))
    ids = np.asarray(groups['evaluation'])
    query = [rows[i] for i in ids]
    y = np.array([r['label'] for r in query])
    scores = {}
    overall, classes, grouped = [], [], []
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            with np.load(STORE / f"{arm}_s{seed}_lr{protocol['scales'][arm]:g}" / 'evaluation.npz') as data:
                if not np.array_equal(ids, data['ids']):
                    raise ValueError('Prediction ordering differs from the frozen evaluation cohort')
                score = data['score'].copy()
            scores[arm, seed] = score
            pred = score.argmax(1)
            precision, recall, f1, count = precision_recall_fscore_support(y, pred, labels=range(8), zero_division=0)
            overall.append(dict(arm=arm, seed=seed, n=len(y), oa=100*np.mean(pred == y), ba=100*recall.mean(), macro_f1=100*f1.mean()))
            for c in range(8):
                classes.append(dict(arm=arm, seed=seed, label=c, class_name=LABELS[c], n=int(count[c]), precision=100*precision[c], recall=100*recall[c], f1=100*f1[c]))
            for field in ('port', 'pass_group'):
                values = np.array([r[field] for r in query])
                for value in sorted(set(values)):
                    mask = values == value
                    present = np.unique(y[mask])
                    correct = pred[mask] == y[mask]
                    grouped.append(dict(level=field, group=value, arm=arm, seed=seed,
                        n=int(mask.sum()), ships=len({query[i]['mmsi'] for i in np.flatnonzero(mask)}),
                        classes=len(present), ports=';'.join(sorted({query[i]['port'] for i in np.flatnonzero(mask)})),
                        oa=100*correct.mean(), ba=100*np.mean([correct[y[mask] == c].mean() for c in present])))
    overall = save('overall_by_seed.csv', overall)
    cls = save('class_by_seed.csv', classes)
    grp = save('scene_by_seed.csv', grouped)
    summary = overall.groupby('arm')[['oa', 'ba', 'macro_f1']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / 'overall_summary.csv', index=False, encoding='utf-8-sig')
    class_summary = cls.groupby(['label', 'class_name', 'n', 'arm'])[['precision', 'recall', 'f1']].agg(['mean', 'std'])
    class_summary.columns = ['_'.join(c) for c in class_summary.columns]
    class_summary.reset_index().to_csv(OUT / 'class_summary.csv', index=False, encoding='utf-8-sig')
    scene_summary = grp.groupby(['level', 'group', 'n', 'ships', 'classes', 'ports', 'arm'])[['oa', 'ba']].agg(['mean', 'std'])
    scene_summary.columns = ['_'.join(c) for c in scene_summary.columns]
    scene_summary.reset_index().to_csv(OUT / 'scene_summary.csv', index=False, encoding='utf-8-sig')
    scene_deltas, class_deltas, port_classes = [], [], []
    for baseline in ('plain', 'se'):
        for level in ('port', 'pass_group'):
            a = grp[(grp.arm == 'full') & (grp.level == level)]
            b = grp[(grp.arm == baseline) & (grp.level == level)]
            paired = a.merge(b, on=['group', 'seed'], suffixes=('_full', '_base'))
            for name, block in paired.groupby('group'):
                scene_deltas.append(dict(baseline=baseline, level=level, group=name, n=int(block.n_full.iloc[0]), classes=int(block.classes_full.iloc[0]),
                    oa_delta=float((block.oa_full-block.oa_base).mean()), ba_delta=float((block.ba_full-block.ba_base).mean()),
                    oa_positive_seeds=int((block.oa_full > block.oa_base).sum()), ba_positive_seeds=int((block.ba_full > block.ba_base).sum())))
        differences = np.stack([(scores['full', s].argmax(1) == y).astype(float)-(scores[baseline, s].argmax(1) == y) for s in protocol['seeds']])
        for c in range(8):
            mask = y == c
            delta = differences[:, mask].mean(1)*100
            class_deltas.append(dict(baseline=baseline, label=c, class_name=LABELS[c], n=int(mask.sum()), recall_delta=delta.mean(), recall_delta_std=delta.std(ddof=1),
                oa_contribution_pp=differences[:, mask].sum(1).mean()/len(y)*100, positive_seeds=int((delta > 0).sum())))
        for port in sorted({r['port'] for r in query}):
            pmask = np.array([r['port'] == port for r in query])
            for c in np.unique(y[pmask]):
                mask = pmask & (y == c)
                port_classes.append(dict(baseline=baseline, port=port, label=int(c), class_name=LABELS[c], n=int(mask.sum()), recall_delta=100*differences[:, mask].mean()))
        if not np.isclose(sum(r['oa_contribution_pp'] for r in class_deltas if r['baseline'] == baseline), differences.mean()*100):
            raise ValueError('Class contributions do not reconstruct the overall difference')
    scene_deltas = save('scene_paired_deltas.csv', scene_deltas)
    class_deltas = save('class_paired_deltas.csv', class_deltas)
    save('port_class_paired_deltas.csv', port_classes)

    cases = json.loads((ROOT / 'cases.json').read_text(encoding='utf-8'))
    by_id = {str(r['id']): r for r in rows}
    query_index = {str(r['id']): i for i, r in enumerate(query)}
    gallery = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>24港全量预测缓存分析</title>',
        '<style>body{font:16px/1.7 system-ui;margin:24px auto;max-width:1250px;padding:0 16px;color:#202124}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{padding:5px 10px;border:1px solid #ccc}section{border-top:1px solid #ddd;margin-top:30px}summary{cursor:pointer}a{color:#146f63}</style>',
        f"<h1>24港全量预测缓存分析</h1><p>仅使用已完成的3种子预测与原始PNG，不训练、不重新推理。{len(query)}条测试观测，{len({r['mmsi'] for r in query})}艘船，{len({r['port'] for r in query})}个港口，{len({r['pass_group'] for r in query})}个完整过境组。</p>",
        '<p><a href="FINDINGS_ZH.md">分析报告</a> · <a href="class_summary.csv">逐类表</a> · <a href="scene_paired_deltas.csv">逐场景配对表</a> · <a href="EXPERIMENT_DESIGN_REVIEW_ZH.md">实验设计核对</a></p>',
        summary.round(3).to_html(),
        '<p>图像为数据集中原有8位PNG显示值，不再次拉伸。上排：原始chip；下排：模型160×160裁剪/补零输入及目标框边界。所有panel保持1图像像素=1输出像素；黑色留白不代表海面。参考图只展示固定16例库的真实成员，不代表已证明其贡献最大。案例按三种子重复性、不同类别选取，不代表总体占比。</p>']
    evidence = []
    (OUT / 'cases').mkdir(exist_ok=True)
    names = {'rescue':'救回', 'harm':'误伤', 'stable_correct':'持续判对', 'stable_wrong':'持续判错'}
    for case in cases:
        row = by_id[str(case['id'])]
        j = query_index[str(case['id'])]
        true = int(y[j])
        f = scores['full', case['seed']][j]
        b = scores[case['baseline'], case['seed']][j]
        rival = int(b.argmax()) if case['kind'] == 'rescue' else int(f.argmax())
        if rival == true:
            rival = int(np.where(np.arange(8) == true, -np.inf, f).argmax())
        refs = [rows[i] for i in groups['references'] if rows[i]['label'] in (true, rival)]
        entries = [row]+refs
        panel_paths, details = [], []
        for k, entry in enumerate(entries):
            views = [prepare_view(entry.get(p)) for p in ('vv', 'vh')]
            width = max([180]+[v['raw'].shape[1]+20 for v in views if v is not None])
            raw_height = max([160]+[v['raw'].shape[0] for v in views if v is not None])
            panel = Image.new('RGB', (2*width, raw_height+220), 'white')
            draw = ImageDraw.Draw(panel)
            for pi, view in enumerate(views):
                x = pi*width+10
                draw.text((x, 3), f"ID {entry['id']} C{entry['label']} {('VV','VH')[pi]}", fill='black')
                if view is None:
                    draw.text((x, 40), 'Unavailable', fill='red')
                    continue
                panel.paste(Image.fromarray(view['raw']).convert('RGB'), (x, 24))
                rgb = np.repeat(view['pixels'][..., None], 3, axis=2)
                mask = view['masks'][0].astype(bool)
                edge = mask & ~(np.roll(mask, 1, 0)&np.roll(mask, -1, 0)&np.roll(mask, 1, 1)&np.roll(mask, -1, 1))
                rgb[edge] = (0, 210, 125)
                draw.text((x, raw_height+32), f"Input; valid={view['valid_target']}", fill='black')
                panel.paste(Image.fromarray(rgb), (x, raw_height+52))
                details.append(dict(id=entry['id'], role='query' if k == 0 else 'source_reference', polarization=('VV','VH')[pi],
                    original_path=entry[('vv','vh')[pi]]['image'], shape=list(view['raw'].shape), valid_target=view['valid_target'], placement=view['placement']))
            name = f"cases/{case['baseline']}_{case['kind']}_{case['id']}_{k}.png"
            panel.save(OUT / name)
            panel_paths.append(name)
        full_margin = float(f[true]-np.max(np.delete(f, true)))
        base_margin = float(b[true]-np.max(np.delete(b, true)))
        evidence.append(dict(**case, pass_group=row['pass_group'], panels=panel_paths, original_images=details,
            full_margin=full_margin, baseline_margin=base_margin, all_reference_ids=[r['id'] for r in refs]))
        gallery += [f"<section><h2>{names[case['kind']]}：{case['id']} · {html.escape(row['port'])} · {LABELS[true]}</h2>",
            f"<p>{html.escape(row['pass_group'])}；对照{case['baseline']}；种子{case['seed']}；该事件在3种子中出现{case['repeated']}次。对照：{LABELS[int(b.argmax())]} → 完整模型：{LABELS[int(f.argmax())]}。真实类对最强错误类logit分差：{base_margin:.3f} → {full_margin:.3f}（不同模型分数尺度可能不同，仅记录判决位置，不能直接当贡献大小）。</p>",
            f"<p><strong>原图观察：</strong>{OBSERVATIONS[str(case['id'])]}</p>",
            f"<img src='{panel_paths[0]}' alt='查询原始双极化图及实际输入'><details><summary>查看固定参考库中真实类与混淆类的全部4例参考</summary>"]
        gallery += [f"<img src='{p}' alt='实际源参考双极化chip'>" for p in panel_paths[1:]]
        gallery += ['</details></section>']
    (OUT / 'cases_local.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    sheets = [Image.open(OUT / case['panels'][0]).convert('RGB') for case in evidence]
    tile_w = max(im.width for im in sheets)
    tile_h = max(im.height for im in sheets)+24
    contact = Image.new('RGB', (tile_w*3, tile_h*4), 'white')
    draw = ImageDraw.Draw(contact)
    for i, (im, case) in enumerate(zip(sheets, evidence)):
        x, yy = i%3*tile_w, i//3*tile_h
        draw.text((x+5, yy+2), f"{case['baseline']} {case['kind']} ID={case['id']}", fill='black')
        contact.paste(im, (x, yy+24))
    contact.save(OUT / 'cases' / 'query_contact.png')
    (OUT / 'index.html').write_text('\n'.join(gallery)+ '</html>', encoding='utf-8')
    stats = []
    for (base, level), block in scene_deltas.groupby(['baseline','level']):
        stats.append(dict(baseline=base, level=level, groups=len(block), positive=int((block.oa_delta > 1e-9).sum()), negative=int((block.oa_delta < -1e-9).sum()), tie=int((block.oa_delta.abs() <= 1e-9).sum()), positive_all_seeds=int((block.oa_positive_seeds == 3).sum())))
    save('scene_direction_counts.csv', stats)
    print(summary.to_string())
    print(class_deltas.to_string(index=False))
    print(pd.DataFrame(stats).to_string(index=False))
    print('Saved', len(cases), 'cases with original VV/VH and all relevant fixed references')


if __name__ == '__main__':
    main()
