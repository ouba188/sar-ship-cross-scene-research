"""Build paper evidence from completed runs; no inference, fitting or tuning."""
from pathlib import Path
import csv
import html
import importlib.util
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
WORK = PAPER.parent
STORE = Path('C:/Users/25791/AppData/Local/sar_paper_experiments_20261110')
OSS2 = PAPER / 'staged_ablation_20261109'
PUBLIC = WORK / 'scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029'
SEEDS = (17, 29, 43)
NAMES4 = ['散货船', '普通货船', '化学品/成品油轮', '集装箱船']
NAMES8 = ['散货船', '渔船', '普通货船', '化学品/成品油轮', '集装箱船', '原油轮', '拖船', '海工船']
TITLES = {'oss1': 'OpenSARShip 1.0', 'oss2': 'OpenSARShip 2.0', 'ports24': '自建24港'}
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 9,
                     'axes.unicode_minus': False, 'pdf.fonttype': 42})
spec = importlib.util.spec_from_file_location('native_input', WORK / 'sar_m123_standalone/sar_m123/input.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def orbit(row):
    return '_'.join(row['pass_group'].split('_')[:2])


def load_dataset(name):
    if name == 'oss2':
        parts = {k: read(PUBLIC / 'manifests' / f'{k}.json') for k in
                 ('source_fit', 'source_adapt', 'validation', 'selection', 'evaluation', 'references')}
        rows = sum((parts[k] for k in ('source_fit', 'source_adapt', 'validation', 'selection', 'evaluation')), [])
        lookup = {r['id']: i for i, r in enumerate(rows)}
        ids = np.array([lookup[r['id']] for r in parts['evaluation']])
        refs = parts['references']
        base = OSS2
    else:
        base = STORE / name
        rows, groups = read(base / 'manifest.json'), read(base / 'groups.json')
        ids = np.array(groups['evaluation'])
        refs = [rows[i] for i in groups['references']]
    scores = {}
    for arm in ('full', 'plain', 'se'):
        all_seeds = []
        for seed in SEEDS:
            path = base / f'{arm}_s{seed}_lr{0.3 if arm == "se" else 1:g}' / 'evaluation.npz'
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['ids'], ids)
                all_seeds.append(saved['score'].copy())
        scores[arm] = np.stack(all_seeds)
    return [rows[i] for i in ids], refs, scores


def margin(score, label):
    return float(score[label] - np.max(np.delete(score, label)))


def select_cases(rows, scores):
    y = np.array([r['label'] for r in rows])
    good = scores['full'].argmax(2) == y
    chosen, used = [], set()
    # Class-balanced illustration, preferring repeatability, never selecting models.
    for arm in ('plain', 'se'):
        rescue = good & (scores[arm].argmax(2) != y)
        counts = rescue.sum(0)
        candidates = [int(i) for i in np.flatnonzero(counts)]
        seen_classes = set()
        for _ in range(3):
            eligible = [i for i in candidates if i not in used]
            if not eligible:
                break
            i = min(eligible, key=lambda j: (y[j] in seen_classes, -counts[j], j))
            seed_pos = int(np.flatnonzero(rescue[:, i])[0])
            chosen.append((arm, i, seed_pos))
            used.add(i)
            seen_classes.add(y[i])
    return chosen


def draw_view(ax, view, title):
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=6)
    if view is None:
        ax.text(.5, .5, '无可用输入', ha='center', va='center')
        return
    ax.imshow(view['pixels'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    if view['valid_target']:
        cx, cy, length, width, angle = view['geometry'][:5]
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * [length / 2, width / 2]
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        box = corners @ rot.T + [cx, cy]
        ax.plot(box[:, 0], box[:, 1], color='#2cbb9b', lw=.7)
    else:
        ax.text(.03, .03, '本极化不进入模型', transform=ax.transAxes, color='#e2ad47', fontsize=8)


def make_panel(name, rows, refs, scores, arm, index, seed_pos, serial, book):
    names = NAMES8 if name == 'ports24' else NAMES4
    row = rows[index]
    y = row['label']
    before, after = scores[arm][seed_pos, index], scores['full'][seed_pos, index]
    wrong = int(before.argmax())
    assert wrong != y and after.argmax() == y
    chosen = [row]
    for cls in (y, wrong):
        chosen.append(next(r for r in refs if r['label'] == cls and
                           str(r['mmsi']) != str(row['mmsi']) and orbit(r) != orbit(row)))
    records = []
    for pos, seed in enumerate(SEEDS):
        b, f = scores[arm][pos, index], scores['full'][pos, index]
        bp, fp = int(b.argmax()), int(f.argmax())
        outcome = ('救回' if bp != y and fp == y else '误伤' if bp == y and fp != y
                   else '均正确' if fp == y else '均错误')
        records.append(dict(seed=seed, before_pred=bp, after_pred=fp, outcome=outcome,
                            before_margin=margin(b, y), after_margin=margin(f, y),
                            before_scores=b.tolist(), after_scores=f.tolist()))
    return render_panel(name, chosen, arm, SEEDS[seed_pos], serial, records, book)


def render_panel(name, chosen, arm, seed, serial, records, book):
    names = NAMES8 if name == 'ports24' else NAMES4
    row = chosen[0]
    y = row['label']
    wrong = next(r['before_pred'] for r in records if r['seed'] == seed)
    identifier = f'{name}_{serial:02d}_{arm}'
    case_dir = ROOT / 'chips' / identifier
    case_dir.mkdir(parents=True, exist_ok=True)
    views = [data.prepare(r) for r in chosen]
    fig = plt.figure(figsize=(10.6, 9.4), layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.09, h_pad=.09)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, .65])
    source_records = []
    for col, (rec, prepared) in enumerate(zip(chosen, views)):
        role = ('查询', '实际同类源参考', '实际混淆类源参考')[col]
        short_id = rec['id'][-6:] if name != 'ports24' else rec['id']
        sources = {}
        for p, view in enumerate(prepared):
            pol = ('VV', 'VH')[p]
            ax = fig.add_subplot(gs[p, col])
            draw_view(ax, view, f'{role} · {pol} · {names[rec["label"]]}\nID {short_id}')
            if view is not None:
                for label, pixels in [('native160', view['pixels']), ('original', view['raw'])]:
                    Image.fromarray(pixels).save(case_dir / f'{col}_{pol}_{label}.png')
                sources[pol] = dict(path=rec[pol.lower()]['image'], valid=bool(view['valid_target']),
                                    original_shape=list(view['raw'].shape), placement=view['placement'])
        source_records.append(dict(role=role, id=rec['id'], label=rec['label'], sources=sources))
    ax = fig.add_subplot(gs[2, :])
    ax.set_axis_off()
    cells = [[r['seed'], names[r['before_pred']], names[r['after_pred']],
              f'{r["before_margin"]:+.3f}', f'{r["after_margin"]:+.3f}', r['outcome']] for r in records]
    table = ax.table(cellText=cells, colLabels=['种子', arm.upper(), 'Full', '对照分差', 'Full分差', '结果'],
                     loc='center', cellLoc='center', colWidths=[.08, .24, .24, .16, .16, .12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#d6dedb')
        if r == 0:
            cell.set_facecolor('#e8efed')
        elif records[r - 1]['outcome'] == '救回':
            cell.set_facecolor('#e9f5ef')
    repeats = sum(r['outcome'] == '救回' for r in records)
    repeat_text = f'{repeats}/3 种子重复' if len(records) == 3 else '已归档的单种子案例'
    query_label = row['id'] if len(row['id']) < 40 else row['id'].split(':', 1)[-1]
    fig.suptitle(f'{TITLES[name]} · Full 相对 {arm.upper()} 救回 · {repeat_text}\n'
                 f'{names[wrong]} → {names[y]} | seed {seed} | 查询 {query_label}', fontsize=12)
    fig.supxlabel('绿色框为既有检测OBB；图像不重新拉伸。参考为库内示例，不是最大贡献参考。\n'
                  '分差 = 真实类别分数 − 该模型最高错误类别分数；不同模型分数幅度不可当作校准概率。', fontsize=8)
    stem = ROOT / 'panels' / identifier
    fig.savefig(stem.with_suffix('.png'), dpi=170)
    fig.savefig(stem.with_suffix('.pdf'))
    book.savefig(fig)
    plt.close(fig)
    return dict(key=identifier, dataset=name, id=row['id'], label=y, true_class=names[y],
                arm=arm, seed=seed, selected_wrong=names[wrong], repeated_rescues=repeats,
                observed_seeds=len(records),
                scene=row.get('scene_id', row['pass_group']), orbit=orbit(row),
                panel=f'panels/{identifier}.png', pdf=f'panels/{identifier}.pdf',
                chips=f'chips/{identifier}', scores=records, sources=source_records)


def build_html(cases):
    parts = []
    for name in TITLES:
        subset = [c for c in cases if c['dataset'] == name]
        parts.append(f'<section id="{name}"><h2>{TITLES[name]} · {len(subset)} 个不同查询</h2>')
        for c in subset:
            parts.append(f'<article><h3>{html.escape(c["id"])} · {c["true_class"]}</h3>'
                         f'<p>Full 相对 {c["arm"].upper()}：'+
                         (f'{c["repeated_rescues"]}/3 种子救回；' if c['observed_seeds'] == 3 else '仅展示已归档seed17，其他种子未在此逐例复核；')+
                         f'图示 seed {c["seed"]}，{c["selected_wrong"]} → {c["true_class"]}。'
                         f'过境 {html.escape(c["orbit"])}。</p>'
                         f'<p><a href="{c["pdf"]}">单例 PDF</a> · <a href="{c["chips"]}/">原图与输入 PNG</a></p>'
                         f'<img loading="lazy" src="{c["panel"]}" alt="{html.escape(c["id"])} 的VV/VH与参考对照"></article>')
        parts.append('</section>')
    document = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>三个数据集：结果与完整模型救回证据</title>
<style>body{max-width:1120px;margin:0 auto;padding:26px 20px;font-family:"Microsoft YaHei",sans-serif;color:#25312f;background:white;line-height:1.7}h1{font-size:26px}h2{font-size:22px;border-bottom:2px solid #26776c;padding-top:20px}h3{font-size:18px}a{color:#17685c}nav{display:flex;gap:22px;flex-wrap:wrap;border-block:1px solid #cedbd6;padding:12px 0}article{border-bottom:1px solid #cedbd6;padding:12px 0 26px}img{display:block;width:100%;height:auto}.note{color:#546461;font-size:14px}table{width:100%;border-collapse:collapse}th,td{padding:8px;border-bottom:1px solid #d7dfdc;text-align:left}@media(max-width:600px){h1{font-size:22px}body{padding:16px 12px}nav{gap:12px}}</style></head><body>
<h1>三个数据集：完整模型的真实救回案例</h1>
<p>同一划分、同一查询、相同训练种子，与配套训练的 Plain / 匹配位置 SE 比较。仅复用已完成的预测，没有重训或调参。</p>
<nav><a href="#oss1">OSS1</a><a href="#oss2">OSS2</a><a href="#ports24">自建24港</a><a href="results.html">正向与负向结果报告</a><a href="RESCUE_CASEBOOK.pdf">全部案例 PDF</a><a href="guide.html">案例观察与图注</a></nav>
<p><strong>总体结论：</strong>Full 在24港、OSS2优于两种对照；OSS1优于Plain但弱于SE。救回不是全面获胜，净收益以全部评价样本的配对统计为准。</p>
<p class="note">OSS2优先展示不同类别、重复救回的查询；OSS1与24港复用已归档的Plain/SE救回各1例，不能推断其单例跨种子重复性。只收录Full相对Plain/SE的真实纠错，不混入关闭单模块的案例。全部评价样本的救回和误伤数量见 paired_outcomes.csv。支持图是合法实际源参考示例，不作为注意力或因果归因图。</p>
'''+''.join(parts)+'</body></html>'
    (ROOT / 'index.html').write_text(document, encoding='utf-8')


def render_documents():
    import markdown
    for source, dest in [('PAPER_RESULTS_ZH.md', 'results.html'), ('CASE_GUIDE_ZH.md', 'guide.html')]:
        path = ROOT / source
        if not path.exists():
            continue
        content = markdown.markdown(path.read_text(encoding='utf-8'), extensions=['tables', 'fenced_code'])
        content = content.replace('href="E:/', 'href="file:///E:/')
        page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三数据集论文证据</title><style>body{max-width:1120px;margin:24px auto;padding:0 20px;font-family:"Microsoft YaHei",sans-serif;line-height:1.8;color:#243630}h1{font-size:26px}h2{font-size:21px;margin-top:28px}h3{font-size:18px}table{border-collapse:collapse;width:100%;display:block;overflow-x:auto}td,th{padding:8px 10px;border-bottom:1px solid #ccd9d4;text-align:left}th{background:#eff4f2}a{color:#1b6f60}code{overflow-wrap:anywhere}blockquote{border-left:3px solid #438b7d;margin-left:0;padding-left:18px;color:#314e45}@media print{body{max-width:none;font-size:10pt}h2{break-after:avoid}tr{break-inside:avoid}}</style></head><body><nav><a href="index.html">救回图库</a> · <a href="results.html">结果报告</a> · <a href="guide.html">案例说明</a></nav>'''+content+'</body></html>'
        (ROOT / dest).write_text(page, encoding='utf-8')


def main():
    (ROOT / 'panels').mkdir(parents=True, exist_ok=True)
    cases, outcomes = [], []
    with PdfPages(ROOT / 'RESCUE_CASEBOOK.pdf') as book:
        for name in TITLES:
            if name != 'oss2' and not (STORE / name / 'manifest.json').exists():
                prior = PAPER / 'dataset_completion_20261110' / name
                saved_cases = [c for c in read(prior / 'case_records.json')
                               if c['arm'] in ('plain', 'se') and c['outcome'] == 'rescue']
                for serial, c in enumerate(saved_cases, 1):
                    before, after = np.array(c['before']), np.array(c['after'])
                    assert before.argmax() != c['label'] and after.argmax() == c['label']
                    np.testing.assert_allclose([margin(before, c['label']), margin(after, c['label'])],
                                               [c['margin_before'], c['margin_after']], atol=1e-5)
                    chosen = c['sources']
                    if name == 'oss1':
                        parts = c['id'].split(':')[0].split('_')
                        pass_group = parts[0] + '_' + parts[6]
                    else:
                        cohorts = read(WORK / 'scene_aware_classification_20261018/model_closeout_20261023/spatial_m12_full_20261029/cohorts.json')
                        pass_group = next(r['pass_group'] for r in cohorts['expanded'] if r['id'] == c['id'])
                    chosen[0]['pass_group'] = pass_group
                    one = dict(seed=c['seed'], before_pred=int(before.argmax()), after_pred=int(after.argmax()),
                               outcome='救回', before_margin=margin(before, c['label']), after_margin=margin(after, c['label']),
                               before_scores=before.tolist(), after_scores=after.tolist())
                    cases.append(render_panel(name, chosen, c['arm'], c['seed'], serial, [one], book))
                with (prior / 'paired_changes.csv').open(encoding='utf-8-sig') as f:
                    for r in csv.DictReader(f):
                        if r['arm'] in ('plain', 'se'):
                            outcomes.append(dict(dataset=name, arm=r['arm'], seed=int(r['seed']),
                                n=63 if name == 'oss1' else 3075, rescued=int(r['rescued']), harmed=int(r['harmed']),
                                stable_correct='', stable_wrong=''))
                print(name, 'archived cases complete; full prediction store absent', flush=True)
                continue
            rows, refs, scores = load_dataset(name)
            y = np.array([r['label'] for r in rows])
            for arm in ('plain', 'se'):
                b, f = scores[arm].argmax(2) == y, scores['full'].argmax(2) == y
                for j, seed in enumerate(SEEDS):
                    outcomes.append(dict(dataset=name, arm=arm, seed=seed, n=len(y),
                        rescued=int((~b[j] & f[j]).sum()), harmed=int((b[j] & ~f[j]).sum()),
                        stable_correct=int((b[j] & f[j]).sum()), stable_wrong=int((~b[j] & ~f[j]).sum())))
            for serial, (arm, index, seed_pos) in enumerate(select_cases(rows, scores), 1):
                cases.append(make_panel(name, rows, refs, scores, arm, index, seed_pos, serial, book))
            print(name, 'panels complete', flush=True)
    write_json(ROOT / 'case_records.json', cases)
    with (ROOT / 'paired_outcomes.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(outcomes[0]))
        writer.writeheader(); writer.writerows(outcomes)
    with (ROOT / 'case_index.csv').open('w', encoding='utf-8-sig', newline='') as f:
        keys = ['key', 'dataset', 'id', 'true_class', 'arm', 'seed', 'repeated_rescues', 'orbit', 'panel', 'pdf']
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(cases)
    build_html(cases)
    render_documents()
    print(json.dumps([dict(dataset=c['dataset'], id=c['id'], label=c['true_class'], arm=c['arm'],
                           repeated=c['repeated_rescues'], file=c['panel']) for c in cases], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
