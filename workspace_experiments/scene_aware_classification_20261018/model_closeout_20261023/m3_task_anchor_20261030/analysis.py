"""Bounded seed17 CPU analysis; reuse reference-episode helpers, never model imports.

python -B analysis.py --ready
python -B analysis.py --gallery
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT = HERE / 'analysis'
spec = importlib.util.spec_from_file_location(
    'task_anchor_episode_cpu', HERE.parent / 'm3_reference_episode_20261030/analysis.py')
episode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(episode)
helper = episode.helper
replay = helper.replay
read, dump, metric = helper.read, helper.dump, helper.metric
METRICS, STATES = helper.METRICS, helper.STATES
ARMS = ('frozen', 'ce', 'task')
COMPARISONS = (('frozen', 'saved_baseline'), ('task', 'saved_baseline'),
               ('task', 'ce'), ('ce', 'saved_baseline'), ('task', 'frozen'))
helper.HERE, helper.OUT, helper.ARMS = HERE, OUT, ('task',)


def required_inputs():
    paths = []
    for dataset in ('private', 'oss2'):
        paths += [HERE / dataset / 'probe' / name for name in
                  ('frozen_s17.npz', 'result_s17.json')]
        for root, arm in ((HERE, 'task'), (helper.CE_ROOT, 'ce')):
            paths += [root / dataset / folder / f'{arm}_s17.{ext}'
                      for folder, ext in (('predictions', 'npz'), ('training', 'json'))]
    return paths


def load_dataset(dataset):
    d = helper.load_dataset(dataset, [17])
    if d['pending']:
        raise RuntimeError(json.dumps(d['pending']))
    pp = HERE / dataset / 'probe' / 'frozen_s17.npz'
    pack, absent = helper.load_pack(pp, d['group'], d['classes'])
    if absent:
        raise ValueError(absent)
    if not np.array_equal(pack['baseline'], d['scores']['saved_baseline', 17]):
        raise ValueError(f'{dataset}: probe and training originals differ')
    tp = HERE / dataset / 'predictions' / 'task_s17.npz'
    with np.load(tp, allow_pickle=False) as raw:
        lookup = {key: i for i, key in enumerate(raw['ids'].astype(str))}
        frozen = raw['frozen'][[lookup[str(r['id'])] for r in d['group']]]
    if not np.array_equal(frozen, pack['score']):
        raise ValueError(f'{dataset}: training frozen endpoint differs from probe')
    d['scores']['frozen', 17] = pack['score']
    rp = HERE / dataset / 'probe' / 'result_s17.json'
    d['probe'] = read(rp)
    report = d['receipts']['task', 17]
    if report['source_count'] != d['probe']['source_count']:
        raise ValueError(f'{dataset}: source cohorts differ')
    d['input_paths'] += [str(pp), str(rp)]
    return d


def summarize(d):
    folder = OUT / d['dataset']
    folder.mkdir(parents=True, exist_ok=True)
    endpoints, paired, tradeoffs, events = helper.previous.paired_tables(d, ARMS, COMPARISONS)
    identities = []
    for arm, reference in COMPARISONS:
        new, old = d['scores'][arm, 17], d['scores'][reference, 17]
        identities.append(dict(arm=arm, reference=reference,
            changed=int((new.argmax(1) != old.argmax(1)).sum()),
            max_abs_logit_delta=float(abs(new-old).max()),
            logits_identical=bool(np.array_equal(new, old))))
    report, probe = d['receipts']['task', 17], d['probe']
    source_weights = []
    for phase, rows in (('original', probe['relations_before']),
                        ('frozen', probe['relations_after']),
                        ('trained_task', report['source_relations'])):
        for row in rows:
            original = next(r for r in probe['relations_before'] if r['pol'] == row['pol'])
            frozen = next(r for r in probe['relations_after'] if r['pol'] == row['pol'])
            if row['count'] != original['count']:
                raise ValueError(f"{d['dataset']}: source weight denominators differ")
            source_weights.append(dict(phase=phase, polarization=('VV', 'VH')[row['pol']], **row,
                delta_original=row['same_mass']-original['same_mass'],
                delta_frozen=row['same_mass']-frozen['same_mass']))
    result = dict(dataset=d['dataset'], seed=17, query_count=len(d['y']),
        metrics=endpoints, paired=paired, prediction_identity=identities,
        source_weights=source_weights,
        source_metrics=dict(original=probe['source_before'], frozen=probe['source_after'],
                            trained_task=report['source_after']),
        source_frozen_top1_changes=probe['source_changed'], input_paths=d['input_paths'],
        training=dict(steps=report['steps'], lr=report['lr'], source_count=report['source_count'],
                      reference_count=report['reference_count']),
        scope='seed17 development/validation only; no order expansion assumed')
    dump(folder / 'summary.json', result)
    for name, rows in (('metrics', endpoints), ('paired', paired), ('class_tradeoffs', tradeoffs),
                       ('events', events), ('prediction_identity', identities), ('source_weights', source_weights)):
        replay.previous.write_csv(folder / f'{name}.csv', rows)
    return result


def select_cases(d, arm, reference):
    events = replay.events(d['scores'][reference, 17], d['scores'][arm, 17], d['group'], 17)
    ce, task = d['scores']['ce', 17], d['scores']['task', 17]
    for i, event in enumerate(events):
        truth, rival = event['label'], event['rival']
        event.update(arm=arm, reference=reference, state_orders=1,
            new_vs_ce=bool(task[i].argmax() != ce[i].argmax()),
            ce_prediction=int(ce[i].argmax()),
            task_ce_margin_delta=float((task[i, truth]-task[i, rival])-(ce[i, truth]-ce[i, rival])))
    rank = lambda e: (not e['new_vs_ce'], -abs(e['task_ce_margin_delta']),
                      -abs(e['margin_change']), e['id'])
    chosen, absent = [], []
    for state in STATES:
        candidates = sorted((e for e in events if e['state'] == state), key=rank)
        if candidates:
            chosen.append(candidates[0])
        else:
            absent.append(dict(state=state, state_events=0, reason='No events relative to original.'))
    used = {e['id'] for e in chosen}
    extras = sorted((e for e in events if e['id'] not in used), key=rank)
    chosen += extras[:4-len(chosen)]
    return chosen, absent


def gallery(d):
    replay.OUT, replay.ARMS = OUT, ARMS
    original_selector = replay.select_cases
    replay.select_cases = select_cases
    try:
        result = replay.gallery(d, 'task', 'saved_baseline')
    finally:
        replay.select_cases = original_selector
    result['rule'] = ('Seed17 only. One query per original-relative state; prioritize task-vs-CE '
        'top1 differences, then absolute task-vs-CE fixed-rival margin change, original-relative '
        'margin change and ID. Fill absent-state slots with remaining ranked distinct queries; max4. No random sampling.')
    folder = OUT / d['dataset'] / 'task_vs_saved_baseline'
    dump(folder / 'selection.json', result)
    # The existing HTML embeds the selector description; keep it consistent with selection.json.
    page = folder / 'index.html'
    import html
    text = page.read_text(encoding='utf-8')
    old_rule = 'One distinct query/state, ordered by state frequency, absolute fixed-rival margin change, ID, seed. No random sampling.'
    page.write_text(text.replace(old_rule, html.escape(result['rule'])), encoding='utf-8')
    return result


def write_final(results):
    lines = ['# Task-anchor seed17 CPU分析', '',
        '只含自建development与OSS2 validation；不假设29/43扩围，不读取最终测试或运行模型。',
        '启动修复（主代理报告）：首次在训练前因通用模块名probe与旧coverage.center导入冲突退出；'
        'train.py改用唯一spec名task_anchor_probe后重启，无超参数变更。这不是一次训练负结果。', '']
    for r in results:
        contrast = next(x for x in r['paired'] if (x['arm'], x['reference']) == ('task', 'ce'))
        baseline = next(x for x in r['paired'] if (x['arm'], x['reference']) == ('task', 'saved_baseline'))
        lines.append(f"{r['dataset']}：task相对CE有{contrast['top1_changes']}条top1差异，"
            f"OA/BA差{contrast['delta_OA']:+.6f}/{contrast['delta_BA']:+.6f}个百分点；"
            f"相对原版为{baseline['delta_OA']:+.6f}/{baseline['delta_BA']:+.6f}个百分点。")
    if all(next(x for x in r['paired'] if (x['arm'], x['reference']) == ('task', 'ce'))['top1_changes'] == 0
           for r in results):
        lines.append('两组训练task与CE逐查询预测相同；源同类参考权重上移尚未转化为额外分类收益。')
    lines.append('')
    for r in results:
        lines += [f"## {r['dataset']}（{r['query_count']}条）", '']
        replay.prior.table(lines, ['方案', 'OA', 'BA', 'NLL', 'Brier'],
            [[x['arm'], *[round(x[k], 6) for k in METRICS]] for x in r['metrics']])
        replay.prior.table(lines, ['比较', 'Delta OA', 'Delta BA', 'Delta NLL', 'Delta Brier', '救回', '误伤', 'CC', 'WW', '翻转'],
            [[x['arm']+' - '+x['reference'], *[round(x['delta_'+k], 6) for k in METRICS],
              *[x[s] for s in STATES], x['top1_changes']] for x in r['paired']])
        lines += ['', '实际源reference_weights同类和：有效槽位内求和、查询内平均，再汇总；balanced为类别等权均值。数值为0–1，非分类准确率。', '']
        replay.prior.table(lines, ['阶段', '极化', '有效查询', 'same mass', 'balanced mass', 'entropy'],
            [[x['phase'], x['polarization'], x['count'], round(x['same_mass'], 7),
              round(x['balanced_same_mass'], 7), round(x['entropy'], 7)] for x in r['source_weights']])
        source = r['source_metrics']
        lines.append('源OA/BA（原版→冻结替换→训练task）：' + ' → '.join(
            f"{source[p]['oa']:.6f}/{source[p]['ba']:.6f}" for p in ('original', 'frozen', 'trained_task')) +
            f"；冻结替换源预测翻转{r['source_frozen_top1_changes']}条。源端权重上移不等于分类收益。")
        identity = next(x for x in r['prediction_identity'] if (x['arm'], x['reference']) == ('task', 'ce'))
        lines.append(f"task vs CE：{identity['changed']}条top1不同；最大logit差{identity['max_abs_logit_delta']:.8g}。")
        if 'gallery' in r:
            g = r['gallery']
            folder = OUT / r['dataset'] / 'task_vs_saved_baseline'
            cases = read(folder / 'cases.json')
            lines += ['', f"原图画廊：{(folder/'index.html').as_posix()}；{g['cases']}查询/{g['panels']}张VV/VH面板。",
                '状态相对原版；各状态内优先task相对CE新差异，缺失状态补其余独立查询。'
                '参考来自实际固定银行，排除同ID/同船，展示前2条合法同类/混淆类，不代表最大注意力。',
                '案例：' + '；'.join(f"{c['id']} {c['state']} new_vs_CE={c['new_vs_ce']}" for c in cases) + '。',
                '原图尺度、裁剪与路径见native_sources.json；面板已生成，视觉判断由主代理完成，未据外观推断物理原因。']
        lines.append('')
    lines += ['## 解释边界', '',
        '冻结锚点替换与后续训练分开列示；训练task对CE才隔离相同训练预算下的锚点路径差异。'
        '同类参考权重的小幅变化不能称显著匹配提升；持续同预测时仍报告概率/分差变化。'
        '这是一组seed17筛查，不能据此声称跨顺序稳定性。OA/BA差为百分点，NLL/Brier越低越好。',
        '本支线仅ID对齐、配对指标、源权重摘要与最多16张原图面板；无GPU、训练、Git或全项目审计。']
    (OUT / 'FINAL_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready', action='store_true')
    parser.add_argument('--gallery', action='store_true')
    args = parser.parse_args()
    missing = [str(p) for p in required_inputs() if not p.is_file()]
    if args.ready or missing:
        print(json.dumps(dict(ready=not missing, missing=missing), indent=2))
        return
    datasets = [load_dataset(name) for name in ('private', 'oss2')]
    print('READY: both seed17 prediction packs, receipts and frozen probes aligned.', flush=True)
    results = []
    for d in datasets:
        result = summarize(d)
        if args.gallery:
            result['gallery'] = gallery(d)
        results.append(result)
    assert 'torch' not in sys.modules, 'CPU sidecar imported model runtime'
    write_final(results)
    dump(OUT / 'run_receipt.json', dict(seed=17, cpu_only=True, torch_imported=False,
        final_test_read=False, source_prediction_npz_read=False, expansion_assumed=False,
        results=[dict(dataset=r['dataset'], prediction_identity=r['prediction_identity'],
                      gallery=r.get('gallery')) for r in results]))
    print(json.dumps([dict(dataset=r['dataset'], comparisons=r['paired']) for r in results], indent=2))


if __name__ == '__main__':
    main()
