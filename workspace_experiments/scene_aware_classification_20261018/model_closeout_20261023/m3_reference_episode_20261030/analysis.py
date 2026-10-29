"""Bounded CPU analysis after READY: python -B analysis.py --gallery.

Defaults to the completed seed17 pilot. No expansion is pending.
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
    'episode_cpu_helpers', HERE.parent / 'm3_query_interaction_20261030/analysis.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
ARMS = ('fixed_alt', 'episodic')
COMPARISONS = tuple((a, b) for a in ARMS for b in ('saved_baseline', 'ce'))
COMPARISONS += (('episodic', 'fixed_alt'),)
helper.HERE, helper.OUT = HERE, OUT
helper.ARMS, helper.COMPARISONS = ARMS, COMPARISONS
load_dataset, load_pack = helper.load_dataset, helper.load_pack
read, dump, metric = helper.read, helper.dump, helper.metric
METRICS, STATES = helper.METRICS, helper.STATES


def summarize(d):
    folder = OUT / d['dataset']
    folder.mkdir(parents=True, exist_ok=True)
    endpoints, paired, tradeoffs, events = helper.previous.paired_tables(
        d, ('ce', *ARMS), COMPARISONS)
    means = []
    for arm, reference in COMPARISONS:
        rows = [r for r in paired if (r['arm'], r['reference']) == (arm, reference)]
        row = dict(arm=arm, reference=reference, seeds=[r['seed'] for r in rows])
        for key in METRICS:
            values = [r['delta_' + key] for r in rows]
            row['delta_' + key] = float(np.mean(values)) if values else None
            row['sd_' + key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        row.update({state: sum(r[state] for r in rows) for state in STATES})
        means.append(row)
    identities = []
    for arm, reference in (('episodic', 'fixed_alt'), ('episodic', 'ce'), ('fixed_alt', 'ce')):
        for seed in helper.previous.paired_seeds(d, arm, reference):
            new, old = d['scores'][arm, seed], d['scores'][reference, seed]
            identities.append(dict(arm=arm, reference=reference, seed=seed,
                changed=int(np.count_nonzero(new.argmax(1) != old.argmax(1))),
                logits_identical=bool(np.array_equal(new, old)),
                max_abs_logit_delta=float(np.max(np.abs(new-old))),
                rms_logit_delta=float(np.sqrt(np.mean((new-old)**2)))))
    summary = dict(dataset=d['dataset'], requested=d['requested'], query_count=len(d['y']),
        pending=d['pending'], metrics=endpoints, paired_means=means,
        prediction_identity=identities, input_paths=d['input_paths'],
        notes=['Completed seed17 pilot; expansion explicitly stopped, not pending.',
               'OA/BA differences are percentage points; NLL/Brier differences are unscaled.',
               'BA uses present classes; one-order SD is null. Query-order events are not independent ships.',
               'Saved baseline logits match exactly after ID alignment across CE and both new arms.',
               'Reference counts and query schedules are matched; unique training reference exposure differs.',
               'No dependence-control files, source predictions or final-test manifests are loaded.'])
    dump(folder / 'summary.json', summary)
    for name, rows in (('metrics', endpoints), ('paired', paired), ('paired_means', means),
                       ('events', events), ('class_tradeoffs', tradeoffs), ('prediction_identity', identities)):
        helper.replay.previous.write_csv(folder / (name + '.csv'), rows)
    return summary


def gallery(d):
    if d['requested'] != [17]:
        raise ValueError('Native gallery is restricted to seed17.')
    replay = helper.replay
    replay.OUT, replay.ARMS = OUT, ('ce', *ARMS)
    # One distinct query per state, with rescue/harm considered before CC/WW.
    return replay.gallery(dict(d, seeds=[17]), 'episodic', 'saved_baseline')


def write_final(results):
    lines = ['# M3参考episode：seed17限界CPU结果', '',
        '四次训练小试结束；主代理明确停止扩围，无29/43待办。仅自建开发160和OSS2源验证228，不是最终测试。', '',
        '以下逐查询核对用于区分相同OA/BA与相同预测；两新臂的比较才回答动态换库是否优于固定新库。', '']
    for result in results:
        dataset = result['dataset']
        lines += [f'## {dataset}', '']
        helper.replay.prior.table(lines, ['方案', '顺序', 'OA', 'BA', 'NLL', 'Brier'],
            [[r['arm'], r['seed'], *[round(r[k], 6) for k in METRICS]] for r in result['metrics']])
        helper.replay.prior.table(lines, ['比较', 'Delta OA', 'Delta BA', 'Delta NLL', 'Delta Brier', '救回', '误伤', 'CC', 'WW'],
            [[r['arm']+' - '+r['reference'], *[round(r['delta_'+k], 6) for k in METRICS],
              *[r[s] for s in STATES]] for r in result['paired_means']])
        for r in result['prediction_identity']:
            lines.append(f"- {r['arm']} 对 {r['reference']}：{r['changed']}/{result['query_count']} 条预测不同；"
                         f"logits逐值相同={r['logits_identical']}，最大绝对差={r['max_abs_logit_delta']:.8g}。")
        probe_path = HERE / dataset / 'source_probe_s17.json'
        if probe_path.is_file():
            probe = read(probe_path)
            lines.append('- 现成source_probe四组源库：相对原库改判数=' +
                str([r['changed'] for r in probe['banks']]) + '；最大logit差=' +
                str([round(r['max_logit_delta'], 6) for r in probe['banks']]) +
                '。这是原模型在源适配查询上的参考敏感性，不是泛化收益。')
        if 'gallery' in result:
            g = result['gallery']
            path = OUT / dataset / 'episodic_vs_saved_baseline' / 'index.html'
            lines += ['', f"原图画廊：[{dataset}]({path.as_posix()})，{g['cases']}条查询/{g['panels']}个VV/VH面板。",
                      '按相对原版救回、误伤、持续对、持续错各最多1条，优先分差变化大者；缺失状态见selection.json。',
                      '展示实际原银行中合法同类/混淆类参考，按银行顺序取前2条，不代表最高注意力。原尺度与路径在native_sources.json；图像由主代理查看，本支线仅生成。']
        lines.append('')
    identical = all(next(r for r in result['prediction_identity'] if
        (r['arm'], r['reference']) == ('episodic', 'fixed_alt'))['changed'] == 0 for result in results)
    if identical:
        lines += ['## 判断', '', '两数据集episodic与fixed_alt逐查询top1完全相同，未见动态参考库优于固定新库的分类证据。'
                  '相对CE的改善不能归因于逐episode变化；概率指标与连续logit差异见上表，不能写成新的分类救回。']
    lines += ['', '只完成对齐、配对指标、预测映射核对和限额画廊。未运行共同偏移分析、模型检查、训练或Git；未推断物理原因。']
    (OUT / 'FINAL_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('private', 'oss2', 'all'), default='all')
    parser.add_argument('--seeds', type=int, nargs='+', choices=helper.SEEDS, default=[17])
    parser.add_argument('--gallery', action='store_true')
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    if args.gallery and seeds != [17]:
        parser.error('Gallery supports seed17 only.')
    results = []
    for dataset in (('private', 'oss2') if args.dataset == 'all' else (args.dataset,)):
        d = load_dataset(dataset, seeds)
        if d['pending']:
            raise RuntimeError('Requested READY inputs incomplete: ' + json.dumps(d['pending']))
        result = summarize(d)
        if args.gallery:
            result['gallery'] = gallery(d)
        results.append(result)
    assert 'torch' not in sys.modules
    if seeds == [17]:
        write_final(results)
    dump(OUT / 'run_receipt.json', dict(seeds=seeds, datasets=[r['dataset'] for r in results],
        cpu_only=True, torch_imported=False, dependence_controls_read=False,
        source_predictions_read=False, final_test_read=False, expansion_pending=False))
    print(json.dumps([dict(dataset=r['dataset'], prediction_identity=r['prediction_identity'],
                          gallery=r.get('gallery')) for r in results], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
