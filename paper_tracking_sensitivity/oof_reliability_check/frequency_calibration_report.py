"""Frozen final-trial metrics and bounded routing-risk diagnostics."""

from frequency_calibration_run import OUT, EXP, PAIR, MODES
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from prepare_learned_frequency import ROOT, SOURCE, DATASETS, load_npz
from frequency_reliability_run import statistics
from frequency_reliability_core import balanced_weights
from frequency_band_report import markdown_table


def report():
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    infos = [json.loads((p/'PREPARED.json').read_text()) for p in folders]
    expected = {(i['dataset'], i['task']) for i in infos}
    found = {(p.parent.parent.name, p.parent.name) for p in (OUT/'targets').glob('*/*/COMPLETE.json')}
    if len(expected) != 25 or found != expected:
        raise ValueError('Freeze all25 predictions before target labels')
    source = [json.loads((OUT/'source'/i['dataset']/i['task']/'SELECTION.json').read_text()) for i in infos]
    metrics, classes, events, aucs, reliability, selections = [], [], [], [], [], []
    for folder, info, selection in zip(folders, infos, source):
        meta = dict(dataset=info['dataset'], task=info['task'])
        suffix = f"{info['dataset']}/{info['task']}"
        p, original = load_npz(OUT/'targets'/suffix/'policies.npz'), load_npz(folder/'target.npz')
        pair = load_npz(PAIR/'targets'/suffix/'policies.npz')
        if tuple(p['modes']) != MODES:
            raise ValueError('Mode mismatch')
        for key in ('rows', 'support', 'eval_masks'):
            np.testing.assert_array_equal(p[key], original[key])
        for key in ('base_predictions', 'predictions', 'frozen_predictions', 'selected', 'choices'):
            np.testing.assert_array_equal(p[key][:, -1], p[key][:, MODES.index(selection['rule'])])
        np.testing.assert_array_equal(p['frozen_predictions'][:, 0], p['predictions'][:, 0])
        manifest, idcol = ((SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
            else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'))
        aligned = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(aligned[idcol].to_numpy(str), original['ids'])
        y_all = aligned.class_id.to_numpy()
        np.testing.assert_array_equal(y_all[p['support']], original['support_labels'])
        selections.append(dict(**meta, rule=selection['rule'], source_gain=selection['source_delta_ba']['calibrated']))
        for draw, ev in enumerate(p['eval_masks']):
            y = y_all[ev]
            for stage, key in (('base', 'base_predictions'), ('full', 'predictions'), ('frozen_main_route', 'frozen_predictions')):
                preds = p[key][draw][:, ev]
                stats = [statistics(y, pred) for pred in preds]
                for j, mode in enumerate(MODES):
                    row = dict(**meta, draw=draw, stage=stage, mode=mode, oa=stats[j][0], ba=stats[j][1], macro_f1_8=stats[j][2])
                    for ref in ('learned', 'standalone_local', 'margin_selected', 'pair_local'):
                        ri = MODES.index(ref)
                        for k, metric in enumerate(('oa', 'ba')):
                            row[f'delta_{metric}_vs_{ref}'] = stats[j][k]-stats[ri][k]
                    metrics.append(row)
                    events.append(dict(**meta, draw=draw, stage=stage, mode=mode,
                        changes=int(np.sum(preds[j] != preds[0])), rescues=int(np.sum((preds[j] == y) & (preds[0] != y))),
                        harms=int(np.sum((preds[j] != y) & (preds[0] == y)))))
                    for c in np.unique(y):
                        mask = y == c
                        classes.append(dict(**meta, draw=draw, stage=stage, mode=mode, class_id=int(c), n=int(mask.sum()),
                            recall=100*float(np.mean(preds[j][mask] == c)),
                            delta_recall=100*float(np.mean(preds[j][mask] == c)-np.mean(preds[0][mask] == c))))
            main = p['base_predictions'][draw, 0, ev]
            bands = p['candidates'][draw, ev]
            probability = p['probabilities'][draw, ev].astype(float)
            weight = balanced_weights(y)
            predictions = np.c_[main, bands]
            for bi in range(5):
                correct = predictions[:, bi] == y
                prob = probability[:, bi]
                bins = np.minimum((prob*10).astype(int), 9)
                ece = sum(float(weight[bins == b].sum()/weight.sum()) *
                    abs(float(np.average(correct[bins == b], weights=weight[bins == b])) -
                        float(np.average(prob[bins == b], weights=weight[bins == b])))
                    for b in np.unique(bins))
                reliability.append(dict(**meta, draw=draw, head=bi,
                    weighted_correctness=float(np.average(correct, weights=weight)),
                    predicted_correctness=float(np.average(prob, weights=weight)),
                    weighted_brier=float(np.average((prob-correct)**2, weights=weight)), weighted_ece=ece))
            for bi in range(4):
                rescue, harm = (main != y) & (bands[:, bi] == y), (main == y) & (bands[:, bi] != y)
                event = rescue | harm
                for name, value in (('calibrated', probability[:, bi+1]-probability[:, 0]),
                    ('pair_local', pair['utilities'][draw, 2, ev, bi]), ('smooth', pair['utilities'][draw, 0, ev, bi])):
                    aucs.append(dict(**meta, draw=draw, band=bi, estimator=name,
                        auc=float(roc_auc_score(rescue[event], value[event])) if rescue.any() and harm.any() else np.nan,
                        positive_fraction=float(np.mean(value[event] > 0)) if event.any() else np.nan))
        print(f'FINAL REPORT {suffix}', flush=True)
    m = pd.DataFrame(metrics)
    m.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    pd.DataFrame(selections).to_csv(OUT/'source_selection.csv', index=False)
    values = [c for c in m if c not in ('dataset', 'task', 'draw', 'stage', 'mode')]
    task = m.groupby(['dataset', 'task', 'stage', 'mode'])[values].mean().reset_index()
    task.to_csv(OUT/'task_means.csv', index=False)
    draw = m.groupby(['dataset', 'draw', 'stage', 'mode'])[values].mean().reset_index()
    draw.to_csv(OUT/'draw_metrics.csv', index=False)
    summary = task.groupby(['dataset', 'stage', 'mode'])[values].mean()
    for ref in ('learned', 'standalone_local', 'margin_selected', 'pair_local'):
        summary[f'positive_draws_ba_vs_{ref}'] = draw[f'delta_ba_vs_{ref}'].gt(0).groupby([draw[k] for k in ('dataset', 'stage', 'mode')]).sum()
    summary.reset_index().to_csv(OUT/'summary.csv', index=False)
    pd.DataFrame(events).groupby(['dataset', 'stage', 'mode'])[['changes', 'rescues', 'harms']].sum().to_csv(OUT/'events.csv')
    for name, rows, group, fields in (
        ('utility', aucs, ['estimator'], ['auc', 'positive_fraction']),
        ('correctness', reliability, ['head'], ['weighted_correctness', 'predicted_correctness', 'weighted_brier', 'weighted_ece'])):
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT/(name+'.csv'), index=False)
        frame.groupby(['dataset', 'task', *group])[fields].mean().groupby(['dataset', *group]).mean().to_csv(OUT/(name+'_summary.csv'))
    main = summary.reset_index()
    show = main[(main.stage == 'full') & main['mode'].isin(['learned', 'calibrated', 'source_selected'])]
    (OUT/'EXPERIMENT_REPORT_ZH.md').write_text('# 最终校准对照\n\n'+markdown_table(show)+'\n\n'
        'OA/BA为百分数，增量为百分点；先平均五组支持抽样，再按任务等权。固定交接使用原learned位置。'
        '只有一个新增校准器，未搜索阈值；source_selected只从原主模型learned和calibrated中选择。'
        '其他历史结果仅作参考，不参与本轮选择。目标真值仅用于全部预测冻结后的评估。'
        'weighted correctness/Brier/ECE以真实评估类平衡权重作事后诊断，不能当作理论校准保证。'
        '收口决策与文献见CLOSEOUT_ZH.md和RELATED_WORK_ZH.md。\n', encoding='utf-8')
    (OUT/'REPORT_COMPLETE.json').write_text(json.dumps(dict(tasks=25, draws=125, labels_aligned=True,
        original_retained=True, target_labels_used_for_selection=False, final_round=True,
        diagnosis_pending=True), indent=2), encoding='utf-8')
    print(show[['dataset', 'mode', 'oa', 'ba', 'delta_ba_vs_learned']].to_string(index=False), flush=True)


if __name__ == '__main__':
    report()
