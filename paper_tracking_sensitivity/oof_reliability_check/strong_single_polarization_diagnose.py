"""Bounded CPU-only 2x2 attribution of the completed polarization control."""

import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'

import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from strong_single_polarization_control import OUT, CACHE, MODES, task_folders
from strong_single_r50_task import (ROOT, SOURCE, geometry, support_scores,
                                    target_design, write_json, metrics)

DEST = OUT/'diagnosis'
PYTHON = ROOT.parent/'envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe'
KEYS = ('source_rows', 'rows', 'ids', 'support', 'support_labels', 'eval_masks')
CELLS = ('P0_R0', 'P1_R0', 'P0_R1', 'P1_R1')
PATHS = ((0, 0), (1, 0), (0, 1), (1, 1))
DUAL = 'current_dual_reference'
COMPARISONS = (
    ('prior_given_R0', 0, 1), ('relation_given_P0', 0, 2),
    ('prior_given_R1', 2, 3), ('relation_given_P1', 1, 3),
    ('joint', 0, 3), ('dual_vs_P0_R0', 0, 4),
    ('dual_vs_P1_R0', 1, 4), ('dual_vs_P0_R1', 2, 4),
    ('dual_vs_P1_R1', 3, 4))
BLOCK = 1024


def manifest(dataset):
    if dataset == 'self24ports':
        return SOURCE/'manifest.csv', 'instance_id'
    folder = ('f0921_detector_recrop_20261007' if dataset == 'f0921'
              else 'opensarship2_transfer_20261007')
    return ROOT/folder/'target_manifest.csv', 'sample_id'


def load_aligned(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    dataset, name = info['dataset'], info['task']
    original = OUT/'targets'/dataset/name
    if not (original/'PREDICTED.json').exists():
        raise ValueError(f'Original task is incomplete: {original}')
    data = target_design(folder/'target.npz')
    with np.load(original/'predictions.npz') as saved:
        for key in KEYS:
            np.testing.assert_array_equal(saved[key], data[key], err_msg=f'control/{key}')
        diagonal = saved['predictions'][:, [list(saved['modes']).index(m) for m in MODES]]
    reference = ROOT/'backbone_ablation_20261010/targets'/dataset/name/'predictions.npz'
    with np.load(reference) as saved:
        reference_keys = [key for key in KEYS if key in saved.files]
        for key in reference_keys:
            np.testing.assert_array_equal(saved[key], data[key], err_msg=f'dual/{key}')
        dual = saved['predictions'][:, list(saved['modes']).index('dual_refit_late')]
    path, idcol = manifest(dataset)
    manifest_ids = pd.read_csv(path, usecols=[idcol])[idcol].to_numpy(str)
    pool = 'source' if dataset == 'self24ports' else dataset
    cached_ids = np.load(CACHE/f'{pool}_ids.npy')
    np.testing.assert_array_equal(cached_ids, manifest_ids)
    np.testing.assert_array_equal(cached_ids[data['rows']], data['ids'])
    raw, weights = [], []
    for mode in MODES:
        receipt = json.loads((OUT/f'{pool}_{mode}_EXTRACTED.json').read_text())
        feature_path = CACHE/f'{pool}_{mode}.npy'
        if Path(receipt['output']) != feature_path:
            raise ValueError('Fresh extraction receipt and feature path disagree')
        features = np.load(feature_path, mmap_mode='r')
        if (features.shape != (len(cached_ids), 2048) or features.dtype != np.float32
                or receipt['rows'] != len(cached_ids)):
            raise ValueError('Unexpected fresh raw-feature shape or precision')
        weight = np.load(original/f'{mode}_source_prior.npy')
        if weight.shape != (8, 2049) or weight.dtype != np.float32:
            raise ValueError('Saved source prior must retain original float32 precision')
        raw.append(features)
        weights.append(torch.from_numpy(weight))
    alignment = dict(control_design_keys=list(KEYS), dual_reference_keys=reference_keys,
        dual_reference_absent_keys=[key for key in KEYS if key not in reference_keys],
        full_cache_ids_equal_manifest=True, selected_ids_equal_design=True,
        source_prior_paths=[str(original/f'{mode}_source_prior.npy') for mode in MODES],
        raw_feature_paths=[str(CACHE/f'{pool}_{mode}.npy') for mode in MODES],
        raw_feature_rows=len(cached_ids))
    return info, data, diagonal, dual, raw, weights, alignment


@torch.no_grad()
def predict(data, raw, weights, name):
    predicted = np.empty((5, 4, len(data['rows'])), dtype=np.int8)
    support_phi, support_prior = [], []
    for support in data['support']:
        phis = [geometry(torch.from_numpy(np.array(r[data['rows'][support]]))) for r in raw]
        support_phi.append(phis)
        support_prior.append([F.linear(phi, w) for phi, w in zip(phis, weights)])
    labels = [torch.as_tensor(y).long() for y in data['support_labels']]
    last = time.perf_counter()
    for lo in range(0, len(data['rows']), BLOCK):
        rows = data['rows'][lo:lo+BLOCK]
        phis = [geometry(torch.from_numpy(np.array(r[rows]))) for r in raw]
        priors = [F.linear(phi, w) for phi, w in zip(phis, weights)]
        for draw in range(5):
            for cell, (pi, ri) in enumerate(PATHS):
                scores = support_scores(support_phi[draw][ri], labels[draw], phis[ri],
                    strength=8., support_prior=support_prior[draw][pi], query_prior=priors[pi])
                predicted[draw, cell, lo:lo+len(rows)] = scores.argmax(1).numpy()
        if time.perf_counter()-last >= 25:
            print(f'{name}: {lo+len(rows)}/{len(data["rows"])} rows, CPU only', flush=True)
            last = time.perf_counter()
    return predicted


def effects(values):
    a, b, c, d = values
    result = dict(prior_given_R0_pp=b-a, relation_given_P0_pp=c-a,
        prior_given_R1_pp=d-c, relation_given_P1_pp=d-b,
        joint_pp=d-a, interaction_pp=d-b-c+a,
        prior_order_mean_pp=((b-a)+(d-c))/2,
        relation_order_mean_pp=((c-a)+(d-b))/2)
    np.testing.assert_allclose(result['joint_pp'], result['prior_given_R0_pp']+
        result['relation_given_P0_pp']+result['interaction_pp'], atol=1e-10)
    return result


def evaluate(info, data, predicted, dual, tables):
    dataset, name = info['dataset'], info['task']
    path, idcol = manifest(dataset)
    frame = pd.read_csv(path, usecols=[idcol, 'class_id', 'class_name'])
    np.testing.assert_array_equal(frame[idcol].to_numpy(str)[data['rows']], data['ids'])
    y = frame.class_id.to_numpy(np.int64)[data['rows']]
    np.testing.assert_array_equal(y[data['support']], data['support_labels'])
    class_names = frame.drop_duplicates('class_id').set_index('class_id').class_name.to_dict()
    names = (*CELLS, DUAL)
    for draw, mask in enumerate(data['eval_masks']):
        truth = y[mask]
        predictions = [predicted[draw, i, mask] for i in range(4)]+[dual[draw, mask]]
        context = dict(dataset=dataset, task=name, draw=draw)
        scores = [metrics(truth, pred) for pred in predictions]
        present = np.unique(truth)
        support_counts = np.bincount(data['support_labels'][draw], minlength=8)
        recalls = {}
        for i, cell in enumerate(names):
            tables['metrics'].append(dict(**context, cell=cell, n=len(truth), **scores[i]))
            for c in present:
                subset = truth == c
                recall = 100*float(np.mean(predictions[i][subset] == c))
                recalls[i, c] = recall
                tables['per_class_metrics'].append(dict(**context, cell=cell,
                    class_id=int(c), class_name=class_names[c], n=int(subset.sum()),
                    support_count=int(support_counts[c]), recall=recall))
        for metric in ('oa', 'ba'):
            tables['interactions'].append(dict(**context, metric=metric,
                **effects([s[metric] for s in scores[:4]])))
        for c in present:
            tables['per_class_interactions'].append(dict(**context, class_id=int(c),
                class_name=class_names[c], n=int((truth == c).sum()),
                support_count=int(support_counts[c]),
                **effects([recalls[i, c] for i in range(4)])))
        for comparison, old_i, new_i in COMPARISONS:
            old, new = predictions[old_i], predictions[new_i]
            rescue = (old != truth) & (new == truth)
            harm = (old == truth) & (new != truth)
            pair = dict(**context, comparison=comparison, reference=names[old_i], candidate=names[new_i])
            gains = []
            for c in present:
                subset = truth == c
                n = int(subset.sum())
                r, h = int(rescue[subset].sum()), int(harm[subset].sum())
                gain = 100*(r-h)/n
                gains.append(gain)
                tables['per_class_gains'].append(dict(**pair, class_id=int(c),
                    class_name=class_names[c], n=n, support_count=int(support_counts[c]),
                    reference_recall=recalls[old_i, c], candidate_recall=recalls[new_i, c],
                    rescues=r, harms=h, net_count=r-h, recall_delta_pp=gain,
                    ba_contribution_pp=gain/len(present)))
            oa, ba = 100*(int(rescue.sum())-int(harm.sum()))/len(truth), float(np.mean(gains))
            for metric, delta in (('oa', oa), ('ba', ba)):
                np.testing.assert_allclose(delta, scores[new_i][metric]-scores[old_i][metric], atol=1e-10)
            tables['rescue_harm'].append(dict(**pair, query_n=len(truth),
                rescues=int(rescue.sum()), harms=int(harm.sum()),
                net_count=int(rescue.sum())-int(harm.sum()), oa_delta_pp=oa, ba_delta_pp=ba))


def aggregate(tables):
    frames = {key: pd.DataFrame(rows) for key, rows in tables.items()}
    for key, table in frames.items():
        table.to_csv(DEST/f'{key}.csv', index=False)
    task_means = frames['metrics'].groupby(['dataset', 'task', 'cell'])[['oa', 'ba']].mean()
    task_means.to_csv(DEST/'task_means.csv')
    summary = task_means.groupby(['dataset', 'cell']).mean()
    summary.to_csv(DEST/'summary.csv')
    effect_columns = [key for key in frames['interactions'].columns if key.endswith('_pp')]
    task_effects = frames['interactions'].groupby(['dataset', 'task', 'metric'])[effect_columns].mean()
    task_effects.to_csv(DEST/'task_interactions.csv')
    effect_summary = task_effects.groupby(['dataset', 'metric']).mean()
    effect_summary.to_csv(DEST/'interaction_summary.csv')
    flow_columns = ['query_n', 'rescues', 'harms', 'net_count', 'oa_delta_pp', 'ba_delta_pp']
    flow_summary = frames['rescue_harm'].groupby(['dataset', 'task', 'comparison'])[flow_columns].mean()
    flow_summary = flow_summary.groupby(['dataset', 'comparison']).mean()
    flow_summary.to_csv(DEST/'rescue_harm_summary.csv')
    # Zero contribution for a class absent from a task is retained in the denominator.
    class_summary = frames['per_class_gains'].groupby(
        ['dataset', 'comparison', 'class_id', 'class_name'])[
        ['rescues', 'harms', 'net_count', 'ba_contribution_pp']].sum().reset_index()
    exposure_counts = frames['metrics'].groupby('dataset').size()/5
    for dataset, exposures in exposure_counts.items():
        selected = class_summary.dataset == dataset
        class_summary.loc[selected, 'ba_contribution_pp'] /= exposures
    class_summary.to_csv(DEST/'class_contributions.csv', index=False)
    # This detects a mistaken macro-port aggregation, not just an argmax replay error.
    original = pd.read_csv(OUT/'summary.csv').set_index(['dataset', 'mode'])
    for dataset in summary.index.get_level_values('dataset').unique():
        for cell, mode in ((CELLS[0], MODES[0]), (CELLS[3], MODES[1]), (DUAL, DUAL)):
            np.testing.assert_allclose(summary.loc[(dataset, cell)].to_numpy(),
                original.loc[(dataset, mode), ['oa', 'ba']].to_numpy(), atol=1e-10, rtol=0)
        for comparison in class_summary.comparison.unique():
            actual = class_summary[(class_summary.dataset == dataset) &
                                   (class_summary.comparison == comparison)].ba_contribution_pp.sum()
            np.testing.assert_allclose(actual, flow_summary.loc[(dataset, comparison), 'ba_delta_pp'],
                                       atol=1e-10, rtol=0)
    return summary, effect_summary, task_effects, flow_summary, class_summary


def write_report(summary, effect_summary, task_effects, flow_summary, classes, elapsed):
    datasets = ('self24ports', 'f0921', 'oss2')
    lines = ['# 极化输入控制：先验与支持关系根因诊断', '',
        '25/25任务完成（23个留出港口、F、OSS2），两个对角格共250个任务/抽样/模式组合的逐样本预测完全复现。',
        '复用两份新提取原始特征及各任务已保存的源先验；原环境CPU、2线程、1024行分块、float32原算子。无新增源拟合、神经训练、GPU前向或调参。仅重算既定64标签残差方程。', '',
        '## 固定2×2结果', '',
        '0=sum，1=sum_difference；P是完整先验函数（自己的特征与已保存的源头），R是支持/查询特征关系。单位为BA（%）。', '',
        '| 数据集 | P0/R0 | P1/R0 | P0/R1 | P1/R1 | 固定双路 |',
        '|---|---:|---:|---:|---:|---:|']
    for dataset in datasets:
        values = [summary.loc[(dataset, cell), 'ba'] for cell in (*CELLS, DUAL)]
        lines.append('| '+dataset+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', '先在每任务内平均原5组64支持抽样，再对23港口等权平均；F与OSS2各自为单任务。五组支持抽样不是五次神经训练。', '',
        '## 路径归因', '',
        '| 数据集 | 先验效应/R0 | 先验效应/R1 | 关系效应/P0 | 关系效应/P1 | 交互项 | 联合变化 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for dataset in datasets:
        e = effect_summary.loc[(dataset, 'ba')]
        cols = ('prior_given_R0_pp', 'prior_given_R1_pp', 'relation_given_P0_pp',
                'relation_given_P1_pp', 'interaction_pp', 'joint_pp')
        lines.append('| '+dataset+' | '+' | '.join(f'{e[c]:+.6f}' for c in cols)+' |')
    lines += ['', '交互项 = BA11 - BA10 - BA01 + BA00；联合变化 = 先验效应/R0 + 关系效应/P0 + 交互项。准确率非线性，不能只拿单个更换顺序当唯一贡献。', '']
    for dataset in datasets:
        e = effect_summary.loc[(dataset, 'ba')]
        prior = [e.prior_given_R0_pp, e.prior_given_R1_pp]
        relation = [e.relation_given_P0_pp, e.relation_given_P1_pp]
        def direction(values):
            if min(values) > 0:
                return '在两个关系/先验背景下均改善'
            if max(values) < 0:
                return '在两个关系/先验背景下均退化'
            return '随另一条路径背景改变，不能作单向解释'
        lines.append(f'- **{dataset}**：先验路径{direction(prior)}；支持关系路径{direction(relation)}。'
            f'对两种更换顺序等权平均，先验{e.prior_order_mean_pp:+.6f}pp、关系{e.relation_order_mean_pp:+.6f}pp，'
            f'合计{e.joint_pp:+.6f}pp；这是本次固定算子的功能分解，不是物理因果或额外实验。')
    port_effects = task_effects.xs(('self24ports', 'ba'), level=('dataset', 'metric'))
    signs = []
    for key, label in (('prior_order_mean_pp', '先验'), ('relation_order_mean_pp', '关系'), ('joint_pp', '联合')):
        values = port_effects[key].to_numpy()
        signs.append(f'{label}正/零/负={int((values > 1e-10).sum())}/{int((np.abs(values) <= 1e-10).sum())}/{int((values < -1e-10).sum())}')
    lines += ['', '23港口逐任务方向（前两项为两种顺序均值）：'+'；'.join(signs)+'。', '',
        '## 单路与双路', '']
    for dataset in datasets:
        before = summary.loc[(dataset, DUAL), 'ba']-summary.loc[(dataset, CELLS[0]), 'ba']
        after = summary.loc[(dataset, DUAL), 'ba']-summary.loc[(dataset, CELLS[3]), 'ba']
        flow = flow_summary.loc[(dataset, 'dual_vs_P1_R1')]
        lines.append(f'- {dataset}：双路减单路BA差距从{before:.6f}pp变为{after:.6f}pp；'
            f'双路相对新单路每任务/抽样平均救回{flow.rescues:.2f}、误改{flow.harms:.2f}，净{flow.net_count:+.2f}个查询。')
    lines += ['', '差分输入盲区被解除后仍有上述双路差距，因而不能把单/双差距全部归因于旧输入只见VV+VH。'
        '固定双路还包含不同预训练和其他模块；本诊断只定位本次输入控制改变了哪条计算路径，不能据此证明双骨干不可替代。'
        '此前全参数CE单R50能够打破通道权重对称，冻结前段的盲区也不能解释全部历史单路结果。', '',
        '## 逐类净纠错证据', '',
        '以下按对数据集宏平均BA的贡献列出主要正负类（pp）；缺席任务贡献为0。完整召回、支持覆盖、救回与误改均已保存。', '']
    for dataset in datasets:
        lines.append(f'**{dataset}**')
        for comparison, label in (('prior_given_R0', '只换先验'), ('relation_given_P0', '只换关系'),
                                  ('joint', '联合'), ('dual_vs_P1_R1', '双路相对新单路')):
            part = classes[(classes.dataset == dataset) & (classes.comparison == comparison)]
            positive = part[part.ba_contribution_pp > 0].nlargest(2, 'ba_contribution_pp')
            negative = part[part.ba_contribution_pp < 0].nsmallest(2, 'ba_contribution_pp')
            pieces = [f'{r.class_name} {r.ba_contribution_pp:+.6f}' for r in pd.concat([positive, negative]).itertuples()]
            lines.append(f'- {label}：'+('；'.join(pieces) if pieces else '无净变化')+'。')
        lines.append('')
    lines += ['## 算子与边界', '',
        '`score(P,R)=prior_P(q)+phi_R(q) @ correction(R, labels-prior_P(s))`。'
        '因此更换先验同时改变查询基础分数与支持残差；更换关系同时改变支持Gram与查询到支持的传播。'
        '两者都保留原类平衡权重、常数坐标、lambda=8及原5×64标签，不是把另一表示的头直接乘到当前特征。',
        'P1/R0与P0/R1需要两份表示，仅作诊断，不能称为可部署单路，也不按目标结果选择交叉格。'
        '没有新增物理分析，没有将编码/归一化输入差解释为标定后散射比。',
        '设计对齐：与原控制和原target.npz逐项核对source_rows、rows、ids、support、support_labels、eval_masks；'
        '固定双路文件只有rows、ids、support、eval_masks四个设计键，全部核对，其缺少的两个键不冒充已验证。'
        '缓存全量ID与manifest一致，支持标签另与真值核对。',
        '所有统计仅用eval_masks查询；跨抽样累计救回/误改是查询暴露次数，不是独立样本量。'
        '逐类净计数同时闭合OA与BA，并与原汇总核对，未更改原预测或报告。', '',
        '## 输出', '',
        '- `targets/<dataset>/<task>/predictions.npz`：四格预测及全部设计键；`REPLAY.json`：两个对角格逐抽样差异数与对齐记录。',
        '- `metrics.csv`、`task_means.csv`、`summary.csv`：四格与固定双路OA/BA。',
        '- `rescue_harm.csv`、`rescue_harm_summary.csv`：条件路径、联合及双路比较的双向纠错。',
        '- `per_class_metrics.csv`、`per_class_gains.csv`、`class_contributions.csv`：逐类召回、增量、支持覆盖与宏BA贡献。',
        '- `interactions.csv`、`task_interactions.csv`、`interaction_summary.csv`、`per_class_interactions.csv`：条件效应及非加性交互。',
        f'- 本次CPU诊断与汇总耗时约{elapsed:.1f}秒；这不是推理部署耗时测试。', '']
    (DEST/'FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    started = time.perf_counter()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f'Use the original environment: {PYTHON}')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    completed = json.loads((OUT/'COMPLETE.json').read_text())
    folders = task_folders()
    if completed['tasks'] != 25 or len(folders) != 25:
        raise ValueError('Require the completed 23-port + F + OSS2 control')
    if (DEST/'COMPLETE.json').exists():
        raise ValueError('This fixed diagnosis is already complete')
    source_ids = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id']).instance_id.to_numpy(str)
    np.testing.assert_array_equal(np.load(CACHE/'source_ids.npy'), source_ids)
    for mode in MODES:
        receipt = json.loads((OUT/f'source_{mode}_EXTRACTED.json').read_text())
        if receipt['rows'] != len(source_ids):
            raise ValueError('Fresh source extraction receipt does not match source IDs')
    DEST.mkdir(parents=True, exist_ok=True)
    write_json(DEST/'PROTOCOL.json', dict(python=sys.executable, torch=torch.__version__,
        numpy=np.__version__, device='cpu', threads=2, interop_threads=2, block_rows=BLOCK,
        dtype='float32', cells=dict(zip(CELLS, PATHS)), modes=list(MODES),
        source_prior='Reuse each saved source_prior.npy; alpha1 original source fits untouched',
        residual_lambda=8., support_draws=5, support_labels=64,
        source_refits=0, neural_training=0, gpu_forwards=0, hyperparameter_search=False,
        cross_cells_diagnostic_only=True, target_labels_used_for_selection=False,
        operator='Original support_scores; solve existing support residual equation, no new training',
        feature_memory='Read-only float32 memmaps, at most 1024 target rows per representation',
        reporting='Query masks only; mean draws within task, then equal mean tasks within dataset'))
    tables = {key: [] for key in ('metrics', 'per_class_metrics', 'interactions',
              'per_class_interactions', 'rescue_harm', 'per_class_gains')}
    receipts = []
    for index, folder in enumerate(folders):
        task_started = time.perf_counter()
        info, data, diagonal, dual, raw, weights, alignment = load_aligned(folder)
        name = f'{info["dataset"]}/{info["task"]}'
        print(f'START {index+1}/25 {name} rows={len(data["rows"])}', flush=True)
        predicted = predict(data, raw, weights, name)
        destination = DEST/'targets'/info['dataset']/info['task']
        destination.mkdir(parents=True, exist_ok=True)
        replay = {}
        for cell, mode_index in ((0, 0), (3, 1)):
            mismatch = predicted[:, cell] != diagonal[:, mode_index]
            replay[CELLS[cell]] = dict(exact=not bool(mismatch.any()),
                mismatches=int(mismatch.sum()), mismatches_per_draw=mismatch.sum(axis=1).tolist(),
                first_mismatches=np.argwhere(mismatch)[:10].tolist())
        receipt = dict(**info, rows=len(data['rows']), alignment=alignment, replay=replay,
                       seconds=time.perf_counter()-task_started)
        write_json(destination/'REPLAY.json', receipt)
        for cell, mode_index in ((0, 0), (3, 1)):
            np.testing.assert_array_equal(predicted[:, cell], diagonal[:, mode_index],
                err_msg=f'{name}/{CELLS[cell]}: exact original diagonal replay failed')
        np.savez_compressed(destination/'predictions.npz', predictions=predicted,
                            cells=np.array(CELLS), paths=np.array(PATHS), **data)
        evaluate(info, data, predicted, dual, tables)
        receipts.append(receipt)
        print(f'DONE {index+1}/25 {name}: both diagonals exact, {time.perf_counter()-task_started:.1f}s', flush=True)
        del data, diagonal, dual, raw, weights, predicted
        gc.collect()
    summary, effect_summary, task_effects, flow_summary, class_summary = aggregate(tables)
    elapsed = time.perf_counter()-started
    write_report(summary, effect_summary, task_effects, flow_summary, class_summary, elapsed)
    write_json(DEST/'COMPLETE.json', dict(tasks=25, support_draws=5, cells=4,
        diagonal_task_draw_mode_checks=250, all_diagonal_predictions_exact=True,
        diagonal_prediction_count=sum(r['rows']*10 for r in receipts),
        original_summary_exact=True, all_available_reference_design_keys_aligned=True,
        rescue_harm_closes_oa_ba=True, per_class_contributions_close_macro_ba=True,
        device='cpu', threads=2, block_rows=BLOCK, source_refits=0, neural_training=0,
        gpu_forwards=0, tuning=False, cross_cells_diagnostic_only=True,
        seconds=elapsed, tasks_completed=[dict(dataset=r['dataset'], task=r['task']) for r in receipts]))
    files = [str(Path(__file__).resolve())]+[str(p.resolve()) for p in sorted(DEST.rglob('*')) if p.is_file()]
    files.append(str((DEST/'CHANGED_FILES.json').resolve()))
    write_json(DEST/'CHANGED_FILES.json', dict(new_files=files, modified_existing_files=[]))
    print(summary.to_string(), flush=True)
    print(effect_summary.to_string(), flush=True)
    print(f'COMPLETE 25/25; elapsed={elapsed:.1f}s; new_files={len(files)}', flush=True)


if __name__ == '__main__':
    main()
