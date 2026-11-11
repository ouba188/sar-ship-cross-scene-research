"""Probability diagnostics from saved scores only; no model imports or execution."""

import json
from pathlib import Path, PureWindowsPath

import numpy as np
from scipy.special import logsumexp


ROOT = Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')
OUT = Path(__file__).resolve().parent
NAMES = ['bulk_carrier', 'fishing_vessel', 'general_cargo',
         'product_chemical_tanker', 'container_ship', 'crude_oil_tanker',
         'tug_towing', 'offshore_supply']
ZH = ['散货', '渔船', '普通货', '化学品/成品油轮', '集装箱', '原油轮', '拖轮/拖带', '海工补给']
THRESHOLDS = [0.8, 0.9, 0.95, 0.99]
BINS = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0000000001]


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def distribution(values):
    if not len(values):
        return {'n': 0, 'mean': None, 'quantiles': None}
    return {'n': len(values), 'mean': float(np.mean(values)),
            'quantiles': dict(zip(['p10', 'p50', 'p90', 'p95', 'p99'],
                                  np.quantile(values, [.1, .5, .9, .95, .99]).tolist()))}


def metrics(run, mask):
    loss, conf, correct = (run[k][mask] for k in ['nll', 'confidence', 'correct'])
    wrong = ~correct
    thresholds = {}
    for t in THRESHOLDS:
        count = int(np.sum(wrong & (conf >= t)))
        thresholds[str(t)] = {
            'count': count, 'fraction_of_errors': count / int(wrong.sum()) if wrong.any() else None,
            'fraction_of_all': count / len(loss),
        }
    return {'n': len(loss), 'nll': float(loss.mean()), 'nll_sum': float(loss.sum()),
            'accuracy': float(correct.mean()), 'errors': int(wrong.sum()),
            'correct_confidence': distribution(conf[correct]),
            'wrong_confidence': distribution(conf[wrong]),
            'correct_nll': distribution(loss[correct]), 'wrong_nll': distribution(loss[wrong]),
            'wrong_high_confidence': thresholds,
            'wrong_confidence_bin_counts': np.histogram(conf[wrong], bins=BINS)[0].tolist()}


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
                '| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]


def number(v, digits=4):
    return '-' if v is None else f'{v:.{digits}f}'


def main():
    manifest, groups, protocol = (read_json(n) for n in ['manifest.json', 'groups.json', 'protocol.json'])
    ids = np.asarray(groups['evaluation'], dtype=np.int64)
    records = [manifest[int(i)] for i in ids]
    y = np.asarray([r['label'] for r in records])
    n, c = len(ids), protocol['classes']
    seeds, arms = protocol['seeds'], protocol['arms']
    # These checks prevent wrong sample alignment or a wrong class mapping from entering comparisons.
    assert len(np.unique(ids)) == n
    assert all(r['split'] == 'evaluation' for r in records)
    assert all(PureWindowsPath(r['vv']['image']).parent.name == NAMES[r['label']] for r in records)
    assert c == len(NAMES)
    metadata = {key: np.asarray([r[key] for r in records]) for key in ['port', 'product', 'pass_group', 'mmsi']}
    orbit_groups = {}
    for group in np.unique(metadata['pass_group']):
        orbit_groups.setdefault(str(group).rsplit('_', 1)[0], []).append(str(group))
    scene_counting = {
        'pure_manifest_pass_group_count': len(np.unique(metadata['pass_group'])),
        'port_x_pass_group_count': len({(r['port'], r['pass_group']) for r in records}),
        'product_count': len(np.unique(metadata['product'])),
        'satellite_absolute_orbit_count': len(orbit_groups),
        'orbit_keys_with_multiple_pass_groups': {k: v for k, v in orbit_groups.items() if len(v) > 1},
        'strata_pass_group_definition': 'Exact manifest pass_group string; no port prefix and no suffix removal',
        'orbit_definition': 'pass_group with final underscore-delimited datatake field removed',
    }
    sources, runs = [], {}
    for arm in arms:
        for seed in seeds:
            scale = format(protocol['scales'][arm], 'g')
            path = ROOT / f'{arm}_s{seed}_lr{scale}' / 'evaluation.npz'
            with np.load(path, allow_pickle=False) as saved:
                saved_ids = saved['ids']
                assert np.array_equal(np.sort(saved_ids), np.sort(ids)), str(path)
                positions = {int(v): k for k, v in enumerate(saved_ids)}
                score = saved['score'][[positions[int(v)] for v in ids]].astype(np.float64)
            assert score.shape == (n, c) and np.isfinite(score).all(), str(path)
            logp = score - logsumexp(score, axis=1, keepdims=True)
            pred = score.argmax(axis=1)
            runs[arm, seed] = {'nll': -logp[np.arange(n), y],
                               'confidence': np.exp(logp.max(axis=1)),
                               'pred': pred, 'correct': pred == y}
            sources.append(str(path))

    pooled = {arm: {key: np.concatenate([runs[arm, s][key] for s in seeds])
                    for key in ['nll', 'confidence', 'pred', 'correct']} for arm in arms}
    yp = np.tile(y, len(seeds))
    all_one, all_pool = np.ones(n, dtype=bool), np.ones(n * len(seeds), dtype=bool)
    result = {
        'scope': 'Saved evaluation scores only; no training, inference, checkpoint loading, or recalibration.',
        'units': {
            'nll': 'natural-log nats per evaluation record per seed; logsumexp(score)-score_true; no clipping',
            'confidence': 'max softmax(saved score), temperature=1, not recalibrated',
            'pooled': 'three-seed record-prediction observations; same evaluation records repeated three times; no ensemble',
            'confusion': 'directed true-class to predicted-class counts; pooled counts are record-seed observations',
            'scene': 'product and exact manifest pass_group are separate strata; pass_group is not port x pass_group or satellite+orbit',
            'nll_contribution': 'class frequency times class mean NLL; delta contribution sums to overall NLL delta',
        },
        'sources': [str(ROOT / f) for f in ['manifest.json', 'groups.json', 'protocol.json']] + sources,
        'protocol': protocol,
        'scene_counting': scene_counting,
        'cohort': {'records': n, 'observations_per_arm': n * len(seeds),
                   **{key + '_count': len(np.unique(val)) for key, val in metadata.items()}},
        'labels': [{'label': k, 'name': NAMES[k], 'zh': ZH[k], 'records': int(np.sum(y == k))} for k in range(c)],
        'confidence_bins': ['[0,0.5)', '[0.5,0.8)', '[0.8,0.9)', '[0.9,0.95)', '[0.95,0.99)', '[0.99,1]'],
        'per_seed': {}, 'pooled': {}, 'comparisons': {},
    }
    for arm in arms:
        result['per_seed'][arm] = {str(s): {
            'overall': metrics(runs[arm, s], all_one),
            'classes': {NAMES[k]: metrics(runs[arm, s], y == k) for k in range(c)},
            'confusion': np.bincount(y * c + runs[arm, s]['pred'], minlength=c*c).reshape(c, c).tolist(),
        } for s in seeds}
        result['pooled'][arm] = {
            'overall': metrics(pooled[arm], all_pool),
            'classes': {NAMES[k]: metrics(pooled[arm], yp == k) for k in range(c)},
        }

    for base in ['plain', 'se']:
        full, other = pooled['full'], pooled[base]
        delta = full['nll'] - other['nll']
        class_rows = []
        for k in range(c):
            mask = yp == k
            sd = [float((runs['full', s]['nll'] - runs[base, s]['nll'])[y == k].mean()) for s in seeds]
            class_rows.append({'label': k, 'class': NAMES[k], 'zh': ZH[k],
                               'delta_nll': float(delta[mask].mean()), 'delta_nll_by_seed': sd,
                               'delta_weighted_contribution': float(delta[mask].sum() / len(delta)),
                               'full_nll_mass_fraction': float(full['nll'][mask].sum() / full['nll'].sum()),
                               'full_nll_contribution': float(full['nll'][mask].sum() / len(delta))})
        positive = sorted([r for r in class_rows if r['delta_weighted_contribution'] > 0],
                          key=lambda r: r['delta_weighted_contribution'], reverse=True)
        positive_total = sum(r['delta_weighted_contribution'] for r in positive)
        top = positive[:2]
        exclude = ~np.isin(yp, [r['label'] for r in top])
        mass_order = sorted(class_rows, key=lambda r: r['full_nll_mass_fraction'], reverse=True)
        transitions = {}
        for name, mask in {
            'both_correct': full['correct'] & other['correct'],
            'both_wrong': ~full['correct'] & ~other['correct'],
            'rescued': full['correct'] & ~other['correct'],
            'harmed': ~full['correct'] & other['correct'],
        }.items():
            transitions[name] = {'observations': int(mask.sum()),
                                 'delta_nll': float(delta[mask].mean()) if mask.any() else None,
                                 'delta_weighted_contribution': float(delta[mask].sum() / len(delta)),
                                 'full_confidence': distribution(full['confidence'][mask]),
                                 'baseline_confidence': distribution(other['confidence'][mask])}

        confusions = []
        for truth in range(c):
            for prediction in range(c):
                if truth == prediction:
                    continue
                entries = []
                for seed in seeds:
                    f = (y == truth) & (runs['full', seed]['pred'] == prediction)
                    b = (y == truth) & (runs[base, seed]['pred'] == prediction)
                    entries.append({'seed': seed, 'full': int(f.sum()), 'baseline': int(b.sum()),
                                    'delta': int(f.sum() - b.sum()),
                                    'entered': int((f & ~b).sum()), 'exited': int((b & ~f).sum()),
                                    'exited_to_correct': int((b & runs['full', seed]['correct']).sum()),
                                    'entered_from_correct': int((f & runs[base, seed]['correct']).sum())})
                ds = [e['delta'] for e in entries]
                confusions.append({'true': NAMES[truth], 'predicted': NAMES[prediction],
                                   'true_zh': ZH[truth], 'predicted_zh': ZH[prediction],
                                   'per_seed': entries, 'pooled_delta': sum(ds),
                                   'pooled_full': sum(e['full'] for e in entries),
                                   'pooled_baseline': sum(e['baseline'] for e in entries),
                                   'delta_true_class_percentage_points': 100 * sum(ds) / (len(seeds) * int((y == truth).sum())),
                                   'all_three_decrease': all(d < 0 for d in ds),
                                   'all_three_increase': all(d > 0 for d in ds)})

        strata = {}
        for key in ['port', 'product', 'pass_group']:
            rows = []
            for value in np.unique(metadata[key]):
                mask = metadata[key] == value
                mp = np.tile(mask, len(seeds))
                rows.append({'name': str(value), 'records': int(mask.sum()),
                             'ports': sorted(set(metadata['port'][mask])),
                             'class_counts': np.bincount(y[mask], minlength=c).tolist(),
                             'full_nll': float(full['nll'][mp].mean()),
                             'baseline_nll': float(other['nll'][mp].mean()),
                             'delta_nll': float(delta[mp].mean()),
                             'delta_weighted_contribution': float(delta[mp].sum() / len(delta)),
                             'delta_nll_by_seed': [float((runs['full', s]['nll'] - runs[base, s]['nll'])[mask].mean()) for s in seeds],
                             'full_errors_by_seed': [int((~runs['full', s]['correct'][mask]).sum()) for s in seeds],
                             'baseline_errors_by_seed': [int((~runs[base, s]['correct'][mask]).sum()) for s in seeds]})
            strata[key] = rows

        cases = []
        for seed in seeds:
            f, b = runs['full', seed], runs[base, seed]
            d = f['nll'] - b['nll']
            for direction, indices in [('increase', np.argsort(d)[-8:][::-1]), ('decrease', np.argsort(d)[:8])]:
                for i in indices:
                    r = records[int(i)]
                    cases.append({'seed': seed, 'direction': direction, 'manifest_index': int(ids[i]),
                                  'id': r['id'], 'mmsi': r['mmsi'], 'port': r['port'], 'product': r['product'],
                                  'true': NAMES[y[i]], 'full_pred': NAMES[f['pred'][i]], 'baseline_pred': NAMES[b['pred'][i]],
                                  'full_nll': float(f['nll'][i]), 'baseline_nll': float(b['nll'][i]),
                                  'delta_nll': float(d[i]), 'full_confidence': float(f['confidence'][i]),
                                  'baseline_confidence': float(b['confidence'][i]),
                                  'vv': r['vv']['image'], 'vh': r['vh']['image'] if r['vh'] else None})

        comp = {'overall_delta_nll': float(delta.mean()),
                'overall_delta_nll_by_seed': [float((runs['full', s]['nll'] - runs[base, s]['nll']).mean()) for s in seeds],
                'classes': class_rows, 'classes_increased': sum(r['delta_nll'] > 0 for r in class_rows),
                'classes_increased_all_seeds': sum(all(d > 0 for d in r['delta_nll_by_seed']) for r in class_rows),
                'positive_delta_contribution_sum': positive_total,
                'negative_delta_contribution_sum': sum(min(0, r['delta_weighted_contribution']) for r in class_rows),
                'top_two_positive_classes': [r['class'] for r in top],
                'top_two_fraction_of_positive_contributions': sum(r['delta_weighted_contribution'] for r in top) / positive_total if positive_total else None,
                'delta_nll_excluding_top_two_positive_classes': float(delta[exclude].mean()) if exclude.any() else None,
                'full_top_two_nll_mass_classes': [r['class'] for r in mass_order[:2]],
                'full_top_two_nll_mass_fraction': sum(r['full_nll_mass_fraction'] for r in mass_order[:2]),
                'paired_outcomes': transitions, 'confusions': confusions, 'strata': strata,
                'extreme_delta_cases_for_chip_followup': cases}
        # Additive decomposition detects aggregation or masking mistakes.
        assert np.isclose(sum(r['delta_weighted_contribution'] for r in class_rows), delta.mean())
        assert np.isclose(sum(r['delta_weighted_contribution'] for r in transitions.values()), delta.mean())
        assert sum(r['pooled_delta'] for r in confusions) == int((~full['correct']).sum() - (~other['correct']).sum())
        result['comparisons'][base] = comp

    result['findings_zh'] = [
        'Full 的 OA 为48.41%，高于 Plain 42.93%和SE 42.31%；NLL为2.0670，高于1.8320和1.6081，且三个种子均同向。',
        'NLL差异具有跨类别普遍性，并非一两个少数类独占：对两基线均有7/8类平均NLL升高；对Plain有4类、对SE有6类三种子均升高。散货三种子均改善。',
        '拖轮/拖带与海工补给的Full类均NLL最高（4.3100/4.4144），但对Plain加权恶化贡献最大的是集装箱、化学品/成品油轮，对SE则是集装箱、海工补给。前两类仅占所有正向贡献49.42%/45.09%；去掉后其余记录ΔNLL仍为+0.1447/+0.2638。',
        'Full错误数较少，但错误置信度均值0.6975高于Plain 0.6004和SE 0.5368；置信度≥0.9错误有1020/503/233次记录×种子观测，占各自错误21.43%/9.55%/4.38%。',
        '同记录同种子配对的持续判错集合，在NLL差的数值分解中贡献+0.3356（对Plain）/+0.5612（对SE）；救回集合贡献-0.3290/-0.2783，误伤集合贡献+0.2722/+0.2493，持续判对集合贡献-0.0438/-0.0733。这是加和分解，不是根因证明。',
        '对Plain三种子一致减少的主要混淆为散货→集装箱（-32/-46/-27，合计-105）、化学品/成品油轮→海工补给（-34/-52/-17，合计-103）；一致增加包括集装箱→普通货（+14/+16/+4，合计+34）、普通货→散货（+8/+9/+14，合计+31）。',
        '对SE三种子一致减少的主要混淆为散货→原油轮（-30/-64/-37，合计-131）、散货→集装箱（-37/-49/-37，合计-123）；一致增加包括集装箱→散货（+21/+18/+32，合计+71）、拖轮/拖带→散货（+17/+7/+18，合计+42）。混淆计数单位均为记录×种子。',
    ]
    lines = ['# 24港缓存概率质量与主要混淆诊断', '',
             '仅读取 manifest.json、groups.json、protocol.json 与九份 evaluation.npz。未重训、未推理、未加载 checkpoint。', '',
             '## 主要发现', '', *['- ' + finding for finding in result['findings_zh']], '',
             '## 计算口径', '',
             f'- 评价单位：{n} 条 manifest 记录，{len(seeds)} 个种子 {seeds}；每个模型合计 {n * len(seeds)} 次“记录×种子”预测观测。同一记录跨种子重复，不能视作独立新增样本，也未做概率集成。',
             f'- 实际覆盖 {len(np.unique(metadata["port"]))} 港、{len(np.unique(metadata["product"]))} 个 product 场景、{len(np.unique(metadata["pass_group"]))} 个 pass_group、{len(np.unique(metadata["mmsi"]))} 个 MMSI。“24港”是数据集名称，不是本评价子集的港口数。',
             f'- 场景口径明确区分：直接按 manifest 完整 pass_group 字符串去重为 {scene_counting["pure_manifest_pass_group_count"]}；按 (port, pass_group) 二元组去重为 {scene_counting["port_x_pass_group_count"]}；按 product 去重为 {scene_counting["product_count"]}。本次三个计数恰好相同，但定义不同；JSON strata.pass_group 始终使用纯完整 pass_group，不拼接港口。',
             f'- 将 pass_group 最后的 datatake 字段去掉后，“卫星＋绝对轨道号”去重为 {scene_counting["satellite_absolute_orbit_count"]}。这能得到此前提及的81这一数值，但未核实此前统计脚本，不能据此断言其原始定义。本报告不将81个轨道键称为85个完整pass_group。',
             '- 四个轨道键各对应两个完整pass_group：' + '；'.join(k + ': ' + ', '.join(v) for k, v in scene_counting['orbit_keys_with_multiple_pass_groups'].items()) + '。',
             '- NLL = logsumexp(score) - score[真类]，自然对数，单位 nats/记录/种子；使用 float64 稳定计算，没有截断概率。总体 NLL 按记录加权，类别宏平均则八类等权。',
             '- 置信度 = max softmax(缓存 score)，温度为 1，未重新校准；高置信错误按 ≥0.8/0.9/0.95/0.99 统计。NLL 与分类正确率是不同统计量，本文不据此作训练机制或物理机制的因果推断。',
             '- 三种子均值等于合并“记录×种子”观测均值，因为各组样本相同。置信度分位数、错误占比则在合并观测上直接计算，不是种子比率的算术平均。',
             '- Full/Plain 使用协议 lr scale=1，SE=0.3；这些是保存协议下的比较。',
             '- 标签由 manifest 图片父目录核实：1=渔船，5=原油轮。', '', '## 总体与各类 NLL', '']
    rows = []
    for arm in arms:
        overall = result['pooled'][arm]['overall']
        macro = np.mean([result['pooled'][arm]['classes'][name]['nll'] for name in NAMES])
        rows.append([arm, *[number(result['per_seed'][arm][str(s)]['overall']['nll']) for s in seeds],
                     number(overall['nll']), number(macro), f'{100*overall["accuracy"]:.2f}%'])
    lines += table(['模型', *[f'NLL seed{s}' for s in seeds], '总体NLL', '类别宏NLL', 'OA'], rows)
    lines += ['', '各类样本数为单个种子的独有评价记录数；贡献为该类 NLL 总和除以所有评价观测数。Δ 均为 Full−基线，正数表示 NLL 较高。', '']
    rows = []
    for k in range(c):
        rows.append([f'{k} {ZH[k]}', int((y == k).sum()),
                     *[number(result['pooled'][a]['classes'][NAMES[k]]['nll']) for a in ['plain', 'se', 'full']],
                     *[number(result['comparisons'][b]['classes'][k]['delta_weighted_contribution']) for b in ['plain', 'se']],
                     f'{100*result["comparisons"]["plain"]["classes"][k]["full_nll_mass_fraction"]:.2f}%'])
    lines += table(['类', 'n', 'Plain NLL', 'SE NLL', 'Full NLL', '对Plain贡献Δ', '对SE贡献Δ', 'Full NLL总量占比'], rows)
    for base, comp in result['comparisons'].items():
        top_names = '、'.join(ZH[NAMES.index(k)] for k in comp['top_two_positive_classes'])
        lines += ['', f'### Full vs {base}', '',
                  f'总体 ΔNLL={comp["overall_delta_nll"]:+.4f}；逐种子为 '+', '.join(f'{x:+.4f}' for x in comp['overall_delta_nll_by_seed'])+'。',
                  f'{comp["classes_increased"]}/8 类的平均 NLL 增加，其中 {comp["classes_increased_all_seeds"]}/8 类在三个种子均增加。正向加权贡献合计 {comp["positive_delta_contribution_sum"]:.4f}，负向贡献合计 {comp["negative_delta_contribution_sum"]:.4f}。',
                  f'正向贡献最大的两类（{top_names}）占所有正向贡献的 {100*comp["top_two_fraction_of_positive_contributions"]:.2f}%；去掉这两类后，其余记录的 ΔNLL={comp["delta_nll_excluding_top_two_positive_classes"]:+.4f}。该占比分母是所有正向贡献，不是正负抵消后的净差。',
                  '']
        lines += table(['类别', *[f'ΔNLL seed{s}' for s in seeds]],
                       [[r['zh'], *[f'{x:+.4f}' for x in r['delta_nll_by_seed']]] for r in comp['classes']])
    mass = result['comparisons']['plain']
    lines += ['', 'Full 自身 NLL 总量最大的两类为 '+ '、'.join(ZH[NAMES.index(k)] for k in mass['full_top_two_nll_mass_classes'])+
              f'，共占 {100*mass["full_top_two_nll_mass_fraction"]:.2f}%。这与“相对基线的恶化贡献”是不同分解。', '', '## 正确与错误置信度', '']
    lines += table(['模型', '正确观测数', '正确均值', '错误观测数', '错误均值', '错误p50', '错误p90', '错误p99'], [
        [a, (m := result['pooled'][a]['overall'])['correct_confidence']['n'], number(m['correct_confidence']['mean']),
         m['errors'], number(m['wrong_confidence']['mean']),
         *[number(m['wrong_confidence']['quantiles'][q]) for q in ['p50', 'p90', 'p99']]] for a in arms])
    lines += ['', '下表高置信错误计数单位均为“记录×种子”；括号中为占该模型所有错误的比例。', '']
    lines += table(['模型', *[f'错误置信度≥{t}' for t in THRESHOLDS]], [
        [a, *[f'{(v := result["pooled"][a]["overall"]["wrong_high_confidence"][str(t)])["count"]} ({100*v["fraction_of_errors"]:.2f}%)' for t in THRESHOLDS]] for a in arms])
    lines += ['', '错误置信度完整分箱（单位同上）：', '']
    lines += table(['模型', *result['confidence_bins']], [[a, *result['pooled'][a]['overall']['wrong_confidence_bin_counts']] for a in arms])
    lines += ['', '逐类置信度与高置信错误：正确/错误均值各自在该模型对应集合内计算，集合会随模型变化；配对集合分析见下一节。', '']
    lines += table(['类别', '模型', '正确置信度均值', '错误置信度均值', '错误数', '≥0.9错误数', '占本类错误%'], [
        [ZH[k], a, number((m := result['pooled'][a]['classes'][NAMES[k]])['correct_confidence']['mean']),
         number(m['wrong_confidence']['mean']), m['errors'], m['wrong_high_confidence']['0.9']['count'],
         number(100*m['wrong_high_confidence']['0.9']['fraction_of_errors']) if m['errors'] else '-'] for k in range(c) for a in arms])
    lines += ['', '## 配对正确性集合', '', '同一记录、同一种子配对。持续判错也包括两模型预测为不同错误类；救回/误伤只表示正确性变化。', '']
    outcome_zh = {'both_correct': '持续判对', 'both_wrong': '持续判错', 'rescued': 'Full救回', 'harmed': 'Full误伤'}
    lines += table(['基线', '集合', '观测数', '集合ΔNLL', '总体加权贡献Δ', 'Full置信度', '基线置信度'], [
        [b, outcome_zh[k], v['observations'], number(v['delta_nll']), number(v['delta_weighted_contribution']),
         number(v['full_confidence']['mean']), number(v['baseline_confidence']['mean'])]
        for b, comp in result['comparisons'].items() for k, v in comp['paired_outcomes'].items()])
    lines += ['', '## 三种子主要有向混淆变化', '',
              '按三种子合计净变化绝对量排序，分别列出减少/增加最多的十项。方向为“真实类→预测类”；Δ=Full−基线。下降可能流向正确类，也可能流向另一错误类，不能将全部混淆下降称为救回。JSON 保留全部56个非对角单元及进入/退出/转对计数。', '']
    for base, comp in result['comparisons'].items():
        for title, sign in [('减少', -1), ('增加', 1)]:
            selected = sorted([r for r in comp['confusions'] if r['pooled_delta'] * sign > 0],
                              key=lambda r: abs(r['pooled_delta']), reverse=True)[:10]
            lines += [f'### 对 {base}：{title}', '']
            lines += table(['真类→预测类', '基线合计', 'Full合计', *[f'Δ{s}' for s in seeds], '合计Δ', '真类内Δ百分点', '三种子同向'], [
                [r['true_zh']+'→'+r['predicted_zh'], r['pooled_baseline'], r['pooled_full'],
                 *[e['delta'] for e in r['per_seed']], r['pooled_delta'], number(r['delta_true_class_percentage_points'], 2),
                 '是' if r['all_three_decrease'] or r['all_three_increase'] else '否'] for r in selected])
            lines += ['']
    lines += ['## 港口与场景分层', '',
              '港口表覆盖全部18个评价港；JSON分别保留全部85个product和85个纯完整pass_group的逐种子NLL变化、类别组成和错误数。此处pass_group不含港口拼接，也未合并为81个卫星＋轨道键。下方场景榜单按product分组。场景间类别组成不同，这些是描述性分层，不能作为已控制类别后的场景效应。', '']
    lines += table(['港口', '记录数', 'Full NLL', '对Plain ΔNLL', '对SE ΔNLL'], [
        [r['name'], r['records'], number(r['full_nll']), number(r['delta_nll']), number(s['delta_nll'])]
        for r, s in zip(result['comparisons']['plain']['strata']['port'], result['comparisons']['se']['strata']['port'])])
    for base, comp in result['comparisons'].items():
        for key in ['port', 'product']:
            rows = comp['strata'][key]
            lines += ['', f'对 {base}，{sum(r["delta_nll"] > 0 for r in rows)}/{len(rows)} 个 {key} 的种子平均 NLL 增加；'
                      f'{sum(all(v > 0 for v in r["delta_nll_by_seed"]) for r in rows)}/{len(rows)} 个在三个种子均增加。']
        scenes = sorted(comp['strata']['product'], key=lambda r: r['delta_weighted_contribution'])
        lines += ['', f'对 {base}，场景加权贡献下降/上升最大的各5项（完整product ID）：', '']
        lines += table(['场景', '港口', 'n', 'ΔNLL', '总体贡献Δ'], [
            [r['name'], ', '.join(r['ports']), r['records'], number(r['delta_nll']), number(r['delta_weighted_contribution'])]
            for r in scenes[:5] + scenes[-5:][::-1]])
    lines += ['', '## 可追溯性与限制', '',
              '- JSON 的 extreme_delta_cases_for_chip_followup 为每个基线、每个种子保存 NLL 增加/减少最大的各8条记录，含 manifest index、ID、MMSI、场景、预测、概率摘要和 VV/VH 路径，供主分析连接原 chip。manifest 中缺失的 VH 路径保留为 null。此概率分支未查看原图，不声称获得图像或物理机制证据。',
              '- 本报告描述已保存结果中的概率质量、错误置信度和混淆变化，不将高置信错误直接归因于某个模块、训练方式或数据机制。未经重校准的 softmax 不能直接解释为已校准的正确概率。',
              '- 验证：九组评价 ID 集合与 groups 一致；按 ID 对齐后计算；分数形状/有限性及 manifest 类别映射通过；逐类/正确性集合加权 NLL 差之和与总体差一致；混淆净变化之和与错误数净变化一致。', '']
    OUT.joinpath('probability_diagnostic.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    OUT.joinpath('PROBABILITY_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'cohort': result['cohort'],
                      'nll': {a: result['pooled'][a]['overall']['nll'] for a in arms},
                      'output': [str(OUT / name) for name in ['probability_diagnostic.json', 'PROBABILITY_ZH.md']]},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
