"""Collect archived summaries only; no model/feature loading or experiments."""

import csv
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = Path(__file__).resolve().parent
PT = ROOT.parent
READS = {}
CHECKS = []


def load(relative):
    path = ROOT / relative
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    READS[path.as_posix()] = path.stat().st_size
    return value


def read_csv(relative):
    path = ROOT / relative
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    READS[path.as_posix()] = path.stat().st_size
    return rows


def absolute(relative):
    return (ROOT / relative).as_posix()


def check_close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f'{label}: {actual} != {expected}')


def verify_means(relative, summaries, arm_key='arm', dataset=None, equal_port=False):
    rows = read_csv(relative)
    if dataset:
        rows = [r for r in rows if r['dataset'] == dataset]
    for summary in summaries:
        matched = [r for r in rows if r[arm_key] == summary['arm']]
        expected_count = 115 if equal_port else 5
        if len(matched) != expected_count:
            raise ValueError(f'{relative}: wrong draw count for {summary["arm"]}')
        for metric in ('acc', 'ba'):
            if equal_port:
                ports = sorted({r['port'] for r in matched})
                actual = mean(mean(float(r[metric]) for r in matched if r['port'] == p) for p in ports)
            else:
                actual = mean(float(r[metric]) for r in matched)
            check_close(actual, summary[metric], f'{relative}/{summary["arm"]}/{metric}')
    CHECKS.append({'check': 'summary_matches_archived_draw_csv', 'path': absolute(relative),
                   'arms': [s['arm'] for s in summaries], 'dataset': dataset})


def base_record(dataset, target, n, q, classes, budget, aggregation, group='main'):
    return dict(dataset=dataset, target=target, group=group, status='completed',
                target_rows=n, evaluated_rows_per_draw=q, target_classes=classes,
                output_classes=classes, support_labels_per_task_per_draw=64,
                budget_unit=budget, support_draws=5, support_seeds=[0, 1, 2, 3, 4],
                aggregation=aggregation, handoff_cap_fraction=0.05)


def main_record(meta, summaries, relative, aliases=None, selector='means'):
    aliases = aliases or ['original_base', 'base_visual64', 'visual_handoff5']
    result = dict(meta, artifact_path=absolute(relative), artifact_selector=selector)
    for label, arm in zip(('source', 'base64', 'full5'), aliases):
        item = next(x for x in summaries if x['arm'] == arm)
        result[label + '_arm'] = arm
        for metric, source_key in [('oa', 'acc'), ('ba', 'ba')]:
            result[f'{label}_{metric}_ratio'] = item[source_key]
            result[f'{label}_{metric}_pct'] = 100 * item[source_key]
    for metric in ('oa', 'ba'):
        result[f'm1_base64_minus_source_{metric}_pp'] = 100 * (
            result[f'base64_{metric}_ratio'] - result[f'source_{metric}_ratio'])
        result[f'm4_full5_minus_base64_{metric}_pp'] = 100 * (
            result[f'full5_{metric}_ratio'] - result[f'base64_{metric}_ratio'])
    result['contrast_scope'] = 'M1/M4 stage contrasts, not factorial ablations'
    return result


main = []
ports_path = 'oof_base_adapt_results_20261007'
port_summary = load(ports_path + '/summary.json')
port_inventory = load(ports_path + '/POLICIES_SAVED.json')
target_rows = sum(x['queries'] for x in port_inventory)
port_meta = base_record('self24ports', '23eligible_equal_port', target_rows,
                       target_rows - 23 * 64, 8, '64_per_target_port',
                       'mean_over_5_draws_within_port_then_equal_mean_over_23_ports')
port_meta.update(corpus_rows=38091, corpus_ports=24, eligible_ports=23,
                 total_support_labels_across_tasks_per_draw=23 * 64,
                 excluded_target_rows=38091 - target_rows,
                 inventory_artifact=absolute(ports_path + '/POLICIES_SAVED.json'),
                 notes='OA is equal-port, not pooled; target_rows excludes the ineligible small port.')
main.append(main_record(port_meta, port_summary['means'], ports_path + '/summary.json',
                        ['original_base', 'base_visual', 'visual_handoff']))
verify_means(ports_path + '/metrics.csv', port_summary['means'], equal_port=True)
port_metrics = read_csv(ports_path + '/metrics.csv')
port_details = []
for item in port_inventory:
    n = item['queries']
    relevant = [r for r in port_metrics if r['port'] == item['port']]
    if {int(r['evaluated_rows']) for r in relevant} != {n - 64}:
        raise ValueError('Port support/query count mismatch')
    means = [dict(arm=a, acc=mean(float(r['acc']) for r in relevant if r['arm'] == a),
                  ba=mean(float(r['ba']) for r in relevant if r['arm'] == a))
             for a in ['original_base', 'base_visual', 'visual_handoff']]
    meta = base_record('self24ports', item['port'], n, n - 64, 8,
                       '64_per_target_port', 'mean_over_5_draws', 'port_detail')
    meta['target_classes'] = None
    meta['output_classes'] = 8
    meta['notes'] = 'Per-port observed class count is not recorded in this metrics CSV; global taxonomy has 8 classes.'
    port_details.append(main_record(meta, means, ports_path + '/metrics.csv',
                                    ['original_base', 'base_visual', 'visual_handoff'],
                                    f'port={item["port"]}; mean over seeds 0..4'))

for dataset, folder, classes, group in [
    ('correctedF0921', 'f0921_detector_recrop_20261007', 8, 'main'),
    ('publicOSS1', 'opensarship1_transfer_20261007', 8, 'main'),
    ('publicOSS2', 'opensarship2_transfer_20261007', 8, 'main'),
    ('FUSAR', 'fusar_transfer_20261007', 7, 'optional_7class_extra'),
]:
    summary = load(folder + '/summary.json')
    meta = base_record(dataset, 'compatible_subset', summary['target_rows'],
                       summary['evaluated_rows_per_draw'], classes,
                       '64_global_per_dataset', 'mean_over_5_draws', group)
    meta['output_classes'] = 8
    meta['total_support_labels_across_tasks_per_draw'] = 64
    meta['notes'] = ('Corrected detector recrops only; production baseline, not matched joint reference.'
                     if dataset == 'correctedF0921' else
                     'Fine-label-compatible subset; not performance on the complete standard taxonomy.')
    if dataset == 'FUSAR':
        meta['notes'] += ' Seven present classes, eight output columns; duplicated single image is not two polarizations.'
    main.append(main_record(meta, summary['means'], folder + '/summary.json'))
    verify_means(folder + '/metrics.csv', summary['means'])

for dataset, folder in [('PACS', 'pacs_handoff_transfer_20261007'),
                         ('OfficeHome', 'officehome_handoff_transfer_20261007')]:
    aggregate = load(folder + '/summary.json')
    for dom in aggregate['domains']:
        directory = dom['target_domain'].replace(' ', '_')
        path = f'{folder}/{directory}/summary.json'
        summary = load(path)
        meta = base_record(dataset, summary['target_domain'], summary['target_rows'],
                           summary['target_rows'] - 64, summary['n_classes'],
                           '64_global_per_target_domain', 'mean_over_5_draws')
        meta['observed_support_classes_by_draw'] = summary['observed_support_classes']
        meta['notes'] = 'Same RGB image, frozen ImageNet R50 plus outer-source-supervised R18; not SAR polarization or joint-module validation.'
        main.append(main_record(meta, summary['means'], path))
        verify_means(f'{folder}/{directory}/metrics.csv', summary['means'])
    for arm in aggregate['domain_macro_means']:
        for metric in ('acc', 'ba'):
            expected = mean(next(x[metric] for x in d['means'] if x['arm'] == arm['arm'])
                            for d in aggregate['domains'])
            check_close(arm[metric], expected, f'{dataset}/equal_domain/{metric}')
    n = sum(d['target_rows'] for d in aggregate['domains'])
    meta = base_record(dataset, 'equal_domain_mean', n, n - 4 * 64,
                       aggregate['domains'][0]['n_classes'], '64_per_domain_4_tasks',
                       'mean_over_5_draws_within_domain_then_equal_mean_over_4_domains')
    meta.update(total_support_labels_across_tasks_per_draw=256, target_domains=4,
                 notes='Counts are sums over four outer tasks, metrics are equal-domain means, not pooled accuracy. Dataset-specific RGB encoders; no new M2/M3/joint validation.')
    main.append(main_record(meta, aggregate['domain_macro_means'], folder + '/summary.json',
                            selector='domain_macro_means'))

by_dataset = {r['dataset']: r for r in main if r['dataset'] not in ('PACS', 'OfficeHome')}
modules = []


def candidate(meta, module, round_id, arm, summary, path, selector, reference=None,
              reference_path=None, reference_arm=None):
    keys = ['dataset', 'target', 'target_rows', 'evaluated_rows_per_draw', 'target_classes',
            'output_classes', 'support_labels_per_task_per_draw', 'budget_unit',
            'support_draws', 'aggregation', 'handoff_cap_fraction']
    row = {k: meta[k] for k in keys}
    row.update(module=module, round_id=round_id, arm=arm, status='completed',
               full_oa_ratio=summary['acc'], full_ba_ratio=summary['ba'],
               full_oa_pct=100 * summary['acc'], full_ba_pct=100 * summary['ba'],
               artifact_path=absolute(path), artifact_selector=selector,
               reference_artifact_path=absolute(reference_path) if reference_path else None,
               reference_arm=reference_arm,
               reference_oa_pct=100 * reference['acc'] if reference else None,
               reference_ba_pct=100 * reference['ba'] if reference else None,
               delta_reference_oa_pp=100 * (summary['acc'] - reference['acc']) if reference else None,
               delta_reference_ba_pp=100 * (summary['ba'] - reference['ba']) if reference else None,
               expert_oa_pct=100 * summary['expert_acc'] if 'expert_acc' in summary else None,
               expert_ba_pct=100 * summary['expert_ba'] if 'expert_ba' in summary else None,
               expert_scope='all-query diagnostic, not 5% deployment performance',
               base64_oa_pct=meta['base64_oa_pct'], base64_ba_pct=meta['base64_ba_pct'])
    return row


for code, dataset in [('f0921', 'correctedF0921'), ('oss1', 'publicOSS1'), ('oss2', 'publicOSS2')]:
    path = f'projection_residual_pilot_20261008/{code}/summary.json'
    summaries = load(path)
    ref = next(s for s in summaries if s['arm'] == 'pca_reference')
    selected = next(s for s in summaries if s['arm'] == 'projection_residual')
    modules.append(candidate(by_dataset[dataset], 'M2', 'standalone_projection_pilot',
                              'projection_residual', selected, path,
                              '[arm=projection_residual]', ref, path, 'pca_reference'))
    verify_means('projection_residual_pilot_20261008/metrics.csv', summaries, dataset=code)

aligned_path = 'aligned_residual_pilot_20261008/residual_global/summary.json'
aligned = load(aligned_path)
frozen_path = 'neural_complement_pilot_20261007/frozen_replay/summary.json'
frozen = load(frozen_path)
old_aligned = next(s for s in aligned['means'] if s['arm'] == 'visual_handoff5')
frozen_ref = next(s for s in frozen['means'] if s['arm'] == 'visual_handoff5')
modules.append(candidate(by_dataset['correctedF0921'], 'M3', 'standalone_aligned_pilot',
                          'residual_global', old_aligned, aligned_path,
                          'means[arm=visual_handoff5]', frozen_ref, frozen_path, 'visual_handoff5'))
verify_means('aligned_residual_pilot_20261008/residual_global/metrics.csv', aligned['means'])

joint_path = 'joint_residual_pilot_20261008/summary.json'
joint_summary = load(joint_path)
joint_reference = next(s for s in joint_summary['means'] if s['arm'] == 'reference')
joint = []
for s in joint_summary['means']:
    row = candidate(by_dataset['correctedF0921'],
                    {'reference': 'reference', 'complement_only': 'M2',
                     'aligned_only': 'M3', 'joint': 'M2+M3'}[s['arm']],
                    'matched_joint_2x2_20261008', s['arm'], s, joint_path,
                    f'means[arm={s["arm"]}]', joint_reference, joint_path, 'reference')
    row.update(complement_enabled=s['arm'] in ('complement_only', 'joint'),
               aligned_enabled=s['arm'] in ('aligned_only', 'joint'),
               selected_rows_per_draw=int(s['selected']),
               notes='M1 and M4 fixed in all four arms; this is the M2 x M3 factorial study.')
    joint.append(row)
verify_means('joint_residual_pilot_20261008/metrics.csv', joint_summary['means'])

# Each requested main dataset/domain gets explicit experiment availability.
coverage = []
for r in main:
    for module in ('M1', 'M2', 'M3', 'M4', 'joint_M2_M3'):
        row = {k: r[k] for k in ('dataset', 'target', 'group', 'target_rows',
                                'evaluated_rows_per_draw', 'target_classes', 'budget_unit', 'aggregation')}
        row.update(module=module, status='not_run', oa_pct=None, ba_pct=None,
                   delta_oa_pp=None, delta_ba_pp=None, artifact_path=None,
                   comparison=None, round_id=None,
                   missing_reason='No completed experiment in the current module reports/artifacts.')
        if module in ('M1', 'M4'):
            label = 'base64' if module == 'M1' else 'full5'
            prefix = 'm1_base64_minus_source' if module == 'M1' else 'm4_full5_minus_base64'
            row.update(status='stage_contrast_only', oa_pct=r[label + '_oa_pct'],
                       ba_pct=r[label + '_ba_pct'], delta_oa_pp=r[prefix + '_oa_pp'],
                       delta_ba_pp=r[prefix + '_ba_pp'], artifact_path=r['artifact_path'],
                       comparison='base64-source' if module == 'M1' else 'full5-base64',
                       round_id='original_main', missing_reason=None)
        elif r['dataset'] == 'correctedF0921':
            arm = {'M2': 'complement_only', 'M3': 'aligned_only', 'joint_M2_M3': 'joint'}[module]
            chosen = next(s for s in joint if s['arm'] == arm)
            row.update(status='completed', oa_pct=chosen['full_oa_pct'], ba_pct=chosen['full_ba_pct'],
                       delta_oa_pp=chosen['delta_reference_oa_pp'], delta_ba_pp=chosen['delta_reference_ba_pp'],
                       artifact_path=chosen['artifact_path'], comparison=arm + '-matched_reference',
                       round_id=chosen['round_id'], missing_reason=None)
        elif module == 'M2' and r['dataset'] in ('publicOSS1', 'publicOSS2'):
            chosen = next(s for s in modules if s['dataset'] == r['dataset'])
            row.update(status='completed', oa_pct=chosen['full_oa_pct'], ba_pct=chosen['full_ba_pct'],
                       delta_oa_pp=chosen['delta_reference_oa_pp'], delta_ba_pp=chosen['delta_reference_ba_pp'],
                       artifact_path=chosen['artifact_path'], comparison='projection_residual-pca_reference',
                       round_id=chosen['round_id'], missing_reason=None)
        coverage.append(row)

inventory = [
    dict(dataset='FMoW', current_method_status='not_found_in_bounded_search',
         oa_pct=None, ba_pct=None, target_rows=None, target_classes=None,
         feature_paths=[str(PT / 'xdom_generic/featsA_fmow_sub.npy'),
                        str(PT / 'xdom_generic/featsB_fmow_clip.npy')],
         inventory_artifact=absolute('opensarship2_transfer_20261007/AVAILABLE_FEATURES.md'),
         historical_results_exist=True,
         historical_artifact=str(PT / 'xdom_generic/fullchain_fmowclip_result.json'),
         historical_protocol_artifact=str(PT / 'xdom_generic/FMOW_CLIP_FINDINGS_v1.md'),
         conclusion='Current-method evidence is feature inventory only. Historical top-8, five-region, 20%-budget CLIP/key experiments exist; these are not current shared64/5% residual-handoff or new joint-module results.'),
    dict(dataset='Camelyon17', current_method_status='not_found_in_bounded_search',
         oa_pct=None, ba_pct=None, target_rows=None, target_classes=None,
         feature_paths=['E:/Hermes/xdomain_data/hf_cam17_feats',
                        'E:/Hermes/xdomain_data/hf_cam17_pca512'],
         data_archive='E:/Hermes/xdomain_data/camelyon17_v1.0/archive.tar.gz',
         inventory_artifact=absolute('opensarship2_transfer_20261007/AVAILABLE_FEATURES.md'),
         historical_results_exist=None, historical_artifact=None,
         conclusion='Feature/data files are present; no current-method metric artifact was found in the bounded search. The inventory location contains a tar.gz archive; actual NPZ feature shards are under hf_cam17_feats/hf_cam17_pca512. No metric, row count, or taxonomy inferred from presence.')
]

search = dict(
    terms=['FMoW', 'Camelyon17', 'cam17'],
    primary_roots=[str(ROOT), str(PT / 'xdom_generic'), str(PT / 'reports'),
                   str(PT / 'internal_mainline'), str(PT / 'external_mainline')],
    supplemental_root=str(PT),
    supplemental_exclusions=['**/repos/**', '**/run_logs/**', '**/envs/**', '**/hf_home/**',
                             '**/cache_home/**', '**/_repair*/**', '**/module_evidence_20261008/**'],
    supplemental_limits='rg text search *.md/*.json/*.py, max file size 300K; file-name/directory inventory only for large arrays and archives',
    outcome='No current shared64/5% method or M2/M3/joint results verified for FMoW/Camelyon17; not a claim of absence elsewhere.',
    loader_scope_artifact=absolute('oof_reliability_check/generic_transfer_data.py'),
    loader_scope='Current generic loader supports PACS and OfficeHome only.')

limitations = [
    'All OA/BA values are percentages; differences are percentage points. JSON retains original ratios and unrounded floating-point values; Markdown rounds to six decimals.',
    'M1 = base64 minus source; M4 = full5 minus base64. These are stage contrasts and do not isolate all internal components. Neither is an M1/M4 2x2 ablation.',
    'Only corrected F0921 enters the main table. Old wrong cached crops are excluded, including their OA17.920356/21.433259/21.834150 mainline.',
    'Production correctedF0921 full5 OA/BA29.633543/25.105149 and matched joint reference29.634353/25.106684 belong to distinct replays. The old standalone M2 and M3 values are retained separately.',
    'Self24ports uses 23 eligible target ports with 64 shared support labels per port per draw. External SAR uses 64 global labels per dataset. Generic tasks use 64 per target domain, 256 across four tasks, not 64 for an entire benchmark.',
    'All evaluation excludes supports; support draws overlap. Row-level SAR support/query is not vessel-disjoint. Historical development exposure and F0921 SSL pretraining exposure remain.',
    'PACS/OfficeHome use their own RGB R50/R18 encoders, not the SAR classifiers/polarizations. Source-head OOF is conditional on representations, not strict encoder OOF. They validate the existing adaptation/handoff mechanism, not new M2/M3/joint modules.',
    'Five percent caps handoff admissions, not feature extraction cost or all changes from the original source predictions.',
    'Missing experiments remain null/not_run; FMoW/Camelyon17 are not_found_in_bounded_search because bounded absence does not prove never run.',
    'No new training, inference, feature extraction, GPU work, significance tests, SOTA or universal-improvement claims.'
]


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: 'null' if row.get(k) is None else
                             json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (list, dict))
                             else row[k] for k in keys})


def fmt(value):
    return 'null' if value is None else f'{value:.6f}'


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def main_table(rows):
    return md_table(['数据集/目标', '目标/查询行', '类', 'Source OA/BA', 'Base64 OA/BA',
                     'Full5 OA/BA', 'M1 OA/BA差值', 'M4 OA/BA差值'], [
        [r['dataset'] + '/' + r['target'], f'{r["target_rows"]}/{r["evaluated_rows_per_draw"]}',
         r['target_classes'],
         fmt(r['source_oa_pct']) + '/' + fmt(r['source_ba_pct']),
         fmt(r['base64_oa_pct']) + '/' + fmt(r['base64_ba_pct']),
         fmt(r['full5_oa_pct']) + '/' + fmt(r['full5_ba_pct']),
         fmt(r['m1_base64_minus_source_oa_pp']) + '/' + fmt(r['m1_base64_minus_source_ba_pp']),
         fmt(r['m4_full5_minus_base64_oa_pp']) + '/' + fmt(r['m4_full5_minus_base64_ba_pp'])]
        for r in rows])


md = ['# 现有模块与数据集证据表（2026-10-08）',
      '仅整理已完成工件，无新实验。OA/BA 为百分数，差值为百分点。机器 JSON 和 CSV 保留未四舍五入值。',
      '## 原主线结果与阶段差值',
      'M1 表示 Base64−Source；M4 表示 Full5−Base64。这是阶段对比，不是 M1/M4 的 2×2 消融，也不等于对其内部全部机制的独立因果归因。',
      main_table([r for r in main if r['group'] == 'main']),
      '## FUSAR 可选附表：7 个实际目标类',
      '保留 8 个输出列，但目标存在 7 类；单图复制输入不构成双极化验证。不要与 8 类 SAR 表合成共同基准。',
      main_table([r for r in main if r['group'] != 'main']),
      '## 单模块历史候选：保留各自匹配参考',
      md_table(['数据集', '模块/轮次', '完整 OA/BA', '参考 OA/BA', '对参考 ΔOA/ΔBA'], [
          [r['dataset'], r['module'] + '/' + r['round_id'],
           fmt(r['full_oa_pct']) + '/' + fmt(r['full_ba_pct']),
           fmt(r['reference_oa_pct']) + '/' + fmt(r['reference_ba_pct']),
           fmt(r['delta_reference_oa_pp']) + '/' + fmt(r['delta_reference_ba_pp'])] for r in modules]),
      'M2 表示补空间候选启用后的完整部署系统，M3 表示部署对齐候选启用后的完整部署系统；不是单独专家在所有查询上的诊断准确率。专家诊断值仅在机器表中另列。',
      '## F0921 本轮匹配 M2×M3 2×2',
      '本轮所有分支固定 M1、Base64、M4 规则、64 个支持标签和 6171/123425 次交接。M2 补空间与 M3 对齐训练是两个实验因子。',
      md_table(['分支', 'M2', 'M3', 'Full OA', 'Full BA', '相对本轮参考 ΔOA', 'ΔBA'], [
          [r['arm'], int(r['complement_enabled']), int(r['aligned_enabled']), fmt(r['full_oa_pct']),
           fmt(r['full_ba_pct']), fmt(r['delta_reference_oa_pp']), fmt(r['delta_reference_ba_pp'])] for r in joint]),
      '必须分开：生产校正参考 29.633543/25.105149；本轮匹配参考 29.634353/25.106684。原独立 M3 的 BA=25.435555，本轮匹配 M3 的 BA=25.434966，不能混拼；独立 M2 的 F0921 结果同样单独保留。',
      '联合分支相对匹配参考 ΔOA=+0.124286、ΔBA=+0.325071；相对 aligned-only ΔOA=+0.432165、ΔBA=−0.003211；相对 complement-only ΔOA=−0.070650、ΔBA=+0.428282。仅为已使用的 F0921 开发结果。',
      '## 逐数据集/域的模块完成矩阵',
      'M1/M4 数字见原主线阶段差值表。下表采用本轮最新匹配 F0921 数字，旧单模块轮次见上表。null/not_run 不以诊断实验或其他模型结果填补。',
      md_table(['数据集/目标', '模块', '状态', 'OA/BA', '对匹配参考 ΔOA/ΔBA'], [
          [r['dataset'] + '/' + r['target'], r['module'], r['status'],
           fmt(r['oa_pct']) + '/' + fmt(r['ba_pct']),
           fmt(r['delta_oa_pp']) + '/' + fmt(r['delta_ba_pp'])]
          for r in coverage if r['module'] not in ('M1', 'M4')]),
      '## 行数、预算与聚合',
      f'自有库共 38091 行/24 港；当前主线纳入 {target_rows} 行/23 个合格目标港，每轮总查询 {target_rows - 23 * 64} 行。每港独立 64 标签，共 1472 个支持标注/轮。先平均港内五次抽样，再等权平均 23 港；OA 也不是汇总样本 OA。具体港口行数和数值见 self23ports_detail.csv。',
      'F0921、publicOSS1、publicOSS2 和可选 FUSAR 每个数据集每轮各用 64 个全局目标标签；两适配分支共享标签。PACS/OfficeHome 每个外层目标域各用 64 标签，四任务合计 256 标签/轮；四域均值等域加权，行数是四任务之和。',
      'PACS 7 类、OfficeHome 65 类均保留。OfficeHome 每轮支持仅覆盖 34–46 类，未观察类不从查询评估中删除。负结果保留：PACS/photo 的 M4 BA 差值 −3.213599，OfficeHome/Art 为 −0.090568。',
      '## 表征与证据边界',
      'PACS/OfficeHome 使用各数据集自己的 RGB 特征：冻结 ImageNet R50 与按外层源域监督微调的 R18 编码同一图像。它们不是 SAR 双极化，不是港口知识验证，也没有验证本轮新增 M2/M3/联合模块。源域内部 head-OOF 以固定表征为条件，不是 encoder-OOF。',
      '外部 SAR 为细标签兼容子集。F0921 继承 SSL 预训练语料暴露，各数据集均有历史开发使用；SAR 支持/查询是行级划分，非同船隔离。5% 是交接次数上限，不是特征计算上限，也不是相对源模型全部预测变化的上限。',
      '## FMoW 与 Camelyon17：有界检索结论',
      'FMoW：当前方法证据仅到特征清单，未发现当前 shared64/5% 残差交接或新联合模块的结果。但已核实旧实验 fullchain_fmowclip_result.json，协议为前 8 高频类、五地区、20% 预算；不能把旧实验填入本表，也不能说 FMoW 从未做过实验。',
      'Camelyon17：已核实 archive.tar.gz 和 hf_cam17_feats/hf_cam17_pca512 下的 NPZ 文件存在；未加载大特征。当前方法 OA/BA、目标行数、类数均 null。未在检索范围发现当前结果不等于断言其他位置绝无实验。',
      '检索范围和限制逐项写入 JSON 的 bounded_search。包含 sensitivity、xdom_generic、reports、internal_mainline、external_mainline，并补充 paper_tracking 中排除第三方仓库、运行日志、环境及备份的有限大小文本检索。',
      '## 精确来源',
      '以下均为已有工件绝对路径；JSON 每行还提供字段选择器。',
      *[f'- `{p}`' for p in sorted({r['artifact_path'] for r in main + modules + joint})],
      '- `' + absolute('joint_residual_pilot_20261008/JOINT_FINDINGS.md') + '`',
      '- `' + absolute('CROSS_DATASET_READOUT_20261007.md') + '`',
      '- `' + absolute('GENERIC_CROSS_DOMAIN_READOUT_20261007.md') + '`',
      '- `' + str(PT / 'xdom_generic/fullchain_fmowclip_result.json') + '`',
      '- `' + absolute('opensarship2_transfer_20261007/AVAILABLE_FEATURES.md') + '`',
      '## 提取验证',
      '从现有逐次 metrics.csv 核对汇总 OA/BA、港口行数和四域均值；不重做模型推理，不运行显著性检验。核对失败会停止生成并定位错误来源。检查记录见 extraction_checks.json。']

payload = dict(date='2026-10-08', metric_units=dict(oa_ba='percent', differences='percentage_points'),
               missing_value_policy='JSON null; CSV literal null; status distinguishes not_run from bounded-search absence',
               main_results=main, self23ports_detail=port_details, standalone_modules=modules,
               matched_joint_2x2=joint, joint_contrasts=joint_summary['contrasts'],
               joint_interaction_pp=joint_summary['interaction_pp'], module_coverage=coverage,
               feature_inventory_only=inventory, bounded_search=search,
               limitations=limitations, primary_artifacts=READS)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'module_dataset_evidence.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
write_csv('main_dataset_results.csv', main)
write_csv('self23ports_detail.csv', port_details)
write_csv('standalone_module_results.csv', modules)
write_csv('f0921_matched_joint_2x2.csv', joint)
write_csv('module_coverage.csv', coverage)
write_csv('feature_inventory_status.csv', inventory)
(OUT / 'MODULE_DATASET_EVIDENCE_ZH.md').write_text('\n\n'.join(md) + '\n', encoding='utf-8')
(OUT / 'extraction_checks.json').write_text(json.dumps(dict(
    status='passed', purpose='Detect summary/CSV mismatch, wrong averaging or support/query scope; stop on mismatch.',
    checks=CHECKS, new_experiments=0, gpu_used=False, feature_arrays_loaded=0,
    main_rows=len(main), standalone_rows=len(modules), joint_rows=len(joint),
    module_coverage_rows=len(coverage), eligible_ports=23,
    eligible_target_rows=target_rows, eligible_query_rows_per_draw=target_rows - 23 * 64
), indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'passed', 'output_dir': OUT.as_posix(), 'main_rows': len(main),
                  'module_rows': len(modules), 'joint_rows': len(joint),
                  'coverage_rows': len(coverage), 'eligible_target_rows': target_rows}))
