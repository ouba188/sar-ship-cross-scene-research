"""CPU analysis of saved class-distance predictions; no model imports.

python -B analysis.py --seeds 17 29 43 --gallery
"""
import argparse
import html
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT = HERE / 'analysis'
spec = importlib.util.spec_from_file_location(
    'class_distance_cpu_episode', HERE.parent / 'm3_reference_episode_20261030/analysis.py')
episode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(episode)
helper = episode.helper
replay = helper.replay
read, dump, metric = helper.read, helper.dump, helper.metric
METRICS, STATES = helper.METRICS, helper.STATES
ARMS = ('aligned', 'pooled')
COMPARISONS = tuple((a, b) for a in ARMS for b in ('saved_baseline', 'ce')) + (
    ('aligned', 'pooled'), ('ce', 'saved_baseline'))
helper.HERE, helper.OUT, helper.ARMS = HERE, OUT, ARMS
helper.COMPARISONS = COMPARISONS


def csv(path, rows):
    replay.previous.write_csv(path, rows)


def load_dataset(dataset, seeds):
    d = helper.load_dataset(dataset, seeds)
    if d['pending']:
        raise RuntimeError('Incomplete requested fits: ' + json.dumps(d['pending']))
    for arm in ARMS:
        for seed in seeds:
            labels = d['receipts'][arm, seed]['reference_labels']
            if labels != [r['label'] for r in d['refs']]:
                raise ValueError('Reference label order differs from the actual bank')
    return d


def load_evaluation(dataset, seeds):
    """Read the completed frozen cohort; never substitute a partial denominator."""
    d = load_dataset(dataset, seeds)
    folder = HERE / dataset / 'evaluation'
    receipt = read(folder / 'results.json')
    if not set(seeds).issubset(receipt['completed_seeds']):
        raise RuntimeError(f'{dataset}: frozen evaluation still in progress')
    path = (HERE.parent / 'm2_preadd_expansion_20261028/query_records.json' if dataset == 'private'
            else Path(d['manifest']) / 'evaluation.json')
    group = replay.prior.rows(path)
    if len(group) != receipt['query_count']:
        raise ValueError('Frozen evaluation manifest count mismatch')
    d.update(group=group, y=np.array([r['label'] for r in group]), scores={},
             input_paths=[str(path), str(folder / 'results.json')], scope='frozen_evaluation',
             evaluation_receipt=receipt)
    for seed in seeds:
        for arm in ('original', 'ce', *ARMS):
            pp = folder / f'{arm}_s{seed}.npz'
            pack, missing = helper.load_pack(pp, group, d['classes'])
            if missing:
                raise RuntimeError(json.dumps(missing))
            name = 'saved_baseline' if arm == 'original' else arm
            if arm == 'original' and not np.array_equal(pack['baseline'], pack['score']):
                raise ValueError('Frozen original and saved baseline differ')
            if ('saved_baseline', seed) in d['scores'] and not np.array_equal(pack['baseline'], d['scores']['saved_baseline', seed]):
                raise ValueError('Frozen arms have different originals')
            record = next(r for r in receipt['records'] if (r['arm'], r['seed']) == (arm, seed))
            for key, value in metric(pack['score'], d['y']).items():
                if not np.isclose(value, record['after'][key.lower()], rtol=0, atol=1e-6):
                    raise ValueError('Frozen evaluation receipt metrics mismatch')
            d['scores'][name, seed] = pack['score']
            d['input_paths'].append(str(pp))
    return d


def paired_summary(d):
    folder = OUT / d['dataset']
    folder.mkdir(parents=True, exist_ok=True)
    endpoints, paired, tradeoffs, events = helper.previous.paired_tables(
        d, ('ce', *ARMS), COMPARISONS)
    means = []
    for arm, reference in COMPARISONS:
        rows = [r for r in paired if (r['arm'], r['reference']) == (arm, reference)]
        entry = dict(arm=arm, reference=reference, seeds=d['requested'])
        for key in METRICS:
            values = [r['delta_' + key] for r in rows]
            entry['delta_' + key] = float(np.mean(values))
            entry['sd_' + key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            entry['positive_orders_' + key] = sum(v > 0 if key in ('OA', 'BA') else v < 0 for v in values)
        entry.update({state: sum(r[state] for r in rows) for state in STATES})
        entry['top1_changes'] = sum(r['top1_changes'] for r in rows)
        means.append(entry)
    identities = []
    for seed in d['requested']:
        for arm, reference in COMPARISONS:
            new, old = d['scores'][arm, seed], d['scores'][reference, seed]
            identities.append(dict(arm=arm, reference=reference, seed=seed,
                changed=int((new.argmax(1) != old.argmax(1)).sum()),
                max_abs_logit_delta=float(abs(new-old).max()),
                logits_identical=bool(np.array_equal(new, old))))
    for name, rows in (('metrics', endpoints), ('paired', paired), ('paired_means', means),
                       ('events', events), ('class_tradeoffs', tradeoffs), ('prediction_identity', identities)):
        csv(folder / f'{name}.csv', rows)
    result = dict(dataset=d['dataset'], seeds=d['requested'], query_count=len(d['y']),
        metrics=endpoints, paired=paired, paired_means=means, prediction_identity=identities,
        input_paths=d['input_paths'])
    dump(folder / 'summary.json', result)
    return result


def source_offsets(d):
    """Fit only an unweighted source-query mean, with fixed coefficient one."""
    folder = OUT / d['dataset']
    results, cases, inputs = [], [], []
    source_cohort = None
    source_baselines = {}
    for seed in d['requested']:
        for arm in ARMS:
            path = HERE / d['dataset'] / 'predictions' / f'{arm}_source_s{seed}.npz'
            with np.load(path, allow_pickle=False) as p:
                ids = p['ids'].astype(str)
                order = np.argsort(ids)
                ids, y_source = ids[order], p['y'][order]
                new_source = p['score'][order].astype(float)
                old_source = p['baseline'][order].astype(float)
            report = d['receipts'][arm, seed]
            if len(set(ids)) != len(ids) or len(ids) != report['source_count']:
                raise ValueError(f'Source cohort count/identity mismatch: {path}')
            if new_source.shape != (len(ids), len(d['classes'])) or not np.isfinite(new_source).all() or not np.isfinite(old_source).all():
                raise ValueError(f'Invalid source scores: {path}')
            cohort = (ids.tolist(), y_source.tolist())
            if source_cohort is None:
                source_cohort = cohort
            elif source_cohort != cohort:
                raise ValueError('Source offset arms/orders use different source cohorts')
            if seed in source_baselines and not np.array_equal(source_baselines[seed], old_source):
                raise ValueError('Source offset arms use different originals')
            source_baselines[seed] = old_source
            for z, phase in ((new_source, 'source_after'), (old_source, 'source_before')):
                actual = metric(z, y_source)
                if any(not np.isclose(actual[k], report[phase][k.lower()], atol=1e-6, rtol=0) for k in METRICS):
                    raise ValueError(f'Source receipt mismatch: {path}, {phase}')
            delta = new_source - old_source
            centered = delta - delta.mean(1, keepdims=True)
            delta_mean = centered.mean(0)
            old, new = d['scores']['saved_baseline', seed], d['scores'][arm, seed]
            reconstructed = old + delta_mean
            base_pred, actual_pred, offset_pred = old.argmax(1), new.argmax(1), reconstructed.argmax(1)
            truth = d['y']
            changed = actual_pred != base_pred
            shared_changed = offset_pred != base_pred
            same = actual_pred == offset_pred
            rescue = (base_pred != truth) & (actual_pred == truth)
            harm = (base_pred == truth) & (actual_pred != truth)
            common_rescue = (base_pred != truth) & (offset_pred == truth)
            common_harm = (base_pred == truth) & (offset_pred != truth)
            source_energy = float(np.square(centered).sum())
            entry = dict(dataset=d['dataset'], arm=arm, seed=seed, source_count=len(ids),
                source_npz=str(path), delta_mean=delta_mean.tolist(), coefficient=1,
                source_common_energy_fraction=float(len(ids)*np.square(delta_mean).sum()/source_energy) if source_energy else None,
                actual_changed=int(changed.sum()), offset_changed=int(shared_changed.sum()),
                same_prediction=int(same.sum()), query_count=len(truth),
                reproduced_actual_changes=int((changed & same).sum()),
                reproduced_change_fraction=float(same[changed].mean()) if changed.any() else None,
                actual_rescued=int(rescue.sum()), reproduced_rescues=int((rescue & common_rescue).sum()),
                actual_harmed=int(harm.sum()), reproduced_harms_same_class=int((harm & same).sum()),
                reproduced_harms_any_wrong_class=int((harm & common_harm).sum()),
                offset_only_changes=int((shared_changed & ~changed).sum()),
                offset_only_rescues=int((common_rescue & ~rescue).sum()),
                offset_only_harms=int((common_harm & ~harm).sum()))
            for name, z in (('baseline', old), ('actual', new), ('source_offset', reconstructed)):
                entry.update({name+'_'+key: value for key, value in metric(z, truth).items()})
            results.append(entry)
            inputs.append(str(path))
            for i, row in enumerate(d['group']):
                if changed[i] or shared_changed[i]:
                    cases.append(dict(dataset=d['dataset'], arm=arm, seed=seed, id=str(row['id']),
                        label=int(truth[i]), baseline=int(base_pred[i]), actual=int(actual_pred[i]),
                        source_offset=int(offset_pred[i]), actual_changed=bool(changed[i]),
                        offset_changed=bool(shared_changed[i]), exact_prediction_reproduced=bool(same[i]),
                        actual_state=replay.prior.state(int(base_pred[i]), int(actual_pred[i]), int(truth[i])),
                        offset_state=replay.prior.state(int(base_pred[i]), int(offset_pred[i]), int(truth[i]))))
    dump(folder / 'source_offset.json', dict(results=results, input_paths=inputs,
        definition='mean_source(center_classes(score-baseline)); selected evaluation-cohort baseline plus this vector; coefficient=1',
        evaluation_scope=d.get('scope', 'development'),
        scope='Source queries only estimate the offset. Target labels only evaluate outcomes. No target fitting.',
        interpretation='Prediction reproduction is a diagnostic, not a unique physical cause or proof of mechanism.'))
    csv(folder / 'source_offset.csv', [{k: v for k, v in r.items() if k != 'delta_mean'} for r in results])
    csv(folder / 'source_offset_events.csv', cases)
    return results


def annotated_events(d, seeds):
    pool = []
    for seed in seeds:
        scores = d['scores']
        old, aligned, pooled, ce = [scores[a, seed] for a in ('saved_baseline', 'aligned', 'pooled', 'ce')]
        for i, e in enumerate(replay.events(old, aligned, d['group'], seed)):
            truth, rival = e['label'], e['rival']
            e.update(arm='aligned', reference='saved_baseline', state_orders=1,
                new_vs_ce=bool(aligned[i].argmax() != ce[i].argmax()),
                aligned_vs_pooled=bool(aligned[i].argmax() != pooled[i].argmax()),
                ce_prediction=int(ce[i].argmax()), pooled_prediction=int(pooled[i].argmax()),
                new_ce_margin_delta=float((aligned[i, truth]-aligned[i, rival])-(ce[i, truth]-ce[i, rival])),
                aligned_pooled_margin_delta=float((aligned[i, truth]-aligned[i, rival])-(pooled[i, truth]-pooled[i, rival])))
            pool.append(e)
    return pool


def make_gallery(d, supplemental=False, excluded=()):
    seeds = [s for s in d['requested'] if s != 17] if supplemental else [17]
    pool = annotated_events(d, seeds)
    rank = lambda e: (not e['aligned_vs_pooled'], not e['new_vs_ce'],
                      -abs(e['aligned_pooled_margin_delta']), -abs(e['new_ce_margin_delta']), e['id'], e['seed'])
    absent, chosen = [], []
    if supplemental:
        seed17 = {e['id']: e for e in annotated_events(d, [17])}
        candidates = [e for e in pool if e['changed'] and not seed17[e['id']]['changed'] and e['id'] not in excluded]
        for e in sorted(candidates, key=rank):
            if e['id'] not in {x['id'] for x in chosen}:
                chosen.append(e)
            if len(chosen) == 2:
                break
    else:
        for state in STATES:
            candidates = sorted((e for e in pool if e['state'] == state), key=rank)
            if candidates:
                chosen.append(candidates[0])
            else:
                absent.append(dict(state=state, state_events=0, reason='No seed17 events relative to original.'))
        used = {e['id'] for e in chosen}
        chosen += sorted((e for e in pool if e['id'] not in used), key=rank)[:4-len(chosen)]
    if not chosen:
        return dict(cases=0, panels=0, selected_ids=[], reason='No additional distinct flips after seed17.')
    replay.OUT = OUT / ('supplement' if supplemental else 'pilot17')
    replay.ARMS = ('ce', *ARMS)
    previous_selector = replay.select_cases
    replay.select_cases = lambda *_: (chosen, absent)
    try:
        result = replay.gallery(dict(d, seeds=d['requested']), 'aligned', 'saved_baseline')
    finally:
        replay.select_cases = previous_selector
    folder = replay.OUT / d['dataset'] / 'aligned_vs_saved_baseline'
    result.update(selected_ids=[e['id'] for e in chosen],
        rule=('At most two additional queries per dataset, changed after seed17 but unchanged at seed17, excluding pilot cases.'
              if supplemental else 'At most four seed17 queries per dataset, one per available original-relative state; fill absent states with distinct queries.'),
        prioritization='aligned-pooled top1 differences, then new-vs-CE differences, absolute margin differences, ID, order; deterministic.',
        all_order_logits_in_cases=True)
    dump(folder / 'selection.json', result)
    page = folder / 'index.html'
    content = page.read_text(encoding='utf-8')
    content = content.replace('One distinct query/state, ordered by state frequency, absolute fixed-rival margin change, ID, seed. No random sampling.',
                              html.escape(result['rule'] + ' ' + result['prioritization']))
    page.write_text(content, encoding='utf-8')
    return dict(result, path=str(folder / 'index.html'))


def evaluation_costs(d):
    rows = []
    records = d['evaluation_receipt']['records']
    for seed in d['requested']:
        original = next(r for r in records if (r['arm'], r['seed']) == ('original', seed))
        ce = next(r for r in records if (r['arm'], r['seed']) == ('ce', seed))
        for arm in ('original', 'ce', *ARMS):
            record = next(r for r in records if (r['arm'], r['seed']) == (arm, seed))
            fit = d['receipts'].get((arm, seed), {})
            latency = record['native_batch4_ms']
            rows.append(dict(arm=arm, seed=seed,
                added_parameters=fit['trainable_parameters'] if arm in ARMS else 0,
                fitted_parameters=fit.get('trainable_parameters', 0),
                native_batch4_ms=latency,
                overhead_original_ms=latency-original['native_batch4_ms'],
                overhead_original_percent=100*(latency/original['native_batch4_ms']-1),
                overhead_ce_ms=latency-ce['native_batch4_ms'],
                fit_receipt_seconds=fit.get('seconds'),
                reference_prefix_cached=record['latency_reference_prefix_cached']))
    means = []
    for arm in ('original', 'ce', *ARMS):
        subset = [r for r in rows if r['arm'] == arm]
        means.append(dict(arm=arm, added_parameters=subset[0]['added_parameters'],
            fitted_parameters=subset[0]['fitted_parameters'],
            **{k: float(np.mean([r[k] for r in subset])) for k in
               ('native_batch4_ms', 'overhead_original_ms', 'overhead_original_percent', 'overhead_ce_ms')}))
    csv(OUT / d['dataset'] / 'costs.csv', rows)
    csv(OUT / d['dataset'] / 'cost_means.csv', means)
    return dict(by_order=rows, means=means,
        measurement='Batch4 mean of 10 timed forwards after warmup; query encoding included, reference prefix cached. Not image I/O/preprocessing or cold-start latency.')


def outcome_groups(d):
    """Additive OA/BA contributions use the full cohort's class denominators."""
    y = d['y']
    classes, counts = np.unique(y, return_counts=True)
    count_by_class = dict(zip(classes, counts))
    ba_weight = np.array([100/(len(classes)*count_by_class[c]) for c in y])
    passes = np.array([str(r.get('pass_group') or 'MISSING') for r in d['group']])
    rows = []
    for seed in d['requested']:
        for arm, reference in COMPARISONS:
            old = d['scores'][reference, seed].argmax(1)
            new = d['scores'][arm, seed].argmax(1)
            rescued = (old != y) & (new == y)
            harmed = (old == y) & (new != y)
            net = rescued.astype(int)-harmed.astype(int)
            groups = [('class', str(c), y == c) for c in classes]
            groups += [('pass', p, passes == p) for p in sorted(set(passes))]
            for kind, key, ix in groups:
                indices = np.flatnonzero(ix)
                row = dict(dataset=d['dataset'], kind=kind, group=key, arm=arm, reference=reference,
                    seed=seed, queries=int(ix.sum()),
                    ships=len({str(d['group'][i]['mmsi']) for i in indices if d['group'][i].get('mmsi')}),
                    rescued=int(rescued[ix].sum()), harmed=int(harmed[ix].sum()), net_correct=int(net[ix].sum()),
                    top1_changes=int((old[ix] != new[ix]).sum()),
                    delta_within_group_OA=float(100*net[ix].mean()),
                    contribution_OA_pp=float(100*net[ix].sum()/len(y)),
                    contribution_BA_pp=float((net[ix]*ba_weight[ix]).sum()),
                    class_counts={str(c):int((y[ix] == c).sum()) for c in classes if np.any(y[ix] == c)})
                if kind == 'class':
                    row['class_name'] = d['classes'][int(key)]
                rows.append(row)
    means = []
    for kind in ('class', 'pass'):
        for key in sorted({r['group'] for r in rows if r['kind'] == kind}):
            for arm, reference in COMPARISONS:
                selected = [r for r in rows if (r['kind'], r['group'], r['arm'], r['reference']) == (kind, key, arm, reference)]
                item = {k:selected[0][k] for k in ('dataset', 'kind', 'group', 'arm', 'reference', 'queries', 'ships', 'class_counts')}
                item['class_name'] = selected[0].get('class_name', '')
                item['seeds'] = d['requested']
                for key_metric in ('rescued', 'harmed', 'net_correct', 'top1_changes', 'contribution_OA_pp', 'contribution_BA_pp'):
                    item['mean_'+key_metric] = float(np.mean([r[key_metric] for r in selected]))
                means.append(item)
    for seed in d['requested']:
        for arm, reference in COMPARISONS:
            a, b = metric(d['scores'][arm, seed], y), metric(d['scores'][reference, seed], y)
            for kind in ('class', 'pass'):
                selected = [r for r in rows if (r['kind'], r['arm'], r['reference'], r['seed']) == (kind, arm, reference, seed)]
                for key in ('OA', 'BA'):
                    if not np.isclose(sum(r['contribution_'+key+'_pp'] for r in selected), a[key]-b[key], atol=1e-10):
                        raise ValueError('Group contributions do not reproduce the overall change')
    for kind in ('class', 'pass'):
        csv(OUT / d['dataset'] / f'{kind}_outcomes.csv', [r for r in rows if r['kind'] == kind])
        csv(OUT / d['dataset'] / f'{kind}_outcome_means.csv', [r for r in means if r['kind'] == kind])
    return dict(means=means, denominator='OA: full N. BA: number of present classes times each full-cohort class count. Pass contributions are additive, not mean pass BA.')


def final_gallery(d):
    """At most two new queries/dataset; prefer net outcomes not reproduced by source offset."""
    shown = set()
    for stage in ('pilot17', 'supplement'):
        path = HERE / 'analysis' / stage / d['dataset'] / 'aligned_vs_saved_baseline/selection.json'
        if path.exists():
            shown.update(read(path).get('selected_ids', []))
    development_name = 'development.json' if d['dataset'] == 'private' else 'validation.json'
    dev_ids = {str(r['id']) for r in replay.prior.rows(Path(d['manifest']) / development_name)}
    offsets = read(OUT / d['dataset'] / 'source_offset.json')['results']
    pool = annotated_events(d, d['requested'])
    lookup = {str(r['id']): i for i, r in enumerate(d['group'])}
    for e in pool:
        i, seed = lookup[e['id']], e['seed']
        row = next(r for r in offsets if (r['arm'], r['seed']) == ('aligned', seed))
        predicted = int((d['scores']['saved_baseline', seed][i]+np.array(row['delta_mean'])).argmax())
        e.update(source_offset_prediction=predicted,
            source_offset_reproduces_actual=predicted == e['new'],
            outside_development=e['id'] not in dev_ids)
    rank = lambda e: (not e['outside_development'], e['source_offset_reproduces_actual'],
                      not e['new_vs_ce'], not e['aligned_vs_pooled'], -abs(e['margin_change']), e['id'], e['seed'])
    chosen = []
    for state in ('rescued', 'harmed'):
        candidates = sorted((e for e in pool if e['state'] == state and e['id'] not in shown), key=rank)
        if candidates:
            chosen.append(candidates[0])
            shown.add(candidates[0]['id'])
    if not chosen:
        candidates = sorted((e for e in pool if e['id'] not in shown), key=rank)
        chosen = candidates[:1]
    if not chosen:
        return dict(cases=0, panels=0, selected_ids=[], reason='No distinct remaining queries.')
    replay.OUT, replay.ARMS = OUT / 'final_cases', ('ce', *ARMS)
    previous_selector = replay.select_cases
    replay.select_cases = lambda *_: (chosen, [])
    try:
        result = replay.gallery(d, 'aligned', 'saved_baseline')
    finally:
        replay.select_cases = previous_selector
    folder = replay.OUT / d['dataset'] / 'aligned_vs_saved_baseline'
    result.update(selected_ids=[e['id'] for e in chosen],
        rule='At most two new queries/dataset: one rescue and one harm when available; exclude all previous gallery queries. Prefer outside-development, source-offset-unreproduced outcomes, then new-vs-CE, aligned-vs-pooled and margin change. Deterministic.',
        path=str(folder / 'index.html'))
    dump(folder / 'selection.json', result)
    page = folder / 'index.html'
    content = page.read_text(encoding='utf-8').replace(
        'One distinct query/state, ordered by state frequency, absolute fixed-rival margin change, ID, seed. No random sampling.',
        html.escape(result['rule']))
    page.write_text(content, encoding='utf-8')
    return result


def write_report(results):
    lines = ['# 类分离 M3：开发集 CPU 配对与源共同偏移', '',
        '仅自建 development160 / OSS2 validation228。三顺序共享既有初始化，不是独立预训练重复；不读取最终测试。',
        '所有预测按 ID 对齐，沿用既有 helper 核对原版基准、receipt 与参考顺序。OA/BA 为百分比，差值为百分点；NLL/Brier 越低越好。', '']
    lines += ['当前已汇总：' + '；'.join(f"{r['dataset']} {r['query_count']}条、顺序{r['seeds']}" for r in results) + '。', '']
    for r in results:
        original = next(x for x in r['paired_means'] if (x['arm'], x['reference']) == ('aligned', 'saved_baseline'))
        ce = next(x for x in r['paired_means'] if (x['arm'], x['reference']) == ('aligned', 'ce'))
        lines.append(f"{r['dataset']} aligned：相对原版均值OA/BA {original['delta_OA']:+.6f}/{original['delta_BA']:+.6f}pp；"
                     f"相对CE {ce['delta_OA']:+.6f}/{ce['delta_BA']:+.6f}pp。")
    if results and results[0].get('scope') == 'frozen_evaluation':
        private = next((r for r in results if r['dataset'] == 'private'), None)
        if private:
            original = next(x for x in private['paired_means'] if (x['arm'], x['reference']) == ('aligned', 'saved_baseline'))
            ce = next(x for x in private['paired_means'] if (x['arm'], x['reference']) == ('aligned', 'ce'))
            if original['delta_OA'] < 0 and ce['delta_BA'] < 0:
                lines += ['', '自建完整范围相对原版OA下降、相对CE的BA下降，因此本轮不能认定短期goal达成。'
                    '保留候选及公开集适用信号，不替换正式模型；不因BA小幅为正掩盖OA损失，也不追加训练寻找正结果。']
    lines.append('')
    for r in results:
        lines += [f"## {r['dataset']}", '']
        replay.prior.table(lines, ['方案', '顺序', *METRICS],
            [[x['arm'], x['seed'], *[round(x[k], 6) for k in METRICS]] for x in r['metrics']])
        replay.prior.table(lines, ['比较', '均值 Delta OA', 'Delta BA', 'Delta NLL', 'Delta Brier', '救回事件', '误伤事件', '翻转事件'],
            [[x['arm']+' - '+x['reference'], *[round(x['delta_'+k], 6) for k in METRICS], x['rescued'], x['harmed'], x['top1_changes']]
             for x in r['paired_means']])
        identities = [x for x in r['prediction_identity'] if (x['arm'], x['reference']) == ('aligned', 'pooled')]
        lines += ['', 'aligned 对 pooled 逐顺序预测差异：' + '；'.join(f"{x['seed']}={x['changed']}条（最大logit差{x['max_abs_logit_delta']:.7g}）" for x in identities) + '。',
            '两臂若分类相同，共有收益不能归功于局部 value 对齐。CE 参数量与新分支不同，CE比较不是等容量对照。', '',
            '### 仅用源查询估计的共同偏移', '',
            '`delta_mean = mean_source((score-baseline) - mean_class(score-baseline))`；开发预测为 `baseline + delta_mean`，系数固定1。',
            '源标签未参与偏移拟合；目标标签仅用于事后判对/错。复现率只在实际翻转中计算，零翻转记为空，不能把大量未翻转的一致率当作收益解释率。', '']
        replay.prior.table(lines, ['臂', '顺序', '实际翻转', '偏移翻转', '复现实际翻转', '实际救回/复现', '实际误伤/同类复现', '偏移额外翻转', '偏移OA', '偏移BA'],
            [[x['arm'], x['seed'], x['actual_changed'], x['offset_changed'], x['reproduced_actual_changes'],
              f"{x['actual_rescued']}/{x['reproduced_rescues']}", f"{x['actual_harmed']}/{x['reproduced_harms_same_class']}",
              x['offset_only_changes'], round(x['source_offset_OA'], 6), round(x['source_offset_BA'], 6)] for x in r['source_offset']])
        for arm in ARMS:
            selected = [x for x in r['source_offset'] if x['arm'] == arm]
            total = lambda key: sum(x[key] for x in selected)
            lines.append(f"{arm}：源共同偏移复现实际翻转{total('reproduced_actual_changes')}/{total('actual_changed')}事件，"
                f"救回{total('reproduced_rescues')}/{total('actual_rescued')}事件，"
                f"误伤且同一预测类别{total('reproduced_harms_same_class')}/{total('actual_harmed')}事件；"
                f"另外产生{total('offset_only_changes')}次实际模型没有的翻转。不同顺序事件不当作独立样本。")
        lines += ['', '偏移未复现部分说明该源均值向量不足以重现完整行为，不自动证明某个局部对齐机制；近似复现也不是唯一物理根因。'
            '源端共同能量比例只描述源增量，不是目标解释比例。逐样本复现与额外救回/误伤见 source_offset_events.csv。', '']
        for tag in ('pilot_gallery', 'supplement_gallery', 'final_gallery'):
            if tag in r:
                g = r[tag]
                lines.append(f"图库 {tag}：{g['cases']}查询/{g['panels']}面板；{g.get('path', g.get('reason', ''))}")
        lines += ['', '案例含 new_vs_ce、aligned_vs_pooled、三顺序分差、原始chip路径与实际合法同类/混淆类参考。'
            '面板由既有渲染器生成，视觉状态保持未审阅，供主代理回看；无外观物理归因。', '']
        if 'outcome_groups' in r:
            lines += ['### OA/BA冲突的类别与过境贡献', '',
                '各类/各pass逐顺序救回、误伤、净正确数见 class_outcomes.csv / pass_outcomes.csv；均值见对应 outcome_means.csv。'
                '贡献分母固定为整个评价队列：OA按总样本数，BA按各类样本数及出现类数；逐类或逐pass贡献求和等于完整指标差。'
                '以下救回/误伤为每顺序平均事件数，不是独立船数。', '']
            for reference in ('saved_baseline', 'ce'):
                selected = [x for x in r['outcome_groups']['means'] if x['arm'] == 'aligned' and x['reference'] == reference]
                lines += [f'相对 {reference} 的全部类别：', '']
                replay.prior.table(lines, ['类', 'N', '平均救回', '平均误伤', '平均净正确', 'OA贡献pp', 'BA贡献pp'],
                    [[x['class_name'], x['queries'], round(x['mean_rescued'], 3), round(x['mean_harmed'], 3),
                      round(x['mean_net_correct'], 3), round(x['mean_contribution_OA_pp'], 6), round(x['mean_contribution_BA_pp'], 6)]
                     for x in selected if x['kind'] == 'class'])
                class_rows = [x for x in selected if x['kind'] == 'class']
                worst = min(class_rows, key=lambda x: x['mean_contribution_OA_pp'])
                best = max(class_rows, key=lambda x: x['mean_contribution_BA_pp'])
                lines += ['', f"样本数贡献最负的是{worst['class_name']}（N={worst['queries']}，每顺序净正确{worst['mean_net_correct']:+.3f}，"
                    f"OA贡献{worst['mean_contribution_OA_pp']:+.6f}pp）；BA贡献最大的是{best['class_name']}"
                    f"（N={best['queries']}，每顺序净正确{best['mean_net_correct']:+.3f}，BA贡献{best['mean_contribution_BA_pp']:+.6f}pp）。"
                    '这是指标分母与真实救回/误伤组成的解释，不是物理根因。']
                pass_rows = sorted((x for x in selected if x['kind'] == 'pass'),
                    key=lambda x: (-abs(x['mean_contribution_BA_pp']), -abs(x['mean_net_correct']), x['group']))[:6]
                lines += ['', '按绝对BA贡献排序的前6个pass（所有pass及pooled同口径明细保留在CSV）：', '']
                replay.prior.table(lines, ['pass', 'N', '平均救回', '平均误伤', '平均净正确', 'OA贡献pp', 'BA贡献pp'],
                    [[x['group'], x['queries'], round(x['mean_rescued'], 3), round(x['mean_harmed'], 3),
                      round(x['mean_net_correct'], 3), round(x['mean_contribution_OA_pp'], 6), round(x['mean_contribution_BA_pp'], 6)] for x in pass_rows])
            lines.append('')
        if 'costs' in r:
            lines += ['### 参数与实测耗时', '']
            replay.prior.table(lines, ['臂', '顺序', '新增参数', '本轮可训练参数', '粗测batch4 ms', '本次较原版多ms'],
                [[x['arm'], x['seed'], x['added_parameters'], x['fitted_parameters'],
                  round(x['native_batch4_ms'], 4), round(x['overhead_original_ms'], 4)] for x in r['costs']['by_order']])
            replay.prior.table(lines, ['臂', '新增参数', '三顺序均值batch4 ms', '平均较原版多ms', '平均较CE多ms'],
                [[x['arm'], x['added_parameters'], round(x['native_batch4_ms'], 4),
                  round(x['overhead_original_ms'], 4), round(x['overhead_ce_ms'], 4)] for x in r['costs']['means']])
            lines += ['', '计时来自主代理已保存原生前向：warmup后10次、batch4，包含query编码、缓存reference prefix；'
                '不含读图/预处理和冷启动，不是整数据集评价耗时。不同臂按顺序粗测，波动明显；'
                '本表不支持稳定百分比加速/减速或aligned必然比pooled慢的结论。CPU支线未自行运行模型计时。', '']
    lines += ['## 限界', '', '本支线只读取已保存预测和源/开发记录、生成限额原图面板；未加载模型、调用GPU、训练或Git。'
        '原版对照、普通续训恢复和新增救回分别记录；案例不是总体比例。完整范围评价另列，不能混作开发结果或独立盲测。']
    if results and results[0].get('scope') == 'frozen_evaluation':
        lines[0] = '# 类分离 M3：冻结完整范围 CPU 配对与源共同偏移'
        lines[2] = '仅冻结自建3075 / OSS2四类378历史队列；三顺序共享既有初始化，不是独立预训练重复或新盲测。'
        lines = [line.replace('开发预测为', '冻结队列预测为').replace('开发翻转中', '冻结队列翻转中') for line in lines]
        lines += ['', '开发报告保留在 analysis/DEVELOPMENT_ZH.md；本报告没有用完整队列选择系数、超参数或训练配置。']
    content = '\n'.join(lines)+'\n'
    (OUT / 'FINAL_ZH.md').write_text(content, encoding='utf-8')
    if results and results[0].get('scope') == 'frozen_evaluation':
        root = HERE / 'analysis'
        old = root / 'FINAL_ZH.md'
        development = root / 'DEVELOPMENT_ZH.md'
        if old.exists() and not development.exists():
            development.write_text(old.read_text(encoding='utf-8'), encoding='utf-8')
        old.write_text(content, encoding='utf-8')


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, choices=(17, 29, 43), default=[17, 29, 43])
    parser.add_argument('--dataset', choices=('private', 'oss2', 'all'), default='all')
    parser.add_argument('--gallery', action='store_true')
    parser.add_argument('--scope', choices=('development', 'evaluation'), default='development')
    parser.add_argument('--final-gallery', action='store_true')
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    if args.gallery and 17 not in seeds:
        parser.error('Gallery requires seed17.')
    if args.final_gallery and args.scope != 'evaluation':
        parser.error('Final gallery requires frozen evaluation.')
    names = ('private', 'oss2') if args.dataset == 'all' else (args.dataset,)
    if args.scope == 'evaluation':
        OUT = HERE / 'analysis' / 'evaluation'
        if args.gallery:
            parser.error('Frozen evaluation does not automatically add more panels.')
    loader = load_evaluation if args.scope == 'evaluation' else load_dataset
    datasets = [loader(name, seeds) for name in names]
    results = []
    for d in datasets:
        result = paired_summary(d)
        result['scope'] = d.get('scope', 'development')
        result['source_offset'] = source_offsets(d)
        if args.scope == 'evaluation':
            result['costs'] = evaluation_costs(d)
            result['outcome_groups'] = outcome_groups(d)
            if args.final_gallery:
                result['final_gallery'] = final_gallery(d)
        if args.gallery:
            result['pilot_gallery'] = make_gallery(d)
            if len(seeds) > 1:
                result['supplement_gallery'] = make_gallery(d, True, result['pilot_gallery']['selected_ids'])
        results.append(result)
        print(json.dumps(dict(dataset=d['dataset'], paired_means=result['paired_means'],
            source_offset=[{k: x[k] for k in ('arm', 'seed', 'actual_changed', 'reproduced_actual_changes', 'offset_only_changes')}
                           for x in result['source_offset']]), ensure_ascii=False), flush=True)
    assert 'torch' not in sys.modules, 'CPU analysis imported model runtime'
    write_report(results)
    dump(OUT / 'run_receipt.json', dict(cpu_only=True, torch_imported=False, final_test_read=args.scope == 'evaluation',
        seeds=seeds, completed_fits=len(names)*len(seeds)*2, results=results))


if __name__ == '__main__':
    main()
