"""Cached continuous-risk and actual-port robustness evidence; no data refits."""

from full_teacher_data import BASE, SOURCE, PORTS

import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp


OUT = BASE/'layered_evaluation_20261012/probability'
DRAW_SEED = 31415
SEEDS = (31415, 27182)
KEYS = ['family', 'layer', 'context', 'training_seed', 'draw_seed', 'draw',
        'arm', 'baseline_arm', 'diagnostic', 'kernel_intervention']
MODEL_KEYS = ['family', 'layer', 'arm', 'baseline_arm', 'diagnostic',
              'kernel_intervention']
VALUES = ['oa', 'nll', 'brier', 'baseline_oa', 'baseline_nll', 'baseline_brier',
          'delta_oa', 'delta_nll', 'delta_brier']


def probability_rows(scores, truth, kind):
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth, dtype=int)
    assert scores.shape == (len(truth), 8)
    assert np.isfinite(scores).all() and ((truth >= 0) & (truth < 8)).all()
    if kind == 'logits':
        logp = scores-logsumexp(scores, axis=1, keepdims=True)
        prob = np.exp(logp)
        nll = -logp[np.arange(len(truth)), truth]
    elif kind == 'probabilities':
        assert (scores >= 0).all() and (scores <= 1).all()
        np.testing.assert_allclose(scores.sum(1), 1., atol=1e-10, rtol=0)
        prob = scores
        with np.errstate(divide='ignore'):
            nll = -np.log(prob[np.arange(len(truth)), truth])
    else:
        raise ValueError(kind)
    target = np.eye(8)[truth]
    return dict(prob=prob, nll=nll, brier=np.square(prob-target).sum(1),
                correct=prob.argmax(1) == truth,
                trueprob=prob[np.arange(len(truth)), truth], maxprob=prob.max(1))


def paired_rows(new, old):
    return pd.DataFrame(dict(
        oa=100.*new['correct'], nll=new['nll'], brier=new['brier'],
        baseline_oa=100.*old['correct'], baseline_nll=old['nll'],
        baseline_brier=old['brier'],
        delta_oa=100.*(new['correct'].astype(float)-old['correct']),
        delta_nll=old['nll']-new['nll'],
        delta_brier=old['brier']-new['brier']))


def reduce_cells(cells):
    """Equal rows, equal classes, or equal ports then observed classes."""
    weighted = cells[VALUES].multiply(cells.n, axis=0)
    micro = weighted.sum()/cells.n.sum()
    classes = weighted.groupby(cells.class_id).sum().div(
        cells.groupby('class_id').n.sum(), axis=0)
    portclasses = weighted.groupby([cells.actual_port, cells.class_id]).sum().div(
        cells.groupby(['actual_port', 'class_id']).n.sum(), axis=0)
    hierarchical = portclasses.groupby(level='actual_port').mean().mean()
    result = {'micro_'+k: float(v) for k, v in micro.items()}
    result.update({'classmacro_'+k: float(v) for k, v in classes.mean().items()})
    result.update({'portclassmacro_'+k: float(v) for k, v in hierarchical.items()})
    result['ba'] = result['classmacro_oa']
    result['baseline_ba'] = result['classmacro_baseline_oa']
    result['delta_ba'] = result['classmacro_delta_oa']
    result['portclass_ba'] = result['portclassmacro_oa']
    return result


def port_metrics(cells):
    records = []
    for port, group in cells.groupby('actual_port', sort=True):
        records.append(dict(actual_port=port, n=int(group.n.sum()),
                            n_classes=int(group.class_id.nunique()),
                            **reduce_cells(group)))
    return pd.DataFrame(records)


def robustness(ports):
    worst = ports.loc[ports.ba.idxmin()]
    nll = ports.loc[ports.micro_nll.idxmax()]
    brier = ports.loc[ports.micro_brier.idxmax()]
    return dict(n_actual_ports=len(ports), worst_actual_port=worst.actual_port,
                worst_actual_port_ba=float(worst.ba),
                actual_port_ba_q10=float(ports.ba.quantile(.1)),
                highest_nll_actual_port=nll.actual_port,
                highest_actual_port_nll=float(nll.micro_nll),
                highest_brier_actual_port=brier.actual_port,
                highest_actual_port_brier=float(brier.micro_brier),
                highest_actual_port_classmacro_nll=float(ports.classmacro_nll.max()),
                highest_actual_port_classmacro_brier=float(ports.classmacro_brier.max()),
                n_ports_ba_improved=int((ports.delta_ba > 0).sum()),
                n_ports_nll_improved=int((ports.micro_delta_nll > 0).sum()),
                n_ports_brier_improved=int((ports.micro_delta_brier > 0).sum()))


def cached(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


class Evaluation:
    def __init__(self):
        self.cells, self.confidence, self.inventory = [], [], []
        self.unique_ids = {}

    def add(self, info, ids, truth, ports, scores, before, kind, path, baseline_path):
        assert len(np.unique(ids)) == len(ids)
        assert not pd.isna(ports).any()
        current = probability_rows(scores, truth, kind)
        baseline = probability_rows(before, truth, kind)
        frame = paired_rows(current, baseline)
        frame['actual_port'], frame['class_id'] = ports, truth
        groups = frame.groupby(['actual_port', 'class_id'], sort=True)
        cells = groups[VALUES].mean()
        cells['n'] = groups.size()
        cells = cells.reset_index()
        for key, value in info.items():
            cells[key] = value
        self.cells.append(cells)
        model_key = tuple(info[k] for k in MODEL_KEYS)
        self.unique_ids.setdefault(model_key, set()).update(map(int, ids))
        self.inventory.append(dict(**info, n=len(ids), score_kind=kind,
                                   cache=str(path), baseline_cache=str(baseline_path)))
        # Candidate-wrong confidence changes membership; baseline-wrong stays paired.
        subsets = dict(all=np.ones(len(ids), bool),
                       candidate_wrong=~current['correct'],
                       baseline_wrong=~baseline['correct'],
                       rescue=~baseline['correct'] & current['correct'],
                       harm=baseline['correct'] & ~current['correct'],
                       both_wrong=~baseline['correct'] & ~current['correct'],
                       both_correct=baseline['correct'] & current['correct'])
        for port in ['__all__', *sorted(np.unique(ports))]:
            group = np.ones(len(ids), bool) if port == '__all__' else ports == port
            for name, subset in subsets.items():
                mask = group & subset
                n, total = int(mask.sum()), int(group.sum())
                row = dict(**info, actual_port=port, subset=name, n=n, total_n=total)
                for label, vector in [
                    ('maxprob', current['maxprob']), ('baseline_maxprob', baseline['maxprob']),
                    ('trueprob', current['trueprob']), ('baseline_trueprob', baseline['trueprob']),
                    ('delta_trueprob', current['trueprob']-baseline['trueprob']),
                    ('nll', current['nll']), ('baseline_nll', baseline['nll']),
                    ('brier', current['brier']), ('baseline_brier', baseline['brier']),
                    ('delta_nll', frame.delta_nll.to_numpy()),
                    ('delta_brier', frame.delta_brier.to_numpy())]:
                    row[label] = float(vector[mask].mean()) if n else np.nan
                row['maxprob_q90'] = float(np.quantile(current['maxprob'][mask], .9)) if n else np.nan
                for risk in ('nll', 'brier'):
                    row['delta_'+risk+'_population_contribution'] = float(
                        frame.loc[mask, 'delta_'+risk].sum()/total)
                self.confidence.append(row)

    def save(self):
        OUT.mkdir(parents=True, exist_ok=True)
        cells = pd.concat(self.cells, ignore_index=True)
        cells.to_csv(OUT/'port_class_draw.csv', index=False)
        pd.DataFrame(self.inventory).to_csv(OUT/'cache_inventory.csv', index=False)
        draw_rows, port_rows, class_rows = [], [], []
        for key, group in cells.groupby(KEYS, sort=True, dropna=False):
            info = dict(zip(KEYS, key))
            ports = port_metrics(group)
            draw_rows.append(dict(**info, n=int(group.n.sum()),
                                  **reduce_cells(group), **robustness(ports)))
            port_rows.extend(dict(**info, **row) for row in ports.to_dict('records'))
            for class_id, cc in group.groupby('class_id'):
                class_rows.append(dict(**info, class_id=class_id, n=int(cc.n.sum()),
                                       **reduce_cells(cc)))
        draws = pd.DataFrame(draw_rows)
        draws.to_csv(OUT/'per_draw.csv', index=False)
        port_draws = pd.DataFrame(port_rows)
        port_draws.to_csv(OUT/'per_port_draw.csv', index=False)
        pd.DataFrame(class_rows).to_csv(OUT/'per_class_draw.csv', index=False)
        summary_rows, summary_ports, summary_classes = [], [], []
        for key, group in cells.groupby(MODEL_KEYS, sort=True, dropna=False):
            info = dict(zip(MODEL_KEYS, key))
            ports = port_metrics(group)
            summary_rows.append(dict(**info, n_row_exposures=int(group.n.sum()),
                                     n_unique_query_rows=len(self.unique_ids[key]),
                                     **reduce_cells(group), **robustness(ports)))
            summary_ports.extend(dict(**info, **row) for row in ports.to_dict('records'))
            for class_id, cc in group.groupby('class_id'):
                summary_classes.append(dict(**info, class_id=class_id,
                                             n=int(cc.n.sum()), **reduce_cells(cc)))
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(OUT/'summary.csv', index=False)
        pd.DataFrame(summary_ports).to_csv(OUT/'per_port_summary.csv', index=False)
        pd.DataFrame(summary_classes).to_csv(OUT/'per_class_summary.csv', index=False)
        metrics = [c for c in draws if c.startswith(('micro_', 'classmacro_', 'portclassmacro_'))]
        metrics += ['ba', 'baseline_ba', 'delta_ba', 'portclass_ba',
                    'worst_actual_port_ba', 'actual_port_ba_q10',
                    'highest_actual_port_nll', 'highest_actual_port_brier']
        spread = []
        for scope, keys, source in [
            ('context_training_seed', MODEL_KEYS+['context', 'training_seed'], draws),
            ('pooled_configurations', MODEL_KEYS, draws),
            ('actual_port_context_training_seed', MODEL_KEYS+['context', 'training_seed', 'actual_port'], port_draws)]:
            for key, group in source.groupby(keys, sort=True, dropna=False):
                info = dict(zip(keys, key))
                for metric in [m for m in metrics if m in group]:
                    spread.append(dict(**info, scope=scope, metric=metric,
                                       n_units=len(group), mean=float(group[metric].mean()),
                                       std_descriptive_ddof0=float(group[metric].std(ddof=0)),
                                       minimum=float(group[metric].min()),
                                       maximum=float(group[metric].max())))
        pd.DataFrame(spread).to_csv(OUT/'draw_spread.csv', index=False)
        # Equal episode means are a secondary historical comparison, not micro risk.
        draws.groupby(MODEL_KEYS, dropna=False)[metrics].mean().reset_index().to_csv(
            OUT/'episode_mean_summary.csv', index=False)
        historical = pd.concat([
            draws[draws.family != 'actual_target'],
            port_draws[port_draws.family == 'actual_target']], ignore_index=True)
        historical.groupby(MODEL_KEYS, dropna=False)[
            [m for m in metrics if not m.startswith(('worst_', 'actual_port_', 'highest_'))]
        ].mean().reset_index().to_csv(OUT/'historical_equal_unit_summary.csv', index=False)
        confidence = pd.DataFrame(self.confidence)
        confidence.to_csv(OUT/'confidence_draw.csv', index=False)
        conf_summary = []
        averages = ['maxprob', 'baseline_maxprob', 'trueprob', 'baseline_trueprob',
                    'delta_trueprob', 'nll', 'baseline_nll', 'brier', 'baseline_brier',
                    'delta_nll', 'delta_brier']
        for key, group in confidence.groupby(MODEL_KEYS+['actual_port', 'subset'], dropna=False):
            info = dict(zip(MODEL_KEYS+['actual_port', 'subset'], key))
            n, total = int(group.n.sum()), int(group.total_n.sum())
            row = dict(**info, n_row_exposures=n, total_row_exposures=total)
            for metric in averages:
                row[metric] = float((group[metric]*group.n).sum()/n) if n else np.nan
            for risk in ('nll', 'brier'):
                col = 'delta_'+risk+'_population_contribution'
                row[col] = float((group[col]*group.total_n).sum()/total)
            conf_summary.append(row)
        pd.DataFrame(conf_summary).to_csv(OUT/'confidence_summary.csv', index=False)
        # The four disjoint transition subsets must reconstruct total risk improvement.
        transitions = confidence[confidence.subset.isin(['rescue', 'harm', 'both_wrong', 'both_correct'])]
        all_rows = confidence[confidence.subset == 'all']
        for risk in ('nll', 'brier'):
            col = 'delta_'+risk+'_population_contribution'
            sums = transitions.assign(weighted=transitions[col]*transitions.total_n).groupby(
                KEYS+['actual_port']).weighted.sum()
            expected = all_rows.assign(weighted=all_rows['delta_'+risk]*all_rows.total_n).groupby(
                KEYS+['actual_port']).weighted.sum()
            np.testing.assert_allclose(sums, expected.loc[sums.index], atol=1e-10, rtol=0)
        return summary, draws


def source_families(evaluation):
    meta = pd.read_csv(BASE/'observations.csv')
    for context in PORTS:
        for draw in range(3):
            name = f'{DRAW_SEED}_{draw}.npz'
            split = cached(BASE/'incidence_transport_20261011'/f'{DRAW_SEED}_source_{draw}_predictions.npz')
            for seed in SEEDS:
                root = BASE/'case_relation_student_20261012'/context/str(seed)
                baseline_path = root/'task'/name
                old = cached(baseline_path)
                ids = old['query_obs_ids']
                np.testing.assert_array_equal(ids, split['query_obs_ids'])
                mm = meta.iloc[ids]
                for arm in ('task', 'score', 'relation', 'joint', 'readout'):
                    path = (BASE/'case_relation_readout_repair_20261012'/context/str(seed)/arm/name
                            if arm == 'readout' else root/arm/name)
                    data = cached(path)
                    np.testing.assert_array_equal(ids, data['query_obs_ids'])
                    info = dict(family='source_students', layer='classifier_continuous_risk',
                                context=context, training_seed=seed, draw_seed=DRAW_SEED, draw=draw,
                                arm=arm, baseline_arm='task', diagnostic=False,
                                kernel_intervention='query_and_support_geometry_fixed_prior')
                    evaluation.add(info, ids, mm.class_id.to_numpy(), mm.port.to_numpy(),
                                   data['scores'], old['scores'], 'logits', path, baseline_path)
            root = BASE/'case_support_completion_20261012'/context
            baseline_path = root/f'{DRAW_SEED}_{draw}_raw.npz'
            old = cached(baseline_path)
            ids = old['query_obs_ids']
            np.testing.assert_array_equal(ids, split['query_obs_ids'])
            np.testing.assert_array_equal(old['support_obs_ids'], split['support_obs_ids'])
            mm = meta.iloc[ids]
            for arm in ('raw', 'class_center', 'image_history', 'class_image_history', 'image_class_target'):
                arm_root = BASE/'case_completion_class_control_20261012'/context if arm == 'image_class_target' else root
                path = arm_root/f'{DRAW_SEED}_{draw}_{arm}.npz'
                data = cached(path)
                for field in ('query_obs_ids', 'support_obs_ids'):
                    np.testing.assert_array_equal(data[field], old[field])
                info = dict(family='source_completion', layer='classifier_continuous_risk',
                            context=context, training_seed=0, draw_seed=DRAW_SEED, draw=draw,
                            arm=arm, baseline_arm='raw', diagnostic=False,
                            kernel_intervention='support_geometry_only_fixed_prior')
                evaluation.add(info, ids, mm.class_id.to_numpy(), mm.port.to_numpy(),
                               data['scores'], old['scores'], 'logits', path, baseline_path)
        print('SOURCE COMPLETE', context, flush=True)


def target_family(evaluation):
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'port'])
    for port in PORTS:
        path = BASE/'support_completion_target_20261012'/port/'predictions.npz'
        data = cached(path)
        mm = manifest.iloc[data['rows']]
        assert (mm.port == port).all()
        for arm in ('base', 'expert_raw', 'expert_corrected', 'fusion_raw', 'fusion_corrected'):
            assert data[arm].shape == (5, len(mm), 8)
        for draw in range(5):
            mask = data['eval_masks'][draw]
            assert not mask[data['support'][draw]].any()
            truth = mm.class_id.to_numpy()[mask]
            variants = {arm: (data[arm][draw][mask], 'probabilities' if arm.startswith('fusion') else 'logits')
                        for arm in ('base', 'expert_raw', 'expert_corrected', 'fusion_raw', 'fusion_corrected')}
            base_prob = probability_rows(variants['base'][0], truth, 'logits')['prob']
            for arm in ('raw', 'corrected'):
                expert_prob = probability_rows(variants['expert_'+arm][0], truth, 'logits')['prob']
                np.testing.assert_allclose((base_prob+expert_prob)/2, variants['fusion_'+arm][0], atol=1e-12, rtol=0)
            old, new = variants['expert_raw'][0], variants['expert_corrected'][0]
            oc, nc = old-old.mean(1, keepdims=True), new-new.mean(1, keepdims=True)
            on, nn = np.linalg.norm(oc, axis=1, keepdims=True), np.linalg.norm(nc, axis=1, keepdims=True)
            assert (on > 0).all() and (nn > 0).all()
            for label, scores in [('shape_only', nc*on/nn), ('amplitude_only', oc*nn/on)]:
                np.testing.assert_array_equal(scores.argmax(1), (new if label == 'shape_only' else old).argmax(1))
                variants['expert_'+label+'_diagnostic'] = (scores, 'logits')
                prob = probability_rows(scores, truth, 'logits')['prob']
                variants['fusion_'+label+'_diagnostic'] = ((base_prob+prob)/2, 'probabilities')
            for arm, (scores, kind) in variants.items():
                baseline_arm = 'base' if arm == 'base' else arm.split('_')[0]+'_raw'
                diagnostic = arm.endswith('_diagnostic')
                info = dict(family='actual_target', layer='full_model' if arm.startswith('fusion') else 'classifier_continuous_risk',
                            context='not_applicable', training_seed=0, draw_seed=0, draw=draw,
                            arm=arm, baseline_arm=baseline_arm, diagnostic=diagnostic,
                            kernel_intervention='post_result_score_diagnostic' if diagnostic else
                            ('unchanged_base' if arm == 'base' else 'support_geometry_only_fixed_prior'))
                evaluation.add(info, data['rows'][mask], truth, mm.port.to_numpy()[mask],
                               scores, variants[baseline_arm][0], kind, path, path)
        print('TARGET COMPLETE', port, flush=True)


def self_test():
    y = np.arange(8)
    uniform = probability_rows(np.full((8, 8), 1/8), y, 'probabilities')
    zero_logits = probability_rows(np.zeros((8, 8)), y, 'logits')
    for result in (uniform, zero_logits):
        np.testing.assert_allclose(result['nll'], np.log(8), atol=1e-14)
        np.testing.assert_allclose(result['brier'], 7/8, atol=1e-14)
    perfect = probability_rows(np.eye(8), y, 'probabilities')
    np.testing.assert_array_equal(perfect['nll'], np.zeros(8))
    np.testing.assert_array_equal(perfect['brier'], np.zeros(8))
    assert perfect['correct'].all()
    # Fixed zero prior, lambda=8, Kss=1, support label=0 gives alpha=t/9.
    # Kqs 0 -> 1 is a hand-computed, exactly re-solved kernel improvement.
    alpha = (2*np.eye(8)[[0]]-1)/9
    baseline = probability_rows(np.zeros((1, 8)), [0], 'logits')
    improved = probability_rows(alpha, [0], 'logits')
    np.testing.assert_allclose(improved['nll'], np.log1p(7*np.exp(-2/9)))
    np.testing.assert_allclose(improved['brier'], 56/(np.exp(2/9)+7)**2)
    delta = paired_rows(improved, baseline)
    assert (delta[['delta_nll', 'delta_brier']].to_numpy() > 0).all()
    np.testing.assert_allclose(paired_rows(baseline, improved).delta_nll, -delta.delta_nll)
    extreme = probability_rows(np.array([[1000., -1000., 0., 0., 0., 0., 0., 0.]]), [1], 'logits')
    np.testing.assert_allclose(extreme['nll'], 2000.)
    cells = pd.DataFrame(dict(actual_port=['A', 'A', 'B'], class_id=[0, 1, 0], n=[9, 1, 1]))
    for value in VALUES:
        cells[value] = [1., 3., 9.]
    result = reduce_cells(cells)
    np.testing.assert_allclose(result['micro_nll'], 21/11)
    np.testing.assert_allclose(result['classmacro_nll'], 2.4)
    np.testing.assert_allclose(result['portclassmacro_nll'], 5.5)
    return ['uniform8_logits_and_probabilities', 'perfect_probabilities',
            'paired_fixed_prior_kernel_risk_sign', 'extreme_logits_logsumexp',
            'unequal_prevalence_and_missing_classes']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test-only', action='store_true')
    args = parser.parse_args()
    tests = self_test()
    print('SELF TEST PASS', ', '.join(tests), flush=True)
    if args.self_test_only:
        return
    evaluation = Evaluation()
    source_families(evaluation)
    target_family(evaluation)
    summary, draws = evaluation.save()
    protocol = dict(
        inputs='Only cached predictions and metadata; full_teacher_data imported first.',
        layers={'representation_utility': 'Retrieval is evaluated by the main workflow; not duplicated here.',
                'classifier_continuous_risk': 'Source students/completion and target base/expert.',
                'full_model': 'Original target equal-softmax base/expert fusion.'},
        temperature=1, classes=8, nll='Natural log; logits use logsumexp; exact zero true probability gives infinity.',
        brier='SUM of eight squared probability errors per row, then mean over rows.',
        cached_fusion='Already probabilities; never softmax again. Verified against original equal-softmax fusion.',
        primary='summary.csv micro_* pools query row exposures across configurations. These are proper scoring-rule row means at observed exposure prevalence, not calibration evidence.',
        secondary='classmacro_* averages observed classes; portclassmacro_* averages observed classes within each ACTUAL port, then equal ports. These reweighted risks do not retain original prevalence and are not claimed calibrated.',
        accuracy='OA and BA in percent; BA is conventional unweighted recall over present classes, pooled across actual ports. portclass_ba instead equal-weights actual ports then present classes. Missing classes excluded, never zero-filled.',
        episode_mean='episode_mean_summary.csv equally averages configuration/draw metrics; target first pools actual ports at each draw. historical_equal_unit_summary.csv instead averages actual-port/draw units for targets, reproducing historical OA/BA aggregation. Neither is pooled micro risk.',
        pairing='delta_nll/brier = baseline risk minus candidate risk; positive is improvement. delta_oa/ba and delta_trueprob = candidate minus baseline.',
        kernel='Current code reviewed: unified_head_core coefficients/score uses balanced residual ridge lambda8; caches already recomputed KSS/KQS and alpha with fixed prior, support identities and labels. Students change query and support geometry; completion changes support geometry only. Fusion deltas include interaction with fixed base, not a standalone expert kernel risk.',
        source_context='LA/Klang/Santos are geometry contexts, never actual source ports. Actual ports/classes from observations.csv.iloc[query_obs_ids].',
        target_context='context=not_applicable; actual ports/classes from SOURCE/manifest.csv.iloc[rows], eval_masks exclude original support rows.',
        seeds='Source draw_seed=31415 only, draws0..2; historical27182 draw tags duplicate these. Student training seeds31415/27182 remain separate. Completion training_seed=0 means no stochastic fit; target draw_seed=0 means original cached index, not an RNG seed.',
        robustness='Worst actual-port BA, q10 across actual-port BA (linear quantile), highest per-port risk. Per-port pooled summaries and per-draw robustness supplied. Highest class-macro risk also supplied.',
        spread='draw_spread.csv std uses ddof0 and is descriptive, never a CI. context_training_seed isolates draw variation; pooled_configurations mixes contexts/training seeds. Target per_draw first pools actual ports at each draw index.',
        confidence='Candidate-wrong maxprob is descriptive on changing membership, with n. Baseline-wrong delta_trueprob is paired on a fixed baseline error set. Rescue/harm and both-wrong/both-correct are outcome-selected diagnostics; contributions sum to total risk change, not causal effects. Aggregated n counts row exposures.',
        diagnostics='Fixed shape-only/new direction old centered norm and amplitude-only/old direction new norm exactly follow diagnose_completion_score_scale.py. Post-result diagnostic, no source-selected deployability/calibration claim; no tuning.',
        limitations='Repeated historical observations, support draws and source contexts are dependent; source priors/readers held out ships but historical encoder/geometry exposure remains. Actual target evaluation retains original row-level support exclusion; query ships can overlap support. Only three actual target ports; no new generalization confirmation.',
        outputs='Tidy aggregated CSVs; no per-sample JSON. No training, data refits, retrieval duplication, memory or report edits.',
        tests=tests, prediction_units=len(evaluation.inventory), metric_draw_rows=len(draws),
        families=summary.family.unique().tolist())
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(
        status='complete', cached_prediction_units=len(evaluation.inventory),
        data_refits=0, neural_fits=0, tests_passed=tests,
        outputs=sorted(p.name for p in OUT.iterdir() if p.suffix == '.csv')), indent=2), encoding='utf-8')
    columns = ['family', 'arm', 'micro_nll', 'micro_brier', 'micro_delta_nll',
               'micro_delta_brier', 'ba', 'delta_ba', 'portclass_ba', 'worst_actual_port_ba']
    print(summary.loc[~summary.diagnostic, columns].to_string(index=False), flush=True)
    print('OUTPUT', str(OUT), flush=True)


if __name__ == '__main__':
    main()
