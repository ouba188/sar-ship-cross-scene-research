"""Cached calibration utility diagnosis; no model fitting or checkpoint search."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from run_training import assessment


def main():
    p = pd.read_csv(PARENT/'calibration/predictions.csv', dtype={'mmsi': str})
    curves = pd.read_csv(PARENT/'calibration/selection_curves.csv')
    selected = pd.read_csv(PARENT/'calibration/selections.csv')
    coverage = pd.read_csv(PARENT.parent/'baseline/split_coverage.csv')
    coverage = coverage[coverage.inner.eq(-1)]
    names = p[['class_id', 'class_name']].drop_duplicates().set_index('class_id').class_name.to_dict()
    assert len(selected) == 9 and p.arm.eq('score_only').all()
    np.testing.assert_array_equal(p[[f'b{c}' for c in range(8)]], p[[f'a{c}' for c in range(8)]])
    np.testing.assert_array_equal(p.prediction, p[[f's{c}' for c in range(8)]].to_numpy().argmax(1))
    metrics, classes, transitions, counts, ports, scope = [], [], [], [], [], []
    for row in coverage.itertuples():
        for c in range(8):
            n = int(getattr(row, f'class_{c}'))
            counts.append(dict(fold=row.fold, role=row.role, class_id=c, class_name=names[c], n=n,
                N=row.n, prevalence=n/row.n, joint_class_weight=.5*n/row.n+1/16,
                one_correct_J_pp=100*(.5/row.n+1/(16*n))))
    pd.DataFrame(counts).to_csv(HERE/'source_role_class_weights.csv', index=False)
    for fold in [-1, 0, 1, 2]:
        data = p if fold == -1 else p[p.fold.eq(fold)]
        for (role, seed), g in data.groupby(['role', 'seed']):
            y = g.class_id.to_numpy()
            a = assessment(y, g[[f'a{c}' for c in range(8)]].to_numpy())
            b = assessment(y, g[[f's{c}' for c in range(8)]].to_numpy())
            meta = dict(fold=fold, role=role, seed=seed)
            metrics.append(dict(**meta, n=len(g), unique_observations=g.instance_id.nunique(),
                **{f'A_{k}': v for k, v in a.items()}, **{f'B_{k}': v for k, v in b.items()},
                **{f'delta_{k}': b[k]-a[k] for k in a if k != 'n'},
                delta_J_pp=50*((b['OA']-a['OA'])/100+(b['BA']-a['BA'])/100)))
            for c, h in g.groupby('class_id'):
                ac, bc = h.base_prediction.eq(c), h.prediction.eq(c)
                dr = 100*(bc.mean()-ac.mean())
                classes.append(dict(**meta, class_id=c, class_name=names[c], n=len(h),
                    unique_observations=h.instance_id.nunique(), mmsi=h.mmsi.nunique(), passes=h.pass_group.nunique(),
                    prevalence=len(h)/len(g), A_correct=int(ac.sum()), B_correct=int(bc.sum()),
                    A_recall=100*ac.mean(), B_recall=100*bc.mean(), delta_recall_pp=dr,
                    rescued=int((bc & ~ac).sum()), harmed=int((ac & ~bc).sum()), net=int(bc.sum()-ac.sum()),
                    A_predicted=int(g.base_prediction.eq(c).sum()), B_predicted=int(g.prediction.eq(c).sum()),
                    OA_contribution_pp=len(h)/len(g)*dr, BA_contribution_pp=dr/8,
                    J_contribution_pp=.5*(len(h)/len(g)+1/8)*dr))
            for (truth, ap, bp), h in g.groupby(['class_id', 'base_prediction', 'prediction']):
                transitions.append(dict(**meta, truth=truth, A_prediction=ap, B_prediction=bp, n=len(h)))
            if seed == 17:
                scope.append(dict(fold=fold, role=role, n=len(g), unique_observations=g.instance_id.nunique(),
                    mmsi=g.mmsi.nunique(), passes=g.pass_group.nunique(), ports=g.port.nunique()))
            if fold != -1:
                for (c, port), h in g.groupby(['class_id', 'port']):
                    ports.append(dict(**meta, class_id=c, class_name=names[c], port=port, n=len(h),
                        A_correct=int(h.base_prediction.eq(c).sum()), B_correct=int(h.prediction.eq(c).sum()),
                        net=int(h.net.sum())))
    mf, cf = pd.DataFrame(metrics), pd.DataFrame(classes)
    mf.to_csv(HERE/'role_fold_metrics.csv', index=False)
    cf.to_csv(HERE/'class_utility.csv', index=False)
    pd.DataFrame(transitions).to_csv(HERE/'prediction_transitions.csv', index=False)
    pd.DataFrame(ports).to_csv(HERE/'class_port_effects.csv', index=False)
    pd.DataFrame(scope).to_csv(HERE/'role_scope.csv', index=False)
    audited, traces = [], []
    for row in selected.itertuples():
        trace = curves[curves.fold.eq(row.fold) & curves.seed.eq(row.seed)].sort_values('epoch').copy()
        zero = trace[trace.epoch.eq(0)].iloc[0]
        chosen = trace[trace.epoch.eq(row.epoch)].iloc[0]
        # Audit the recorded source choice only; never score other epochs on test.
        earliest = int(trace[np.isclose(trace.J, trace.J.max(), rtol=0, atol=1e-12)].epoch.iloc[0])
        assert earliest == row.epoch
        observed = mf[mf.fold.eq(row.fold) & mf.seed.eq(row.seed) & mf.role.eq('validation')].iloc[0]
        assert abs(chosen.J-row.validation_J) < 1e-12
        assert abs(observed.B_J-chosen.J) < 1e-12
        trace['delta_OA_pp'], trace['delta_BA_pp'] = trace.OA-zero.OA, trace.BA-zero.BA
        trace['delta_J_pp'] = 100*(trace.J-zero.J)
        trace['selected'] = trace.epoch.eq(row.epoch)
        trace['OA_up_BA_down'] = trace.delta_OA_pp.gt(0) & trace.delta_BA_pp.lt(0)
        traces.append(trace)
        audited.append(dict(fold=row.fold, seed=row.seed, epoch=row.epoch,
            selected_delta_OA_pp=chosen.OA-zero.OA, selected_delta_BA_pp=chosen.BA-zero.BA,
            selected_delta_J_pp=100*(chosen.J-zero.J),
            baseline_NLL=zero.NLL, selected_NLL=chosen.NLL,
            baseline_balanced_NLL=zero.balanced_NLL, selected_balanced_NLL=chosen.balanced_NLL,
            earliest_max_J_epoch=earliest, source_replay_J_error=observed.B_J-chosen.J,
            checked_epochs=len(trace), OA_up_BA_down_epochs=int(trace.OA_up_BA_down.sum()),
            J_up_BA_down_epochs=int((trace.delta_J_pp.gt(0) & trace.delta_BA_pp.lt(0)).sum())))
    pd.DataFrame(audited).to_csv(HERE/'selection_audit.csv', index=False)
    pd.concat(traces, ignore_index=True).to_csv(HERE/'validation_curve_tradeoffs.csv', index=False)
    decomp, detail = [], []
    for (fold, seed), group in cf.groupby(['fold', 'seed']):
        v = group[group.role.eq('validation')].set_index('class_id').sort_index()
        t = group[group.role.eq('test')].set_index('class_id').sort_index()
        pv, pt = v.prevalence.to_numpy(), t.prevalence.to_numpy()
        rv, rt = v.delta_recall_pp.to_numpy(), t.delta_recall_pp.to_numpy()
        composition = (pt-pv)*rv
        within = pt*(rt-rv)
        ba_within = (rt-rv)/8
        assert abs((pt*rt).sum()-(pv*rv).sum()-composition.sum()-within.sum()) < 1e-10
        decomp.append(dict(fold=fold, seed=seed, validation_delta_OA_pp=(pv*rv).sum(),
            test_delta_OA_pp=(pt*rt).sum(), validation_delta_BA_pp=rv.mean(), test_delta_BA_pp=rt.mean(),
            validation_effect_test_prevalence_OA_pp=(pt*rv).sum(),
            OA_gap_composition_pp=composition.sum(), OA_gap_within_class_pp=within.sum(),
            BA_gap_within_class_pp=ba_within.sum(),
            J_gap_composition_pp=.5*composition.sum(), J_gap_within_class_pp=.5*(within.sum()+ba_within.sum())))
        for c in range(8):
            detail.append(dict(fold=fold, seed=seed, class_id=c, class_name=names[c],
                validation_prevalence=pv[c], test_prevalence=pt[c], validation_delta_recall_pp=rv[c],
                test_delta_recall_pp=rt[c], OA_composition_pp=composition[c],
                OA_within_class_pp=within[c], BA_within_class_pp=ba_within[c]))
    pd.DataFrame(decomp).to_csv(HERE/'validation_test_decomposition.csv', index=False)
    pd.DataFrame(detail).to_csv(HERE/'validation_test_class_decomposition.csv', index=False)
    cf['group'] = np.where(cf.class_id.isin([0, 3]), 'bulk_and_chemical', 'other_six')
    cf.groupby(['fold', 'role', 'seed', 'group'])[['n', 'net', 'OA_contribution_pp', 'BA_contribution_pp',
        'J_contribution_pp']].sum().to_csv(HERE/'majority_minority_tradeoff.csv')
    # Deduplicate validation exposures by weighting each ID once, without choosing a fold.
    sensitivity = []
    for seed, g in p[p.role.eq('validation')].groupby('seed'):
        g = g.copy()
        g['difference'] = g.prediction.eq(g.class_id).astype(int)-g.base_prediction.eq(g.class_id).astype(int)
        objects = g.groupby(['instance_id', 'class_id']).difference.mean().reset_index()
        sensitivity.append(dict(seed=seed, observations=len(objects),
            delta_OA_pp=100*objects.difference.mean(),
            delta_BA_pp=100*objects.groupby('class_id').difference.mean().mean()))
    pd.DataFrame(sensitivity).to_csv(HERE/'validation_object_weighted_sensitivity.csv', index=False)
    receipt = dict(status='complete', training=False, inference=False, reselected=False,
        selected_heads=9, audited_source_selections=True,
        source_selected_BA_down=int(sum(r['selected_delta_BA_pp'] < 0 for r in audited)),
        decomp='source-recall-anchored algebraic standardization; within-class term is not a causal domain-shift estimate',
        fold_minus_one='pooled fold exposures; validation repeats objects, test is mutually exclusive',
        code_review='joint_loss=.5 ordinary CE+.5 macro-class CE; exact_j=.5OA+.5BA, strict greater, epoch0 included')
    (HERE/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(pd.DataFrame(audited).to_string(index=False))
    print(pd.DataFrame(decomp).query('fold == -1').to_string(index=False))
    print(cf.query('fold == -1')[['role','seed','class_id','class_name','n','net','delta_recall_pp']].to_string(index=False))


if __name__ == '__main__':
    main()
