"""Frozen support-bank correction in the original full target-port pipeline."""

from full_teacher_data import BASE, ROOT, SOURCE, PORTS, write
from case_support_completion import COLS, predict, classify, adapt_support
from unified_head_core import score, late, coefficients
import json
import time
import numpy as np
import pandas as pd

OUT = BASE/'support_completion_target_20261012'
CONTROL = BASE/'case_completion_class_control_20261012'
ORIGINAL = BASE/'full_teacher_integration_20261012'


def run_port(port, dose, meta):
    dest=OUT/port
    dest.mkdir(parents=True,exist_ok=True)
    with np.load(ORIGINAL/port/'no_training/predictions.npz') as z:
        old={k:z[k] for k in z.files}
    with np.load(ROOT/'learned_frequency_residual_20261009/tasks/self24ports'/port/'target.npz') as z:
        data={k:z[k] for k in ('rows','support','support_labels','eval_masks','expected_base','expected_full','source_rows')}
    for key in ('rows','support','eval_masks'):
        np.testing.assert_array_equal(old[key],data[key])
    base=np.load(ORIGINAL/port/'base_scores.npy')
    with np.load(ROOT/'frozen_comparison_20261010/targets/self24ports'/port/'policies.npz') as z:
        modes=list(z['modes'])
        historical_base=z['predictions'][:,modes.index('ours_base')]
        historical_fusion=z['predictions'][:,modes.index('softmax_average_all')]
        for key in ('rows','support','eval_masks'):
            np.testing.assert_array_equal(data[key],z[key])
    np.testing.assert_array_equal(base.argmax(2),historical_base)
    with np.load(CONTROL/port/'final_model.npz') as z:
        model={k:z[k] for k in z.files}
    with np.load(BASE/'case_relation_headroom_20261012'/port/'maps.npz') as z:
        fit_obs=z['obs_ids'][:1296][z['multi'][:1296]]
    fitting=meta.iloc[fit_obs]
    assert len(fitting)==1040
    assert not np.intersect1d(fitting.manifest_row,data['rows']).size
    assert np.isin(fitting.manifest_row,data['source_rows']).all()
    assert not (fitting.port==port).any()
    phi,prior=old['phi'],old['prior']
    assert model['image_coef'].shape==(len(COLS),len(COLS))
    raw,corrected,raw_fusion,new_fusion=[],[],[],[]
    diagnostics=[]
    for draw,s in enumerate(data['support']):
        sy=data['support_labels'][draw]
        assert len(s)==64 and not data['eval_masks'][draw][s].any()
        original=score(phi@phi[s].T,prior,s,sy)[0]
        np.testing.assert_allclose(original,old['expert'][draw],atol=1e-10,rtol=0)
        original_fusion=late(base[draw],original)
        np.testing.assert_allclose(original_fusion,old['fusion'][draw],atol=1e-10,rtol=0)
        np.testing.assert_array_equal(original_fusion.argmax(1),historical_fusion[draw])
        start=time.perf_counter()
        # Image-only predictor ignores labels; pass no true query labels anywhere.
        delta=predict(model,phi[s][:,COLS],np.zeros(len(s),dtype=int),'image_history')
        ps=adapt_support(phi[s],delta,dose)
        np.testing.assert_allclose(ps[:,COLS].mean(0),phi[s][:,COLS].mean(0),atol=1e-12,rtol=0)
        unchanged=np.setdiff1d(np.arange(phi.shape[1]),COLS)
        np.testing.assert_array_equal(ps[:,unchanged],phi[s][:,unchanged])
        alpha=coefficients(ps@ps.T,prior[s],sy,8.)
        result=prior+(phi@ps.T)@alpha
        elapsed=time.perf_counter()-start
        # Independent helper parity detects a role/normalization integration mismatch.
        expected=classify(phi[s],phi,prior[s],prior,sy,delta,dose)
        np.testing.assert_allclose(result,expected,atol=1e-10,rtol=0)
        raw.append(original);corrected.append(result)
        raw_fusion.append(original_fusion);new_fusion.append(late(base[draw],result))
        diagnostics.append(dict(draw=draw,expert_baseline_max_error=float(np.max(abs(original-old['expert'][draw]))),
            fusion_baseline_max_error=float(np.max(abs(original_fusion-old['fusion'][draw]))),
            selected_dose=dose,support_shift_rms=float(np.sqrt(np.mean((ps-phi[s])**2))),
            correction_and_readout_seconds=elapsed,
            query_features_unchanged=True,prior_unchanged=True,base_unchanged=True))
        np.savez_compressed(dest/f'support_{draw}.npz',support=s,support_rows=data['rows'][s],
            support_phi=ps,alpha=alpha)
    np.savez_compressed(dest/'predictions.npz',rows=data['rows'],support=data['support'],
        eval_masks=data['eval_masks'],base=base,expert_raw=np.stack(raw),
        expert_corrected=np.stack(corrected),fusion_raw=np.stack(raw_fusion),fusion_corrected=np.stack(new_fusion))
    write(dest/'REPLAY.json',diagnostics)
    write(dest/'FITTING.json',dict(fit_observations=len(fitting),fit_ships=int(fitting.mmsi.nunique()),
        fit_ports=sorted(fitting.port.unique().tolist()),no_target_port_in_completion_fit=True,
        fitted_model=str(CONTROL/port/'final_model.npz'),source_dose_file=str(CONTROL/'FROZEN_CHOICE.json')))
    print('FROZEN',port,len(data['rows']),flush=True)


def report():
    manifest=pd.read_csv(SOURCE/'manifest.csv',usecols=['class_id','mmsi','port'])
    meta=pd.read_csv(BASE/'observations.csv')
    records,scope=[],[]
    for port in PORTS:
        with np.load(OUT/port/'predictions.npz') as z:
            data={k:z[k] for k in z.files}
        rows=data['rows'];y=manifest.iloc[rows].class_id.to_numpy()
        ships=manifest.iloc[rows].mmsi.to_numpy()
        with np.load(BASE/'case_relation_headroom_20261012'/port/'maps.npz') as z:
            fit_obs=z['obs_ids'][:1296][z['multi'][:1296]]
        trainships=meta.iloc[fit_obs].mmsi.to_numpy()
        for draw,s in enumerate(data['support']):
            mask=data['eval_masks'][draw]
            for arm in ('base','expert_raw','expert_corrected','fusion_raw','fusion_corrected'):
                pred=data[arm][draw].argmax(1)[mask];truth=y[mask]
                records.append(dict(port=port,draw=draw,arm=arm,n=int(mask.sum()),
                    oa=float(100*np.mean(pred==truth)),
                    ba=float(100*np.mean([np.mean(pred[truth==c]==c) for c in np.unique(truth)]))))
            scope.append(dict(port=port,draw=draw,target_rows=len(rows),query_rows=int(mask.sum()),
                support_unique_ships=int(pd.Series(ships[s]).nunique()),query_unique_ships=int(pd.Series(ships[mask]).nunique()),
                support_missing_identity_rows=int(pd.isna(ships[s]).sum()),query_missing_identity_rows=int(pd.isna(ships[mask]).sum()),
                query_rows_sharing_support_ship=int(np.isin(ships[mask],ships[s]).sum()),
                query_rows_sharing_completion_fit_ship=int(np.isin(ships[mask],trainships).sum()),
                query_classes=len(np.unique(y[mask]))))
    frame=pd.DataFrame(records);frame.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(scope).to_csv(OUT/'scope.csv',index=False)
    summary=frame.groupby('arm')[['oa','ba']].mean()
    summary.to_csv(OUT/'summary.csv')
    frame.groupby(['port','arm'])[['oa','ba']].mean().to_csv(OUT/'ports.csv')
    write(OUT/'COMPLETE.json',dict(target_ports=list(PORTS),support_draws=5,support_labels=64,
        neural_fits=0,linear_fits=0,frozen_prior_and_backbones=True,original_fusion_ratio=.5,
        final_fusion_evaluated=True,other_datasets_evaluated=False,target_parameter_selection=False))
    print(summary.to_string(),flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    dose=float(json.loads((CONTROL/'FROZEN_CHOICE.json').read_text())['dose'])
    assert dose==1.
    write(OUT/'PROTOCOL.json',dict(
        question='Does frozen source-class support completion improve original target expert AND final fusion?',
        model='Reuse existing other-vessel same-class residual predictor; no refits or hyperparameter changes',
        dose=dose,dose_selection='Previous source-vessel-fold BA only, before this target evaluation',
        scope='Los Angeles, Port Klang, Santos actual target ports, five original 64-label draws',
        baseline='Original no-training full expert and unchanged base; not prior teacher/task continuation',
        fixed='Queries, base, source prior, backbones, non-R18 coordinates, ridge8, support labels and evaluation masks',
        replay_reference='frozen_comparison_20261010 policies ours_base/softmax_average_all; target.npz expected fields are older pipeline',
        operation='Only support R18-dependent 576 coords corrected; original support centroid/energy restored; recompute both KSS and KQS',
        fusion='Original equal softmax fusion; no temperature, ratio, adaptive routing or 5% budget selection',
        prediction_inputs='Single SAR and existing acquisition fields only; no new AIS/MMSI/history or query labels',
        limitation='Repeated historical target data and original row-level support exclusion; not fresh validation',
        phases='Freeze all three ports before query-label evaluation; preserve old artifacts'))
    meta=pd.read_csv(BASE/'observations.csv')
    for port in PORTS:
        run_port(port,dose,meta)
    write(OUT/'PREDICTIONS_FROZEN.json',dict(ports=list(PORTS),all_targets_complete=True))
    report()


if __name__=='__main__':
    main()
