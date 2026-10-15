"""Check class exclusion, vessel isolation, gate wiring and metric denominators."""

import json
import pickle
import numpy as np
import pandas as pd
from scipy.special import expit
from run import OUT, METHODS, GROUPS, core


def main():
    frame=pd.read_csv(OUT/'input_replay/source_manifest.csv',dtype={'identity':str})
    source=pd.read_csv(OUT/'source_contexts.csv')
    plans=json.loads((OUT/'SOURCE_PLAN.json').read_text(encoding='utf-8'))
    gate_ids={r:set() for r in GROUPS}
    for row in source.itertuples():
        z=np.load(OUT/row.path); q=z['query']; s=z['supports']
        np.testing.assert_array_equal(z['labels'],frame.class_id.to_numpy()[q])
        assert row.outer not in z['labels'] and row.outer not in z['support_labels']
        assert row.pseudo not in z['support_labels'] and row.pseudo in z['labels']
        np.testing.assert_array_equal(z['classes'],np.setdiff1d(np.arange(8),[row.outer,row.pseudo]))
        assert len(s)==64 and len(set(frame.identity.iloc[s]))==64
        assert not set(frame.identity.iloc[s])&set(frame.identity.iloc[q])
        assert set(frame.port.iloc[np.r_[s,q]]).issubset(GROUPS[row.role])
        if row.arm=='missing':assert row.missing not in z['support_labels']
        gate_ids[row.role].update(frame.identity.iloc[q])
    assert not gate_ids['train']&gate_ids['calibration']
    with (OUT/'frozen_gates.pkl').open('rb') as h:policies=pickle.load(h)
    contexts=pd.read_csv(OUT/'target_contexts.csv'); error=0.
    for row in contexts.itertuples():
        z=np.load(OUT/row.path); manifest=pd.read_csv(OUT/(row.task.replace('/','_')+'_manifest.csv'),dtype={'identity':str}).set_index('local_row')
        support_ids=set(manifest.loc[z['supports'],'identity'])
        query_ids=set(manifest.loc[z['query'],'identity'])
        assert len(support_ids)==64 and not support_ids&query_ids
        assert not set(frame.identity)&(support_ids|query_ids)
        assert row.outer not in z['support_labels'] and row.outer not in z['pred']
        counts=np.bincount(z['support_labels'],minlength=8)
        np.testing.assert_allclose(np.expm1(z['inputs'][:,3]),counts[z['pred']],atol=1e-10)
        p=policies[row.outer]
        direct=expit(((z['inputs']-p['scaler'].mean_)/p['scaler'].scale_)@p['gate'].coef_[0]+p['gate'].intercept_[0])
        error=max(error,float(abs(direct-z['score'][:,-1]).max()))
        np.testing.assert_allclose(direct,z['score'][:,-1],atol=1e-12)
        np.testing.assert_array_equal(z['source_thresholds'],p['thresholds'])
    sm=pd.read_csv(OUT/'source_metrics.csv')
    calibration_frr=sm[sm.role=='calibration'].groupby(['outer','method']).known_false_reject.mean()
    assert calibration_frr.max()<=5.+1e-8
    summary=pd.read_csv(OUT/'summary.csv'); bootstrap=pd.read_csv(OUT/'query_ship_bootstrap.csv')
    for row in bootstrap.itertuples():
        expected=summary[(summary.task==row.task)&(summary.method==row.method)&
                         (summary.arm=='random')&(summary.calibration==row.calibration)]
        col='utility_05' if row.prevalence==.05 else 'utility_20'
        np.testing.assert_allclose(row.utility,expected[col].iloc[0],atol=1e-10)
    core.write(OUT/'VERIFIED.json',dict(source_contexts=len(source),target_contexts=len(contexts),
        gate_training_calibration_query_ship_overlap=0,source_target_ship_overlap=0,
        max_gate_direct_error=error,max_source_calibration_macro_frr=float(calibration_frr.max()),
        query_bootstrap_aggregate_matches=True))
    print('VERIFIED',len(source),'source contexts,',len(contexts),'target contexts; gate error',error)


if __name__=='__main__':main()
