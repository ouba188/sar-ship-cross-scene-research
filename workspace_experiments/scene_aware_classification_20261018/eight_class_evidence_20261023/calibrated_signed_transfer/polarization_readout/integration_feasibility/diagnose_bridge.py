"""Read existing score/partition artifacts; write only local feasibility tables."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EIGHT = HERE.parents[2]
WORK = EIGHT.parents[1]
OLD = WORK / 'same_vessel_gallery_20261019'
HELD = OLD / 'held_vessel_gate_20261022'
BASE = EIGHT / 'baseline'
sys.path.insert(0, str(OLD / 'conditional_evidence_pilot'))
from run_pilot import indices


def passes(f):
    return f['product'].map(lambda s: '_'.join([s.split('_')[i] for i in (0, 6, 7)]))


def main():
    old = pd.read_csv(OLD / 'conditional_evidence_pilot/cohort.csv', dtype={'mmsi': str})
    new = pd.read_csv(EIGHT / 'cohort.csv', dtype={'mmsi': str})
    excluded = pd.read_csv(OLD / 'local_condition_module/EXCLUSIONS.csv').obs_id
    assert old.obs_id.is_unique and new.obs_id.is_unique
    old['old_row'] = np.arange(len(old))
    new['new_row'] = np.arange(len(new))
    old['pass_group'], new['pass_group'] = passes(old), passes(new)
    join = old.merge(new, on='obs_id', how='outer', suffixes=('_old', '_new'), indicator=True)
    common = join.loc[join['_merge'].eq('both')].copy()
    fields = ['instance_id', 'mmsi', 'class_id', 'fold', 'product', 'time', 'image_vv', 'image_vh']
    mismatch = {key: int((common[key+'_old'].fillna('') != common[key+'_new'].fillna('')).sum()) for key in fields}
    assert not any(mismatch.values()), mismatch
    join[['obs_id', 'old_row', 'new_row', 'instance_id_old', 'instance_id_new', '_merge']].to_csv(HERE/'row_mapping.csv', index=False)
    mapping = old.set_index('obs_id').old_row.loc[new.obs_id].to_numpy()
    inventory, outer_rows, grid, inner_rows, metrics, memberships = [], [], [], [], [], []
    exposure, bridge_counts = [], []
    saved_partitions = pd.read_csv(HELD/'inner_partitions.csv', dtype={'mmsi':str})
    def inventory_item(path, purpose):
        with np.load(path, allow_pickle=False) as z:
            inventory.append(dict(path=str(path), purpose=purpose, bytes=path.stat().st_size,
                                  shapes=json.dumps({k:list(z[k].shape) for k in z.files})))
    def metric(fold, seed, role, rows, scores):
        y = old.iloc[rows].class_id.to_numpy()
        p = scores.argmax(1)
        metrics.append(dict(fold=fold, seed=seed, role=role, n=len(y),
            OA=100*float((p==y).mean()), BA=100*float(np.mean([(p[y==c]==c).mean() for c in range(8)]))))
        return y,p
    pooled = {s: [] for s in range(3)}
    worst_replay = 0.
    for fold in range(3):
        split = tuple(ii[~old.iloc[ii].obs_id.isin(excluded).to_numpy()] for ii in indices(old, fold))
        with np.load(BASE/f'fold{fold}_inputs.npz') as z:
            nd = {k:z[k] for k in ('train','validation','test','inner_bin','instance_ids','valid')}
        np.testing.assert_array_equal(nd['instance_ids'], new.instance_id.to_numpy())
        for role, oi in zip(('train','validation','test'), split):
            ni = nd[role]
            a,b = set(old.iloc[oi].obs_id), set(new.iloc[ni].obs_id)
            outer_rows.append(dict(fold=fold, role=role, old_n=len(oi), new_n=len(ni), common=len(a&b),
                old_only=len(a-b),new_only=len(b-a), same_order=bool(np.array_equal(old.iloc[oi].obs_id,new.iloc[ni].obs_id)),
                ships=old.iloc[oi].mmsi.nunique(), passes=old.iloc[oi].pass_group.nunique(),
                classes=old.iloc[oi].class_id.nunique(), valid_vh=int(nd['valid'][ni,1].sum()),
                date_min=old.iloc[oi].time.min(), date_max=old.iloc[oi].time.max()))
            assert a == b
        tr, val, test = split
        opath = HELD/f'oof_f{fold}.npz'
        inventory_item(opath,'old source OOF A=base, D=residual; seed axis 0/1/2; rows are OLD positions')
        with np.load(opath) as z:
            oo = dict(z)
        np.testing.assert_array_equal(oo['rows'],tr)
        recorded = saved_partitions.loc[saved_partitions.fold.eq(fold)]
        np.testing.assert_array_equal(recorded['row'], tr)
        np.testing.assert_array_equal(recorded['inner'], oo['inner'])
        assert np.isfinite(oo['base']).all() and np.isfinite(oo['residual']).all()
        query_failures = {'old':0, 'new':0}
        def exposure_record(pipeline, inner, fit_frame, held_frame):
            own_query_in_fit = int(held_frame.mmsi.isin(fit_frame.mmsi).sum())
            own_query_in_selection = int(held_frame.mmsi.isin(old.iloc[val].mmsi).sum())
            test_ships_in_fit = len(set(fit_frame.mmsi)&set(old.iloc[test].mmsi))
            test_ships_in_selection = len(set(old.iloc[val].mmsi)&set(old.iloc[test].mmsi))
            test_obs_in_fit = len(set(fit_frame.obs_id)&set(old.iloc[test].obs_id))
            test_pass_in_fit = len(set(fit_frame.pass_group)&set(old.iloc[test].pass_group))
            query_failures[pipeline] += own_query_in_fit + own_query_in_selection
            exposure.append(dict(fold=fold,pipeline=pipeline,inner=inner,queries=len(held_frame),
                query_rows_with_own_mmsi_in_fit=own_query_in_fit,
                query_rows_with_own_mmsi_in_source_selection=own_query_in_selection,
                outer_test_mmsi_in_fit=test_ships_in_fit,
                outer_test_mmsi_in_source_selection=test_ships_in_selection,
                outer_test_obs_in_fit=test_obs_in_fit,outer_test_pass_in_fit=test_pass_in_fit))
            assert not any((own_query_in_fit,own_query_in_selection,test_ships_in_fit,
                            test_ships_in_selection,test_obs_in_fit,test_pass_in_fit))
        for k in range(3):
            held = tr[oo['inner']==k]
            fit = tr[oo['inner']!=k]
            exposure_record('old',k,old.iloc[fit],old.iloc[held])
            inner_rows.append(dict(fold=fold,pipeline='old',inner=k,fit_n=len(fit),held_n=len(held),
                fit_classes=old.iloc[fit].class_id.nunique(),held_classes=old.iloc[held].class_id.nunique(),
                held_fit_mmsi_overlap=len(set(old.iloc[fit].mmsi)&set(old.iloc[held].mmsi)),
                held_fit_pass_overlap=len(set(old.iloc[fit].pass_group)&set(old.iloc[held].pass_group))))
            for j in range(4):
                with np.load(BASE/'models'/f'fold{fold}_inner{j}.npz') as z:
                    nf,nh = z['fit'],z['held']
                a,b = set(old.iloc[held].obs_id), set(new.iloc[nh].obs_id)
                grid.append(dict(fold=fold,old_inner=k,new_inner=j,old_held_n=len(a),new_held_n=len(b),
                    held_intersection=len(a&b),same_held=a==b,
                    old_held_rows_in_new_fit=len(a&set(new.iloc[nf].obs_id)),
                    old_held_ships_in_new_fit=len(set(old.iloc[held].mmsi)&set(new.iloc[nf].mmsi)),
                    new_held_rows_in_old_fit=len(b&set(old.iloc[fit].obs_id)),
                    new_held_ships_in_old_fit=len(set(new.iloc[nh].mmsi)&set(old.iloc[fit].mmsi))))
        for j in range(4):
            with np.load(BASE/'models'/f'fold{fold}_inner{j}.npz') as z:
                nf,nh = z['fit'],z['held']
            np.testing.assert_array_equal(nh,nd['train'][nd['inner_bin'][nd['train']]==j])
            exposure_record('new',j,new.iloc[nf],new.iloc[nh])
            inner_rows.append(dict(fold=fold,pipeline='new',inner=j,fit_n=len(nf),held_n=len(nh),
                fit_classes=new.iloc[nf].class_id.nunique(),held_classes=new.iloc[nh].class_id.nunique(),
                held_fit_mmsi_overlap=len(set(new.iloc[nf].mmsi)&set(new.iloc[nh].mmsi)),
                held_fit_pass_overlap=len(set(new.iloc[nf].pass_group)&set(new.iloc[nh].pass_group))))
        new_lookup = new.set_index('obs_id').new_row
        for oi,k in zip(tr,oo['inner']):
            ni = int(new_lookup.loc[old.iloc[oi].obs_id])
            memberships.append(dict(fold=fold,obs_id=int(old.iloc[oi].obs_id),instance_id=old.iloc[oi].instance_id,
                old_row=int(oi),new_row=ni,old_inner=int(k),new_inner=int(nd['inner_bin'][ni])))
        with np.load(BASE/'models'/f'fold{fold}_outer.npz') as z:
            np.testing.assert_array_equal(z['fit'],nd['train'])
        bridge_counts.append(dict(fold=fold,source_rows=len(tr),old_oof_rows=len(oo['rows']),
            new_oof_rows=sum(r['held_n'] for r in inner_rows if r['fold']==fold and r['pipeline']=='new'),
            paired_source_rows=len(tr),old_query_identity_exposure=query_failures['old'],
            new_query_identity_exposure=query_failures['new'],validation_rows=len(val),test_rows=len(test)))
        for seed in range(3):
            path = OLD/'geometry_evidence_gate_20261021'/f'inputs_f{fold}_s{seed}.npz'
            inventory_item(path,'old frozen ALL rows A=base,D=residual; use base+residual, NOT content or new A')
            with np.load(path) as z:
                full = z['base']+z['residual']
            assert full.shape == (len(old),8) and np.isfinite(full).all()
            reference = OLD/'neighborhood_level_split_20261021'/f'scores_f{fold}_separated_s{seed}.npz'
            inventory_item(reference,'old frozen TEST reference; scores already A+D')
            with np.load(reference) as z:
                np.testing.assert_array_equal(z['rows'],test)
                error = float(np.abs(full[test]-z['scores']).max())
            assert error < 2e-5
            worst_replay=max(worst_replay,error)
            for role,rows in zip(('train_in_sample','validation','test'),split):
                yp=metric(fold,seed,role,rows,full[rows])
                if role=='test': pooled[seed].append(yp)
            metric(fold,seed,'train_oof',tr,oo['base'][seed]+oo['residual'][seed])
    for seed,parts in pooled.items():
        y=np.concatenate([v[0] for v in parts]);p=np.concatenate([v[1] for v in parts])
        metrics.append(dict(fold='pooled',seed=seed,role='test',n=len(y),OA=100*float((p==y).mean()),
            BA=100*float(np.mean([(p[y==c]==c).mean() for c in range(8)]))))
    for name,rows in [('score_inventory',inventory),('outer_partition_counts',outer_rows),
                      ('inner_crosswalk_counts',grid),('inner_coverage',inner_rows),
                      ('old_score_metrics',metrics),('source_inner_membership',memberships)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    pd.DataFrame(exposure).to_csv(HERE/'identity_exposure_counts.csv',index=False)
    pd.DataFrame(bridge_counts).to_csv(HERE/'bridge_counts.csv',index=False)
    cache_path=EIGHT/'calibrated_signed_transfer/signed_features.npz'
    with np.load(cache_path) as z:
        np.testing.assert_array_equal(z['instance_ids'],new.instance_id.to_numpy())
        moments=z['moments_pol']
        reader_names=z['reader_names'].tolist()
        shape=list(moments.shape)
        assert np.isfinite(moments).all()
    with np.load(HELD/'inner_f0_k0_normalization.npz') as z:
        old_coef_shape=list(z['ridge_coef'].shape)
    summary=dict(status='complete_read_only',old_cohort=len(old),new_cohort=len(new),common_rows=len(common),
        old_only_obs=join.loc[join['_merge'].eq('left_only'),'obs_id'].tolist(),
        new_only_obs=join.loc[join['_merge'].eq('right_only'),'obs_id'].tolist(),metadata_mismatch=mismatch,
        shifted_row_positions=int((mapping!=np.arange(len(new))).sum()),
        outer_partitions_exact=True,old_inner_groups=3,new_inner_groups=4,
        same_inner_held_sets=sum(v['same_held'] for v in grid),tested_inner_pairs=len(grid),
        independent_branch_oof_pairing_legal=True,paired_source_fold_exposures=sum(r['source_rows'] for r in bridge_counts),
        query_identity_exposure_failures=sum(v['query_rows_with_own_mmsi_in_fit']+v['query_rows_with_own_mmsi_in_source_selection'] for v in exposure),
        outer_test_identity_exposure_failures=sum(v['outer_test_mmsi_in_fit']+v['outer_test_mmsi_in_source_selection'] for v in exposure),
        old_score_reference_max_error=worst_replay,old_ridge_coef_shape=old_coef_shape,
        signed_reader_names=reader_names,moments_pol_shape=shape,
        new_fits=0,new_extractions=0,parent_writes=0,
        note='Independent branch cross-fitting is legal: join fold+instance, use EACH branch own held assignment, not same inner number. Full stack is identity-OOF but not pass-OOF; source calibration remains a supervised second stage, not nested calibration OOF.')
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(pd.DataFrame(outer_rows).to_string(index=False))
    print(pd.DataFrame(metrics).query("fold == 'pooled'").to_string(index=False))


if __name__=='__main__':
    main()
