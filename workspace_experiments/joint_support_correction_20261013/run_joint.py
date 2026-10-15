"""Frozen B16 factorial and new-support follow-up, without backbone training."""

import argparse
import json
from pathlib import Path
import sys
import time

import joint_core as j
from joint_core import np, c, utility
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
OLD = ROOT/'classwise_utility_adaptation_20261012'
BUDGET = ROOT/'frozen_candidate_coverage_20261012/budget'
sys.path.insert(0, str(BUDGET))
from strong_external import CachedStrongFeatures
sys.path.insert(0, str(ROOT/'residual_strength_path_20261013'))
from run_path import measures

DATASETS = ('f0921', 'oss2')
ARMS = ('raw', 'initial', 'light', 'combined', 'initial_guard')
KEY = ['dataset', 'cohort', 'draw']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def prepare():
    units, datasets = [], []
    for di, ds in enumerate(DATASETS):
        directory = BUDGET/'strong_external'/ds/ds
        data = load(c.TASKS/ds/'target.npz', ['rows', 'support'])
        np.testing.assert_array_equal(data['rows'], np.arange(len(data['rows'])))
        pool = np.unique(data['support'])
        q = load(directory/'0_16.npz', ['query_ids'])['query_ids']
        assert not np.intersect1d(pool, q).size
        support_sets = set()
        for cohort in ('legacy', 'new'):
            for draw in range(5):
                seed = None
                if cohort == 'legacy':
                    z = load(directory/f'{draw}_16.npz', ['support_ids', 'query_ids'])
                    np.testing.assert_array_equal(q, z['query_ids'])
                    s = z['support_ids']
                else:
                    seed = 2026101300 + 100*di + draw
                    s = np.random.default_rng(seed).choice(pool, 16, replace=False)
                key = tuple(sorted(s.tolist()))
                assert key not in support_sets
                support_sets.add(key)
                dest = OUT/'episodes'/ds/f'{cohort}_{draw}.npz'
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, support_ids=s, query_ids=q)
                units.append(dict(dataset=ds, cohort=cohort, draw=draw, budget=16,
                    path=str(dest), seed=seed, query_n=len(q)))
        datasets.append(dict(dataset=ds, pool_n=len(pool), query_n=len(q),
            manifest=read(directory/'PREDICTIONS_FROZEN.json')['target_manifest']))
    assert read(ROOT/'residual_strength_path_20261013/FROZEN_ALPHA.json')['alphas']['joint'] == j.ALPHA
    assert read(OLD/'FROZEN_CHOICE.json')['taus']['joint'] == j.TAU
    write('FROZEN_PROTOCOL.json', dict(units=units, datasets=datasets, alpha=j.ALPHA,
        tau=j.TAU, dose=1., arms=ARMS, query_labels_used=False, no_parameter_search=True))
    print('EPISODES FROZEN', len(units), flush=True)


def support_metadata(manifest, union):
    columns = pd.read_csv(manifest, nrows=0).columns
    identity = next(k for k in ('mmsi', 'mmsi_or_ship_id') if k in columns)
    keep = set(union.tolist())
    table = pd.read_csv(manifest, usecols=['class_id', identity], dtype=str,
        keep_default_na=False, skiprows=lambda row: row > 0 and row-1 not in keep)
    assert len(table) == len(union)
    table.index = union
    values = table[identity].str.strip().str.replace(r'\.0$', '', regex=True)
    known = values.str.fullmatch(r'[0-9]{9}') & values.ne('000000000')
    table['group'] = np.where(known, 'ship:'+values, 'row:'+pd.Series(union, index=union).astype(str))
    return table


def predict():
    started = time.perf_counter()
    protocol = read(OUT/'FROZEN_PROTOCOL.json')
    times, gate_rows, replay_rows, outputs = [], [], [], []
    for ds in DATASETS:
        directory = BUDGET/'strong_external'/ds/ds
        folder = c.TASKS/ds
        dataset = next(d for d in protocol['datasets'] if d['dataset'] == ds)
        units = [u for u in protocol['units'] if u['dataset'] == ds]
        episodes = [load(u['path']) for u in units]
        union = np.unique(np.concatenate([z['support_ids'] for z in episodes]))
        smeta = support_metadata(dataset['manifest'], union)
        data = load(folder/'target.npz', ['rows', 'phi', 'mean', 'conditions', 'prior'])
        embedding = load(folder/'selective_embedding.npz', ['rows', 'embedding'])
        np.testing.assert_array_equal(embedding['rows'], data['rows'])
        np.testing.assert_array_equal(data['rows'], np.arange(len(data['rows'])))
        feat = CachedStrongFeatures(ds)
        np.testing.assert_array_equal(feat.target_rows, data['rows'])
        model = load(directory/'initial.npz')
        eta = read(folder/'PREPARED.json')['eta']
        for unit, episode in zip(units, episodes):
            key = {k: unit[k] for k in KEY}
            s, q = episode['support_ids'], episode['query_ids']
            sy = smeta.loc[s, 'class_id'].to_numpy(int)
            groups = smeta.loc[s, 'group'].to_numpy(str)
            raw, prior = feat.target(s)
            start = time.perf_counter()
            kb = c.base_block(data, embedding['embedding'], np.empty(0, int), s, eta)
            kernel_seconds = time.perf_counter()-start
            start = time.perf_counter()
            if unit['cohort'] == 'legacy':
                oldfold = read(OLD/'target_oof'/ds/ds/f"{unit['draw']}_16_folds.json")
                parts, notes = oldfold['parts'], oldfold['warnings']
                np.testing.assert_array_equal(groups, oldfold['groups'])
            else:
                parts, notes = j.make_parts(sy, groups)
            split_seconds = time.perf_counter()-start
            fitted, timing, optim = j.fit_pair(raw, prior, kb, data['prior'][s], sy, groups, model, parts)
            timing.update(support_base_kernel_seconds=kernel_seconds, fold_setup_seconds=split_seconds)
            legacy = None
            errors = {}
            if unit['cohort'] == 'legacy':
                legacy = load(directory/f"{unit['draw']}_16.npz",
                    ['base', 'expert_raw', 'expert_initial', 'fusion_raw', 'fusion_initial', 'support_labels'])
                np.testing.assert_array_equal(sy, legacy['support_labels'])
                oo = load(OLD/'target_oof'/ds/ds/f"{unit['draw']}_16.npz")
                for ours, old in [('oof_prior','prior'), ('oof_raw','residual'), ('oof_base','base_probability'),
                                  ('oof_labels','labels'), ('gate_light','gate_joint')]:
                    err = float(np.abs(fitted[ours]-oo[old]).max())
                    np.testing.assert_allclose(fitted[ours], oo[old], atol=1e-9, rtol=0)
                    errors[ours] = err
            dest = OUT/'weights'/ds/f"{unit['cohort']}_{unit['draw']}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **fitted, support_ids=s, support_labels=sy)
            write(str(dest.relative_to(OUT)).replace('.npz', '_folds.json'),
                  dict(parts=parts, groups=groups.tolist(), warnings=notes, optim=optim))
            for arm, residual in [('light','oof_raw'), ('combined','oof_initial')]:
                gate = fitted['gate_'+arm]
                risk0 = utility.loss_grad(np.ones(8), fitted['oof_prior'], fitted[residual],
                    fitted['oof_base'], fitted['oof_labels'], 0.)[0]
                risk1 = utility.loss_grad(1+j.ALPHA*(gate-1), fitted['oof_prior'], fitted[residual],
                    fitted['oof_base'], fitted['oof_labels'], 0.)[0]
                for cls in range(8):
                    gate_rows.append(dict(**key, arm=arm, class_id=cls,
                        support_n=int((sy==cls).sum()), gamma=float(gate[cls]),
                        effective_gamma=float(1+j.ALPHA*(gate[cls]-1)), oof_risk_gain=risk0-risk1))
            probabilities = {arm: np.empty((len(q), 8)) for arm in ARMS}
            base_scores = np.empty((len(q), 8))
            joined = np.concatenate([fitted[k] for k in ('raw_weight','initial_weight','guard_weight')], axis=1)
            predict_start = time.perf_counter()
            for lo in range(0, len(q), 2048):
                ids = q[lo:lo+2048]
                xq, pq = feat.target(ids)
                bkernel = c.base_block(data, embedding['embedding'], ids, s, eta)[len(s):]
                base = data['prior'][ids]+bkernel@fitted['base_coef']
                residuals = (xq@joined).reshape(len(ids), 3, 8)
                pp = j.scores(base, pq, residuals[:,0], residuals[:,1], residuals[:,2],
                    fitted['gate_light'], fitted['gate_combined'])
                for arm in ARMS:
                    probabilities[arm][lo:lo+len(ids)] = pp[arm]
                base_scores[lo:lo+len(ids)] = base
                if legacy is not None:
                    for name, actual in [('base',base), ('expert_raw',pq+residuals[:,0]),
                        ('expert_initial',pq+residuals[:,1]), ('fusion_raw',pp['raw']),
                        ('fusion_initial',pp['initial'])]:
                        expected = legacy[name][lo:lo+len(ids)]
                        errors[name] = max(errors.get(name,0.), float(abs(actual-expected).max()))
                        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
                        np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
            timing['query_all_arms_seconds'] = time.perf_counter()-predict_start
            if legacy is not None:
                oldlight = load(ROOT/'frozen_scope_closeout_20261013/predictions'/ds/ds/f"{unit['draw']}_16.npz", ['joint25'])['joint25']
                oldguard = load(Path('C:/Users/25791/AppData/Local/Temp/SAR_prior_protection_20261012')/ds/f"{unit['draw']}_16.npz", ['initial_guard'])['initial_guard']
                oldguard = .5*(j.softmax(legacy['base'],axis=1)+j.softmax(oldguard,axis=1))
                for arm, expected in [('light', oldlight), ('initial_guard', oldguard)]:
                    errors[arm] = float(abs(probabilities[arm]-expected).max())
                    np.testing.assert_allclose(probabilities[arm], expected, atol=1e-9, rtol=0)
                    np.testing.assert_array_equal(probabilities[arm].argmax(1), expected.argmax(1))
                replay_rows.append(dict(**key, **errors))
            predpath = OUT/'predictions'/ds/f"{unit['cohort']}_{unit['draw']}.npz"
            predpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(predpath, **probabilities, base=base_scores, support_ids=s, query_ids=q)
            times.append(dict(**key, **timing))
            outputs.append(dict(**unit, prediction_path=str(predpath), weights_path=str(dest)))
            print('PREDICTED', ds, unit['cohort'], unit['draw'], 'replay', max(errors.values(), default=0.), flush=True)
    pd.DataFrame(times).to_csv(OUT/'timings.csv', index=False)
    pd.DataFrame(gate_rows).to_csv(OUT/'gates.csv', index=False)
    pd.DataFrame(replay_rows).to_csv(OUT/'replays.csv', index=False)
    write('PREDICTIONS_FROZEN.json', dict(units=outputs, datasets=protocol['datasets'],
        seconds=time.perf_counter()-started, query_labels_used=False, new_neural_training=0,
        source_mapper_fits=0, target_gate_fits=2*len(outputs), source_choice_unchanged=True))


def evaluate():
    frozen = read(OUT/'PREDICTIONS_FROZEN.json')
    records, classes, flows = [], [], []
    label_tables = {d['dataset']: pd.read_csv(d['manifest'],usecols=['class_id']).class_id.to_numpy(int)
                    for d in frozen['datasets']}
    for unit in frozen['units']:
        key = {k:unit[k] for k in KEY}
        z = load(unit['prediction_path'])
        y = label_tables[unit['dataset']][z['query_ids']]
        w = utility.class_weights(y)*100.
        raw, initial, combined = (z[a].argmax(1)==y for a in ('raw','initial','combined'))
        initial_rescue, initial_harm = (~raw)&initial, raw&(~initial)
        flow = dict(initial_rescue_kept=float(w@(initial_rescue&combined)),
            initial_rescue_lost=float(w@(initial_rescue&~combined)),
            novel_rescue=float(w@(~raw&~initial&combined)),
            initial_harm_kept=float(w@(initial_harm&~combined)),
            initial_harm_avoided=float(w@(initial_harm&combined)),
            novel_harm=float(w@(raw&initial&~combined)))
        np.testing.assert_allclose(flow['initial_harm_avoided']+flow['novel_rescue']-
            flow['initial_rescue_lost']-flow['novel_harm'], w@(combined.astype(float)-initial), atol=1e-10)
        flows.append(dict(**key, **flow))
        sy = load(unit['weights_path'],['support_labels'])['support_labels']
        for arm in ARMS:
            mm, cc = measures(y,z[arm],z['raw'])
            records.append(dict(**key,arm=arm,**mm))
            losses = np.square(z[arm]-np.eye(8)[y]).sum(1)
            for v in cc:
                classes.append(dict(**key,arm=arm,support_n=int((sy==v['class_id']).sum()),
                    brier=float(losses[y==v['class_id']].mean()),**v))
    f = pd.DataFrame(records)
    f.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'classes.csv',index=False)
    flow = pd.DataFrame(flows)
    flow.to_csv(OUT/'flows.csv',index=False)
    flow.groupby(['dataset','cohort']).mean(numeric_only=True).to_csv(OUT/'flow_summary.csv')
    values = ['oa','ba','macro_brier','delta_oa','delta_ba','delta_macro_brier','rescue_ba','harm_ba']
    f.groupby(['dataset','cohort','arm'])[values].mean().reset_index().to_csv(OUT/'summary.csv',index=False)
    pairs=[]
    for candidate,reference in [('combined','raw'),('combined','initial'),('combined','light'),
        ('combined','initial_guard'),('initial','raw'),('light','raw'),('initial_guard','raw')]:
        left=f[f.arm==candidate][KEY+['oa','ba','macro_brier']]
        right=f[f.arm==reference][KEY+['oa','ba','macro_brier']]
        both=left.merge(right,on=KEY,suffixes=('_new','_reference'),validate='one_to_one')
        row=both[KEY].copy()
        row['candidate'],row['reference']=candidate,reference
        for m in ('oa','ba','macro_brier'):
            row['delta_'+m]=both[m+'_new']-both[m+'_reference']
        pairs.append(row)
    pair=pd.concat(pairs,ignore_index=True)
    pair.to_csv(OUT/'paired.csv',index=False)
    gates = pd.read_csv(OUT/'gates.csv')
    transfer = gates.groupby(KEY+['arm']).agg(oof_risk_gain=('oof_risk_gain','first'),
        absent_classes=('support_n',lambda x:int((x==0).sum())),
        effective_gamma_min=('effective_gamma','min'),
        effective_gamma_max=('effective_gamma','max')).reset_index()
    links = pair[((pair.candidate=='combined') & (pair.reference=='initial')) |
                 ((pair.candidate=='light') & (pair.reference=='raw'))].rename(columns={'candidate':'arm'})
    transfer.merge(links,on=KEY+['arm'],validate='one_to_one').to_csv(OUT/'gate_transfer.csv',index=False)
    summary=pair.groupby(['dataset','cohort','candidate','reference']).agg(
        delta_oa=('delta_oa','mean'),delta_ba=('delta_ba','mean'),
        delta_macro_brier=('delta_macro_brier','mean'),ba_sd=('delta_ba','std'),
        ba_positive=('delta_ba',lambda x:int((x>1e-10).sum())),
        ba_negative=('delta_ba',lambda x:int((x < -1e-10).sum()))).reset_index()
    summary.to_csv(OUT/'paired_summary.csv',index=False)
    interaction=f.pivot(index=KEY,columns='arm',values='ba')
    interaction['interaction_ba']=interaction.combined-interaction.initial-interaction.light+interaction.raw
    interaction.to_csv(OUT/'interaction.csv')
    timing=pd.read_csv(OUT/'timings.csv')
    costs=[]
    for row in timing.to_dict('records'):
        common=row['support_base_kernel_seconds']+row['base_fit_seconds']
        raw=common+row['raw_fit_seconds']
        initial=common+row['initial_fit_seconds']
        light=raw+row['fold_setup_seconds']+row['oof_base_seconds']+row['oof_raw_seconds']+row['light_gate_seconds']
        combined=initial+row['fold_setup_seconds']+row['oof_base_seconds']+row['oof_initial_seconds']+row['combined_gate_seconds']
        for arm,seconds in [('raw',raw),('initial',initial),('light',light),
            ('combined',combined),('initial_guard',initial+row['guard_extra_seconds'])]:
            costs.append({**{k:row[k] for k in KEY},'arm':arm,'support_solver_seconds':seconds})
    pd.DataFrame(costs).to_csv(OUT/'adaptation_cost.csv',index=False)
    pd.DataFrame(costs).groupby(['dataset','cohort','arm']).support_solver_seconds.agg(['mean','min','max']).to_csv(OUT/'adaptation_cost_summary.csv')
    write('EVALUATION_COMPLETE.json',dict(units=len(frozen['units']), legacy_units=10,new_draw_units=10,
        arms=ARMS,new_neural_training=0,target_gate_fits=40,main_model_changed=False))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','predict','evaluate','all'])
    args=parser.parse_args()
    with c.threadpool_limits(limits=2):
        if args.stage in ('prepare','all'): prepare()
        if args.stage in ('predict','all'): predict()
        if args.stage in ('evaluate','all'): evaluate()
