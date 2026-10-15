"""Freeze simple source-trained choices, then evaluate existing held tasks."""

import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT/'joint_support_correction_20261013'))
import joint_core as j
from joint_core import c, np, softmax
import pandas as pd
from sklearn.metrics import roc_auc_score
from choice_core import (NAMES, features, support_profile, gain_labels, class_weights,
                         fit_probe, predict_probe, selected_probabilities)

STUDY = ROOT/'same_vessel_condition_study_20261010'
REFERENCE = STUDY/'support_budget_fusion_select_20261012'
BUDGET = ROOT/'frozen_candidate_coverage_20261012/budget'
JOINT = ROOT/'joint_support_correction_20261013'
CONTEXTS = ('Los Angeles', 'Port Klang', 'Santos')
MODELS = {'score_only': 8, 'contextual': 16}


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(name, value):
    p = OUT/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def store_unit(units, key, raw, initial, bp, profile, support, query, extras=None):
    uid = '__'.join(str(key[k]).replace(' ', '_') for k in ('family','task','unit'))
    path = OUT/'inputs'/f'{uid}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, raw=raw, initial=initial, base_probability=bp,
        x=features(raw,initial,bp,profile), support_ids=support, query_ids=query,
        **{'profile_'+k:v for k,v in profile.items()}, **(extras or {}))
    units.append(dict(**key,path=str(path),uid=uid))


def make_profile(raw, prior, kb, base_prior, sy, groups, model, parts=None):
    if parts is None:
        parts, notes = j.make_parts(sy,groups)
    else:
        notes = []
    values, oy, timing = j.paired_oof(raw,prior,kb,base_prior,sy,groups,model,parts)
    np.testing.assert_array_equal(oy,np.tile(sy,3))
    return support_profile(*values,sy), dict(parts=parts,warnings=notes,timing=timing)


def prepare_source():
    started = time.perf_counter()
    meta = pd.read_csv(STUDY/'observations.csv',dtype={'mmsi':str})
    folds = load(STUDY/'crossfit_evidence_20261011/source_folds.npz')
    fit_ids = folds['fit_obs_ids']
    assert not set(meta.iloc[fit_ids].port) & set(CONTEXTS)
    units, replays = [], []
    for context in CONTEXTS:
        maps = load(STUDY/'case_relation_headroom_20261012'/context/'maps.npz',
                    ['obs_ids','single','prior'])
        bd = load(REFERENCE/'base_cache'/context/'base.npz')
        np.testing.assert_array_equal(maps['obs_ids'],bd['obs_ids'])
        np.testing.assert_array_equal(maps['obs_ids'][:len(fit_ids)],fit_ids)
        index = pd.Index(maps['obs_ids'])
        fitprior = load(STUDY/'case_relation_student_20261012'/context/'source_readers.npz',
                        ['fit_prior'])['fit_prior']
        for family, folder in [('source_fit','selection'),('source_held','source')]:
            for path in sorted((REFERENCE/folder/context).glob('*_16.npz')):
                old = load(path)
                s,q = index.get_indexer(old['support_ids']),index.get_indexer(old['query_ids'])
                assert (s>=0).all() and (q>=0).all()
                if family == 'source_fit':
                    assert max(s.max(),q.max()) < len(fit_ids)
                    fold = int(path.stem.split('_')[0])
                    model = load(STUDY/'case_completion_class_control_20261012'/context/f'fold_{fold}_model.npz')
                    prior,bprior = fitprior,bd['prior_oof']
                    raw,initial = old['fusion'][0],old['fusion'][8]
                    expected_raw,expected_initial = old['expert'][0],old['expert'][8]
                else:
                    assert min(s.min(),q.min()) >= len(fit_ids)
                    model = load(STUDY/'case_completion_class_control_20261012'/context/'final_model.npz')
                    prior,bprior = maps['prior'],bd['prior_held']
                    raw,expected_raw = old['fusion_raw'],old['expert_raw']
                sy = meta.iloc[old['support_ids']].class_id.to_numpy(int)
                groups = meta.iloc[old['support_ids']].mmsi.to_numpy(str)
                assert not set(groups) & set(meta.iloc[old['query_ids']].mmsi)
                xs,xq = maps['single'][s],maps['single'][q]
                kb = c.base_kernel({k:bd[k][s] for k in ('phi','mean','conditions')},
                    bd['embedding'][s].astype(float),np.arange(16),float(bd['eta']))
                profile, detail = make_profile(xs,prior[s],kb,bprior[s],sy,groups,model)
                changed = j.corrected_support(xs,model)
                ex_raw = prior[q]+(xq@xs.T)@c.coefficients(xs@xs.T,prior[s],sy,2.)
                ex_initial = prior[q]+(xq@changed.T)@c.coefficients(changed@changed.T,prior[s],sy,2.)
                np.testing.assert_allclose(ex_raw,expected_raw,atol=1e-9,rtol=0)
                if family == 'source_fit':
                    np.testing.assert_allclose(ex_initial,expected_initial,atol=1e-9,rtol=0)
                else:
                    initial = .5*(softmax(old['base'],axis=1)+softmax(ex_initial,axis=1))
                bp = softmax(old['base'],axis=1)
                key = dict(family=family,task=context,unit=path.stem,dataset='source',cohort=folder,
                           label_path=str(STUDY/'observations.csv'))
                store_unit(units,key,raw,initial,bp,profile,old['support_ids'],old['query_ids'])
                write('oof/'+units[-1]['uid']+'.json',detail)
                replays.append(dict(uid=units[-1]['uid'],raw_max_error=float(abs(ex_raw-expected_raw).max())))
            print('SOURCE PREPARED',context,family,flush=True)
    write('SOURCE_INPUTS.json',dict(units=units,seconds=time.perf_counter()-started))
    pd.DataFrame(replays).to_csv(OUT/'source_replays.csv',index=False)


def fit():
    started = time.perf_counter()
    units = [u for u in read(OUT/'SOURCE_INPUTS.json')['units'] if u['family']=='source_fit']
    meta = pd.read_csv(STUDY/'observations.csv',dtype={'mmsi':str})
    xx,gg,ww,tasks = [],[],[],[]
    events = []
    for u in units:
        z = load(u['path'])
        qm = meta.iloc[z['query_ids']]
        y = qm.class_id.to_numpy(int)
        gain = gain_labels(z['raw'],z['initial'],y)
        weight = class_weights(y)/len(units)
        xx.append(z['x']);gg.append(gain);ww.append(weight)
        tasks.extend([u['task']]*len(gain))
        for pos in np.flatnonzero(gain):
            events.append(dict(uid=u['uid'],obs_id=int(z['query_ids'][pos]),
                mmsi=str(qm.iloc[pos].mmsi),port=qm.iloc[pos].port,
                gain=int(gain[pos]),weight=float(weight[pos])))
    x,g,w = np.concatenate(xx),np.concatenate(gg),np.concatenate(ww)
    constants,coefficients,metadata = {},[],[]
    for context in ('pooled',)+CONTEXTS:
        use = np.ones(len(g),bool) if context=='pooled' else np.asarray(tasks)==context
        constants[context] = bool(w[use]@g[use] > 0)
        for name,width in MODELS.items():
            model = fit_probe(x[use],g[use],w[use],width)
            np.savez(OUT/(context+'__'+name+'.npz'),**model)
            metadata.append(dict(context=context,name=name,width=width,events=model['events'],iterations=model['iterations']))
            coefficients.extend(dict(context=context,model=name,feature=NAMES[k],standardized_coef=float(v))
                                for k,v in enumerate(model['coef']))
    pd.DataFrame(coefficients).to_csv(OUT/'coefficients.csv',index=False)
    ef = pd.DataFrame(events)
    ef.to_csv(OUT/'training_events.csv',index=False)
    write('FROZEN_MODELS.json',dict(models=metadata,C=.1,threshold=.5,
        features=NAMES,constant_initial=constants,source_units=len(units),
        source_exposures=len(g),informative_events=len(ef),unique_event_obs=ef.obs_id.nunique(),
        unique_event_ships=ef.mmsi.nunique(),source_mean_ba_gain=float(100*w@g),
        source_held_labels_used=False,target_labels_used=False,seconds=time.perf_counter()-started))
    print('MODELS FROZEN',len(g),'exposures',len(ef),'events',ef.mmsi.nunique(),'event ships',flush=True)


def prepare_targets():
    assert (OUT/'FROZEN_MODELS.json').exists()
    from run_joint import support_metadata
    started = time.perf_counter()
    units,replays = [],[]
    for context in CONTEXTS:
        directory = BUDGET/'tasks/self24ports'/context
        folder = c.TASKS/'self24ports'/context
        receipt = read(directory/'PREDICTIONS_FROZEN.json')
        olds = [load(directory/f'{draw}_16.npz') for draw in range(5)]
        union = np.unique(np.concatenate([z['support_ids'] for z in olds]))
        smeta = support_metadata(receipt['target_manifest'],union)
        feat = c.Features(folder)
        index = pd.Index(feat.target_rows)
        data = load(folder/'target.npz',['rows','phi','mean','conditions','prior'])
        embedding = load(folder/'selective_embedding.npz',['rows','embedding'])
        np.testing.assert_array_equal(data['rows'],embedding['rows'])
        np.testing.assert_array_equal(data['rows'],feat.target_rows)
        model = load(directory/'initial.npz')
        eta = read(folder/'PREPARED.json')['eta']
        for draw,old in enumerate(olds):
            s,q = index.get_indexer(old['support_ids']),index.get_indexer(old['query_ids'])
            assert (s>=0).all() and (q>=0).all()
            sy = old['support_labels']
            np.testing.assert_array_equal(sy,smeta.loc[old['support_ids'],'class_id'].to_numpy(int))
            groups = smeta.loc[old['support_ids'],'group'].to_numpy(str)
            xs,ps = feat.target(s)
            kb = c.base_block(data,embedding['embedding'],np.empty(0,int),s,eta)
            foldfile = ROOT/'classwise_utility_adaptation_20261012/target_oof/self24ports'/context/f'{draw}_16_folds.json'
            parts = read(foldfile)['parts']
            profile,detail = make_profile(xs,ps,kb,data['prior'][s],sy,groups,model,parts)
            changed = j.corrected_support(xs,model)
            # Replaying a query block catches row-coordinate and readout mismatches.
            qx,pq = feat.target(q[:32])
            for arm,sp in [('raw',xs),('initial',changed)]:
                actual = pq+(qx@sp.T)@c.coefficients(sp@sp.T,ps,sy,2.)
                expected = old['expert_'+arm][:32]
                np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0)
                replays.append(dict(task=context,draw=draw,arm=arm,max_error=float(abs(actual-expected).max())))
            key = dict(family='held_port',task=context,unit=str(draw),dataset='self24ports',
                       cohort='legacy',label_path=receipt['target_manifest'])
            store_unit(units,key,old['fusion_raw'],old['fusion_initial'],softmax(old['base'],axis=1),
                       profile,old['support_ids'],old['query_ids'])
            write('oof/'+units[-1]['uid']+'.json',detail)
        print('PORT PREPARED',context,flush=True)
    protocol = read(JOINT/'FROZEN_PROTOCOL.json')
    for ds in ('f0921','oss2'):
        manifest = next(d['manifest'] for d in protocol['datasets'] if d['dataset']==ds)
        for cohort in ('legacy','new'):
            for draw in range(5):
                z = load(JOINT/'predictions'/ds/f'{cohort}_{draw}.npz')
                w = load(JOINT/'weights'/ds/f'{cohort}_{draw}.npz')
                np.testing.assert_array_equal(w['support_ids'],z['support_ids'])
                profile = support_profile(w['oof_prior'],w['oof_raw'],w['oof_initial'],
                                          w['oof_base'],w['support_labels'])
                key = dict(family='external',task=ds,unit=f'{cohort}_{draw}',dataset=ds,
                           cohort=cohort,label_path=manifest)
                store_unit(units,key,z['raw'],z['initial'],softmax(z['base'],axis=1),profile,
                    z['support_ids'],z['query_ids'],{a:z[a] for a in ('light','combined','initial_guard')})
            print('EXTERNAL PREPARED',ds,cohort,flush=True)
    pd.DataFrame(replays).to_csv(OUT/'target_replays.csv',index=False)
    write('TARGET_INPUTS.json',dict(units=units,query_labels_used=False,seconds=time.perf_counter()-started))


def predict():
    frozen = read(OUT/'FROZEN_MODELS.json')
    units = read(OUT/'SOURCE_INPUTS.json')['units']+read(OUT/'TARGET_INPUTS.json')['units']
    models = {context:{name:load(OUT/(context+'__'+name+'.npz')) for name in MODELS}
              for context in ('pooled',)+CONTEXTS}
    timings = []
    for u in units:
        z = load(u['path'])
        start = time.perf_counter()
        context = 'pooled' if u['family']=='external' else u['task']
        values = {name:predict_probe(model,z['x']) for name,model in models[context].items()}
        values['confidence'] = z['initial'].max(1)-z['raw'].max(1)
        elapsed = time.perf_counter()-start
        dest = OUT/'decisions'/(u['uid']+'.npz')
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,**values,query_ids=z['query_ids'])
        timings.append(dict(uid=u['uid'],query_n=len(z['query_ids']),two_probes_seconds=elapsed))
    pd.DataFrame(timings).to_csv(OUT/'decision_timings.csv',index=False)
    write('PREDICTIONS_FROZEN.json',dict(units=units,constant_initial=frozen['constant_initial'],
        held_or_target_query_labels_used=False,neural_fits=0,selector_fits=8))
    print('DECISIONS FROZEN',len(units),flush=True)


def metrics(y,p,raw,initial):
    r,i,chosen = raw.argmax(1)==y,initial.argmax(1)==y,p.argmax(1)==y
    w = class_weights(y)*100
    rescue,harm = ~r&i,r&~i
    losses = np.square(p-np.eye(8)[y]).sum(1)
    return dict(oa=float(100*chosen.mean()),ba=float(w@chosen),
        delta_oa=float(100*(chosen.astype(float)-r).mean()),
        delta_ba=float(w@(chosen.astype(float)-r)),
        versus_initial_oa=float(100*(chosen.astype(float)-i).mean()),
        versus_initial_ba=float(w@(chosen.astype(float)-i)),
        rescue_kept_ba=float(w@(rescue&chosen)),rescue_lost_ba=float(w@(rescue&~chosen)),
        harm_avoided_ba=float(w@(harm&chosen)),harm_kept_ba=float(w@(harm&~chosen)),
        brier=float(losses.mean()),macro_brier=float(w@losses/100))


def evaluate():
    frozen = read(OUT/'PREDICTIONS_FROZEN.json')
    label_tables = {}
    records,diagnostics,classes,ports,feature_auc,excluded = [],[],[],[],[],[]
    sm = pd.read_csv(STUDY/'observations.csv',dtype={'mmsi':str})
    fit_ids = load(STUDY/'crossfit_evidence_20261011/source_folds.npz')['fit_obs_ids']
    fit_ships = set(sm.iloc[fit_ids].mmsi.str.replace(r'\.0$','',regex=True))
    for u in frozen['units']:
        if u['label_path'] not in label_tables:
            cols = pd.read_csv(u['label_path'],nrows=0).columns
            extra = [k for k in ('port','mmsi','mmsi_or_ship_id') if k in cols]
            label_tables[u['label_path']] = pd.read_csv(u['label_path'],
                usecols=['class_id']+extra,dtype={k:str for k in extra})
        z = load(u['path'])
        qm = label_tables[u['label_path']].iloc[z['query_ids']]
        y = qm.class_id.to_numpy(int)
        d = load(OUT/'decisions'/(u['uid']+'.npz'))
        np.testing.assert_array_equal(z['query_ids'],d['query_ids'])
        r,i = z['raw'].argmax(1),z['initial'].argmax(1)
        gain = gain_labels(z['raw'],z['initial'],y)
        event = gain != 0
        weights = class_weights(y)
        def auc(value):
            if len(np.unique(gain[event])) != 2:
                return np.nan
            return float(roc_auc_score(gain[event]>0,value[event],sample_weight=weights[event]))
        key = {k:u[k] for k in ('family','dataset','cohort','task','unit','uid')}
        diagnostics.append(dict(**key,query_n=len(y),class_n=len(np.unique(y)),
            disagreement=int((r!=i).sum()),rescue=int((gain==1).sum()),harm=int((gain==-1).sum()),
            initial_gain_ba=float(100*weights@gain),
            oracle_ba=float(100*weights@((r==y)|(i==y))),
            oracle_gain_over_initial=float(100*weights@(gain==-1)),
            **{name+'_auc':auc(d[name]) for name in MODELS},confidence_auc=auc(d['confidence'])))
        for col,name in enumerate(NAMES):
            feature_auc.append(dict(**key,feature=name,auc=auc(z['x'][:,col])))
        choices = {name:d[name]>.5 for name in MODELS}
        choices['confidence'] = d['confidence']>0
        context = 'pooled' if u['family']=='external' else u['task']
        choices['constant'] = np.full(len(y),frozen['constant_initial'][context])
        arms = {a:z[a] for a in ('raw','initial','light','combined','initial_guard') if a in z}
        arms.update({name:selected_probabilities(z['raw'],z['initial'],pick) for name,pick in choices.items()})
        for arm,p in arms.items():
            mm = metrics(y,p,z['raw'],z['initial'])
            if arm in choices:
                np.testing.assert_allclose(mm['versus_initial_ba'],
                    mm['harm_avoided_ba']-mm['rescue_lost_ba'],atol=1e-10)
            records.append(dict(**key,arm=arm,**mm,
                select_initial_fraction=float(choices[arm].mean()) if arm in choices else np.nan,
                changed_decision_fraction=float(np.mean(p.argmax(1)!=r))))
            if u['family']=='external':
                col = next(k for k in ('mmsi','mmsi_or_ship_id') if k in qm)
                ships = qm[col].fillna('').str.strip().str.replace(r'\.0$','',regex=True)
                known = ships.str.fullmatch(r'[0-9]{9}') & ships.ne('000000000')
                keep = ~ships.isin(fit_ships).to_numpy()
                excluded.append(dict(**key,arm=arm,kept_n=int(keep.sum()),removed_n=int((~keep).sum()),
                    unknown_kept=int((~known.to_numpy()&keep).sum()),
                    **metrics(y[keep],p[keep],z['raw'][keep],z['initial'][keep])))
            for cls in np.unique(y):
                mask = y==cls
                classes.append(dict(**key,arm=arm,class_id=int(cls),n=int(mask.sum()),
                    recall=float(100*np.mean(p.argmax(1)[mask]==cls)),
                    delta_recall=float(100*np.mean((p.argmax(1)[mask]==cls).astype(float)-(r[mask]==cls)))))
            if u['family']=='source_held':
                for port in sorted(qm.port.unique()):
                    mask = qm.port.to_numpy()==port
                    ports.append(dict(**key,actual_port=port,arm=arm,n=int(mask.sum()),
                        **metrics(y[mask],p[mask],z['raw'][mask],z['initial'][mask])))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv',index=False)
    pd.DataFrame(feature_auc).to_csv(OUT/'feature_auc.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'classes.csv',index=False)
    pd.DataFrame(ports).to_csv(OUT/'source_held_actual_ports.csv',index=False)
    ex = pd.DataFrame(excluded)
    ex.to_csv(OUT/'external_source_identity_excluded.csv',index=False)
    ex.groupby(['dataset','cohort','arm'])[['oa','ba','delta_oa','delta_ba','versus_initial_oa',
        'versus_initial_ba','kept_n','removed_n','unknown_kept']].mean().reset_index().to_csv(
            OUT/'external_identity_excluded_summary.csv',index=False)
    group = ['family','dataset','cohort','task','arm']
    per_task = frame.groupby(group).agg(oa=('oa','mean'),ba=('ba','mean'),
        delta_oa=('delta_oa','mean'),delta_ba=('delta_ba','mean'),
        versus_initial_oa=('versus_initial_oa','mean'),versus_initial_ba=('versus_initial_ba','mean'),
        versus_initial_ba_sd=('versus_initial_ba','std'),
        positive_vs_initial=('versus_initial_ba',lambda x:int((x>1e-10).sum())),
        units=('ba','size'),harm_avoided_ba=('harm_avoided_ba','mean'),
        rescue_lost_ba=('rescue_lost_ba','mean'),
        select_initial_fraction=('select_initial_fraction','mean')).reset_index()
    per_task.to_csv(OUT/'task_summary.csv',index=False)
    numeric=['oa','ba','delta_oa','delta_ba','versus_initial_oa','versus_initial_ba',
             'harm_avoided_ba','rescue_lost_ba','select_initial_fraction']
    summary = per_task.groupby(['family','dataset','cohort','arm'])[numeric].mean().reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    write('COMPLETE.json',dict(status='complete',source_fit_units=27,source_held_units=9,
        held_port_units=15,external_units=20,selector_fits=8,neural_fits=0,
        scope='B16 frozen raw/initial classifiers; reused development data and support draws',
        main_model_replaced=False,query_labels_used_for_selector_training_or_selection=False))
    print(summary[['family','dataset','cohort','arm','delta_ba','versus_initial_ba']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['source','fit','targets','predict','evaluate','all'])
    args=parser.parse_args()
    with c.threadpool_limits(limits=2):
        if args.stage in ('source','all'): prepare_source()
        if args.stage in ('fit','all'): fit()
        if args.stage in ('targets','all'): prepare_targets()
        if args.stage in ('predict','all'): predict()
        if args.stage in ('evaluate','all'): evaluate()
