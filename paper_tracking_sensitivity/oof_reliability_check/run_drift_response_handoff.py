"""Six-port paired pilot: latent-response routing with classifiers held fixed."""

import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from drift_response_handoff import (FEATURE_NAMES, response_features, confidence_features,
                                    fit_net_gate, predict_net_gate)
from evidence_aware_handoff import route_scores
from latent_usage_attribution import raw_readout_gradients
from projection_residual_expert import discarded_coordinates, fit_kernel_correction
from protected_drift_activation import protected_drift_basis
from residual_expert import fit_correction
from run_ports_projection_completion import matched_views
from run_protected_drift_activation import cell_means
from run_generic_transfer import metrics
from sar_joint_completion import SOURCE, write_json


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT/'drift_response_handoff_pilot_20261008'
PARENT = ROOT/'module_completion_20261008/ports_projection'
EVIDENCE = ROOT/'evidence_aware_handoff_20261008'
PROTECTED = ROOT/'protected_drift_activation_20261008/ports'
PILOT = ('Antwerp-Bruges','Jebel Ali','Port Klang','Santos','Singapore','Tanger Med')
GATES = ('confidence_net','generic_response','protected_response')
ARMS = ('reference',*GATES)


def archive(path):
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def protocol():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'direction_cache').mkdir(exist_ok=True)
    value = dict(date='2026-10-08',ports=list(PILOT),draws=5,labels_per_draw=64,budget=.05,
        arms=list(ARMS),source_episodes='All22 existing source-held episodes; original seed0 64 supports',
        classifiers='Frozen historical base64 and current strong expert; no score recalibration or retraining',
        geometry='Inner-source q and moments exclude both outer and held source ports; existing identity/product purges',
        covariance='Equalclass/equalport sameclass mean deviations; top16 covariance factors',
        generic='Top16 directions of U',protected='Top16 directions of U projected outside candidate Qs',
        perturbation='Symmetric +/-sqrt(rank) covariance axes; fixed amplitude1; queryR50 only, support and R18 fixed',
        features=list(FEATURE_NAMES),gate='C1 logistic netutility P(rescue)-P(harm), includes neutral outcomes',
        weights='Equal episode then true class within sourcequery, normalized on eligible training rows',
        diagnostic='Three blocked source-episode gate folds; not fully nested encoder/geometry across meta folds',
        inference='No target query labels; disagreement, observed expert class, positive score, 5percent cap',
        encoder_boundary='Current M3 encoder saw source episode labels; not encoder-inner-OOF',
        hyperparameter_selection=False,all_target_policies_before_readout=True,
        main_model_changed=False,independent_confirmation=False,physical_intervention=False)
    path = OUT/'PROTOCOL.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=value:
        raise ValueError('Changed protocol requires separate output')
    write_json(path,value)


def covariance_factors(key, raw, rows, labels, domains, center, scale, qs):
    path = OUT/'direction_cache'/(key+'.npz')
    if path.exists():
        saved = archive(path)
        np.testing.assert_array_equal(saved['train_rows'],rows)
        return saved
    means,classes = cell_means(raw,rows,labels,domains,center,scale)
    info = protected_drift_basis(means,classes,qs,rank=16)
    _,singular,vt = np.linalg.svd(info['u'],full_matrices=False)
    rank = min(16,int(np.sum(singular>singular[0]*1e-6)))
    rd = info['q_drift'].shape[1]
    generic = vt[:rank].T*singular[:rank]
    protected = info['q_drift']*info['singular_values'][:rd]
    np.testing.assert_allclose(qs.T@protected,0,atol=1e-8)
    result = dict(train_rows=rows,generic=generic,protected=protected,
        class_port_cells=len(means),generic_energy=np.sum(generic**2),
        protected_energy=np.sum(protected**2),overlap=info['nuisance_energy_in_stable_fraction'])
    np.savez_compressed(path,**result)
    return result


def source_gradients(ep, fitted, geometry, ed, extra_support):
    support = ep['support']
    xs = ep['xs'][support]
    phi = np.column_stack(((xs-fitted['base_residual_center'])/
                           fitted['base_residual_scale']/np.sqrt(128),np.ones(len(xs))))
    bcoef = fit_correction(phi,ep['labels'][support],ed['base_prior'][support])
    np.testing.assert_allclose(ed['base_prior'][support]+phi@bcoef,ed['base_scores'][support],atol=1e-8)
    z = ((extra_support-ep['ssl_mean'])@ep['ssl_components'].T).astype(np.float32)
    xe = (np.column_stack((xs,z))-ep['expert_center'])/ep['expert_scale']
    prior = (xe@ep['expert_coef'].T+ep['expert_intercept']).astype(np.float32).astype(float)
    retained = (xe/np.sqrt(192)).astype(float)
    ssl_pca = SimpleNamespace(mean_=ep['ssl_mean'],components_=ep['ssl_components'])
    extra_discarded = (discarded_coordinates(extra_support,ssl_pca)-ep['discarded_center'][2048:])/ep['discarded_scale'][2048:]
    complement = np.column_stack((ep['base_discarded'][support],extra_discarded))/np.sqrt(2560)
    kernel = .5*(retained@retained.T)+.5*(complement@complement.T)+1.
    beta = fit_kernel_correction(kernel,ep['labels'][support],prior)
    np.testing.assert_allclose(prior+kernel@beta,ed['expert_scores'][support],atol=2e-6,rtol=2e-6)
    model = dict(pca=SimpleNamespace(components_=geometry['base_components']),scale=geometry['base_scale'],
        base=SimpleNamespace(coef_=fitted['base_coef']),expert=SimpleNamespace(coef_=ep['expert_coef']),
        expert_scale=ep['expert_scale'])
    gradients = raw_readout_gradients(model,bcoef,fitted['base_residual_scale'],retained,complement,beta,ep['discarded_scale'])
    return (gradients['base_source']+gradients['base_adaptation'],
            gradients['expert_source']+gradients['expert_retained']+gradients['expert_complement'])


def feature_key(name):
    return {'confidence_net':'confidence','generic_response':'generic','protected_response':'protected'}[name]


def concatenate(groups, omit_fold=None):
    selected = [g for i,g in enumerate(groups) if omit_fold is None or i%3!=omit_fold]
    return {k:np.concatenate([g[k] for g in selected]) for k in
            ('base','expert','labels','eligible','weights','confidence','generic','protected')}


def train_source(port, raw, frame):
    folder = OUT/'ports'/port
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'SOURCE_FITTED.json').exists():
        with (folder/'gates.pkl').open('rb') as f:
            return pickle.load(f)
    descriptor = json.loads((EVIDENCE/'source_episodes'/port/'COMPLETE.json').read_text(encoding='utf-8'))
    original = json.loads((PARENT/port/'EPISODES_COMPLETE.json').read_text(encoding='utf-8'))
    originals = {x['port']:x for x in original['episodes']}
    source_rows = np.load(EVIDENCE/'source_features'/port/'rows.npy')
    source_extra = np.load(EVIDENCE/'source_features'/port/'features.npy',mmap_mode='r')
    labels,domains = frame.class_id.to_numpy(),frame.port.to_numpy()
    groups = []
    start = time.monotonic()
    for i,item in enumerate(descriptor['episodes']):
        path = folder/(Path(item['path']).stem+'_source_response.npz')
        if path.exists():
            groups.append(archive(path))
            continue
        ep,ed = archive(Path(originals[item['port']]['path'])),archive(Path(item['path']))
        key = item['pair_key']
        geometry = archive(PARENT/'episode_geometry_cache'/(key+'.npz'))
        fitted = archive(EVIDENCE/'source_pair_cache'/(key+'.npz'))
        np.testing.assert_array_equal(fitted['train_rows'],ep['train_rows'])
        np.testing.assert_array_equal(ed['rows'],ep['rows'])
        if np.isin(domains[fitted['train_rows']],[port,item['port']]).any():
            raise ValueError('Held port entered direction fitting')
        factor = covariance_factors(key,raw,fitted['train_rows'],labels,domains,
            fitted['raw_center'],fitted['raw_scale'],fitted['q'])
        support_global = ep['rows'][ep['support']]
        local = np.searchsorted(source_rows,support_global)
        np.testing.assert_array_equal(source_rows[local],support_global)
        gb,ge = source_gradients(ep,fitted,geometry,ed,source_extra[local])
        b,e = ed['base_scores'],ed['expert_scores']
        query = ed['query']
        support_labels = ep['labels'][ep['support']]
        observed = np.bincount(support_labels,minlength=8)>0
        eligible = (b.argmax(1)!=e.argmax(1))&observed[e.argmax(1)]
        _,inverse,counts = np.unique(ed['labels'][query],return_inverse=True,return_counts=True)
        data = dict(base=b[query],expert=e[query],labels=ed['labels'][query],
            eligible=eligible[query],weights=1./(len(descriptor['episodes'])*len(counts)*counts[inverse]),
            confidence=confidence_features(b,e)[query],row_ids=ed['rows'][query],support_labels=support_labels,
            held_port=np.array(item['port']))
        for name in ('generic','protected'):
            data[name] = response_features(b,e,gb,ge,fitted['raw_scale'],factor[name])[query]
        np.savez_compressed(path,**data)
        groups.append(data)
        del ep,ed,geometry,fitted,gb,ge
        if (i+1)%4==0:
            print(f'SOURCE {port}: {i+1}/22 {time.monotonic()-start:.1f}s',flush=True)
    full = concatenate(groups)
    gates = {name:fit_net_gate(full[feature_key(name)],full['base'],full['expert'],full['labels'],
                             full['eligible'],full['weights']) for name in GATES}
    diagnostic = []
    for fold in range(3):
        train = concatenate(groups,omit_fold=fold)
        fold_gates = {name:fit_net_gate(train[feature_key(name)],train['base'],train['expert'],train['labels'],
                                      train['eligible'],train['weights']) for name in GATES}
        for i,data in enumerate(groups):
            if i%3!=fold:
                continue
            for name,gate in fold_gates.items():
                score = predict_net_gate(gate,data[feature_key(name)])
                route = route_scores(data['base'],data['expert'],score,data['support_labels'],
                                     np.ones(len(score),dtype=bool),data['row_ids'])
                m = metrics(data['labels'],route['predictions'],8)
                bpred = data['base'].argmax(1)
                diagnostic.append(dict(port=port,held_port=str(data['held_port']),arm=name,
                    acc=m['acc'],ba=m['ba'],selected=int(route['selected'].sum()),
                    rescued=int(np.sum((route['predictions']==data['labels'])&(bpred!=data['labels']))),
                    harmed=int(np.sum((route['predictions']!=data['labels'])&(bpred==data['labels'])))))
    pd.DataFrame(diagnostic).to_csv(folder/'source_gate_crossfit.csv',index=False,encoding='utf-8-sig')
    with (folder/'gates.pkl').open('wb') as f:
        pickle.dump(gates,f)
    write_json(folder/'SOURCE_FITTED.json',dict(port=port,episodes=len(groups),source_query_rows=len(full['labels']),
        gate_counts={name:gate.get('counts',{}) for name,gate in gates.items()},
        feature_sd={name:gate.get('feature_sd',np.array([])).tolist() for name,gate in gates.items()},
        source_only=True,seconds=time.monotonic()-start))
    print(f'SOURCE READY {port}: {time.monotonic()-start:.1f}s',flush=True)
    return gates


def verify_query_perturbation(model,stats,adapted,old,draw,xs,retained,complement,prior,gb,ge,raw_scale,factor):
    ix = np.flatnonzero(old['eval_mask'][draw])[:7]
    support = np.searchsorted(old['query_rows'],old['support_rows'][draw])
    delta = raw_scale*factor[:,0]*.25
    p = np.asarray(model['pca'].components_,dtype=float)
    dx = (delta@p.T)/model['scale']
    phi = np.column_stack(((xs[ix]-adapted['source_center'])/adapted['source_scale']/np.sqrt(128),np.ones(len(ix))))
    phi_delta = np.r_[dx/adapted['source_scale']/np.sqrt(128),0.]
    bc = adapted['visual_correction'][draw]
    before = model['base'].decision_function(xs[ix])+phi@bc
    after = model['base'].decision_function(xs[ix]+dx)+(phi+phi_delta)@bc
    b_error = float(np.max(np.abs(after-before-delta@gb)))
    dr = np.r_[dx/model['expert_scale'][:128]/np.sqrt(192),np.zeros(64)]
    dc = np.r_[(delta-(delta@p.T)@p)/stats['discarded_scale'][:2048]/np.sqrt(2560),np.zeros(512)]
    xe = retained[ix]*np.sqrt(192)
    p0 = model['expert'].decision_function(xe).astype(np.float32).astype(float)
    p1 = model['expert'].decision_function(xe+dr*np.sqrt(192)).astype(np.float32).astype(float)
    kernel_delta = .5*(dr@retained[support].T)+.5*(dc@complement[support].T)
    actual = p1-p0+kernel_delta@old['coefficients'][draw]
    e_error = float(np.max(np.abs(actual-delta@ge)))
    if max(b_error,e_error)>5e-5:
        raise ValueError(f'Gradient disagrees with actual query perturbation: {b_error},{e_error}')
    return dict(base_error=b_error,expert_error=e_error)


def freeze_target(port,gates,raw,frame):
    folder = OUT/'ports'/port
    if (folder/'TARGET_FROZEN.json').exists():
        return
    old = archive(PARENT/port/'matched_joint/joint_policies.npz')
    adapted = archive(ROOT/'oof_base_adapt_results_20261007'/port/'policies.npz')
    with (PROTECTED/port/'geometry.pkl').open('rb') as f:
        g = pickle.load(f)
    model,stats = g['model'],g['stats']
    factor = covariance_factors('outer_'+port,raw,old['source_rows'],frame.class_id.to_numpy(),frame.port.to_numpy(),
                               g['feature_center'],g['feature_scale'],g['qs'])
    rows = old['query_rows']
    bank = np.load(PROTECTED/port/'target_bank.npy',mmap_mode='r')
    xs = np.asarray(bank[:,:128],dtype=np.float32)
    extra = np.load(PARENT/port/'matched_joint/joint_extra.npy',mmap_mode='r')
    bd = ((discarded_coordinates(raw[rows],model['pca'])-stats['discarded_center'][:2048])/
          stats['discarded_scale'][:2048])
    prior,retained,complement = matched_views(xs,bd,extra,model,stats,True)
    del bd
    routes,gate_scores,feature_list = [],[],[]
    checks = []
    for draw,ev in enumerate(old['eval_mask']):
        support = np.searchsorted(rows,old['support_rows'][draw])
        np.testing.assert_array_equal(rows[support],old['support_rows'][draw])
        grads = raw_readout_gradients(model,adapted['visual_correction'][draw],adapted['source_scale'],
            retained[support],complement[support],old['coefficients'][draw],stats['discarded_scale'])
        gb = grads['base_source']+grads['base_adaptation']
        ge = grads['expert_source']+grads['expert_retained']+grads['expert_complement']
        checks.append(verify_query_perturbation(model,stats,adapted,old,draw,xs,retained,complement,prior,
                                               gb,ge,g['feature_scale'],factor['protected']))
        b,e = old['base_scores'][draw],old['expert_scores'][draw]
        features = dict(confidence=confidence_features(b,e))
        for name in ('generic','protected'):
            features[name] = response_features(b,e,gb,ge,g['feature_scale'],factor[name])
        support_labels = frame.class_id.to_numpy()[old['support_rows'][draw]]
        results = [dict(predictions=old['predictions'][draw],selected=old['selected'][draw],cap=old['cap'][draw],
                        base_predictions=b.argmax(1),expert_predictions=e.argmax(1))]
        scores = [old['gate_scores'][draw]]
        for name in GATES:
            score = predict_net_gate(gates[name],features[feature_key(name)])
            r = route_scores(b,e,score,support_labels,ev,rows)
            if r['selected'][~ev].any() or r['selected'].sum()>old['cap'][draw]:
                raise ValueError('Support exclusion or budget changed')
            results.append(r)
            scores.append(score)
        routes.append(results)
        gate_scores.append(np.stack(scores))
        feature_list.append(np.stack((features['generic'],features['protected'])))
    np.savez_compressed(folder/'policies.npz',query_rows=rows,source_rows=old['source_rows'],
        support_rows=old['support_rows'],eval_mask=old['eval_mask'],arm_names=np.array(ARMS),
        base_scores=old['base_scores'],expert_scores=old['expert_scores'],gate_scores=np.stack(gate_scores),
        response_features=np.stack(feature_list),
        **{k:np.stack([[r[k] for r in draw] for draw in routes]) for k in routes[0][0]})
    write_json(folder/'TARGET_FROZEN.json',dict(port=port,draws=5,query_labels_used=False,
        classifiers_unchanged=True,query_perturbation_checks=checks))
    print('TARGET FROZEN '+port,flush=True)


def evaluate(frame):
    if not all((OUT/'ports'/p/'TARGET_FROZEN.json').exists() for p in PILOT):
        raise ValueError('All six policies must be frozen before target readout')
    records = []
    for port in PILOT:
        p = archive(OUT/'ports'/port/'policies.npz')
        labels = frame.class_id.to_numpy()[p['query_rows']]
        for draw,ev in enumerate(p['eval_mask']):
            b,e = p['base_scores'][draw].argmax(1),p['expert_scores'][draw].argmax(1)
            utility = (e==labels).astype(int)-(b==labels).astype(int)
            observed = np.unique(frame.class_id.to_numpy()[p['support_rows'][draw]])
            eligible = ev&(b!=e)&np.isin(e,observed)
            useful = eligible&(utility!=0)
            for j,arm in enumerate(ARMS):
                pred = p['predictions'][draw,j]
                score = p['gate_scores'][draw,j]
                m = metrics(labels[ev],pred[ev],8)
                auc = (roc_auc_score(utility[useful]>0,score[useful])
                       if len(np.unique(utility[useful]))==2 else np.nan)
                records.append(dict(port=port,draw=draw,arm=arm,acc=m['acc'],ba=m['ba'],
                    selected=int(p['selected'][draw,j].sum()),cap=int(p['cap'][draw,j]),
                    rescued=int(np.sum(ev&(pred==labels)&(b!=labels))),
                    harmed=int(np.sum(ev&(pred!=labels)&(b==labels))),
                    neutral_admitted=int(np.sum(p['selected'][draw,j]&(utility==0))),
                    informative_auc=auc,
                    changed_from_reference=float(np.mean(pred[ev]!=p['predictions'][draw,0,ev]))))
    table = pd.DataFrame(records)
    table.to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    pm = table.groupby(['port','arm'],as_index=False).mean(numeric_only=True).drop(columns='draw')
    pm.to_csv(OUT/'port_means.csv',index=False,encoding='utf-8-sig')
    means = pm.groupby('arm').mean(numeric_only=True).reset_index()
    means.to_csv(OUT/'equal_port_means.csv',index=False,encoding='utf-8-sig')
    contrasts = []
    for a,b in [('confidence_net','reference'),('generic_response','confidence_net'),
                ('protected_response','confidence_net'),('protected_response','generic_response'),
                ('protected_response','reference')]:
        x,y = pm[pm.arm==a].set_index('port'),pm[pm.arm==b].set_index('port')
        delta = x.ba-y.ba
        contrasts.append(dict(candidate=a,control=b,delta_ba_pp=100*delta.mean(),
            delta_oa_pp=100*(x.acc-y.acc).mean(),positive_ports=int((delta>1e-12).sum()),
            negative_ports=int((delta < -1e-12).sum())))
    source = pd.concat([pd.read_csv(OUT/'ports'/p/'source_gate_crossfit.csv') for p in PILOT])
    source.to_csv(OUT/'source_gate_crossfit.csv',index=False,encoding='utf-8-sig')
    source_means = source.groupby(['port','arm']).mean(numeric_only=True).groupby('arm').mean().reset_index()
    write_json(OUT/'COMPLETE.json',dict(status='complete',ports=6,draws=5,
        means=means.to_dict('records'),contrasts=contrasts,source_crossfit=source_means.to_dict('records'),
        main_model_unchanged=True,independent_confirmation=False))
    print(means.to_string(index=False),flush=True)
    print(json.dumps(contrasts,indent=2),flush=True)


def main():
    protocol()
    if (OUT/'COMPLETE.json').exists():
        print('Completed pilot preserved',flush=True)
        return
    frame = pd.read_csv(SOURCE/'manifest.csv',usecols=['class_id','port'])
    raw = np.load(SOURCE/'pooled_features.npy',mmap_mode='r')
    with threadpool_limits(limits=2):
        for port in PILOT:
            if (OUT/'ports'/port/'TARGET_FROZEN.json').exists():
                continue
            gates = train_source(port,raw,frame)
            freeze_target(port,gates,raw,frame)
    write_json(OUT/'ALL_TARGETS_FROZEN.json',dict(ports=list(PILOT)))
    evaluate(frame)


if __name__=='__main__':
    main()
