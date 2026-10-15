"""Source-episode learned directional fusion and matched handoff diagnostics."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import gc
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from run_c4_base_trial import OUT as C4, SOURCE, ROOT, GEOMETRY, PARENT, REFERENCE, base_phi, replace_r50, write_json
from direct_incidence_conditioning import condition_features, extend_transforms, predict_conditioned
from protected_drift_activation import fit_source_heads
from run_protected_drift_activation import make_transforms, make_views, gate_features
from residual_expert import fit_correction
from evidence_aware_handoff import fit_gate, predict_gate, route_scores
from run_generic_transfer import metrics

OUT = ROOT/'directional_fusion_trial_20261009'
EPISODES = ROOT/'module_completion_20261008/ports_projection'
LEARNED = ('global_mix', 'class_mix', 'quality_mix', 'class_quality_mix', 'class_geometry_mix', 'class_bias')
MODES = ('original', 'c4', 'half_mix', *LEARNED, 'c4_gate_refit', 'c4_direction_gate')


def archive(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def inputs():
    from run_conditioned_vessel_drift import SIDECAR
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'port', 'class_id'])
    side = pd.read_csv(SIDECAR, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    relative = pd.read_csv(GEOMETRY/'relative_conditions.csv')
    np.testing.assert_array_equal(frame.instance_id, relative.instance_id)
    valid = relative.relative_status.eq('ok').to_numpy()
    angle = np.deg2rad(relative.relative_signed_deg.fillna(0).to_numpy()*2)
    u = condition_features(frame.incidence_deg.to_numpy())
    context = np.c_[u, np.isfinite(frame.incidence_deg), valid, np.where(valid, np.cos(angle), 0), np.where(valid, np.sin(angle), 0)]
    return frame, context


def source_port(port, frame, context, raw, averaged):
    destination = OUT/'ports'/port
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'SOURCE_READY.json').exists():
        return
    started = time.perf_counter()
    folder = PARENT/port
    with (folder/'geometry.pkl').open('rb') as f:
        g = pickle.load(f)
    sr = np.load(folder/'source_rows.npy')
    bank = np.load(folder/'source_bank.npy', mmap_mode='r')
    c4bank = replace_r50(averaged[sr].astype(float), bank, g)
    su = context[sr, :2]
    augmented, c4augmented = np.c_[bank, su], np.c_[c4bank, su]
    transforms = extend_transforms({k:v for k,v in make_transforms(g).items() if k.startswith('joint_')})
    extra = np.load(ROOT/'evidence_aware_handoff_20261008/source_features'/port/'features.npy', mmap_mode='r')
    items = json.loads((EPISODES/port/'EPISODES_COMPLETE.json').read_text(encoding='utf-8'))['episodes']
    records = {k:[] for k in ['original', 'c4', 'expert', 'labels', 'rows', 'episode', 'weights', 'eligible', 'context']}
    source_metrics = []
    for i, item in enumerate(items):
        ep = archive(item['path'])
        train, rows, support, query = ep['train_rows'], ep['rows'], ep['support'], ep['query']
        ti, hi = np.searchsorted(sr, train), np.searchsorted(sr, rows)
        np.testing.assert_array_equal(sr[ti], train)
        np.testing.assert_array_equal(sr[hi], rows)
        if frame.loc[train, 'port'].isin([port, item['port']]).any() or frame.loc[rows, 'port'].eq(port).any():
            raise ValueError('Outer target or inner held port enters source head fitting')
        if np.intersect1d(support, query).size or np.intersect1d(train, rows).size:
            raise ValueError('Episode support/query/train overlap')
        heads = fit_source_heads(augmented[ti], frame.loc[train,'class_id'].to_numpy(), transforms)
        ch = fit_source_heads(c4augmented[ti], frame.loc[train,'class_id'].to_numpy(), {'joint_base':transforms['joint_base']})['joint_base']
        views = make_views(np.asarray(bank[hi]), raw[rows], extra[hi], g)
        labels = frame.loc[rows, 'class_id'].to_numpy()
        original, expert = predict_conditioned(bank[hi], views, heads, support, labels[support], su[hi])
        phi = base_phi(c4bank[hi], g, su[hi])
        prior = c4augmented[hi] @ ch['coef']+ch['intercept']
        c4 = prior+phi @ fit_correction(phi[support], labels[support], prior[support])
        _, inverse, counts = np.unique(labels[query], return_inverse=True, return_counts=True)
        weights = 1./(len(items)*len(counts)*counts[inverse])
        observed = np.bincount(labels[support], minlength=8)>0
        eligible = (c4.argmax(1)!=expert.argmax(1)) & observed[expert.argmax(1)]
        values = dict(original=original[query], c4=c4[query], expert=expert[query], labels=labels[query],
            rows=rows[query], episode=np.full(len(query), i), weights=weights, eligible=eligible[query], context=context[rows[query]])
        for key, value in values.items():
            records[key].append(value)
        for name, scores in [('original', original), ('c4', c4), ('expert', expert)]:
            score = metrics(labels[query], scores[query].argmax(1), 8)
            source_metrics.append(dict(outer_port=port, held_port=item['port'], mode=name, oa=score['acc'], ba=score['ba']))
        del heads, ch, views
        if (i+1)%6 == 0:
            print(f'SOURCE {port} {i+1}/{len(items)} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    np.savez_compressed(destination/'source_predictions.npz', **{k:np.concatenate(v) for k,v in records.items()})
    pd.DataFrame(source_metrics).to_csv(destination/'source_episode_metrics.csv', index=False)
    write_json(destination/'SOURCE_READY.json', dict(port=port, episodes=len(items), seconds=time.perf_counter()-started,
        target_labels_used=0, heads_inner_port_held=True, geometry_inner_port_held=False,
        encoder_inner_port_held=False, support_per_episode=64))
    print(f'SOURCE READY {port} elapsed={time.perf_counter()-started:.1f}s', flush=True)


def fit_port(port, frame, context):
    from direction_score_fusion import fit_fusions, predict_fusion, quality_features
    folder = OUT/'ports'/port
    if (folder/'TARGET_FROZEN.json').exists():
        return
    started = time.perf_counter()
    source = archive(folder/'source_predictions.npz')
    models = fit_fusions(source['original'], source['c4'], source['labels'], source['weights'], source['context'])
    write_json(folder/'fusion_models.json',models)
    source_gate_features = gate_features(source['c4'], source['expert'])
    source_quality = quality_features(source['original'], source['c4'])
    gates = {name: fit_gate(feature, source['c4'], source['expert'], source['labels'], source['eligible'], source['weights'])
        for name, feature in [('c4_gate_refit',source_gate_features), ('c4_direction_gate',np.c_[source_gate_features,source_quality])]}
    with (folder/'learned.pkl').open('wb') as f:
        pickle.dump(dict(fusions=models, gates=gates), f)
    with (PARENT/port/'source_models.pkl').open('rb') as f:
        old_gate = pickle.load(f)['gates']['joint']
    prior, parent = archive(REFERENCE/port/'policies.npz'), archive(C4/'ports'/port/'policies.npz')
    slot = list(prior['mode_names']).index('direct_condition')
    cslot = list(parent['base_modes']).index('matched_c4')
    for key in ['query_rows','support_rows','eval_mask']:
        np.testing.assert_array_equal(parent[key], prior[key])
    rows = parent['query_rows']
    all_bases, all_pred, all_selected, alpha_records = [], [], [], []
    replay = 0
    for draw, ev in enumerate(parent['eval_mask']):
        old = prior['base_scores'][draw,slot]
        c4 = parent['base_scores'][draw,cslot]
        expert = prior['expert_scores'][draw,slot]
        labels = frame.loc[parent['support_rows'][draw], 'class_id'].to_numpy()
        if np.intersect1d(parent['support_rows'][draw], rows[ev]).size:
            raise ValueError('Target support/query overlap')
        predictions = {'original':old, 'c4':c4, 'half_mix':.5*(old+c4), 'c4_gate_refit':c4, 'c4_direction_gate':c4}
        for name in LEARNED:
            predictions[name], alpha = predict_fusion(models[name], old, c4, context[rows])
            if alpha.size:
                alpha_records.append(dict(draw=draw, mode=name, mean=float(alpha[ev].mean()),
                    **{f'class_{c}':float(alpha[ev,c].mean()) for c in range(8)}))
        bases, fulls, selected = [], [], []
        for name in MODES:
            base = predictions[name]
            feature = gate_features(base, expert)
            if name == 'c4_direction_gate':
                feature = np.c_[feature,quality_features(old,c4)]
            gate = gates[name] if name in gates else old_gate
            routed = route_scores(base, expert, predict_gate(gate, feature), labels, ev, rows)
            if routed['selected'][~ev].any() or routed['selected'].sum()>int(.05*ev.sum()):
                raise ValueError('Handoff budget error')
            if name in ['original','c4']:
                wanted = prior['predictions'][draw,slot] if name=='original' else parent['full_predictions'][draw,1]
                replay += int(np.count_nonzero(wanted!=routed['predictions']))
            bases.append(base); fulls.append(routed['predictions']); selected.append(routed['selected'])
        all_bases.append(np.stack(bases)); all_pred.append(np.stack(fulls)); all_selected.append(np.stack(selected))
    if replay:
        raise ValueError(f'Frozen parent prediction replay failed {port}: {replay}')
    np.savez_compressed(folder/'policies.npz', query_rows=rows,support_rows=parent['support_rows'],eval_mask=parent['eval_mask'],
        modes=np.asarray(MODES), base_scores=np.stack(all_bases),predictions=np.stack(all_pred),selected=np.stack(all_selected))
    pd.DataFrame(alpha_records).to_csv(folder/'alpha_summary.csv', index=False)
    source_eval=[]
    for name in ['c4','half_mix',*LEARNED]:
        scores=source['c4'] if name=='c4' else .5*(source['original']+source['c4']) if name=='half_mix' else predict_fusion(models[name],source['original'],source['c4'],source['context'])[0]
        for episode in np.unique(source['episode']):
            mask=source['episode']==episode
            score=metrics(source['labels'][mask],scores[mask].argmax(1),8)
            source_eval.append(dict(port=port,episode=int(episode),mode=name,oa=score['acc'],ba=score['ba']))
    pd.DataFrame(source_eval).to_csv(folder/'source_fusion_fit_metrics.csv',index=False)
    write_json(folder/'TARGET_FROZEN.json',dict(port=port,seconds=time.perf_counter()-started,parent_prediction_mismatches=replay,
        query_labels_used=0,source_metamodel_metrics_are_fit_not_validation=True))
    print(f'TARGET FROZEN {port} elapsed={time.perf_counter()-started:.1f}s',flush=True)


def report(frame):
    records, classes, transitions, alphas, fit_metrics = [], [], [], [], []
    metadata = pd.read_csv(GEOMETRY/'relative_conditions.csv')
    for folder in sorted((OUT/'ports').iterdir()):
        p=archive(folder/'policies.npz')
        rows=p['query_rows']; truth=frame.loc[rows,'class_id'].to_numpy()
        measured=metadata.loc[rows,'relative_status'].eq('ok').to_numpy()
        angle=metadata.loc[rows,'relative_acute_deg'].to_numpy()
        alphas.append(pd.read_csv(folder/'alpha_summary.csv').assign(port=folder.name))
        fit_metrics.append(pd.read_csv(folder/'source_fusion_fit_metrics.csv'))
        for draw,ev in enumerate(p['eval_mask']):
            for stage,predictions in [('full',p['predictions'][draw]),('base',p['base_scores'][draw].argmax(2))]:
                ref=predictions[list(MODES).index('c4')]
                for mi,name in enumerate(MODES):
                    pred=predictions[mi]
                    for subset,mask in [('all',ev),('measurable',ev&measured)]:
                        if not mask.any(): continue
                        score=metrics(truth[mask],pred[mask],8)
                        records.append(dict(port=folder.name,draw=draw,stage=stage,mode=name,subset=subset,n=int(mask.sum()),
                            oa=score['acc'],ba=score['ba'],selected=int((p['selected'][draw,mi]&mask).sum()),
                            rescue=int((mask&(ref!=truth)&(pred==truth)).sum()),harm=int((mask&(ref==truth)&(pred!=truth)).sum())))
                    for cl in np.unique(truth[ev]):
                        mask=ev&(truth==cl)
                        classes.append(dict(port=folder.name,draw=draw,stage=stage,mode=name,class_id=int(cl),n=int(mask.sum()),
                            recall=float((pred[mask]==cl).mean()),rescue=float(((ref[mask]!=cl)&(pred[mask]==cl)).mean()),harm=float(((ref[mask]==cl)&(pred[mask]!=cl)).mean())))
                    if stage=='base' and name in ['c4','half_mix','class_mix','quality_mix','class_quality_mix','class_geometry_mix','class_bias']:
                        for j in np.flatnonzero(ev&measured):
                            transitions.append(dict(manifest_row=int(rows[j]),port=folder.name,draw=draw,mode=name,class_id=int(truth[j]),
                                relative_angle=float(angle[j]),prediction=int(pred[j]),correct=int(pred[j]==truth[j])))
    metrics_table=pd.DataFrame(records); metrics_table.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'port_class_draw.csv',index=False)
    pd.DataFrame(transitions).to_csv(OUT/'angle_predictions.csv',index=False)
    pd.concat(alphas).to_csv(OUT/'alpha_summary.csv',index=False)
    pd.concat(fit_metrics).to_csv(OUT/'source_fit_metrics.csv',index=False)
    pm=metrics_table.groupby(['port','stage','mode','subset']).mean(numeric_only=True).reset_index()
    pm.to_csv(OUT/'port_means.csv',index=False)
    eight=frame.groupby('port').class_id.nunique(); summaries=[]
    for group,mask in [('all23',np.ones(len(pm),dtype=bool)),('strict8_12',pm.port.map(eight).eq(8))]:
        table=pm[mask].groupby(['stage','mode','subset']).agg(ports=('port','nunique'),oa=('oa','mean'),ba=('ba','mean')).reset_index()
        table.insert(0,'group',group); summaries.append(table)
    summary=pd.concat(summaries,ignore_index=True); summary.to_csv(OUT/'summary.csv',index=False)
    write_json(OUT/'COMPLETE.json',dict(ports=int(pm.port.nunique()),draws=5,modes=list(MODES),
        target_selection=False,independent_confirmation=False,main_model_unchanged=True))
    print(summary[(summary.stage=='full')&(summary.subset=='all')].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['source','fit','report'],required=True)
    parser.add_argument('--ports',nargs='*')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'PROTOCOL.json').exists():
        write_json(OUT/'PROTOCOL.json',dict(date='2026-10-09',modes=list(MODES),source='Original outer-purged inner-held source-port episodes',
            heads_inner_held=True,geometry_inner_held=False,encoder_inner_held=False,target_labels=64,draws=5,budget=.05,
            features='Original and C4 scores;4 quality features; context incidence t,t2,known,relative-known,cos2relative,sin2relative',
            objective='Weighted multiclass CE with equal episode/class weights; NOT direct BA maximization',
            controls='Half blend, learned scalar, class coefficients, quality-shared, class-quality, class-geometry, class-bias, gate-only refit',
            heldout_target_selection=False,source_meta_fit_metrics_not_independent_validation=True,
            fixed_ridge='source1/support8',expert_unchanged=True,main_model_unchanged=True,independent_confirmation=False))
    frame,context=inputs()
    ports=[p.name for p in sorted(PARENT.iterdir()) if (p/'policies.npz').exists()]
    if args.ports: ports=[p for p in ports if p in args.ports]
    if args.stage=='source':
        raw=np.load(SOURCE/'pooled_features.npy',mmap_mode='r')
        averaged=np.load(C4/'pooled_c4.npy',mmap_mode='r')
        for port in ports:
            source_port(port,frame,context,raw,averaged)
            gc.collect()
    elif args.stage=='fit':
        for port in ports:
            fit_port(port,frame,context)
            gc.collect()
        if all((OUT/'ports'/p/'TARGET_FROZEN.json').exists() for p in [q.name for q in PARENT.iterdir() if (q/'policies.npz').exists()]):
            write_json(OUT/'ALL_FROZEN.json',dict(ports=23,target_query_labels_used=0))
    else:
        if not (OUT/'ALL_FROZEN.json').exists(): raise ValueError('Freeze all targets before reporting')
        report(frame)


if __name__=='__main__': main()
