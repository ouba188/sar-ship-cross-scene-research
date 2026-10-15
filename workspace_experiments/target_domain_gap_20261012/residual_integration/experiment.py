"""Bounded source-selected reuse of an existing 64-label prior increment."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent/'full_readout_integration'
sys.path.insert(0, str(PREVIOUS))
from core import source_statistics, solve, Y8, write
import run_full as previous
from run_full import ExpertContext, load_task, head_inputs, base_cross, coefficients, ROOT, SOURCE, FOLDERS, DATASETS
from compact_fusion_source import _episodes, load_npz, _expert_kernel
from unified_head_expert import _conditions

ARMS = ('original', 'resolve', 'retain_half', 'retain_full')


def increment(stats, hs, ys, hall):
    n = stats['n']
    counts = np.bincount(ys, minlength=8)
    w = (n.sum()/(8.*(n+counts)))[ys]
    head = solve(stats['g0']+hs.T@(w[:, None]*hs),
                 stats['c0']+hs.T@(w[:, None]*Y8[ys]))
    return hall@(head-stats['theta0'])


def integrate(prior, kernel, support, labels, delta):
    old = prior+kernel@coefficients(kernel[support], prior[support], labels, 8.)
    new_prior = prior+delta
    resolved = new_prior+kernel@coefficients(kernel[support], new_prior[support], labels, 8.)
    return np.stack((old, resolved, old+.5*delta, old+delta))


def metric(y, p):
    correct = p.argmax(1) == y
    losses = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.))
    classes = np.unique(y)
    return dict(oa=100.*correct.mean(), ba=100.*np.mean([correct[y==c].mean() for c in classes]),
                nll=float(losses.mean()), macro_nll=float(np.mean([losses[y==c].mean() for c in classes])),
                brier=float(np.mean(np.sum((p-np.eye(8)[y])**2, axis=1))))


def rows_metrics(y, probabilities, meta):
    records = []
    for stage, values in probabilities.items():
        before = values[0].argmax(1)
        for arm, p in zip(ARMS, values):
            pred = p.argmax(1)
            records.append(dict(**meta, stage=stage, arm=arm, n=len(y), classes=len(np.unique(y)),
                **metric(y, p), rescues=int(np.sum((before!=y)&(pred==y))),
                harms=int(np.sum((before==y)&(pred!=y)))))
    return records


def choose(frame):
    # Equal held-port weighting, not an aggregate dominated by large ports.
    by_port = frame.groupby(['held_port', 'arm'])[['oa','ba','macro_nll']].mean()
    mean = by_port.groupby('arm').mean().reindex(ARMS)
    eligible = [a for a in ARMS if mean.loc[a, 'ba'] >= mean.loc['original', 'ba']-1e-10]
    order = {a:i for i,a in enumerate(ARMS)}
    winner = min(eligible, key=lambda a: (-round(mean.loc[a,'oa'],10), -round(mean.loc[a,'ba'],10),
                                         round(mean.loc[a,'macro_nll'],10), order[a]))
    return winner, mean.reset_index().to_dict('records')


def source_phase(reuse_ports=False):
    records, receipts = [], []
    for folder in FOLDERS:
        info = json.loads((folder/'PREPARED.json').read_text())
        if reuse_ports and info['dataset']=='self24ports':
            for ei, ep, ix, base in _episodes(folder, info):
                saved = load_npz(OUT/'source'/info['task']/f'episode_{ei:02d}.npz')
                np.testing.assert_array_equal(saved['rows'], ep['rows'][ix[64:]])
                np.testing.assert_array_equal(saved['y'], ep['labels'][ix[64:]])
                np.testing.assert_array_equal(saved['arms'], np.array(ARMS))
                records.extend(rows_metrics(saved['y'],
                    {'expert_prior':saved['prior'],'expert':saved['expert'],'full':saved['full']},
                    dict(task=info['task'],episode=ei,held_port=str(ep['held_port']))))
                receipts.append(dict(task=info['task'],episode=ei,held_port=str(ep['held_port']),
                                     reused_already_replayed_port_episode=True,query_n=len(base)))
            print('SOURCE REUSED',info['task'],flush=True)
            continue
        ctx = ExpertContext(folder)
        ctx._prepare_source()
        if ctx.dataset == 'self24ports':
            xs = ctx.augmented@ctx.transforms['joint_expert']
        else:
            xs = previous._expert_inputs(ctx.raw[ctx.source_rows], ctx.extra, ctx.model)
        h = np.c_[xs.astype(float), np.ones(len(xs))]
        eps = _episodes(folder, ctx.info)
        for index, (ei, ep, ix, base) in enumerate(eps):
            held = str(ep['held_port'])
            rows = ep['rows'][ix]
            pos = ctx.locations.get_indexer(rows)
            if (pos < 0).any():
                raise ValueError('Episode rows not in permitted source pool')
            labels = ep['labels'][ix].astype(int)
            np.testing.assert_array_equal(labels, ctx.source_labels[pos])
            if ctx.dataset == 'self24ports':
                old_ep = load_npz(Path(ctx.items[ei]['path']))
                np.testing.assert_array_equal(old_ep['rows'], ep['rows'])
                np.testing.assert_array_equal(old_ep['support'], ix[:64])
                train = ctx.locations.get_indexer(old_ep['train_rows'])
                if (train < 0).any() or np.isin(ctx.source_ports[train], [held, ctx.task]).any():
                    raise ValueError('Excluded source port entered readout fit')
                kernel = ctx._port_kernel(ctx.bank[pos], ctx.raw[rows], ctx.extra[pos],
                                          _conditions()[rows], np.arange(64))
            else:
                train = np.flatnonzero(ctx.source_ports != held)
                kernel = _expert_kernel(ctx.raw[rows], ctx.extra[pos], xs[pos], ctx.model, ctx.stats)
            if np.intersect1d(train, pos).size:
                raise ValueError('Readout training contains held episode samples')
            stats = source_statistics(h[train], ctx.source_labels[train])
            prior = h[pos]@stats['theta0']
            head_gap = 0.
            if ctx.dataset != 'self24ports':
                from sklearn.base import clone
                ref = clone(ctx.model['expert']).fit(xs[train], ctx.source_labels[train])
                expected = ref.decision_function(xs[pos])
                head_gap = float(np.max(np.abs(prior-expected)))
                if index == 0:
                    double_ref = clone(ctx.model['expert']).fit(xs[train].astype(float), ctx.source_labels[train])
                    np.testing.assert_allclose(prior, double_ref.decision_function(xs[pos].astype(float)),
                                               atol=1e-8, rtol=1e-7)
                # Anchor the original FP32 fit; only the adaptation increment uses FP64.
                prior = expected.astype(np.float32).astype(float)
            delta = increment(stats, h[pos[:64]], labels[:64], h[pos])
            expert = integrate(prior, kernel, np.arange(64), labels[:64], delta)
            replay = None
            if ctx.dataset == 'self24ports':
                expected = ctx._expected_query(ei, rows[64:])
                replay = float(np.max(np.abs(expert[0,64:]-expected)))
                np.testing.assert_allclose(expert[0,64:], expected, atol=3e-5, rtol=1e-6)
            ep_prob = softmax(expert[:,64:], axis=-1)
            full = .5*(softmax(base,axis=1)[None]+ep_prob)
            priors = softmax(np.stack((prior,prior+delta,prior+.5*delta,prior+delta))[:,64:],axis=-1)
            records.extend(rows_metrics(labels[64:], {'expert_prior':priors,'expert':ep_prob,'full':full},
                                         dict(task=ctx.task,episode=ei,held_port=held)))
            dest = OUT/'source'/ctx.task
            dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/f'episode_{ei:02d}.npz', rows=rows[64:], y=labels[64:],
                                arms=np.array(ARMS), prior=priors, expert=ep_prob, full=full,
                                held_port=np.array(held))
            receipts.append(dict(task=ctx.task,episode=ei,held_port=held,source_n=len(train),
                                 query_n=len(base),replay_error=replay,source_refit_gap=head_gap))
            print(f'SOURCE {ctx.task} {index+1}/{len(eps)} {held}',flush=True)
        del ctx, h, xs, stats
        gc.collect()
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'source_metrics.csv',index=False)
    selected, loo = {}, []
    for task, part in frame.query("stage=='full'").groupby('task',sort=False):
        winner, summary = choose(part)
        selected[task] = dict(arm=winner,source_summary=summary)
        for held in part.held_port.unique():
            picked, _ = choose(part[part.held_port!=held])
            selected_rows = part[(part.held_port==held)&(part.arm==picked)]
            baseline = part[(part.held_port==held)&(part.arm=='original')]
            for key in ('oa','ba'):
                loo.append(dict(task=task,held_port=held,arm=picked,metric=key,
                                delta=float(selected_rows[key].mean()-baseline[key].mean())))
    pd.DataFrame(loo).to_csv(OUT/'source_leave_port_selection.csv',index=False)
    write(OUT/'SOURCE_SELECTION.json',dict(selected=selected,episodes=len(receipts),receipts=receipts,
        independent_validation=False,target_labels_used=False,neural_training=False))
    print('SOURCE SELECTED', {t:v['arm'] for t,v in selected.items()},flush=True)


def target_phase():
    selection = json.loads((OUT/'SOURCE_SELECTION.json').read_text())['selected']
    receipts = []
    for folder in FOLDERS:
        info = json.loads((folder/'PREPARED.json').read_text())
        task = info['task']
        previous.MATCHED = info['dataset']=='oss2'
        ctx = ExpertContext(folder)
        data = load_task(folder)
        hs, ht = head_inputs(ctx,data)
        stats = source_statistics(hs,ctx.source_labels)
        baseline_folder = PREVIOUS/('matched_full' if previous.MATCHED else 'full')/task
        ref = load_npz(baseline_folder/'predictions.npz')
        np.testing.assert_array_equal(ref['rows'],data['rows'])
        np.testing.assert_array_equal(ref['support'],data['support'])
        np.testing.assert_array_equal(ref['eval_masks'],data['eval_masks'])
        matched = None
        if previous.MATCHED:
            matched = load_npz(ROOT/'backbone_ablation_20261010/targets/oss2/oss2/coefficients.npz')
        expert_all, full_all, prior_all, deltas_all = [], [], [], []
        for draw, support in enumerate(data['support']):
            labels = data['support_labels'][draw]
            if data['eval_masks'][draw,support].any():
                raise ValueError('Support row entered evaluation')
            prior, kernel = ctx.target_components(data,support)
            if matched is not None:
                prior = (ht[:,:-1]@matched['dual_coef']+matched['dual_intercept']).astype(np.float32).astype(float)
            delta = increment(stats,ht[support],labels,ht)
            expert = integrate(prior,kernel,support,labels,delta)
            kb = base_cross(data,slice(None),support,info['eta'])
            base = data['prior']+kb@coefficients(kb[support],data['prior'][support],labels,8.)
            ep = softmax(expert,axis=-1)
            full = .5*(softmax(base,axis=1)[None]+ep)
            for stage, values in [('expert',ep),('full',full)]:
                for ai, old_arm in [(0,'original'),(1,'source_fixed')]:
                    ri = list(ref['arms']).index(old_arm)
                    np.testing.assert_allclose(values[ai],ref[stage][draw,ri],atol=1e-11,rtol=1e-9)
            expert_all.append(ep)
            full_all.append(full)
            prior_all.append(softmax(np.stack((prior,prior+delta,prior+.5*delta,prior+delta)),axis=-1))
            # Compare centered score changes, which discard softmax-invisible offsets.
            d = delta-delta.mean(1,keepdims=True)
            de = expert-expert[0:1]
            de -= de.mean(-1,keepdims=True)
            deltas_all.append(dict(task=task,draw=draw,prior_rms=float(np.sqrt(np.mean(d*d))),
                                  resolve_rms=float(np.sqrt(np.mean(de[1]**2))),
                                  retain_half_rms=float(np.sqrt(np.mean(de[2]**2))),
                                  retain_full_rms=float(np.sqrt(np.mean(de[3]**2)))))
        dest = OUT/'target'/task
        dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'predictions.npz',expert_prior=np.array(prior_all),expert=np.array(expert_all),
            full=np.array(full_all),arms=np.array(ARMS),rows=data['rows'],ids=data['ids'],
            support=data['support'],support_labels=data['support_labels'],eval_masks=data['eval_masks'],
            common_eval=data['eval_masks'].all(0))
        pd.DataFrame(deltas_all).to_csv(dest/'score_changes.csv',index=False)
        receipts.append(dict(task=task,selected_arm=selection[task]['arm'],source_n=len(hs),
                             old_baseline_and_resolve_replayed=True))
        print('TARGET FROZEN',task,selection[task]['arm'],flush=True)
        del ctx,hs,ht,stats,data,ref
        gc.collect()
    write(OUT/'TARGET_FROZEN.json',dict(tasks=receipts,neural_training=False,target_selection=False))


def evaluate():
    selection = json.loads((OUT/'SOURCE_SELECTION.json').read_text())['selected']
    records,events = [],[]
    for folder in FOLDERS:
        info=json.loads((folder/'PREPARED.json').read_text())
        task=info['task']
        z=load_npz(OUT/'target'/task/'predictions.npz')
        if info['dataset']=='self24ports':
            frame=pd.read_csv(SOURCE/'manifest.csv',usecols=['instance_id','class_id']).iloc[z['rows']]
            idcol='instance_id'
        else:
            frame=pd.read_csv(ROOT/DATASETS['oss2'][0]/'target_manifest.csv',usecols=['sample_id','class_id']).iloc[z['rows']]
            idcol='sample_id'
        np.testing.assert_array_equal(frame[idcol].to_numpy(str),z['ids'])
        y=frame.class_id.to_numpy(int)
        np.testing.assert_array_equal(y[z['support']],z['support_labels'])
        for draw in range(5):
            for scope,mask in [('original',z['eval_masks'][draw]),('common',z['common_eval'])]:
                probabilities={s:z[s][draw][:,mask] for s in ('expert_prior','expert','full')}
                meta=dict(task=task,draw=draw,scope=scope)
                records.extend(rows_metrics(y[mask],probabilities,meta))
                q=np.flatnonzero(mask)
                for ai,arm in enumerate(ARMS[1:],1):
                    before=z['full'][draw,0].argmax(1)
                    after=z['full'][draw,ai].argmax(1)
                    expert0=z['expert'][draw,0].argmax(1)
                    expert1=z['expert'][draw,ai].argmax(1)
                    for pos in q[before[q]!=after[q]]:
                        events.append(dict(**meta,arm=arm,position=int(pos),sample_id=str(z['ids'][pos]),
                            true_class=int(y[pos]),before=int(before[pos]),after=int(after[pos]),
                            event='rescue' if after[pos]==y[pos] else ('harm' if before[pos]==y[pos] else 'wrong_to_wrong'),
                            expert_argmax_changed=bool(expert0[pos]!=expert1[pos])))
    metrics=pd.DataFrame(records)
    metrics.to_csv(OUT/'target_metrics.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'target_events.csv',index=False)
    key=['task','draw','scope','stage']
    baseline=metrics[metrics.arm=='original'][key+['oa','ba','nll','macro_nll','brier']]
    contrast=metrics.merge(baseline,on=key,suffixes=('','_baseline'),validate='many_to_one')
    for k in ('oa','ba','nll','macro_nll','brier'):
        contrast['delta_'+k]=contrast[k]-contrast[k+'_baseline']
    contrast['selected']=[a==selection[t]['arm'] for t,a in zip(contrast.task,contrast.arm)]
    contrast.to_csv(OUT/'target_contrasts.csv',index=False)
    cols=['oa','ba','delta_oa','delta_ba','delta_nll','delta_brier','rescues','harms']
    summary=contrast.groupby(['task','scope','stage','arm'],sort=False)[cols].mean().reset_index()
    summary.to_csv(OUT/'target_summary.csv',index=False)
    print(summary.query("scope=='original' and stage=='full'").to_string(index=False),flush=True)
    write(OUT/'COMPLETE.json',dict(source_episodes=sum(len(list((OUT/'source'/t).glob('episode_*.npz'))) for t in selection),
        target_tasks=4,support_draws=20,arms=list(ARMS),selected={t:v['arm'] for t,v in selection.items()},
        query_labels_for_adaptation_or_selection=False,neural_training=False,main_thread_unchanged=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['source','target','evaluate'])
    parser.add_argument('--reuse-ports',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        {'source':lambda:source_phase(args.reuse_ports),'target':target_phase,'evaluate':evaluate}[args.phase]()
