"""Distinguish target marginal statistics from centered feature-label association."""

from run_diagnosis import (
    PARENT, OUT as ROOT, SUPPORT, TASKS, SEEDS, DRAWS,
    weights, sufficient, solve, scored, softmax, residual, write,
    np, pd, threadpool_limits,
)

OUT = ROOT / 'association'
ARMS = ('B', 'G', 'M', 'A', 'F')


def freeze():
    frame = pd.read_csv(PARENT / 'manifest.csv', dtype={'mmsi': str})
    label = frame.class_id.to_numpy()
    audits = []
    for task in TASKS:
        for seed in SEEDS:
            folder = PARENT / task / str(seed)
            split = dict(np.load(folder / 'split.npz'))
            z = dict(np.load(folder / 'features.npz'))
            loc = pd.Index(z.pop('rows'))
            features = {'r50': z['r50'], 'dual': np.c_[z['r50'],z['r18']] / np.sqrt(2.)}
            train = loc.get_indexer(split['source'])
            ys = label[split['source']]
            for draw in DRAWS:
                old = ROOT / task / str(seed) / str(draw)
                with np.load(old / 'predictions.npz') as z:
                    query, support = z['query_rows'], z['support_rows']
                    outputs = {f'{stage}_{a}':z[f'{stage}_{a}'] for stage in ('prior','residual') for a in ('B','G','F')}
                si, qi = loc.get_indexer(support), loc.get_indexer(query)
                yt = label[support]
                w = weights(np.r_[ys,yt],len(ys))
                ws, wt = w[:len(ys)], w[len(ys):]
                by_mode, head_arrays = {}, {}
                with np.load(old / 'heads.npz') as z:
                    previous = dict(z)
                for mode, x in features.items():
                    h = np.c_[x.astype(np.float64),np.ones(len(x))]
                    gs, cs = sufficient(h[train],ys,ws)
                    gt, ct = sufficient(h[si],yt,wt)
                    target = 2.*np.eye(8)[yt]-1.
                    mean_h = np.average(h[si],axis=0,weights=wt)
                    mean_y = np.average(target,axis=0,weights=wt)
                    marginal = wt.sum()*np.outer(mean_h,mean_y)
                    centered = ct-marginal
                    heads = {'M':solve(gs+gt,cs+marginal),
                             'A':solve(gs+gt,cs+centered)}
                    err = float(np.max(np.abs(heads['M']+heads['A']-
                        previous[mode+'_G']-previous[mode+'_F'])))
                    if err > 1e-10:
                        raise ValueError('Centered/marginal decomposition does not reconstruct joint readout')
                    audits.append(dict(task=task,split_seed=seed,draw_seed=draw,mode=mode,
                        theta_additivity_error=err, marginal_norm=float(np.linalg.norm(marginal)),
                        centered_norm=float(np.linalg.norm(centered))))
                    by_mode[mode] = {}
                    for arm, theta in heads.items():
                        head_arrays[mode+'_'+arm] = theta
                        prior = h@theta
                        adapted = residual(h[:,:-1],prior,si,yt)
                        by_mode[mode]['prior_'+arm] = prior[qi]
                        by_mode[mode]['residual_'+arm] = adapted[qi]
                for name in by_mode['r50']:
                    outputs[name] = .5*softmax(by_mode['r50'][name])+.5*softmax(by_mode['dual'][name])
                dest = OUT/task/str(seed)/str(draw)
                dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/'predictions.npz',query_rows=query,support_rows=support,**outputs)
                np.savez_compressed(dest/'heads.npz',**head_arrays)
            print('ASSOCIATION FROZEN',task,seed,flush=True)
    pd.DataFrame(audits).to_csv(OUT/'decomposition_check.csv',index=False)
    write(OUT/'PREDICTIONS_FROZEN.json',dict(classifier_solutions=160,support_sets=40,
        neural_fits=0,post_initial_result_diagnosis=True))


def evaluate():
    frame = pd.read_csv(PARENT/'manifest.csv',dtype={'mmsi':str})
    metrics, factors, contrasts = [],[],[]
    pairs = {'full_vs_original':('B','F'),'mean_only_vs_original':('B','M'),
             'association_only_vs_original':('B','A'),
             'association_given_mean':('M','F'),'mean_given_association':('A','F')}
    for task in TASKS:
        for seed in SEEDS:
            for draw in DRAWS:
                meta = dict(task=task,split_seed=seed,draw_seed=draw)
                with np.load(OUT/task/str(seed)/str(draw)/'predictions.npz') as z:
                    query = frame.iloc[z['query_rows']]
                    yy, ships = query.class_id.to_numpy(),query.mmsi.to_numpy()
                    for stage in ('prior','residual'):
                        by_arm = {}
                        for arm in ARMS:
                            mm,_ = scored(yy,z[stage+'_'+arm],ships)
                            metrics.append(dict(**meta,stage=stage,arm=arm,**mm))
                            by_arm[arm] = mm
                        for name,(a,b) in pairs.items():
                            contrasts.append(dict(**meta,stage=stage,comparison=name,
                                **{'delta_'+k:by_arm[b][k]-by_arm[a][k] for k in ('oa','ba','nll','brier')}))
                        for score in ('oa','ba','nll','brier'):
                            g,m,a,f = (by_arm[arm][score] for arm in ('G','M','A','F'))
                            cm,ca = .5*((m-g)+(f-a)),.5*((a-g)+(f-m))
                            if abs(cm+ca-(f-g)) > 1e-10:
                                raise ValueError('Metric attribution does not sum')
                            factors.append(dict(**meta,stage=stage,metric=score,
                                marginal=cm,association=ca,total=f-g,interaction=f-m-a+g))
    for name, rows, keys, values in (
        ('metrics',metrics,['task','stage','arm'],['oa','ba','nll','brier']),
        ('contrasts',contrasts,['task','stage','comparison'],['delta_oa','delta_ba','delta_nll','delta_brier']),
        ('factors',factors,['task','stage','metric'],['marginal','association','total','interaction']),
    ):
        data = pd.DataFrame(rows)
        data.to_csv(OUT/(name+'.csv'),index=False)
        records = []
        for group, part in data.groupby(keys,sort=False):
            record = dict(zip(keys,group))
            for col in values:
                record[col] = float(part[col].mean())
                record[col+'_support_sd'] = float(np.sqrt(part.groupby('split_seed')[col].var(ddof=1).mean()))
                if name == 'contrasts':
                    record[col+'_positive'] = int((part[col]>1e-10).sum())
                    record[col+'_negative'] = int((part[col]<-1e-10).sum())
            records.append(record)
        summary = pd.DataFrame(records)
        summary.to_csv(OUT/(name+'_summary.csv'),index=False)
        print(name.upper(),flush=True)
        print(summary.loc[summary.stage.eq('residual')].to_string(index=False),flush=True)
    write(OUT/'COMPLETE.json',dict(support_sets=40,new_classifier_solutions=160,
        query_labels_used_only_for_evaluation=True,main_model_changed=False))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        freeze()
        evaluate()
