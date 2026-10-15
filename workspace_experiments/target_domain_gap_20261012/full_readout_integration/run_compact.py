"""Replay the same class-mixture interventions in the prior 40 compact episodes."""

from core import *
import sys
sys.path.insert(0,str(OUT.parent))
from experiment import residual

PARENT=OUT.parent
ROOT=PARENT/'readout_root_diagnosis'
DEST=OUT/'compact'
TASKS=('Los Angeles','Port Klang','Santos','OpenSARShip2')
SEEDS=(31415,27182)
DRAWS=(1201,1202,1203,1204,1205)
ARMS=('original','joint','source_fixed','SS','TS','ST')


def freeze():
    frame=pd.read_csv(PARENT/'manifest.csv')
    labels=frame.class_id.to_numpy()
    checks=[]
    for task in TASKS:
        for seed in SEEDS:
            folder=PARENT/task/str(seed)
            split=dict(np.load(folder/'split.npz'))
            z=dict(np.load(folder/'features.npz'))
            loc=pd.Index(z.pop('rows'))
            train=loc.get_indexer(split['source'])
            features={'r50':z['r50'],'dual':np.c_[z['r50'],z['r18']]/np.sqrt(2.)}
            cache={}
            for mode,x in features.items():
                h=np.c_[x.astype(float),np.ones(len(x))]
                cache[mode]=(h,source_statistics(h[train],labels[split['source']]))
            for draw in DRAWS:
                parent=ROOT/task/str(seed)/str(draw)
                with np.load(parent/'predictions.npz') as z:
                    support,query=z['support_rows'],z['query_rows']
                    old={a:z['residual_'+b] for a,b in [('original','B'),('joint','F'),('source_fixed','T')]}
                si,qi=loc.get_indexer(support),loc.get_indexer(query)
                sy=labels[support]
                by_mode={}
                for mode,(h,stats) in cache.items():
                    heads,details=fit_variants(stats,h[si],sy)
                    by_mode[mode]={}
                    for arm,theta in heads.items():
                        prior=h@theta
                        by_mode[mode][arm]=residual(h[:,:-1],prior,si,sy)[qi]
                    checks.append(dict(task=task,split_seed=seed,draw=draw,mode=mode,**details))
                fused={a:.5*softmax(by_mode['r50'][a],axis=1)+.5*softmax(by_mode['dual'][a],axis=1) for a in ARMS}
                for arm,reference in old.items():
                    np.testing.assert_allclose(fused[arm],reference,atol=1e-10,rtol=1e-10)
                    np.testing.assert_array_equal(fused[arm].argmax(1),reference.argmax(1))
                dest=DEST/task/str(seed)/str(draw)
                dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/'predictions.npz',rows=query,support=support,**fused)
            print('COMPACT FROZEN',task,seed,flush=True)
    write(DEST/'PREDICTIONS_FROZEN.json',dict(support_sets=40,neural_fits=0,
        source_solves=16,adapted_solves=400,original_joint_fixed_replayed=True,details=checks))


def evaluate():
    frame=pd.read_csv(PARENT/'manifest.csv')
    results=[[] for _ in range(5)]
    for task in TASKS:
        for seed in SEEDS:
            for draw in DRAWS:
                with np.load(DEST/task/str(seed)/str(draw)/'predictions.npz') as z:
                    y=frame.iloc[z['rows']].class_id.to_numpy()
                    values=np.stack([z[a] for a in ARMS])
                rows=evaluate_arrays(dict(task=task,split_seed=seed,draw=draw),y,
                    {'fixed_query':np.ones(len(y),bool)},{'full':values},('full',),ARMS)
                for accumulator,part in zip(results,rows):
                    accumulator.extend(part)
    frames=save_reports(DEST,results)
    print('COMPACT ALL',flush=True)
    print(frames['metrics'].groupby(['task','arm'],sort=False)[['oa','ba']].mean().to_string(),flush=True)
    write(DEST/'COMPLETE.json',dict(support_sets=40,neural_fits=0,query_label_fitting=False))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        freeze()
        evaluate()
