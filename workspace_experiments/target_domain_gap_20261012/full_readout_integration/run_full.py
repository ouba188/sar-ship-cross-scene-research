"""Expert-prior adaptation inside the unchanged historical complete system."""

from core import *
import sys
import gc

HELPERS=Path('E:/Hermes/paper_tracking/sensitivity/oof_reliability_check')
sys.path.insert(0,str(HELPERS))
from unified_head_expert import ExpertContext,_conditions,_expert_inputs
from dual_evidence_core import load_task,base_cross
from unified_head_core import coefficients
from compact_fusion_source import ROOT,SOURCE
from prepare_learned_frequency import DATASETS

TASK_ROOT=ROOT/'learned_frequency_residual_20261009/tasks'
FOLDERS=[TASK_ROOT/'self24ports'/p for p in ('Los Angeles','Port Klang','Santos')]+[TASK_ROOT/'oss2']
ARMS=('original','joint','source_fixed','SS','TS','ST')
STAGES=('expert_prior','expert','full')
DEST=OUT/'full'
MATCHED=False


def head_inputs(ctx,data):
    ctx._prepare_source()
    np.testing.assert_array_equal(ctx.source_rows,data['source_rows'])
    if ctx.dataset=='self24ports':
        transform=ctx.transforms['joint_expert']
        xs=ctx.augmented@transform
        bank=np.load(ctx.parent/'target_bank.npy',mmap_mode='r')
        xt=np.c_[bank,_conditions()[data['rows']]]@transform
    else:
        xs=_expert_inputs(ctx.raw[ctx.source_rows],ctx.extra,ctx.model) if MATCHED else ctx.fit_inputs
        protocol=json.loads((ctx.joint.parent/'protocol.json').read_text())
        with np.load(Path(protocol['target'])/'features.npz') as z:
            raw=z['paired_r50']
        extra=np.load(ctx.joint/'target_extra.npy',mmap_mode='r')
        xt=_expert_inputs(raw,extra,ctx.model)
    return np.c_[xs.astype(float),np.ones(len(xs))],np.c_[xt.astype(float),np.ones(len(xt))]


def freeze():
    receipts=[]
    for folder in FOLDERS:
        info=json.loads((folder/'PREPARED.json').read_text())
        task=info['task']
        data=load_task(folder)
        ctx=ExpertContext(folder)
        hs,ht=head_inputs(ctx,data)
        stats=source_statistics(hs,ctx.source_labels)
        old_path=(ROOT/'backbone_ablation_20261010/targets'/info['dataset']/task/'predictions.npz') if MATCHED \
            else (ROOT/'unified_head_20261010/targets'/info['dataset']/task/'policies.npz')
        with np.load(old_path) as z:
            old={k:z[k] for k in z.files}
        if MATCHED:
            with np.load(old_path.parent/'coefficients.npz') as z:
                matched_head={k:z[k] for k in z.files}
        np.testing.assert_array_equal(old['rows'],data['rows'])
        np.testing.assert_array_equal(old['support'],data['support'])
        n=len(data['rows'])
        probabilities={s:np.empty((5,len(ARMS),n,8),float) for s in STAGES}
        delta_stats=[]
        for draw,support in enumerate(data['support']):
            labels=data['support_labels'][draw]
            if data['eval_masks'][draw,support].any():
                raise ValueError('Support enters query evaluation')
            pe,ke=ctx.target_components(data,support)
            if MATCHED:
                pe=(ht[:,:-1]@matched_head['dual_coef']+matched_head['dual_intercept']).astype(np.float32).astype(float)
            kb=base_cross(data,slice(None),support,info['eta'])
            base=data['prior']+kb@coefficients(kb[support],data['prior'][support],labels,8.)
            bp=softmax(base,axis=1)
            heads,details=fit_variants(stats,ht[support],labels)
            theta0=stats['theta0']
            record=dict(task=task,draw=draw,head_width=ht.shape[1]-1,
                source_head_prediction_gap=float(np.max(np.abs(ht@theta0-pe))),**details)
            for ai,arm in enumerate(ARMS):
                delta=ht@(heads[arm]-theta0)
                prior=pe+delta
                if arm=='original':
                    np.testing.assert_array_equal(prior,pe)
                adapted=prior+ke@coefficients(ke[support],prior[support],labels,8.)
                ep=softmax(adapted,axis=1)
                probabilities['expert_prior'][draw,ai]=softmax(prior,axis=1)
                probabilities['expert'][draw,ai]=ep
                probabilities['full'][draw,ai]=.5*(bp+ep)
                delta_stats.append(dict(task=task,draw=draw,arm=arm,
                    prior_delta_rms=float(np.sqrt(np.mean(delta**2))),
                    support_prior_delta_rms=float(np.sqrt(np.mean(delta[support]**2)))))
                if arm=='original':
                    if MATCHED:
                        expected=pe+ke@matched_head[f'{draw}_dual_refit_expert']
                    else:
                        with np.load(folder/'target.npz') as z:
                            expected=z['expert'][draw]
                    np.testing.assert_allclose(adapted,expected,atol=3e-5,rtol=1e-6)
                    record['expert_replay_error']=float(np.max(np.abs(adapted-expected)))
                    names=('base','dual_refit_expert','dual_refit_late') if MATCHED else ('base','expert','late_softmax_fixed')
                    for values,name in zip((bp,ep,.5*(bp+ep)),names):
                        np.testing.assert_array_equal(values.argmax(1),old['predictions'][draw,list(old['modes']).index(name)])
            record['original_classes_exact']=True
            receipts.append(record)
        dest=DEST/task
        dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'predictions.npz',**probabilities,arms=np.array(ARMS),
            rows=data['rows'],ids=data['ids'],support=data['support'],support_labels=data['support_labels'],
            eval_masks=data['eval_masks'],common_eval=data['eval_masks'].all(0))
        pd.DataFrame(delta_stats).to_csv(dest/'prediction_changes.csv',index=False)
        write(dest/'FROZEN.json',dict(task=task,dataset=info['dataset'],source_n=len(hs),
            query_pool=n,source_head_delta_anchor=True,neural_fits=0,
            historical_source_fit_recipe_preserved=True,matched_post_tail_source=MATCHED))
        print('FULL FROZEN',task,len(hs),n,flush=True)
        del ctx,hs,ht,stats,probabilities,data
        gc.collect()
    write(DEST/'PREDICTIONS_FROZEN.json',dict(tasks=len(FOLDERS),support_sets=5*len(FOLDERS),source_solves=len(FOLDERS),
        adapted_solves=25*len(FOLDERS),neural_fits=0,all_baseline_predictions_exact=True,details=receipts))


def evaluate():
    results=[[] for _ in range(5)]
    inventories=[]
    for folder in FOLDERS:
        info=json.loads((folder/'PREPARED.json').read_text())
        task=info['task']
        with np.load(DEST/task/'predictions.npz') as z:
            saved={k:z[k] for k in z.files}
        if info['dataset']=='self24ports':
            frame=pd.read_csv(SOURCE/'manifest.csv',usecols=['instance_id','class_id','mmsi'],
                              dtype={'mmsi':str},keep_default_na=False).iloc[saved['rows']]
            idcol,shipcol='instance_id','mmsi'
        else:
            frame=pd.read_csv(ROOT/DATASETS['oss2'][0]/'target_manifest.csv',keep_default_na=False,
                              dtype={'mmsi_or_ship_id':str}).iloc[saved['rows']]
            idcol,shipcol='sample_id','mmsi_or_ship_id'
        np.testing.assert_array_equal(frame[idcol].to_numpy(str),saved['ids'])
        y=frame.class_id.to_numpy()
        np.testing.assert_array_equal(y[saved['support']],saved['support_labels'])
        for draw,support in enumerate(saved['support']):
            masks={'original':saved['eval_masks'][draw],'common':saved['common_eval']}
            outputs={s:saved[s][draw] for s in STAGES}
            meta=dict(task=task,split_seed=0,draw=draw)
            values=evaluate_arrays(meta,y,masks,outputs,STAGES,ARMS)
            for accumulator,rows in zip(results,values):
                accumulator.extend(rows)
            ship=frame[shipcol].astype(str)
            known=ship.str.fullmatch(r'[0-9]{9}') & ship.ne('000000000')
            support_ships=set(ship.iloc[support][known.iloc[support]])
            for scope,mask in masks.items():
                inventories.append(dict(task=task,draw=draw,scope=scope,query_n=int(mask.sum()),
                    support_unique_known_ships=len(support_ships),
                    support_unknown=int((~known.iloc[support]).sum()),
                    query_known_ship_overlap=int((mask & known.to_numpy() & ship.isin(support_ships).to_numpy()).sum())))
    frames=save_reports(DEST,results)
    pd.DataFrame(inventories).to_csv(DEST/'cohort_inventory.csv',index=False)
    chosen=frames['metrics'].query("scope=='original' and stage=='full' and arm in ['original','joint','source_fixed']")
    print('FULL PRIMARY',flush=True)
    print(chosen.groupby(['task','arm'],sort=False)[['oa','ba','nll','brier']].mean().to_string(),flush=True)
    write(DEST/'COMPLETE.json',dict(tasks=len(FOLDERS),support_sets=5*len(FOLDERS),neural_fits=0,
        main_model_unchanged=True,query_label_fitting=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--matched-oss2',action='store_true')
    args=parser.parse_args()
    if args.matched_oss2:
        MATCHED=True
        FOLDERS=[TASK_ROOT/'oss2']
        DEST=OUT/'matched_full'
    with threadpool_limits(limits=2):
        freeze()
        evaluate()
