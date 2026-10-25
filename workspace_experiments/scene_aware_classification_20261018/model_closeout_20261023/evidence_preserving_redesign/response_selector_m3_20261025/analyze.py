"""Matched source/development decisions and original-scale query/reference panels."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
import json
import numpy as np
import pandas as pd
import torch
import train as run

native = run.load('selector_native_analysis', HERE.parent/'anchored_m3_20261025/analyze.py')
stats = native.previous
native.HERE = stats.HERE = HERE
stats.PAIRS = [(a, 'old_full') for a in run.ARMS] + [
    ('response', 'summary'), ('combined', 'summary'), ('combined', 'response'),
    ('response', 'fixed_action'), ('combined', 'fixed_action')]


def action_name(a):
    if a == 0:
        return 'no_op'
    t = int(a)-1
    return f"{('VV','VH')[t//9]}_{('body','sides','ends')[(t%9)//3]}_gabor{(4,8,16)[t%3]}"


def load_inputs(seed):
    with np.load(HERE/'results'/f'inputs_{seed}.npz') as z:
        return {k:z[k].copy() for k in z.files}


@torch.inference_mode()
def source_final(data, seed, candidates):
    arrays = load_inputs(seed)
    x = torch.tensor(arrays['x'], device='cuda')
    extra = torch.tensor(arrays['extra'], device='cuda')
    valid = torch.tensor(arrays['valid'], device='cuda')
    response, _ = run.response_features(candidates, data.tr,
        (arrays['response_mu'], arrays['response_sd']))
    y = data.table.iloc[:data.n].class_id.to_numpy(int)
    ce, risk, scale = run.risk_targets(candidates, y, data.tr)
    rows = []
    for arm in run.LEARNED:
        for state in ('selected', 'final'):
            path = HERE/'models'/f"{'final_' if state == 'final' else ''}{arm}_{seed}.pt"
            ck = torch.load(path, map_location='cpu', weights_only=False)
            model = run.make_model(arm, seed); model.load_state_dict(ck['state']); model.eval()
            for role, ids in (('utility_train', data.tr), ('selection', data.va)):
                pred, actions, prob = run.choose(model, x, response, extra, valid, candidates, ids)
                rows.append(dict(seed=seed, arm=arm, state=state, role=role,
                    step=100 if state == 'final' else ck['receipt']['selected_step'],
                    expected_NLL=float((prob*ce[ids].cpu().numpy()).sum(1).mean()),
                    expected_weighted_CE_delta=float((prob*risk[ids].cpu().numpy()).sum(1).mean()*scale),
                    action_fraction=float((actions!=0).mean()),
                    **(stats.effect(pred,candidates[ids,0],y[ids]) | run.metrics(y[ids],pred))))
    return rows


def select_cases(data, scores, actions):
    ids=data.te; y=data.table.iloc[ids].class_id.to_numpy(int)
    chosen=[]; used=set()
    for arm in ('combined','response','summary'):
        for kind in ('rescue','harm'):
            found=[]
            for seed in run.SEEDS:
                new=scores[seed,arm][ids].argmax(1)==y
                old=scores[seed,'old_full'][ids].argmax(1)==y
                take=new & ~old if kind=='rescue' else ~new & old
                for k in np.flatnonzero(take):
                    j=int(ids[k]); obs=int(data.table.iloc[j].obs_id)
                    if obs not in used:found.append((seed,obs,j))
            if found:
                seed,obs,j=sorted(found)[0]; used.add(obs)
                chosen.append(dict(seed=seed,arm=arm,obs_id=obs,index=j,case=kind,
                                   action=int(actions[seed,arm][j])))
    # Cover remaining combined-policy correctness flips, not only one exemplar.
    for seed in run.SEEDS:
        new=scores[seed,'combined'][ids].argmax(1)==y
        old=scores[seed,'old_full'][ids].argmax(1)==y
        for k in np.flatnonzero(new!=old):
            j=int(ids[k]);obs=int(data.table.iloc[j].obs_id)
            if obs not in used:
                used.add(obs)
                chosen.append(dict(seed=seed,arm='combined',obs_id=obs,index=j,
                    case='rescue' if new[k] else 'harm',action=int(actions[seed,'combined'][j])))
    for kind in ('CC','WW'):
        found=[]
        for seed in run.SEEDS:
            arm='combined'
            new=scores[seed,arm][ids].argmax(1)==y
            old=scores[seed,'old_full'][ids].argmax(1)==y
            take=new & old if kind=='CC' else ~new & ~old
            change=stats.margin(scores[seed,arm][ids],y)-stats.margin(scores[seed,'old_full'][ids],y)
            for k in np.flatnonzero(take):
                j=int(ids[k]); obs=int(data.table.iloc[j].obs_id)
                if obs not in used:found.append((-abs(float(change[k])),seed,obs,j))
        if found:
            _,seed,obs,j=sorted(found)[0]; used.add(obs)
            chosen.append(dict(seed=seed,arm='combined',obs_id=obs,index=j,case=kind,
                               action=int(actions[seed,'combined'][j])))
    return chosen


def main():
    assert (HERE/'complete.json').exists()
    torch.set_num_threads(2)
    data=run.Data(); scores={}; actions={}; paired=[]; use=[]; oracle=[]; diagnostics=[]
    for seed in run.SEEDS:
        with np.load(HERE/'results'/f'candidates_{seed}.npz') as z:
            np.testing.assert_array_equal(z['up_index'],data.up[:data.n])
            cand=z['score'].copy()
        valid=load_inputs(seed)['valid']
        diagnostics.extend(source_final(data,seed,cand))
        for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
            y=data.table.iloc[ids].class_id.to_numpy(int)
            correct=cand[ids].argmax(-1)==y[:,None]
            rescue=(correct[:,1:] & ~correct[:,:1] & valid[ids,1:]).any(1)
            harm=(~correct[:,1:] & correct[:,:1] & valid[ids,1:]).any(1)
            oracle.append(dict(seed=seed,role=role,n=len(ids),rescuable=int(rescue.sum()),harmable=int(harm.sum()),
                               note='Label-oracle headroom, not learned performance'))
        for arm in ('old_full',)+run.ARMS:
            with np.load(HERE/'results'/f'{arm}_{seed}.npz') as z:
                scores[seed,arm]=z['score'].copy(); actions[seed,arm]=z['action'].copy()
            np.testing.assert_array_equal(scores[seed,arm],cand[np.arange(data.n),actions[seed,arm]])
            assert valid[np.arange(data.n),actions[seed,arm]].all()
            for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
                y=data.table.iloc[ids].class_id.to_numpy(int)
                pred=scores[seed,arm][ids]; old=cand[ids,0]
                paired.append(dict(seed=seed,arm=arm,role=role,**stats.effect(pred,old,y),
                    mean_margin_change=float((stats.margin(pred,y)-stats.margin(old,y)).mean())))
                for a in range(19):
                    take=actions[seed,arm][ids]==a
                    if take.any():
                        use.append(dict(seed=seed,arm=arm,role=role,action=a,name=action_name(a),
                            **stats.effect(pred[take],old[take],y[take])))
    receipt=stats.statistics(data,scores)
    pd.DataFrame(paired).to_csv(HERE/'source_vs_development.csv',index=False)
    pd.DataFrame(use).to_csv(HERE/'action_usage.csv',index=False)
    pd.DataFrame(oracle).to_csv(HERE/'oracle_headroom.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(HERE/'source_soft_hard.csv',index=False)
    chosen=select_cases(data,scores,actions)
    geo=run.load('selector_geo',run.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up=pd.read_csv(run.BASE/'regional_utility_pilot/upstream/table.csv')
    view_valid=np.load(run.BASE/'regional_utility_pilot/upstream/summaries.npz')['valid']
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs=np.load(run.LOCAL/'scores_layer1.npz')['refs']; records=[]
    for case in chosen:
        seed,j,arm=case['seed'],case['index'],case['arm']
        before,after=scores[seed,'old_full'][j],scores[seed,arm][j]
        path,provenance,true,rival=native.panel(case,data,before,after,geo,up,view_valid,snapshots,refs)
        case.update(action_name=action_name(case['action']),panel=str(path),provenance=provenance,
            true=true,rival=rival,old_prediction=int(before.argmax()),new_prediction=int(after.argmax()),
            before=before,after=after,score_delta=after-before,
            old_margin=float(stats.margin(before[None],np.array([true]))[0]),
            new_margin=float(stats.margin(after[None],np.array([true]))[0]))
        records.append(case)
    run.write_json(HERE/'analysis/cases.json',records)
    run.write_json(HERE/'analysis_receipt.json',dict(**receipt,panels=len(records),
        note='Panels generated; manual visual inspection recorded separately.'))
    print(pd.DataFrame(paired).query("role=='evaluation'").to_string(index=False))
    print('PANELS', [c['panel'] for c in records],flush=True)


if __name__=='__main__':main()
