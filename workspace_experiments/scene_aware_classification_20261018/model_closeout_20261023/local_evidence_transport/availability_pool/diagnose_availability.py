"""Source-only fixed-reference availability accounting. No fitting or training."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
UP = PARENT.parent / 'regional_utility_pilot/upstream'
sys.path.insert(0, str(PARENT))
import match_readout as match

ARMS = ('pooled', 'independent', 'anchored', 'body_match')
REGION_NAMES = ('body', 'side', 'end')
POL_NAMES = ('VV', 'VH')


def readouts(q, s, qm, sm):
    q = q[:,None,None].expand(*s.shape[:-2],q.shape[-2],q.shape[-1])
    qm = qm[:,None,None].expand(*sm.shape[:-1],qm.shape[-1])
    sims = [torch.einsum('...id,...jd->...ij',q[...,r,:],s[...,r,:]) for r in match.REGIONS]
    qa, sa = [qm[...,r] for r in match.REGIONS], [sm[...,r] for r in match.REGIONS]
    pooled, independent, ts = [], [], []
    for sim, a, b in zip(sims,qa,sa):
        aa = a/a.sum(-1,keepdim=True).clamp_min(1e-20)
        bb = b/b.sum(-1,keepdim=True).clamp_min(1e-20)
        pooled.append(match.pair_score(aa[..., :,None]*bb[...,None,:],sim,a,b))
        t,_,_,_ = match.sinkhorn(sim,a,b)
        ts.append(t)
        independent.append(match.pair_score(t,sim,a,b))
    end_sim = torch.einsum('...id,...jd->...ij',q[...,match.END,:],s[...,match.END,:])
    end_flag = torch.as_tensor(match.IS_END,device=q.device)
    anchored = [match.pair_score(ts[0],sims[0],qa[0],sa[0]),
                match.pair_score(ts[0],sims[1],qa[1],sa[1]),
                match.pair_score(ts[0],end_sim,qm[...,match.END]*end_flag,sm[...,match.END]*end_flag)]
    out = dict(pooled=torch.stack(pooled,-2), independent=torch.stack(independent,-2),
               anchored=torch.stack(anchored,-2))
    out['body_match'] = out['anchored'].clone()
    out['body_match'][...,1:,:] = 0
    return out


def available(region_exists, refs, arm):
    query = region_exists.copy()
    support = region_exists[refs].copy()
    if arm in ('anchored', 'body_match'):
        query &= query[...,0,None]
        support &= support[...,0,None]
    if arm == 'body_match':
        query[...,1:] = False
        support[...,1:] = False
    return query, support, query[:,None,None]&support


def coverage_rows(table, layer, arm, query, support, both, true_labels, train):
    rows = []
    for role in table.role.unique():
        ids = np.flatnonzero(table.role.eq(role))
        for c in range(8):
            for p in range(2):
                for r in range(1 if arm=='body_match' else 3):
                    q = query[ids,p,r]
                    rc = support[ids,c,:,p,r].sum(1)
                    ac = both[ids,c,:,p,r].sum(1)
                    rows.append(dict(layer=layer,arm=arm,role=role,scope='candidate_class',class_id=c,
                        pol=POL_NAMES[p],region=REGION_NAMES[r],n_queries=len(ids),query_available=int(q.sum()),
                        query_missing=int((~q).sum()),reference_count0=int((rc==0).sum()),
                        reference_count1=int((rc==1).sum()),reference_count2=int((rc==2).sum()),
                        joint_count0=int((ac==0).sum()),joint_count1=int((ac==1).sum()),joint_count2=int((ac==2).sum()),
                        query_available_ref0=int((q&(rc==0)).sum()),query_available_ref1=int((q&(rc==1)).sum()),
                        query_available_ref2=int((q&(rc==2)).sum())))
    for c in range(8):
        ids = train[true_labels==c]
        for p in range(2):
            for r in range(1 if arm=='body_match' else 3):
                q = query[ids,p,r]
                rc = support[ids,c,:,p,r].sum(1)
                ac = both[ids,c,:,p,r].sum(1)
                rows.append(dict(layer=layer,arm=arm,role='utility_train',scope='true_class',class_id=c,
                    pol=POL_NAMES[p],region=REGION_NAMES[r],n_queries=len(ids),query_available=int(q.sum()),
                    query_missing=int((~q).sum()),reference_count0=int((rc==0).sum()),
                    reference_count1=int((rc==1).sum()),reference_count2=int((rc==2).sum()),
                    joint_count0=int((ac==0).sum()),joint_count1=int((ac==1).sum()),joint_count2=int((ac==2).sum()),
                    query_available_ref0=int((q&(rc==0)).sum()),query_available_ref1=int((q&(rc==1)).sum()),
                    query_available_ref2=int((q&(rc==2)).sum())))
    return rows


def label_diagnostics(layer, arm, old, new, count, table, train, y, rival, base_correct):
    grouped, cases = [], []
    before = old[train,y]-old[train,rival]
    after = new[train,y]-new[train,rival]
    true_count, rival_count = count[train,y], count[train,rival]
    for c in [-1]+list(range(8)):
        take = np.ones(len(train),bool) if c==-1 else y==c
        for state in ('all','base_correct','base_wrong'):
            use = take if state=='all' else take&(base_correct if state=='base_correct' else ~base_correct)
            if not use.any():
                continue
            for p in range(2):
                for r in range(1 if arm=='body_match' else 3):
                    for j,stat in enumerate(('similarity_mass','matched_mass')):
                        b,a = before[use,p,r,j],after[use,p,r,j]
                        d = a-b
                        grouped.append(dict(layer=layer,arm=arm,true_class=c,base_state=state,
                            pol=POL_NAMES[p],region=REGION_NAMES[r],stat=stat,n=int(use.sum()),
                            current_descriptor_margin_mean=float(b.mean()),valid_descriptor_margin_mean=float(a.mean()),
                            mean_change=float(d.mean()),increased=int((d>1e-7).sum()),decreased=int((d< -1e-7).sum()),
                            unchanged=int((abs(d)<=1e-7).sum()),nonpositive_to_positive=int(((b<=0)&(a>0)).sum()),
                            positive_to_nonpositive=int(((b>0)&(a<=0)).sum()),
                            true_ref_count1=int((true_count[use,p,r]==1).sum()),
                            rival_ref_count1=int((rival_count[use,p,r]==1).sum())))
    for j,i in enumerate(train):
        for p in range(2):
            for r in range(1 if arm=='body_match' else 3):
                tc,rc = int(true_count[j,p,r]),int(rival_count[j,p,r])
                if tc!=1 and rc!=1:
                    continue
                cases.append(dict(layer=layer,arm=arm,source_index=int(i),obs_id=int(table.iloc[i].obs_id),
                    true_class=int(y[j]),fixed_B_rival=int(rival[j]),B_correct=bool(base_correct[j]),
                    pol=POL_NAMES[p],region=REGION_NAMES[r],true_available_refs=tc,rival_available_refs=rc,
                    current_similarity_margin=float(before[j,p,r,0]),valid_similarity_margin=float(after[j,p,r,0]),
                    current_matched_mass_margin=float(before[j,p,r,1]),valid_matched_mass_margin=float(after[j,p,r,1])))
    return grouped,cases


def main():
    tick = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    table = pd.read_csv(UP/'table.csv',dtype={'mmsi':str,'instance_id':str})
    train = np.flatnonzero(table.role.eq('utility_train'))
    assert len(train)==644
    y = table.iloc[train].class_id.to_numpy()
    base = np.load(UP/'strategies_seed17.npz')['logits'][train,2].astype(np.float32)
    rival_scores = base.copy()
    rival_scores[np.arange(len(train)),y] = -np.inf
    rival = rival_scores.argmax(1)
    base_correct = base.argmax(1)==y
    index = table[['row_index','obs_id','instance_id','role','mmsi','pass_group']].copy()
    index.insert(0,'source_index',np.arange(len(table)))
    index.to_csv(HERE/'source_index.csv',index=False)
    src = np.load(PARENT/'tokens.npz')
    np.testing.assert_array_equal(src['instance_ids'].astype(str),table.instance_id.to_numpy())
    mass, valid = src['mass'],src['valid'].astype(bool)
    region_exists = np.stack([mass[...,r].sum(-1)>0 for r in match.REGIONS],-1)&valid[...,None]
    coverage, margins, cases, bank_rows, receipts = [], [], [], [], []
    for layer in ('stem','layer1'):
        stage = time.perf_counter()
        raw = src[layer]
        proj = np.load(PARENT/f'projection_{layer}.npz')
        flat = raw.reshape(-1,raw.shape[-1])
        x = flat@proj['components'].T
        x -= proj['mean'].reshape(1,-1)@proj['components'].T
        x = match.unit(x).reshape(*raw.shape[:-1],32).astype(np.float32)
        x[mass<=0] = 0
        parent = np.load(PARENT/f'scores_{layer}.npz')
        refs = parent['refs'].copy()
        bank = proj['bank']
        for c in range(8):
            assert np.isin(refs[:,c],bank[c]).all()
            for p in range(2):
                for r in range(3):
                    bank_rows.append(dict(layer=layer,class_id=c,pol=POL_NAMES[p],region=REGION_NAMES[r],
                        references=8,region_available=int(region_exists[bank[c],p,r].sum()),
                        body_and_region_available=int((region_exists[bank[c],p,0]&region_exists[bank[c],p,r]).sum())))
        per = {a:np.zeros((len(table),8,2,2,3,2),np.float32) for a in ARMS}
        with torch.inference_mode():
            for start in range(0,len(table),32):
                sl = slice(start,start+32)
                r = refs[sl]
                args = [torch.tensor(v,device='cuda') for v in (x[sl],x[r],mass[sl],mass[r])]
                values = readouts(*args)
                for arm,value in values.items():
                    per[arm][sl] = value.cpu().numpy()
                if start%1024==0:
                    print(layer,start,'/',len(table),'seconds',round(time.perf_counter()-stage,1),flush=True)
        original, corrected, per_avail = {}, {}, {}
        layer_checks = []
        for arm in ARMS:
            current = per[arm].mean(2)
            baseline = parent['anchored' if arm=='body_match' else arm].copy()
            if arm=='body_match':
                baseline[:,:,:,1:] = 0
            replay = float(abs(current-baseline).max())
            assert replay<2e-6,(layer,arm,replay)
            query,support,avail = available(region_exists,refs,arm)
            count = avail.sum(2)
            numerator = (per[arm]*avail[...,None]).sum(2)
            new = np.divide(numerator,count[...,None],out=np.zeros_like(numerator),where=count[...,None]>0)
            unavailable_max = float(abs(per[arm][~avail]).max(initial=0))
            assert unavailable_max<1e-6,(layer,arm,unavailable_max)
            assert np.array_equal(new[count==2],current[count==2])
            assert not new[count==0].any()
            half_error = float(abs(new[count==1]-2*current[count==1]).max(initial=0))
            assert half_error<2e-6
            original[arm],corrected[arm] = current,new
            original[f'{arm}_count'] = count.astype(np.uint8)
            corrected[f'{arm}_count'] = count.astype(np.uint8)
            per_avail[f'{arm}_available'] = avail
            coverage.extend(coverage_rows(table,layer,arm,query,support,avail,y,train))
            rows,detail = label_diagnostics(layer,arm,current,new,count,table,train,y,rival,base_correct)
            margins.extend(rows);cases.extend(detail)
            true_counts = count[train,y]
            enabled = slice(0,1) if arm=='body_match' else slice(None)
            slots = true_counts[:,:,enabled]
            lowmass = per[arm][avail,1]
            layer_checks.append(dict(arm=arm,current_cache_replay_max_error=replay,
                unavailable_score_max_abs=unavailable_max,one_ref_double_max_error=half_error,
                utility_true_class_slots_total=int(slots.size),utility_true_class_slots_count1=int((slots==1).sum()),
                utility_queries_any_true_count1=int((slots==1).any((1,2)).sum()),
                structurally_available_semantic_mass_quantiles=np.quantile(lowmass,[0,.01,.5,1]).tolist()))
        common = dict(refs=refs,indices=table.row_index.to_numpy(),instance_ids=table.instance_id.to_numpy(str),
            region_names=np.array(REGION_NAMES),pol_names=np.array(POL_NAMES))
        np.savez_compressed(HERE/f'per_reference_{layer}.npz',**per,**per_avail,region_exists=region_exists,**common)
        np.savez_compressed(HERE/f'scores_current_{layer}.npz',**original,**common)
        np.savez_compressed(HERE/f'scores_valid_{layer}.npz',**corrected,**common)
        receipts.append(dict(layer=layer,seconds=time.perf_counter()-stage,checks=layer_checks))
        print(layer,'complete',flush=True)
    pd.DataFrame(coverage).to_csv(HERE/'structural_coverage.csv',index=False)
    pd.DataFrame(bank_rows).to_csv(HERE/'bank_coverage.csv',index=False)
    pd.DataFrame(margins).to_csv(HERE/'utility_descriptor_margins.csv',index=False)
    pd.DataFrame(cases).to_csv(HERE/'utility_one_reference_cases.csv',index=False)
    summary = dict(seconds=time.perf_counter()-tick,source_observations=len(table),utility_train=len(train),
        label_scope='utility_train only; seed17 full fixes the highest nontrue competitor',
        structural_cache_scope='all source roles, no held-role label diagnostics',
        target_data_read=False,new_fits=0,new_selections=0,backbone_extractions=0,
        denominator='count of structurally available fixed references; never semantic matched mass',layers=receipts)
    (HERE/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
