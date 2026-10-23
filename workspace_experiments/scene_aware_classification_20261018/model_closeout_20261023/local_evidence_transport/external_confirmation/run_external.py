"""Frozen source-pilot replay and isolated external inference, never fitting."""
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
from torchvision.models import resnet50

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent
UP = LOCAL.parent / 'regional_utility_pilot/upstream'
sys.path.insert(0, str(UP))
sys.path.insert(0, str(LOCAL))
import pipeline as pipe
import extract_source as spatial
import extract_tokens as tokens
ROOT, OLD, geo = pipe.ROOT, pipe.OLD, tokens.geo


def save_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def error(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.max(np.abs(a.astype(float)-b.astype(float))))


def prepare():
    full = pd.read_csv(ROOT / 'cohort.csv', dtype={'mmsi': str, 'instance_id': str})
    full['row_index'] = np.arange(len(full))
    full['pass_group'] = full['product'].map(lambda v: '_'.join([v.split('_')[0], v.split('_')[6], v.split('_')[7]]))
    source = pd.read_csv(UP / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
    folds = [np.load(ROOT / f'baseline/fold{f}_inputs.npz')['test'] for f in range(3)]
    ids = np.unique(np.concatenate(folds))
    assert len(ids) == 4744 and sum(map(len, folds)) == len(ids)
    audit = full.iloc[ids].copy()
    audit['original_test_fold'] = -1
    for f, ix in enumerate(folds):
        audit.loc[audit.row_index.isin(ix), 'original_test_fold'] = f
    reasons = [[] for _ in range(len(audit))]
    counts = []
    for role, group in source.groupby('role'):
        for field in ('mmsi', 'pass_group'):
            hit = audit[field].isin(group[field]).to_numpy()
            counts.append(dict(role=role, field=field, overlapping_rows=int(hit.sum())))
            for j in np.flatnonzero(hit):
                reasons[j].append(f'{role}:{field}')
    audit['exclusion_reason'] = [';'.join(r) for r in reasons]
    audit['eligible'] = audit.exclusion_reason.eq('')
    audit.to_csv(HERE / 'eligibility_4744.csv', index=False)
    audit.loc[~audit.eligible].to_csv(HERE / 'excluded.csv', index=False)
    target = audit.loc[audit.eligible].drop(columns=['eligible', 'exclusion_reason']).reset_index(drop=True)
    target['role'] = 'external_confirmation'
    assert len(target) == 3075
    for field in ('obs_id', 'mmsi', 'pass_group'):
        assert not set(target[field]) & set(source[field]), field
    target.to_csv(HERE / 'table.csv', index=False)
    save_json('isolation.json', dict(fold_counts=list(map(len, folds)), original_union=len(audit),
        retained=len(target), excluded=len(audit)-len(target), overlaps=counts,
        retained_by_original_fold={str(k):int(v) for k,v in target.original_test_fold.value_counts().items()},
        policy='Exclude identity/pass overlap with ALL source roles, including developed evaluation.', blind_test=False))
    with np.load(ROOT / 'features.npz', allow_pickle=True) as z:
        np.testing.assert_array_equal(z['instance_ids'].astype(str), full.instance_id.to_numpy())
        valid = z['valid'].astype(bool)
    np.savez_compressed(HERE / 'valid.npz', valid=valid[target.row_index])
    return source, target, valid


class FrozenChain:
    def __init__(self):
        self.ck, self.models = {}, {}
        classes = dict(five=pipe.Five, neighborhood=pipe.Neighborhood,
                       calibration=pipe.Calibration, attention=pipe.AttentionHead,
                       body=pipe.RegionHead, full=pipe.RegionHead)
        for seed in pipe.SEEDS:
            for stage, cls in classes.items():
                key = (stage, seed)
                ck = torch.load(UP / 'models' / f'{stage}_seed{seed}.pt', map_location='cpu', weights_only=False)
                model = cls().eval().requires_grad_(False)
                model.load_state_dict(ck['state'])
                self.ck[key], self.models[key] = ck, model

    def residual(self, stage, seed, raw, standardized=False):
        ck = self.ck[stage, seed]
        x = raw if standardized else (raw-ck['mu'])/ck['scale']
        return pipe.predict(self.models[stage, seed], x.astype(np.float32))

    def base(self, table):
        oldids = pd.read_csv(OLD / 'conditional_evidence_pilot/cohort.csv', usecols=['obs_id'])
        oldrow = oldids.reset_index().set_index('obs_id')['index'].loc[table.obs_id].to_numpy()
        with np.load(OLD / 'native_axis_probe_20261020/descriptors.npz') as z:
            h = np.concatenate((z['base'][oldrow], z['axis'][oldrow]), 1)
        with np.load(UP / 'models/axis_ridge.npz') as z:
            x = (h-z['mean'])/z['std']/np.sqrt(h.shape[1])
            ridge = x @ z['coef'].T + z['intercept']
        five = np.load(OLD / 'sampling_preservation_20261020/keep_refs.npy', mmap_mode='r')[oldrow].reshape(len(table),60,64)
        with np.load(OLD / 'neighborhood_level_split_20261021/features.npz') as z:
            dc = {k:z[k][oldrow] for k in ('centered','means','usable')}
        feat, level = np.zeros_like(dc['centered']), np.zeros_like(dc['means'])
        with np.load(UP / 'models/neighborhood_scaler.npz') as z:
            for p in range(2):
                feat[:,p] = (dc['centered'][:,p]-z['mean'][p])/z['std'][p]*dc['usable'][:,p,None]
                level[:,p] = (dc['means'][:,p]-z['level_mean'][p])/z['level_std'][p]*dc['usable'][:,p]
        xd = np.concatenate((feat.reshape(len(table),-1), level, dc['usable'].astype(np.float32)),1).astype(np.float32)
        out = {}
        for seed in pipe.SEEDS:
            a = ridge + self.residual('five', seed, five)
            d = self.residual('neighborhood', seed, xd, standardized=True)
            u = a + d
            b = u + self.residual('calibration', seed, pipe.center(u))
            out[seed] = dict(axis=ridge, A=a, D=d, U=u, B=b)
        return out

    def strategies(self, base, summary):
        out = {}
        for seed in pipe.SEEDS:
            b = base[seed]['B']
            raw = pipe.attention_raw(b, summary['mean3'], summary['hb3'], summary['valid'])
            s0 = b + self.residual('attention', seed, raw)
            values = [s0]
            for arm in ('body', 'full'):
                raw = pipe.regional_raw(s0, summary['summary1'], summary['quality'], summary['geometry'], arm)
                values.append(s0+self.residual(arm, seed, raw))
            out[seed] = dict(logits=np.stack(values,1), regional_raw=raw)
            assert np.isfinite(out[seed]['logits']).all()
        return out


class Extractor:
    def __init__(self, wanted):
        self.snapshots = {s['obs_id']:s for s in json.loads((OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if s['obs_id'] in wanted}
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.model = resnet50(weights=None)
        self.model.load_state_dict(torch.load(geo.WEIGHTS, map_location='cpu', weights_only=True))
        self.model = self.model.cuda().float().eval().requires_grad_(False)
        self.W, self.intercept = {}, {}
        for layer, dim in [('layer1',256),('layer3',1024)]:
            with np.load(UP / 'models' / f'reader_{layer}.npz') as z:
                self.W[layer] = torch.tensor(z['W'].reshape(8,2,dim).transpose(1,0,2), device='cuda', dtype=torch.float64)
                self.intercept[layer] = z['intercept'].copy()
        self.tau = {k:torch.tensor(v,device='cuda',dtype=torch.float64) for k,v in json.loads((UP/'scales.json').read_text())['tau'].items()}
        self.mean = torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
        self.std = torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]

    def run(self, table, valid, name):
        tick = last = time.perf_counter()
        n = len(table)
        summary = dict(indices=table.row_index.to_numpy(), mean3=np.zeros((n,2,8)), hb3=np.zeros((n,2,8,2)),
            summary1=np.zeros((n,2,3,8,3)), quality=np.zeros((n,2,3,4)), geometry=np.zeros((n,2,4)),
            region_valid=np.zeros((n,2,3),bool), valid=valid.copy())
        tok = dict(stem=np.zeros((n,2,20,64),np.float32), layer1=np.zeros((n,2,20,256),np.float32),
            mass=np.zeros((n,2,20),np.float32), neff=np.zeros((n,2,2,20),np.float32),
            geometry=np.zeros((n,2,6),np.float32), valid=np.zeros((n,2),bool),
            instance_ids=table.instance_id.to_numpy(str), regions=np.r_[np.zeros(8,int),np.full(8,2),np.ones(4,int)])
        gaps = dict(layer1=np.zeros((n,2,256)), layer3=np.zeros((n,2,1024)))
        polmean1 = np.zeros((n,2,8))
        seen = np.zeros((n,2),int)
        with torch.inference_mode():
            for first in range(0,n,8):
                images, fps, slot_masks, rw, addresses = [], [], [], [], []
                for i in range(first,min(first+8,n)):
                    for p in range(2):
                        if not valid[i,p]:
                            continue
                        pix, fp, g, place = geo.native(table, self.snapshots, i, p)
                        images.append(pix[None].astype(np.float32)/255)
                        fps.append(fp)
                        addresses.append((i,p))
                        masks = geo.soft_regions(g)
                        q = geo.region_quality(masks,fp,g)
                        flags = (q['area']>0)&(q['min_neff']>=4)
                        summary['region_valid'][i,p] = flags
                        summary['quality'][i,p] = np.stack([q['area'],q['coverage'],q['min_neff'],flags],-1)
                        sm = tokens.slots(g,fp)
                        slot_masks.append(sm)
                        tok['mass'][i,p] = sm.sum((-2,-1))
                        if g is not None:
                            summary['geometry'][i,p] = np.r_[g[2:4],g[5],1.]
                            tok['geometry'][i,p] = g
                            tok['valid'][i,p] = True
                        rw.append(np.stack([geo.cell_mass(np.rot90(masks*fp,k,(-2,-1))) for k in range(4)]))
                if not addresses:
                    continue
                b = len(addresses)
                x = torch.tensor(np.stack(images),device='cuda')
                pol = torch.tensor([p for i,p in addresses],device='cuda')
                rw = np.stack(rw)
                eff = {key:[] for key in ('stem','layer1')}
                for k in range(4):
                    xx = x.rot90(k,(-2,-1)).expand(-1,3,-1,-1)
                    stem = self.model.relu(self.model.bn1(self.model.conv1((xx-self.mean)/self.std)))
                    z1 = self.model.layer1(self.model.maxpool(stem))
                    z3 = self.model.layer3(self.model.layer2(z1))
                    for layer,z,stride in [('stem',stem,2),('layer1',z1,4)]:
                        w = np.stack([tokens.cell_mass(np.rot90(m,k,(-2,-1)),stride) for m in slot_masks]).reshape(b,20,-1)
                        total = w.sum(-1)
                        eff[layer].append(total**2/np.maximum((w*w).sum(-1),1e-20))
                        norm = torch.tensor(w/np.maximum(total[...,None],1e-20),device='cuda',dtype=torch.float32)
                        desc = torch.einsum('btu,bdu->btd',norm,z.flatten(2)).cpu().numpy()
                        for j,(i,p) in enumerate(addresses):
                            tok[layer][i,p] += desc[j]/4
                    weights = dict(layer1=np.stack([geo.cell_mass(np.rot90(fp,k)) for fp in fps]),
                        layer3=np.stack([np.rot90(fp,k).reshape(10,16,10,16).mean((1,3)) for fp in fps]))
                    for layer,z in [('layer1',z1),('layer3',z3)]:
                        zz = z.flatten(2).double()
                        weight = torch.tensor(weights[layer].reshape(b,1,-1),device='cuda',dtype=torch.float64)
                        norm = weight/weight.sum(-1,keepdim=True)
                        gap = (zz*norm).sum(-1).cpu().numpy()
                        response = torch.einsum('bck,bku->bcu',self.W[layer][pol],zz)
                        mu,_ = spatial.centered_response(response,norm)
                        mu = mu[:,0].cpu().numpy()
                        tau = self.tau[layer][pol,None,None,None]
                        if layer=='layer3':
                            packed = spatial.attention(response,weight,tau,recenter=True)[:,0].cpu().numpy()
                        else:
                            weight = torch.tensor(rw[:,k].reshape(b,3,-1),device='cuda')
                            packed = spatial.attention(response,weight,tau).cpu().numpy()
                        for j,(i,p) in enumerate(addresses):
                            gaps[layer][i,p] += gap[j]/4
                            if layer=='layer3':
                                summary['mean3'][i,p] += mu[j]/4
                                summary['hb3'][i,p] += packed[j,:,1:]/4
                            else:
                                polmean1[i,p] += mu[j]/4
                                packed[j,~summary['region_valid'][i,p]] = 0
                                summary['summary1'][i,p] += packed[j]/4
                for j,(i,p) in enumerate(addresses):
                    seen[i,p] += 1
                    for l,layer in enumerate(('stem','layer1')):
                        tok['neff'][i,p,l] = np.min(eff[layer],axis=0)[j]
                if time.perf_counter()-last>20:
                    print(name,min(first+8,n),'/',n,'seconds',round(time.perf_counter()-tick,1),flush=True)
                    last = time.perf_counter()
        np.testing.assert_array_equal(seen, valid.astype(int))
        for key in ('mean3','hb3','summary1','quality','geometry'):
            assert np.isfinite(summary[key]).all(), key
        for layer in ('stem','layer1'):
            assert np.isfinite(tok[layer]).all() and not tok[layer][~tok['valid']].any()
        assert not summary['hb3'][~valid].any()
        assert not summary['summary1'][~summary['region_valid']].any()
        checks = []
        for layer,means in [('layer1',polmean1),('layer3',summary['mean3'])]:
            w = self.W[layer].cpu().numpy().transpose(1,0,2).reshape(8,-1)
            closure = error(gaps[layer].reshape(n,-1)@w.T+self.intercept[layer],means.sum(1)+self.intercept[layer])
            assert closure<1e-8, (layer,closure)
            checks.append(dict(layer=layer,linear_closure=closure))
        with np.load(ROOT/'features.npz',allow_pickle=True) as z:
            prior = z['layer3'][table.row_index.to_numpy()].reshape(n,2,1024)
        gaperr = error(gaps['layer3'],prior)
        assert gaperr<1e-4, gaperr
        receipt = dict(n=n,pol_images=int(valid.sum()),seconds=time.perf_counter()-tick,
            original_layer3_GAP_max_error=gaperr,reader_closure=checks,
            gpu_peak_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
            native_no_resize=True,one_shared_C4_forward=True,coarse_region_min_neff=4,fine_token_min_neff=None)
        return summary,tok,receipt


def replay(chain, extractor, source, allvalid):
    take = np.unique(np.r_[np.arange(8),np.flatnonzero(source.obs_id.isin([220,1128,150,11556]))])
    probe = source.iloc[take].reset_index(drop=True)
    valid = allvalid[probe.row_index.to_numpy()]
    summary,tok,receipt = extractor.run(probe,valid,'source replay')
    checks = []
    with np.load(UP/'summaries.npz') as z:
        for key in ('mean3','hb3','summary1','quality','geometry','region_valid','valid'):
            e = error(summary[key],z[key][take])
            limit = 1e-8 if key in ('quality','geometry','region_valid','valid') else 1e-4
            checks.append(dict(kind='summary',field=key,max_error=e,tolerance=limit))
            assert e<=limit,(key,e)
    with np.load(LOCAL/'tokens.npz') as z:
        for key in ('stem','layer1','mass','neff','geometry','valid'):
            e = error(tok[key],z[key][take])
            limit = 1e-4 if key in ('stem','layer1') else 1e-8
            checks.append(dict(kind='tokens',field=key,max_error=e,tolerance=limit))
            assert e<=limit,(key,e)
    bases = chain.base(probe)
    strategies = chain.strategies(bases,summary)
    for seed in pipe.SEEDS:
        with np.load(UP/f'base_seed{seed}.npz') as z:
            for key,value in bases[seed].items():
                e = error(value,z[key][take])
                checks.append(dict(kind='base',seed=seed,field=key,max_error=e,tolerance=5e-5))
                assert e<=5e-5,(seed,key,e)
        with np.load(UP/f'strategies_seed{seed}.npz') as z:
            e = error(strategies[seed]['logits'],z['logits'][take])
            same = np.array_equal(strategies[seed]['logits'].argmax(-1),z['logits'][take].argmax(-1))
            assert e<=5e-5 and same,(seed,e,same)
            checks.append(dict(kind='strategies',seed=seed,max_error=e,top1_exact=same,tolerance=5e-5))
    save_json('replay_checks.json',dict(obs_ids=probe.obs_id.tolist(),source_rows=take.tolist(),checks=checks,extraction=receipt,passed=True))
    print('SOURCE REPLAY PASSED',len(probe),'observations; max strategy error',max(c['max_error'] for c in checks if c['kind']=='strategies'),flush=True)


def main():
    tick = time.perf_counter()
    source,target,valid = prepare()
    print('ISOLATION',len(target),'retained from 4744; 1669 excluded',flush=True)
    chain = FrozenChain()
    extractor = Extractor(set(source.obs_id)|set(target.obs_id))
    replay(chain,extractor,source,valid)
    summary,tok,receipt = extractor.run(target,valid[target.row_index.to_numpy()],'external')
    np.savez_compressed(HERE/'summaries.npz',**summary)
    np.savez_compressed(HERE/'tokens.npz',**tok)
    bases = chain.base(target)
    strategies = chain.strategies(bases,summary)
    for seed in pipe.SEEDS:
        np.savez_compressed(HERE/f'base_seed{seed}.npz',indices=target.row_index.to_numpy(),**bases[seed])
        np.savez_compressed(HERE/f'strategies_seed{seed}.npz',indices=target.row_index.to_numpy(np.int64),
            role=target.role.to_numpy(str),instance_ids=target.instance_id.to_numpy(str),
            strategy_names=np.array(['s0','body','full']),class_ids=np.arange(8),seed=seed,**strategies[seed])
    # Only coverage uses target labels; no target metric is used or selected here.
    coverage = target.groupby(['class_id','class_name']).agg(n=('obs_id','size'),ships=('mmsi','nunique'),passes=('pass_group','nunique')).reset_index()
    coverage['valid_vv'] = [int(summary['valid'][target.class_id.eq(c),0].sum()) for c in coverage.class_id]
    coverage['valid_vh'] = [int(summary['valid'][target.class_id.eq(c),1].sum()) for c in coverage.class_id]
    coverage.to_csv(HERE/'coverage_by_class.csv',index=False)
    target.groupby(['original_test_fold','port']).agg(n=('obs_id','size'),ships=('mmsi','nunique')).reset_index().to_csv(HERE/'coverage_by_fold_port.csv',index=False)
    for name in ('summaries.npz','tokens.npz'):
        with np.load(HERE/name) as z:
            for key in z.files:
                a = z[key]
                if np.issubdtype(a.dtype,np.number):
                    assert np.isfinite(a).all(),(name,key)
    for seed in pipe.SEEDS:
        with np.load(HERE/f'strategies_seed{seed}.npz') as z:
            np.testing.assert_array_equal(z['indices'],target.row_index)
            np.testing.assert_array_equal(z['instance_ids'],target.instance_id)
            np.testing.assert_array_equal(z['logits'],strategies[seed]['logits'])
    receipt.update(completed_utc=pipe.utc(),total_seconds=time.perf_counter()-tick,seeds=list(pipe.SEEDS),
        ships=int(target.mmsi.nunique()),passes=int(target.pass_group.nunique()),
        valid_vv=int(summary['valid'][:,0].sum()),valid_vh=int(summary['valid'][:,1].sum()),
        original_candidates=4744,excluded=1669,checkpoints_origin=str(UP/'models'),
        tau_origin=str(UP/'scales.json'),new_fits=0,new_selections=0,matching_or_local_head_inference=False,
        historical_not_blind=True,all_saved_arrays_finite=True,serialized_scores_exact=True)
    save_json('complete.json',receipt)
    print('COMPLETE',json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
