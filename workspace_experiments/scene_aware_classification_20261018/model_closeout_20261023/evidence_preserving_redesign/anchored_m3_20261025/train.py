"""Bounded evidence-conditioned refinement of a retained both metric and reader."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
LOCAL = BASE / 'local_evidence_transport'
NATIVE = HERE.parent / 'native_validation'
SEEDS = (17, 29, 43)
ARMS = ('old_continue', 'content_both', 'typed_transport', 'typed_both')
CHECKPOINTS = (0, 20, 40, 60, 80, 100)
MICRO = 8


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load('anchored_old_metric', LOCAL / 'pair_conditioned_joint_metric/train.py')
typed = load('anchored_typed_utils', HERE.parent / 'typed_m3_20261025/train.py')
metrics, write_json = typed.metrics, typed.write_json


class Data:
    def __init__(self):
        self.table = pd.read_csv(NATIVE / 'table.csv', dtype={'mmsi': str})
        self.n = 1741
        self.up = self.table.up_index.to_numpy(int)
        self.tr = np.flatnonzero(self.table.role.eq('utility_train'))
        self.va = np.flatnonzero(self.table.role.eq('selection'))
        self.te = np.flatnonzero(self.table.role.eq('evaluation'))
        assert len(self.table) == 1805 and len(self.tr) == 644
        with np.load(HERE / 'features_hard.npz') as z:
            np.testing.assert_array_equal(z['up_index'], self.up)
            desc, mass = z['desc'].copy(), z['mass'].copy()
        assert desc.shape == (1805, 18, 8, 91)
        mu, sd, active = [], [], []
        for t in range(18):
            x = desc[self.tr, t][mass[self.tr, t] > 0]
            m, s = x.mean(0), x.std(0)
            a = s > 1e-6
            mu.append(m); sd.append(np.where(a, s, 1.)); active.append(a)
        mu, sd, active = np.array(mu), np.array(sd), np.array(active)
        desc = np.where(active[None, :, None], (desc-mu[None, :, None])/sd[None, :, None], 0.)
        desc[mass <= 0] = 0.
        assert np.isfinite(desc).all()
        self.scaler = dict(mu=mu, scale=sd, active=active)
        self.desc = torch.tensor(desc.reshape(1805, 2, 3, 3, 8, 91), device='cuda')
        self.valid = torch.tensor((mass > 0).reshape(1805, 2, 3, 3, 8), device='cuda')
        for a, b in ((self.tr,self.va), (self.tr,self.te), (self.va,self.te),
                     (np.arange(self.n), np.arange(self.n,1805))):
            for field in ('mmsi', 'pass_group'):
                assert not set(self.table.iloc[a][field]) & set(self.table.iloc[b][field]), field


class Correction(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.project = nn.Linear(91, 4, bias=False)
        self.embedding = nn.Embedding(18, 4)
        self.output = nn.Linear(12, 32, bias=False)
        nn.init.zeros_(self.output.weight)
        self.register_buffer('dim_mask', torch.cat((torch.ones(32), torch.zeros(59))))

    def projected(self, desc):
        if self.arm == 'content_both':
            desc = desc*self.dim_mask
        return self.project(desc)

    def change(self, q, s, validq, valids, region):
        # VV/VH, role and band remain distinct; no absolute heading is supplied.
        hi = q[:, None, None, :, :, :, None, :]
        hj = s[..., None, :, :]
        tid = (torch.arange(2, device=q.device)[:, None]*9 + region*3
               + torch.arange(3, device=q.device)[None])
        emb = self.embedding(tid)[None, None, None, :, :, None, None, :]
        emb = emb.expand(*torch.broadcast_shapes(hi.shape[:-1], hj.shape[:-1]), 4)
        phi = torch.cat(((hi+hj)*.5, (hi-hj).abs()*.5, emb), -1)
        keep = validq[:, None, None, :, :, :, None] & valids[..., None, :]
        band_change = self.output(F.gelu(phi)).tanh()*keep[..., None]
        return .25*band_change.mean(4)


class Reader:
    def __init__(self, data, seed):
        self.data, self.seed = data, seed
        self.old = prior.Reader('pair_joint', seed)
        lookup = {int(v):i for i,v in enumerate(self.old.ids)}
        self.ix = np.array([lookup[int(v)] for v in data.up])
        reverse = np.full(len(self.old.ids), -1)
        reverse[self.ix] = np.arange(len(self.ix))
        self.refs = torch.tensor(reverse[self.old.refs[self.ix].cpu().numpy()], device='cuda')
        assert (self.refs[:data.n] >= data.n).all()
        self.frozen = prior.Metric('pair_joint').cuda()
        ck = torch.load(LOCAL/'pair_conditioned_joint_metric/models'/f'pair_joint_{seed}.pt',
                        map_location='cpu', weights_only=False)
        self.frozen.load_state_dict(ck['state'])
        self.frozen.eval().requires_grad_(False)
        self.old_state = ck['state']
        with torch.no_grad():
            self.h = self.frozen.project(self.old.d)
        with np.load(LOCAL/'pair_conditioned_joint_metric/results'/f'pair_joint_{seed}.npz') as z:
            self.baseline = z['score'][self.ix[:data.n]].copy()
            self.local0 = z['local'][self.ix[:data.n]].copy()

    def forward(self, ids, model, mode=None, diagnostic=False):
        ids = torch.as_tensor(ids, device='cuda')
        idx = torch.tensor(self.ix, device='cuda')[ids]
        if isinstance(model, prior.Metric):
            result = self.old.forward(idx, model, diagnostic)
            if diagnostic:
                score, local, _ = result
                ref = self.old.refs[idx]
                h1 = model.project(self.old.d)
                details = []
                for r in prior.old.REGIONS:
                    q = self.old.x[idx][:, :, r][:, None, None, :, :, None, :]
                    s = self.old.x[ref][..., r, :][..., None, :, :]
                    hq0 = self.h[idx][:, :, r][:, None, None, :, :, None, :]
                    hs0 = self.h[ref][..., r, :][..., None, :, :]
                    hq1 = h1[idx][:, :, r][:, None, None, :, :, None, :]
                    hs1 = h1[ref][..., r, :][..., None, :, :]
                    k0 = prior.similarity(q,s,self.frozen.weights(hq0,hs0))
                    k1 = prior.similarity(q,s,model.weights(hq1,hs1))
                    qa = self.old.mass[idx][:,None,None,:,r]
                    sa = self.old.mass[ref][...,r]
                    t0,_,_,_ = prior.old.sinkhorn(k0,qa,sa)
                    t1,_,_,_ = prior.old.sinkhorn(k1,qa,sa)
                    keep = (qa>0)[...,None] & (sa>0)[...,None,:]
                    details.append(torch.stack(((t1-t0).abs().sum((-1,-2)),
                        ((k1-k0).abs()*keep).sum((-1,-2))/keep.sum((-1,-2)).clamp_min(1),
                        (t1*k1).sum((-1,-2))-(t0*k0).sum((-1,-2))),-1))
                return score,local,torch.stack(details,-2).mean(2)
            return result
        ref = self.old.refs[idx]
        q, s = self.old.x[idx], self.old.x[ref]
        hq, hs = self.h[idx], self.h[ref]
        a, b = self.old.mass[idx], self.old.mass[ref]
        h = model.projected(self.data.desc)
        nq, ns = h[ids], h[self.refs[ids]]
        vq, vs = self.data.valid[ids], self.data.valid[self.refs[ids]]
        groups, details = [], []
        selected_mode = mode or ('transport' if model.arm == 'typed_transport' else 'both')
        # Old local-head order is body, side, end; typed descriptor order is body, end, side.
        for r, role in zip(prior.old.REGIONS, (0, 2, 1)):
            count = len(r)
            qi = q[:, :, r][:, None, None, :, :, None, :]
            sj = s[..., r, :][..., None, :, :]
            hi = hq[:, :, r][:, None, None, :, :, None, :]
            hj = hs[..., r, :][..., None, :, :]
            w0 = self.frozen.weights(hi, hj)
            k0 = prior.similarity(qi, sj, w0)
            change = model.change(nq[:, :, role, :, :count], ns[..., role, :, :count, :],
                                  vq[:, :, role, :, :count], vs[..., role, :, :count], role)
            k1 = prior.similarity(qi, sj, w0*(1+change))
            kt = k0 if selected_mode in ('initial', 'score') else k1
            ks = k0 if selected_mode in ('initial', 'transport') else k1
            qa, sa = a[:, None, None, :, r], b[..., r]
            t, _, _, _ = prior.old.sinkhorn(kt, qa, sa)
            groups.append(prior.old.pair_score(t, ks, qa, sa))
            if diagnostic:
                t0, _, _, _ = prior.old.sinkhorn(k0, qa, sa)
                keep = (qa>0)[...,None] & (sa>0)[...,None,:]
                details.append(torch.stack(((t-t0).abs().sum((-1,-2)),
                    ((k1-k0).abs()*keep).sum((-1,-2))/keep.sum((-1,-2)).clamp_min(1),
                    (t*ks).sum((-1,-2))-(t0*k0).sum((-1,-2))), -1))
        local = torch.stack(groups, -2).mean(2)
        score = self.old.anchor[idx]+self.old.delta(local, idx)-self.old.old_delta[idx]
        if diagnostic:
            return score, local, torch.stack(details, -2).mean(2)
        return score

    @torch.no_grad()
    def predict(self, ids, model, mode=None, diagnostic=False, batch=MICRO):
        values = [self.forward(ids[i:i+batch], model, mode, diagnostic)
                  for i in range(0, len(ids), batch)]
        if diagnostic:
            return tuple(np.concatenate([x[j].cpu().numpy() for x in values]) for j in range(3))
        return np.concatenate([x.cpu().numpy() for x in values])


def make_model(reader, arm, seed):
    torch.manual_seed(seed)
    if arm == 'old_continue':
        model = prior.Metric('pair_joint').cuda()
        model.load_state_dict(reader.old_state)
    else:
        model = Correction(arm).cuda()
    return model


def checks(data):
    reader = Reader(data, 17)
    ids = data.tr[:3]
    y = torch.tensor(data.table.iloc[ids].class_id.to_numpy(), device='cuda')
    verification = {}
    for arm in ARMS:
        model = make_model(reader, arm, 17)
        score, local, _ = reader.predict(ids, model, diagnostic=True)
        replay = float(abs(score-reader.baseline[ids]).max())
        le = float(abs(local-reader.local0[ids]).max())
        assert replay < 3e-5 and le < 2e-5, (arm, replay, le)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(reader.forward(ids, model), y).backward()
        p = model.output.weight
        ix = int(p.grad.abs().argmax())
        auto = float(p.grad.flatten()[ix]); assert abs(auto)>1e-9, (arm, auto)
        value = float(p.detach().flatten()[ix]); eps = .01; losses = []
        with torch.no_grad():
            for sign in (1, -1):
                p.flatten()[ix] = value + sign*eps
                losses.append(float(F.cross_entropy(reader.forward(ids, model), y)))
            p.flatten()[ix] = value
        numeric = (losses[0]-losses[1])/(2*eps)
        assert abs(auto-numeric)<max(1e-4,abs(auto)*.1), (arm,auto,numeric)
        assert all(p.grad is None for p in reader.frozen.parameters())
        assert all(p.grad is None for p in reader.old.head.parameters())
        verification[arm] = dict(zero_replay=replay, local_replay=le,
                                 gradient_auto=auto, gradient_numeric=numeric)
    return verification


def fit(reader, arm, seed):
    data = reader.data
    start = time.perf_counter()
    model = make_model(reader, arm, seed)
    y = data.table.class_id.to_numpy(int)
    yt = torch.tensor(y, device='cuda')
    counts = np.bincount(y[data.tr], minlength=8)
    weights = torch.tensor(.5+.5*len(data.tr)/(8*counts), device='cuda', dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    rng = np.random.default_rng(seed)
    best = prior.old.exact_j(y[data.va], reader.baseline[data.va].argmax(1))
    chosen = 0
    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    curve = [dict(seed=seed, arm=arm, step=0, loss=None,
                  **metrics(y[data.va],reader.baseline[data.va]))]
    for step in range(1,101):
        batch = rng.choice(data.tr,32,replace=False)
        opt.zero_grad(set_to_none=True); total = 0.
        for first in range(0,32,MICRO):
            ids = batch[first:first+MICRO]
            score = reader.forward(ids, model)
            loss = (F.cross_entropy(score,yt[ids],reduction='none')*weights[yt[ids]]).sum()/32
            assert torch.isfinite(loss)
            loss.backward(); total += float(loss.detach())
        opt.step()
        if step in CHECKPOINTS:
            score = reader.predict(data.va, model)
            m = metrics(y[data.va], score)
            j = prior.old.exact_j(y[data.va], score.argmax(1))
            curve.append(dict(seed=seed,arm=arm,step=step,loss=total,**m))
            if j > best:
                best=j; chosen=step
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('FIT',arm,seed,step,'source_J',round(m['J'],6),
                  'seconds',round(time.perf_counter()-start,1),flush=True)
    receipt = dict(arm=arm,seed=seed,selected_step=chosen,steps=100,batch=32,micro=MICRO,
                   trainable_parameters=sum(p.numel() for p in model.parameters()),
                   seconds=time.perf_counter()-start)
    torch.save(dict(state=state,receipt=receipt), HERE/'models'/f'{arm}_{seed}.pt')
    pd.DataFrame(curve).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv', index=False)
    return receipt


def main(check_only=False):
    start=time.perf_counter(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    for name in ('models','results'): (HERE/name).mkdir(exist_ok=True)
    data=Data(); np.savez_compressed(HERE/'scaler.npz',**data.scaler)
    verification=checks(data); write_json(HERE/'checks.json',verification)
    print('CHECKS',verification,flush=True)
    if check_only:return
    receipts=[]
    for seed in SEEDS:
        reader=Reader(data,seed)
        for arm in ARMS:
            receipts.append(fit(reader,arm,seed))
            pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
        del reader
    # Freeze every source-selected checkpoint before opening development results.
    rows=[]; timings=[]
    for seed in SEEDS:
        reader=Reader(data,seed)
        np.savez_compressed(HERE/'results'/f'old_full_{seed}.npz',score=reader.baseline,
                            local=reader.local0,up_index=data.up[:data.n])
        for arm in ('old_full',)+ARMS:
            if arm=='old_full':score=reader.baseline
            else:
                model=make_model(reader,arm,seed)
                ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(ck['state'])
                torch.cuda.synchronize(); tick=time.perf_counter()
                score,local,diagnostic=reader.predict(np.arange(data.n),model,diagnostic=True)
                torch.cuda.synchronize()
                timings.append(dict(seed=seed,arm=arm,n=data.n,seconds=time.perf_counter()-tick))
                np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',score=score,local=local,
                                    diagnostic=diagnostic,up_index=data.up[:data.n])
            for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
                rows.append(dict(seed=seed,arm=arm,role=role,
                                 **metrics(data.table.iloc[ids].class_id.to_numpy(int),score[ids])))
    frame=pd.DataFrame(rows);frame.to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(timings).to_csv(HERE/'inference_receipts.csv',index=False)
    write_json(HERE/'complete.json',dict(fits=len(receipts),seconds=time.perf_counter()-start,
                                       peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    print(frame.query("role=='evaluation'").groupby('arm')[['OA','BA','J','NLL','Brier']].mean().to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true')
    main(parser.parse_args().check_only)
