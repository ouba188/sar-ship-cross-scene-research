"""Source-pool rebuild of the actual A/D/T -> attention -> regional chain."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import json
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'eight_class_evidence_20261023'
OLD = ROOT.parents[1] / 'same_vessel_gallery_20261019'
SEEDS = (17, 29, 43)
torch.set_num_threads(2)


def save_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def load():
    table = pd.read_csv(HERE / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
    z = np.load(HERE / 'indices.npz')
    tr, va = z['fit_train'], z['fit_validation']
    assert table.iloc[np.r_[tr, va]].role.eq('strategy_fit').all()
    assert not set(table.iloc[tr].mmsi) & set(table.iloc[va].mmsi)
    assert not set(table.iloc[tr].pass_group) & set(table.iloc[va].pass_group)
    return table, tr, va


def weight(table, tr):
    f = table.iloc[tr]
    return len(tr) / (8 * f.groupby('mmsi').mmsi.transform('size').to_numpy() *
                      f.groupby('class_id').mmsi.transform('nunique').to_numpy())


def metrics(y, s):
    p = s.argmax(1)
    lp = s - logsumexp(s, axis=1, keepdims=True)
    loss = -lp[np.arange(len(y)), y]
    cls = np.unique(y)
    oa = float((p == y).mean())
    ba = float(np.mean([(p[y == c] == c).mean() for c in cls]))
    return dict(n=len(y), classes=len(cls), OA=oa, BA=ba, J=(oa+ba)/2,
                NLL=float(loss.mean()), balanced_NLL=float(np.mean([loss[y == c].mean() for c in cls])),
                Brier=float(((np.exp(lp) - np.eye(8)[y])**2).sum(1).mean()))


def exact_j(y, p):
    cls = np.unique(y)
    return (Fraction(int((p == y).sum()), len(y)) +
            sum(Fraction(int((p[y == c] == c).sum()), int((y == c).sum())) for c in cls) / len(cls)) / 2


def joint_loss(s, y):
    loss = F.cross_entropy(s, y, reduction='none')
    return .5 * loss.mean() + .5 * torch.stack([loss[y == c].mean() for c in y.unique()]).mean()


def center(x):
    return x - x.mean(1, keepdims=True)


def scaled(raw, tr, floor=1e-8):
    axes = (0, 1) if raw.ndim == 3 and raw.shape[1] == 8 else 0
    mu = raw[tr].mean(axis=axes, keepdims=True)
    sd = np.maximum(raw[tr].std(axis=axes, keepdims=True), floor)
    return ((raw-mu)/sd).astype(np.float32), mu, sd


def branch(dim, width, output=1):
    net = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, output))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class Five(nn.Module):
    def __init__(self):
        super().__init__()
        self.content = nn.Linear(64, 8)
        self.output = nn.Linear(60*8, 8)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(F.gelu(self.content(x)).flatten(1))


class Neighborhood(nn.Module):
    def __init__(self):
        super().__init__()
        self.content = nn.Linear(260, 32)
        self.output = nn.Linear(32, 8)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(F.gelu(self.content(x)))


class Calibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = branch(8, 16, 8)

    def forward(self, x):
        return self.net(x)


class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = branch(7, 12)
        self.local = branch(11, 12)

    def forward(self, x):
        a = self.mean(x[..., [0,1,2,7,8,13,14]])
        b = self.local(x[..., [0,3,4,5,6,9,10,11,12,13,14]])
        z = (a+b).squeeze(-1)
        return z-z.mean(1, keepdim=True)


class RegionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = branch(69, 12)

    def forward(self, x):
        z = self.net(x).squeeze(-1)
        return z-z.mean(1, keepdim=True)


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def predict(model, x):
    with torch.inference_mode():
        return np.concatenate([model(torch.tensor(x[i:i+256], dtype=torch.float32)).numpy()
                               for i in range(0, len(x), 256)]).astype(float)


def train_head(name, seed, model, x, base, table, tr, va, mu, scale, mse=False):
    start = time.perf_counter()
    # Labels of utility/selection/evaluation are never materialized by the trainer.
    y = torch.tensor(table.iloc[tr].class_id.to_numpy(), dtype=torch.long)
    yv = table.iloc[va].class_id.to_numpy()
    xt, xv = torch.tensor(x[tr]), torch.tensor(x[va])
    bt = torch.tensor(base[tr], dtype=torch.float32)
    initial = state(model)
    with torch.no_grad():
        assert not model(xt).any() and not model(xv).any()
    best_state, chosen = initial, 0
    met = metrics(yv, base[va])
    best = (met['BA'], -met['Brier']) if mse else exact_j(yv, base[va].argmax(1))
    curves = [dict(stage=name, seed=seed, epoch=0, train_loss=None, **met)]
    opt = torch.optim.AdamW(model.parameters(), lr=.001 if mse else .003, weight_decay=.001 if mse else .01)
    rng = np.random.default_rng(seed)
    weights = torch.tensor(weight(table, tr), dtype=torch.float32)
    steps = 150 if mse else 200
    for step in range(1, steps+1):
        opt.zero_grad(set_to_none=True)
        if mse:
            pick = rng.integers(len(tr), size=128)
            target = 2*F.one_hot(y[pick], 8).float()-1
            loss = ((bt[pick]+model(xt[pick])-target).square().mean(1)*weights[pick]).mean()
        else:
            loss = joint_loss(bt+model(xt), y)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        if step % (50 if mse else 5):
            continue
        with torch.no_grad():
            score = base[va] + model(xv).numpy().astype(float)
        met = metrics(yv, score)
        curves.append(dict(stage=name, seed=seed, epoch=step, train_loss=float(loss.detach()), **met))
        rank = (met['BA'], -met['Brier']) if mse else exact_j(yv, score.argmax(1))
        if rank > best:
            best, chosen, best_state = rank, step, state(model)
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = dict(stage=name, seed=seed, state=best_state, initial_state=initial,
        epoch=chosen, mu=mu, scale=scale, fit_indices=table.iloc[tr].row_index.to_numpy(),
        validation_indices=table.iloc[va].row_index.to_numpy(), parameters=sum(p.numel() for p in model.parameters()),
        upstream_training_predictions_oof=False, selected_utc=utc())
    path = HERE / 'models' / f'{name}_seed{seed}.pt'
    torch.save(checkpoint, path)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(restored['state'])
    out = base + predict(model, x)
    met = metrics(yv, out[va])
    with torch.no_grad():
        selected_val = base[va] + model(xv).numpy().astype(float)
    selected_met = metrics(yv, selected_val)
    replay = (selected_met['BA'], -selected_met['Brier']) if mse else exact_j(yv, selected_val.argmax(1))
    assert replay == best
    batch_error = float(abs(out[va]-selected_val).max())
    assert batch_error < 2e-5
    np.testing.assert_array_equal(out[va].argmax(1), selected_val.argmax(1))
    pd.DataFrame(curves).to_csv(HERE / f'curves_{name}_{seed}.csv', index=False)
    receipt = dict(stage=name, seed=seed, epoch=chosen, parameters=checkpoint['parameters'],
                   seconds=time.perf_counter()-start, fit_validation=met, replay_selected=True, zero_step_exact=True,
                   saved_vs_selection_batch_max_error=batch_error)
    save_json(f'selection_{name}_{seed}.json', receipt)
    print(name, seed, 'selected', chosen, 'valJ', met['J'], 'seconds', round(receipt['seconds'],2), flush=True)
    return out


def fit_reader(name, h, table, tr, va):
    tick = time.perf_counter()
    scale = StandardScaler().fit(h[tr])
    pca = PCA(n_components=128, svd_solver='randomized', random_state=17, iterated_power=4).fit(scale.transform(h[tr]))
    post = StandardScaler().fit(pca.transform(scale.transform(h[tr])))
    def project(x):
        return post.transform(pca.transform(scale.transform(x)))
    ytr, yv = table.iloc[tr].class_id.to_numpy(), table.iloc[va].class_id.to_numpy()
    candidates, logs = [], []
    for alpha in (1.,10.,100.):
        head = Ridge(alpha=alpha).fit(project(h[tr]), np.eye(8)[ytr], sample_weight=weight(table,tr))
        met = metrics(yv, head.predict(project(h[va]))*3)
        candidates.append((met['balanced_NLL'], alpha, head))
        logs.append(dict(reader=name, alpha=alpha, **met))
    _, alpha, head = min(candidates, key=lambda x: x[:2])
    W = ((head.coef_/post.scale_)@pca.components_)/scale.scale_*3
    intercept = head.predict(project(np.zeros((1,h.shape[1]))))[0]*3
    error = float(abs(h@W.T+intercept-head.predict(project(h))*3).max())
    assert error < 1e-8
    np.savez_compressed(HERE/'models'/f'{name}.npz', W=W, intercept=intercept, alpha=alpha,
        scale_mean=scale.mean_, scale_scale=scale.scale_, pca_mean=pca.mean_, pca_components=pca.components_,
        post_mean=post.mean_, post_scale=post.scale_, head_coef=head.coef_, head_intercept=head.intercept_,
        fit_indices=table.iloc[tr].row_index.to_numpy(), validation_indices=table.iloc[va].row_index.to_numpy())
    pd.DataFrame(logs).to_csv(HERE/f'{name}_alpha.csv', index=False)
    save_json(f'{name}_receipt.json', dict(alpha=alpha, linear_replay_max_error=error, seconds=time.perf_counter()-tick))
    print(name, 'alpha', alpha, 'linear replay', error, flush=True)


def fit_base():
    table, tr, va = load()
    (HERE/'models').mkdir(exist_ok=True)
    assert set(table.iloc[tr].class_id) == set(range(8))
    save_json('frozen_config.json', dict(created_utc=utc(), seeds=list(SEEDS), source_fold=0, backbone_training=False,
        old_A='stem axis Ridge + keep_refs five-point MSE residual', old_D='separated R-context MSE residual',
        T='score_only 8-16-8 residual', attention='mean7-12-1 + local11-12-1 (h,b)',
        strategies=['s0','body','full'], region_head='69-12-1',
        upstream_roles=['fit_train','fit_validation'], external_roles=['utility_train','selection','evaluation'],
        fit_train_n=len(tr), fit_validation_n=len(va), whole_stack_oof=False,
        supervised_fit_scope='All fit labels/scalers limited to fit_train; all upstream selection labels limited to fit_validation.',
        legacy_steps=150, new_steps=200, no_historical_models_modified=True,
        layer1_stride=4, layer1_origin=0, layer3_pool='original area footprint', rhoL=.25,rhoW=.5,edge=1,min_C4_neff=4,
        tau_scope='Median C4-averaged global sigma of fit_train for the new reader, valid pol/classes; in-sample within strategy_fit.',
        internal_protocol_change='No nested whole-stack OOF inside strategy_fit; same in-sample upstream training predictions at successive stages.'))
    oldids = pd.read_csv(OLD/'conditional_evidence_pilot/cohort.csv', usecols=['obs_id'])
    oi = oldids.reset_index().set_index('obs_id')['index'].loc[table.obs_id].to_numpy()
    raw = np.load(OLD/'native_axis_probe_20261020/descriptors.npz')
    h = np.concatenate((raw['base'][oi],raw['axis'][oi]),1)
    mu, sd = h[tr].mean(0), np.maximum(h[tr].std(0),1e-6)
    x = (h-mu)/sd/np.sqrt(h.shape[1])
    candidates, logs = [], []
    for alpha in (.1,1.,10.):
        model = RidgeClassifier(alpha=alpha, solver='cholesky').fit(x[tr], table.iloc[tr].class_id, sample_weight=weight(table,tr))
        met = metrics(table.iloc[va].class_id.to_numpy(),model.decision_function(x[va]))
        candidates.append((met['BA'],-met['Brier'],alpha,model))
        logs.append(dict(alpha=alpha, **met))
    _,_,alpha,model = max(candidates,key=lambda v:v[:2])
    ridge = model.decision_function(x).astype(float)
    np.savez_compressed(HERE/'models/axis_ridge.npz',mean=mu,std=sd,alpha=alpha,coef=model.coef_,intercept=model.intercept_,fit_indices=table.iloc[tr].row_index)
    pd.DataFrame(logs).to_csv(HERE/'axis_alpha.csv',index=False)
    five = np.load(OLD/'sampling_preservation_20261020/keep_refs.npy', mmap_mode='r')[oi].reshape(len(table),60,64)
    fm, fs = five[tr].mean(0), np.maximum(five[tr].std(0),1e-6)
    xf = ((five-fm)/fs).astype(np.float32)
    with np.load(OLD/'neighborhood_level_split_20261021/features.npz') as z:
        dc = {k:z[k][oi] for k in ('centered','means','usable')}
    feat = np.zeros_like(dc['centered']); level = np.zeros_like(dc['means'])
    dm, ds = np.zeros((2,128)), np.ones((2,128))
    lm, ls = np.zeros(2),np.ones(2)
    for p in range(2):
        rows = tr[dc['usable'][tr,p].astype(bool)]
        dm[p] = dc['centered'][rows,p].mean(0)
        ds[p] = np.maximum(dc['centered'][rows,p].std(0),1e-5)
        lm[p], ls[p] = dc['means'][rows,p].mean(),max(dc['means'][rows,p].std(),1e-5)
        feat[:,p] = (dc['centered'][:,p]-dm[p])/ds[p]*dc['usable'][:,p,None]
        level[:,p] = (dc['means'][:,p]-lm[p])/ls[p]*dc['usable'][:,p]
    xd = np.concatenate((feat.reshape(len(table),-1),level,dc['usable'].astype(np.float32)),1).astype(np.float32)
    np.savez_compressed(HERE/'models/neighborhood_scaler.npz',mean=dm,std=ds,level_mean=lm,level_std=ls)
    for seed in SEEDS:
        torch.manual_seed(seed)
        a = train_head('five',seed,Five(),xf,ridge,table,tr,va,fm,fs,mse=True)
        torch.manual_seed(seed)
        u = train_head('neighborhood',seed,Neighborhood(),xd,a,table,tr,va,None,None,mse=True)
        xt,tm,ts = scaled(center(u),tr)
        torch.manual_seed(seed)
        b = train_head('calibration',seed,Calibration(),xt,u,table,tr,va,tm,ts)
        np.savez_compressed(HERE/f'base_seed{seed}.npz',indices=table.row_index,axis=ridge,A=a,D=u-a,U=u,B=b)
    fit_aux()


def fit_aux():
    table,tr,va = load()
    rows = table.row_index.to_numpy()
    with np.load(ROOT/'features.npz',allow_pickle=True) as z:
        assert np.array_equal(z['instance_ids'][rows].astype(str),table.instance_id.to_numpy())
        fit_reader('reader_layer3',z['layer3'][rows].astype(float),table,tr,va)
        np.savez_compressed(HERE/'valid.npz',valid=z['valid'][rows])
    with np.load(ROOT/'obb_dynamic_readout/features_source.npz') as z:
        assert z['computed'][rows].all()
        fit_reader('reader_layer1',z['layer1'][rows].astype(float),table,tr,va)
    save_json('base_complete.json',dict(completed_utc=utc(),new_legacy_heads=6,new_T_heads=3,new_readers=2))


def attention_raw(base, means, hb, valid):
    cols = [center(base)]
    for p in range(2):
        for value in (means[:,p],hb[:,p,:,0],hb[:,p,:,1]):
            cols.extend([value,center(value)])
    cols.extend([np.broadcast_to(valid[:,p,None],base.shape) for p in range(2)])
    return np.stack(cols,-1)


def regional_raw(base, summary, quality, geometry, arm='full'):
    value = summary.copy()
    if arm == 'body':
        value[:,:,1:] = 0
    cols = [center(base)]
    for p in range(2):
        for r in range(3):
            for stat in range(3):
                v = value[:,p,r,:,stat]
                cols.extend([v,center(v)])
    fields = []
    for p in range(2):
        g,q = geometry[:,p],quality[:,p]
        fields.extend([np.log1p(g[:,0]),np.log1p(g[:,1]),g[:,2],g[:,3]])
        for r in range(3):
            fields.extend([np.log1p(q[:,r,0]),q[:,r,1],np.log1p(q[:,r,2]),q[:,r,3]])
    q = np.stack(fields,-1)
    raw = np.concatenate((np.stack(cols,-1),np.broadcast_to(q[:,None],(len(base),8,32))),-1)
    assert raw.shape == (len(base),8,69) and np.isfinite(raw).all()
    return raw


def fit_strategies():
    table,tr,va = load()
    with np.load(HERE/'summaries.npz') as z:
        means,hb,summary,quality,geometry,valid = [z[k] for k in ('mean3','hb3','summary1','quality','geometry','valid')]
    receipts = []
    for seed in SEEDS:
        b = np.load(HERE/f'base_seed{seed}.npz')['B']
        raw = attention_raw(b,means,hb,valid)
        x,mu,sd = scaled(raw,tr)
        torch.manual_seed(seed)
        s0 = train_head('attention',seed,AttentionHead(),x,b,table,tr,va,mu,sd)
        scores = [s0]
        initials, qscale = [], []
        for arm in ('body','full'):
            raw = regional_raw(s0,summary,quality,geometry,arm)
            x,mu,sd = scaled(raw,tr)
            torch.manual_seed(seed)
            model = RegionHead()
            initials.append(state(model))
            qscale.append((mu[...,-32:],sd[...,-32:]))
            scores.append(train_head(arm,seed,model,x,s0,table,tr,va,mu,sd))
        assert all(torch.equal(initials[0][k],initials[1][k]) for k in initials[0])
        for j in range(2):
            np.testing.assert_array_equal(qscale[0][j],qscale[1][j])
        raw = regional_raw(s0,summary,quality,geometry)
        logits = np.stack(scores,1)
        assert logits.shape == (len(table),3,8) and np.isfinite(logits).all()
        path = HERE/f'strategies_seed{seed}.npz'
        np.savez_compressed(path,indices=table.row_index.to_numpy(np.int64),role=table.role.to_numpy(str),
            logits=logits,regional_raw=raw,strategy_names=np.array(['s0','body','full']),
            class_ids=np.arange(8),seed=seed,instance_ids=table.instance_id.to_numpy(str))
        with np.load(path) as z:
            np.testing.assert_array_equal(z['logits'],logits)
            np.testing.assert_array_equal(z['regional_raw'],raw)
        receipts.append(dict(seed=seed,n=len(table),path=path.name,finite=True,serialization_exact=True,
            common_quality_scaler_exact=True,body_full_initial_state_equal=True))
        print('READY',path.name,logits.shape,raw.shape,flush=True)
    save_json('strategies_complete.json',dict(completed_utc=utc(),seeds=list(SEEDS),receipts=receipts,
        evaluation_labels_used_by_upstream=False,external_labels_used_by_upstream=False,upstream_fully_oof=False,
        fit_scope='All upstream supervision only fit_train/fit_validation within strategy_fit.',
        historical_test_used=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['base','readers','strategies'])
    args = parser.parse_args()
    if args.stage == 'base':
        fit_base()
    elif args.stage == 'readers':
        fit_aux()
    else:
        fit_strategies()
