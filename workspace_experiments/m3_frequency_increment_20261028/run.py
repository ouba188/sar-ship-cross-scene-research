"""Isolated frozen-system M3 descriptor experiment. No parent files are changed."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
PACKAGE = Path('E:/临时会话/sar_m123_current_20261027')
sys.path.insert(0, str(PACKAGE))
from sar_m123.current import data, model as current
from sar_m123 import frequency, tokens
from sar_m123.regions import soft_regions
from sar_m123.pool_replacement import grid_occupancy, regional_mean
from sar_m123.selective_matching import SelectiveReferenceMatch

ARMS = ('A_no_frequency', 'B_current', 'C_band_layout', 'D_spatial_layout')
SEEDS = (17, 29, 43)
EPOCHS = 4
DEVICE = 'cuda'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def records():
    fit = data.read_manifest(PACKAGE/'manifests/source_fit.json')
    dev = data.read_manifest(PACKAGE/'manifests/development.json')
    refs = data.read_manifest(PACKAGE/'manifests/references.json')
    for key in ('id', 'mmsi', 'pass_group'):
        assert not {r[key] for r in fit} & {r[key] for r in dev}, key
    lookup = {r['id']: i for i, r in enumerate(fit+dev)}
    rid = np.array([lookup[r['id']] for r in refs])
    assert (rid < len(fit)).all()
    return fit+dev, len(fit), refs, rid


def local_layout(fields, slots, geometry):
    cx, cy, length, width, angle = geometry[:5]
    yy, xx = np.mgrid[:160, :160]
    u = (xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle)
    v = -(xx-cx)*np.sin(angle)+(yy-cy)*np.cos(angle)
    result = np.zeros((20, 3, 16), np.float32)
    occupied = np.zeros((20, 16), bool)
    counts = np.zeros(20, int)
    for s, mask in enumerate(slots):
        live = mask > 0
        counts[s] = live.sum()
        if not live.any():
            continue
        a, b = u[live], v[live]
        ix = np.minimum(((a-a.min())/max(float(np.ptp(a)), 1e-6)*4).astype(int), 3)
        iy = np.minimum(((b-b.min())/max(float(np.ptp(b)), 1e-6)*4).astype(int), 3)
        bins, weight = iy*4+ix, mask[live]
        mass = np.bincount(bins, weights=weight, minlength=16)
        for c in range(3):
            numerator = np.bincount(bins, weights=weight*fields[c][live], minlength=16)
            result[s, c] = numerator/np.maximum(mass, 1e-12)
        occupied[s] = mass > 0
    # No reliable bow/stern: preserve layout modulo a 180-degree reversal.
    amount = occupied.astype(float)+occupied[:, ::-1]
    result = (result+result[:, :, ::-1])/np.maximum(amount[:, None], 1)
    return result.astype(np.float32), amount > 0, counts


def prepare_plus(record):
    views = data.prepare(record)
    item = {k: np.zeros(shape, np.float32) for k, shape in data.SHAPES.items()}
    layouts = np.zeros((2, 2, 20, 3, 16), np.float32)
    occupancy = np.zeros((2, 20, 16), bool)
    count = np.zeros((2, 20), int)
    for p, view in enumerate(views):
        if view is None or not view['valid_target']:
            continue
        item['available'][p] = 1
        item['images'][p, 0] = view['pixels'].astype(np.float32)/255
        item['valid'][p, 0] = view['footprint']
        item['masks'][p] = soft_regions(view['geometry'])*view['footprint']
        masks = tokens.slots(view['geometry'], view['footprint'])
        field = frequency.extract(view['raw'], masks, view['placement'])['energy_field']
        native = torch.from_numpy(field).float()
        valid = torch.from_numpy(view['footprint']).float()[None, None]
        product = (torch.from_numpy(masks).float()[:, None]*native[None]).flatten(0, 1)[None]
        for stage, size, key in ((2, 20, 'local_frequency'), (3, 10, 'frequency')):
            item['local_slot_masks' if stage == 2 else 'slot_masks'][p] = grid_occupancy(torch.from_numpy(masks)[None], (size, size))[0].numpy()
            item[key][p] = grid_occupancy(product, (size, size))[0].reshape(20, 3, size, size).numpy()
            cov = grid_occupancy(valid, (size, size))
            item[f'field{stage}'][p] = (grid_occupancy(native[None], (size, size))/cov.clamp_min(1e-8))[0].numpy()
        layouts[p, 0], occupancy[p], count[p] = local_layout(field, masks, view['geometry'])
        raw = view['pixels'].astype(np.float32)/255
        # Smooth on the original available image, never across the padded border.
        ys, xs = np.where(view['footprint'] > 0)
        y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        spatial = np.zeros((3, 160, 160), np.float32)
        for k, sigma in enumerate((0, 2, 4)):
            spatial[k, y0:y1, x0:x1] = gaussian_filter(raw[y0:y1, x0:x1], sigma=sigma)
        layouts[p, 1] = local_layout(spatial, masks, view['geometry'])[0]
    return item, layouts, occupancy, count


def prepare():
    dest = HERE/'cache'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists():
        return
    rows, nf, refs, rid = records()
    write(HERE/'records.json', rows)
    model = current.load_model(PACKAGE/'weights/selected_s17.pt', DEVICE)
    ritems = [prepare_plus(r)[0] for r in refs]
    bank = data.batch(ritems, DEVICE)
    with torch.no_grad():
        bank['prefix'] = model.prefix(bank['images'], stop=2)
    arrays, errors = {}, []
    tick = time.perf_counter()
    for first in range(0, len(rows), 16):
        part = rows[first:first+16]
        prepared = [prepare_plus(r) for r in part]
        values = data.batch([x[0] for x in prepared], DEVICE)
        with torch.no_grad():
            prefix = model.prefix(values['images'], stop=2)
            x2, x, r, anchor, ra = model.stage3_paths(prefix, values['valid'], values['available'],
                values['field2'], values['field3'], bank, bank['valid'], values['slot_masks'], values['frequency'], {}, [])
            old = [op.descriptors(anchor[:, p], prefix[:, p], values['slot_masks'][:, p],
                values['local_slot_masks'][:, p], values['valid'][:, p], values['frequency'][:, p],
                values['local_frequency'][:, p]) for p, op in enumerate(model.m3[0])]
            low = []
            for p in range(2):
                masks, energy = model.region_integrals(values['local_slot_masks'][:, p], values['local_frequency'][:, p])
                low.append(model.m2[p](x2[:, p], masks, values['valid'][:, p], energy)['pooled'])
            output = dict(x=x, low=torch.stack(low, 1), deep=torch.stack([a[0][0] for a in old], 1),
                fine=torch.stack([torch.stack(a[0][1:], 2) for a in old], 1),
                mass=torch.stack([a[1] for a in old], 1), slots=values['slot_masks'], energy=values['frequency'],
                valid=values['valid'].to(torch.uint8), available=values['available'])
        output = {k:v.cpu().numpy() for k,v in output.items()}
        output.update(layout=np.stack([a[1] for a in prepared]), occupied=np.stack([a[2] for a in prepared]),
                      native_count=np.stack([a[3] for a in prepared]))
        for key, value in output.items():
            if key not in arrays:
                arrays[key] = np.lib.format.open_memmap(dest/f'{key}.npy', mode='w+', dtype=value.dtype,
                    shape=(len(rows),)+value.shape[1:])
            arrays[key][first:first+len(part)] = value
        if first == 0:
            original = data.prepare_record(part[0])
            errors = {k: float(np.max(np.abs(original[k]-prepared[0][0][k]))) for k in data.SHAPES}
            assert max(errors.values()) < 1e-5, errors
            with torch.no_grad():
                true = current.forward_batch(model, values, bank, data.pair_mask(part, refs, DEVICE)).cpu().numpy()
            np.save(dest/'parity_score.npy', true)
        if first % 128 == 0:
            print('PREPARE', first+len(part), len(rows), round(time.perf_counter()-tick, 1), flush=True)
    for a in arrays.values():
        a.flush()
    write(dest/'complete.json', dict(n=len(rows), fit=nf, dev=len(rows)-nf, reference_indices=rid.tolist(),
        prepare_input_max_errors=errors, seconds=time.perf_counter()-tick))


def load():
    rows, nf, refs, rid = records()
    arrays = {p.stem:np.load(p, mmap_mode='r') for p in (HERE/'cache').glob('*.npy') if p.stem != 'parity_score'}
    return rows, nf, refs, rid, arrays


class CachedMatch(SelectiveReferenceMatch):
    def descriptors(self, anchors, local, slots, local_slots, valid, energy, local_energy):
        return anchors, local


def make(arm, seed, arrays, nf):
    model = current.load_model(PACKAGE/'weights/selected_s17.pt', DEVICE)
    for p in range(2):
        op = CachedMatch()
        op.load_state_dict(model.m3[0][p].state_dict())
        op.requires_grad_(False)
        if seed is not None:
            torch.manual_seed(seed+100*p)
            op.selector = nn.Sequential(nn.Linear(5,16), nn.GELU(), nn.Linear(16,5))
            nn.init.zeros_(op.selector[-1].weight)
            nn.init.zeros_(op.selector[-1].bias)
            op.branch_bias = nn.Parameter(torch.tensor([.5,.25,1/12,1/12,1/12]).log())
            op.selector.requires_grad_(True)
            op.branch_bias.requires_grad_(True)
        op.frequency = arm != 'A_no_frequency'
        model.m3[0][p] = op.to(DEVICE)
    model.eval()
    normalized = None
    if arm in ('C_band_layout','D_spatial_layout'):
        layout = np.asarray(arrays['layout'][:, :, 0 if arm == 'C_band_layout' else 1]).copy()
        occupied = np.asarray(arrays['occupied'])[:, :, :, None, :]
        denom = np.maximum(occupied[:nf].sum((0,2)), 1)
        mean = (layout[:nf]*occupied[:nf]).sum((0,2))/denom
        centered = (layout-mean[None,:,None])*occupied
        # One scale per polarization/branch, avoiding amplification of near-constant bins.
        scale = np.sqrt((centered[:nf]**2).sum((0,2,4))/np.maximum(occupied[:nf].sum((0,2,4)),1))
        normalized = centered/np.maximum(scale[None,:,None,:,None],1e-6)
        normalized = normalized.astype(np.float32)
    return model, normalized


def tensor(a, ids):
    return torch.as_tensor(np.array(a[ids], copy=True), device=DEVICE).float()


def describe(arrays, ids, p, normalized):
    deep = tensor(arrays['deep'], ids)[:,p]
    fine = tensor(arrays['fine'], ids)[:,p]
    other = fine[:,:,1:] if normalized is None else F.normalize(tensor(normalized, ids)[:,p], dim=-1)
    return [deep, fine[:,:,0]]+list(other.unbind(2))


def forward(model, arrays, ids, rid, rows, normalized=None, details=False):
    x, slots, valid, energy = [tensor(arrays[k],ids) for k in ('x','slots','valid','energy')]
    rx, rs, rv = [tensor(arrays[k],rid) for k in ('x','slots','valid')]
    mass, rm = tensor(arrays['mass'],ids), tensor(arrays['mass'],rid)
    avail, ravail = tensor(arrays['available'],ids), tensor(arrays['available'],rid)
    legal = data.pair_mask([rows[i] for i in ids], [rows[i] for i in rid], DEVICE)
    outputs, info = [], []
    for p, op in enumerate(model.m3[0]):
        bank = dict(features=rx[:,p], anchor=describe(arrays,rid,p,normalized), local=rm[:,p],
            slots=rs[:,p], local_slots=None, valid=rv[:,p], energy=None, local_energy=None)
        y, d = op(x[:,p], describe(arrays,ids,p,normalized), mass[:,p], slots[:,p], None,
            valid[:,p], None, None, bank, legal & avail[:,p,None].bool() & ravail[None,:,p].bool(), return_details=details)
        outputs.append(y)
        info.append(d)
    updated = torch.stack(outputs,1)
    deep = model.backbone.model.layer4(updated.flatten(0,1))
    glob = regional_mean(deep, valid.flatten(0,1), valid.flatten(0,1))[0][:,0].reshape(len(ids),2,-1)
    low = tensor(arrays['low'],ids)
    pooled = []
    for p in range(2):
        reg, en = model.region_integrals(slots[:,p], energy[:,p])
        extra = model.deep_pool[p](updated[:,p],reg,valid[:,p],en)
        pooled.append((low[:,p]+model.deep_output[p](extra['pooled'])).flatten(1))
    values = torch.cat((glob,torch.stack(pooled,1)),-1)*avail[...,None]
    score = model.head(torch.cat((values.flatten(1),avail),-1))
    return score, info


def metrics(y, score):
    prob = torch.from_numpy(score).double().softmax(-1).numpy()
    pred = score.argmax(1)
    recalls = [float((pred[y==c]==c).mean()) for c in np.unique(y)]
    return dict(oa=float((pred==y).mean()*100), ba=float(np.mean(recalls)*100),
        nll=float(-np.log(prob[np.arange(len(y)),y].clip(1e-20)).mean()),
        brier=float(((prob-np.eye(8)[y])**2).sum(1).mean()))


def relation_loss(info, target, refs):
    losses, active = [], []
    for d in info:
        valid, score = d['reference_valid'], d['reference_logits']
        positive = (target[:,None]==refs[None]) & valid
        live = positive.any(1) & valid.any(1)
        losses.append(-(score.masked_fill(~valid,-1e4).log_softmax(1)*positive).sum(1)/positive.sum(1).clamp_min(1)*live)
        active.append(live)
    return torch.stack(losses).sum(0)/torch.stack(active).sum(0).clamp_min(1)


@torch.no_grad()
def predict(model, arrays, ids, rid, rows, norm):
    scores, relations, validity = [], [], []
    for first in range(0,len(ids),24):
        score, ds = forward(model,arrays,ids[first:first+24],rid,rows,norm)
        scores.append(score.cpu().numpy())
        live = torch.stack([d['reference_valid'] for d in ds])
        rel = (torch.stack([d['reference_logits'] for d in ds])*live).sum(0)/live.sum(0).clamp_min(1)
        relations.append(rel.cpu().numpy())
        validity.append(live.any(0).cpu().numpy())
    return dict(score=np.concatenate(scores), relation=np.concatenate(relations), valid=np.concatenate(validity))


def check():
    rows,nf,refs,rid,arrays = load()
    model,norm = make('B_current',None,arrays,nf)
    with torch.no_grad():
        score,_ = forward(model,arrays,np.arange(16),rid,rows,norm)
    error = float(abs(score.cpu().numpy()-np.load(HERE/'cache/parity_score.npy')).max())
    assert error < 1e-5, error
    receipts = []
    for arm in ARMS:
        del model
        model,norm = make(arm,17,arrays,nf)
        ids=np.arange(16)
        score,info=forward(model,arrays,ids,rid,rows,norm)
        y=torch.tensor([rows[i]['label'] for i in ids],device=DEVICE)
        ry=torch.tensor([rows[i]['label'] for i in rid],device=DEVICE)
        loss=F.cross_entropy(score,y)+.1*relation_loss(info,y,ry).mean()
        loss.backward()
        grads={k:float(v.grad.norm()) for k,v in model.named_parameters() if v.grad is not None}
        assert grads and all(k.startswith('m3.') and ('.selector.' in k or '.branch_bias' in k) for k in grads)
        assert any(v>0 for v in grads.values())
        receipts.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),gradients=grads))
    write(HERE/'checks.json',dict(full_forward_error=error,arms=receipts))
    print('CHECK',error,flush=True)


def train():
    rows,nf,refs,rid,arrays=load()
    y=np.array([r['label'] for r in rows])
    fit=np.arange(nf)
    dev=np.arange(nf,len(rows))
    frame=pd.DataFrame(rows[:nf])
    weight=nf/(8*frame.groupby('label').mmsi.transform('nunique').to_numpy()*frame.groupby('mmsi').mmsi.transform('size').to_numpy())
    for seed in SEEDS:
        for arm in ARMS:
            dest=HERE/'training'/f'{arm}_{seed}'
            dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():
                continue
            model,norm=make(arm,seed,arrays,nf)
            initial=predict(model,arrays,dev,rid,rows,norm)
            np.savez_compressed(dest/'initial.npz',**initial)
            parameters=[p for p in model.parameters() if p.requires_grad]
            opt=torch.optim.AdamW(parameters,lr=.001,weight_decay=.01)
            rng=np.random.default_rng(seed)
            curve=[]
            tick=time.perf_counter()
            for epoch in range(1,EPOCHS+1):
                losses=[]
                for first in range(0,nf,24):
                    if first==0:
                        order=rng.permutation(fit)
                    ids=order[first:first+24]
                    opt.zero_grad(set_to_none=True)
                    score,info=forward(model,arrays,ids,rid,rows,norm)
                    target=torch.tensor(y[ids],device=DEVICE)
                    refs_y=torch.tensor(y[rid],device=DEVICE)
                    w=torch.tensor(weight[ids],device=DEVICE,dtype=torch.float32)
                    loss=((F.cross_entropy(score,target,reduction='none')+.1*relation_loss(info,target,refs_y))*w).mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.detach()))
                outcome=predict(model,arrays,dev,rid,rows,norm)
                curve.append(dict(epoch=epoch,loss=float(np.mean(losses)),**metrics(y[dev],outcome['score'])))
                pd.DataFrame(curve).to_csv(dest/'curve.csv',index=False)
                print('TRAIN',arm,seed,epoch,curve[-1],round(time.perf_counter()-tick,1),flush=True)
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()
                if k.startswith('m3.') and ('.selector.' in k or '.branch_bias' in k)},dest/'selector.pt')
            np.savez_compressed(dest/'final.npz',**outcome)
            write(dest/'complete.json',dict(arm=arm,seed=seed,epochs=EPOCHS,seconds=time.perf_counter()-tick,
                trainable=sum(p.numel() for p in parameters),metrics=metrics(y[dev],outcome['score'])))
            del model,opt,parameters,score,info,loss
            torch.cuda.empty_cache()


if __name__=='__main__':
    current.setup(DEVICE)
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','check','train','all'))
    args=parser.parse_args()
    if args.action in ('prepare','all'): prepare()
    if args.action in ('check','all'): check()
    if args.action in ('train','all'): train()
