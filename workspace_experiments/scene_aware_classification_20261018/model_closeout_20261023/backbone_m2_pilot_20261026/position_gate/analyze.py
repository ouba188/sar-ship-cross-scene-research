"""Paired results, feedback ablations, and native-image cases."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from experiment import HERE, PILOT, CONFIGS, old, op, setup, load, predict, restore


def result(stage, gate, seed):
    if stage == 2 and gate in ('none', 'dot'):
        folder = PILOT/'training'/f"{'none' if gate == 'none' else 'regional'}_{seed}"
    else:
        folder = HERE/'training'/f's{stage}_{gate}_{seed}'
    with np.load(folder/'evaluation.npz') as data:
        return {k: data[k].copy() for k in data.files}


def margin(scores, y):
    other = scores.copy()
    other[np.arange(len(y)), y] = -np.inf
    return scores[np.arange(len(y)), y]-other.max(1)


def mean_gate_hook(block, args, output):
    # Fixed trained weights; keep each image/region's occupancy-weighted gate
    # mean, but remove its spatial variation. This is not a retrained baseline.
    x, masks, valid = args
    masks = F.interpolate(masks, x.shape[-2:], mode='area')
    valid = F.interpolate(valid, x.shape[-2:], mode='area')
    masks = torch.minimum(masks, valid)
    z = F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)
    tokens = torch.einsum('brhw,bchw->brc', masks, z)/masks.sum((-2, -1)).clamp_min(1e-6)[..., None]
    q, k = F.normalize(block.query(z), dim=1), F.normalize(block.key(tokens), dim=-1)
    gate = torch.sigmoid(torch.einsum('bdhw,brd->brhw', q, k)/block.temperature)
    occupancy = masks/masks.sum(1, keepdim=True).clamp_min(1)
    mean = (gate*occupancy).sum((-2, -1), keepdim=True)/occupancy.sum((-2, -1), keepdim=True).clamp_min(1e-6)
    values = torch.einsum('brd,rde->bre', block.value(tokens), block.region_value)
    update = block.output(torch.einsum('brhw,brd->bdhw', mean*occupancy, values))
    return x+block.gain*update


@torch.no_grad()
def main():
    setup()
    table, _ = old.load()
    images = np.load(PILOT/'cache/images.npy', mmap_mode='r')
    full = pd.read_csv(HERE/'metrics.csv')
    original = pd.read_csv(PILOT/'metrics.csv')
    original = original[original['mode'].isin(['none', 'regional'])].copy()
    original['stage'] = 2
    original['gate'] = original['mode'].map({'none': 'none', 'regional': 'dot'})
    full = pd.concat([full, original], ignore_index=True)
    full.groupby(['stage', 'gate'])[['OA', 'BA', 'J', 'NLL', 'Brier', 'validation_j', 'seconds', 'trainable']].agg(['mean', 'std']).to_csv(HERE/'summary.csv')
    chosen = json.loads((HERE/'source_selected.json').read_text())
    cases, paired, perclass, effects, gates, uncertainty, predictions = [], [], [], [], [], [], []
    for stage, gate in [x for x in CONFIGS if x[1] != 'none']:
        _, arrays = load(stage)
        changes = []
        for seed in old.SEEDS:
            before, after = result(stage, 'none', seed), result(stage, gate, seed)
            ids, y = after['ids'], after['y']
            assert np.array_equal(before['ids'], ids) and np.array_equal(before['y'], y)
            a, b = before['score'].argmax(1) == y, after['score'].argmax(1) == y
            changes.append(b.astype(float)-a)
            mb, ma = op.metrics(y, before['score']), op.metrics(y, after['score'])
            paired.append(dict(stage=stage, gate=gate, seed=seed, rescued=int((~a & b).sum()), harmed=int((a & ~b).sum()),
                               delta_oa=ma['OA']-mb['OA'], delta_ba=ma['BA']-mb['BA']))
            for c in range(8):
                use = y == c
                perclass.append(dict(stage=stage, gate=gate, seed=seed, class_id=c, n=int(use.sum()),
                                     recall=float(b[use].mean()), delta=float(b[use].mean()-a[use].mean())))
            model = restore(stage, gate, seed)
            model.stages = ()
            bypass = predict(model, arrays, ids, stage)
            model.stages = (stage,)
            variants = {'no_feedback': bypass}
            if gate == 'cosine':
                handles = [block.register_forward_hook(mean_gate_hook) for block in model.blocks.values()]
                variants['spatial_mean_gate'] = predict(model, arrays, ids, stage)
                for handle in handles:
                    handle.remove()
            for name, score in variants.items():
                m = op.metrics(y, score)
                effects.append(dict(stage=stage, gate=gate, seed=seed, intervention=name,
                    original_oa=ma['OA'], original_ba=ma['BA'], **m,
                    changed=int((score.argmax(1) != after['score'].argmax(1)).sum()),
                    score_difference=float(abs(score-after['score']).max())))
            for role in ('fit_validation', 'evaluation'):
                use = np.flatnonzero(table.role.eq(role))
                stats = [[], []]
                for first in range(0, len(use), 24):
                    x, masks, valid, av = old.batch(arrays, use[first:first+24])
                    for p in range(2):
                        _, detail = model.blocks[f'{p}_{stage}'](x[:, p], masks[:, p], valid[:, p], return_details=True)
                        active = masks[:, p] > .01
                        g = detail['gates'][active].cpu().numpy()
                        rr = masks[:, p].sum(1) > .01
                        ratio = (detail['update'].norm(dim=1)[rr]/x[:, p].norm(dim=1)[rr].clamp_min(1e-6)).cpu().numpy()
                        stats[p].append((g, ratio))
                for p in range(2):
                    gg, rr = [np.concatenate([x[k] for x in stats[p]]) for k in range(2)]
                    gates.append(dict(stage=stage, gate=gate, seed=seed, role=role, pol=('VV','VH')[p],
                        mean=float(gg.mean()), std=float(gg.std()), below001=float((gg<.01).mean()), above099=float((gg>.99).mean()),
                        update_mean=float(rr.mean()), update_median=float(np.median(rr))))
            del model
            for i, idx in enumerate(ids):
                predictions.append(dict(stage=stage, gate=gate, seed=seed, obs_id=int(table.iloc[idx].obs_id),
                    y=int(y[i]), before=int(before['score'][i].argmax()), after=int(after['score'][i].argmax()),
                    margin_before=float(margin(before['score'][i:i+1],y[i:i+1])[0]), margin_after=float(margin(after['score'][i:i+1],y[i:i+1])[0])))
            if stage == chosen['stage'] and gate == chosen['gate']:
                delta = margin(after['score'], y)-margin(before['score'], y)
                for name, flag in [('rescued',~a & b), ('harmed',a & ~b), ('CC',a & b), ('WW',~a & ~b)]:
                    pool = np.flatnonzero(flag)
                    if len(pool):
                        k = int(pool[np.argmax(abs(delta[pool]))])
                        cases.append(dict(kind=name, seed=seed, k=k, size=float(abs(delta[k]))))
        frame = table.iloc[ids]
        groups = [np.flatnonzero(frame.pass_group.eq(g)) for g in frame.pass_group.unique()]
        rng, samples, ba_samples, delta = np.random.default_rng(171), [], [], np.mean(changes, axis=0)
        for _ in range(2000):
            take = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            samples.append(float(delta[take].mean()))
            if len(np.unique(y[take])) == 8:
                ba_samples.append(float(np.mean([delta[take][y[take] == c].mean() for c in range(8)])))
        uncertainty.append(dict(stage=stage, gate=gate, oa_low=float(np.quantile(samples, .025)),
             oa_high=float(np.quantile(samples, .975)), ba_conditional_low=float(np.quantile(ba_samples, .025)),
             ba_conditional_high=float(np.quantile(ba_samples, .975)), ba_fullclass_draws=len(ba_samples),
             missing_class_draws=2000-len(ba_samples), ba_note='Conditional on all eight classes occurring in resampled passes.'))
    for name, rows in [('paired',paired),('by_class',perclass),('interventions',effects),('gates',gates),('uncertainty',uncertainty),('predictions',predictions)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv', index=False)
    stage, gate = chosen['stage'], chosen['gate']
    _, arrays = load(stage)
    panels = HERE/'analysis'
    panels.mkdir(exist_ok=True)
    selected, interventions = [], []
    for kind in ('rescued','harmed','CC','WW'):
        pool = [c for c in cases if c['kind'] == kind]
        if not pool:
            continue
        case = max(pool, key=lambda c:c['size'])
        seed, k = case['seed'], case['k']
        before, after = result(stage,'none',seed), result(stage,gate,seed)
        idx, y = int(after['ids'][k]), int(after['y'][k])
        model = restore(stage, gate, seed)
        x, masks, valid, av = old.batch(arrays,[idx])
        fig, axes = plt.subplots(2,5,figsize=(14,7.5),constrained_layout=True)
        for p, pol in enumerate(('VV','VH')):
            _, d = model.blocks[f'{p}_{stage}'](x[:,p],masks[:,p],valid[:,p],return_details=True)
            img = images[idx,p,0]
            axes[p,0].imshow(img,cmap='gray',vmin=0,vmax=255)
            axes[p,0].set_title(f'{pol} native crop/pad')
            maps = [d['update'][0].norm(dim=0).cpu().numpy()]+[d['weights'][0,r].cpu().numpy() for r in range(3)]
            for j, (values, title) in enumerate(zip(maps,('update norm','body weight','end weight','side weight')),1):
                axes[p,j].imshow(img,cmap='gray',vmin=0,vmax=255)
                axes[p,j].imshow(values,extent=(0,160,160,0),cmap='magma' if j==1 else 'viridis',vmin=0,
                                  vmax=None if j==1 else 1,alpha=.65,interpolation='nearest')
                axes[p,j].set_title(f'{title}; max={values.max():.3f}')
        for a in axes.flat:
            a.axis('off')
        obs = int(table.iloc[idx].obs_id)
        fig.suptitle(f'{kind}: obs {obs}, seed {seed}, stage {stage}, {gate}\ntrue {y}, '
                     f"pred {before['score'][k].argmax()} -> {after['score'][k].argmax()}")
        filename = f'{kind}_{obs}_{seed}.png'
        fig.savefig(panels/filename,dpi=130)
        plt.close(fig)
        for arm in ('full','no_feedback','no_body','no_end','no_side','no_vv','no_vh'):
            mm = masks.clone()
            if arm == 'no_feedback': mm.zero_()
            if arm in ('no_body','no_end','no_side'): mm[:,:,('no_body','no_end','no_side').index(arm)] = 0
            if arm in ('no_vv','no_vh'): mm[:,('no_vv','no_vh').index(arm)] = 0
            score = model.forward_from(x,mm,valid,av,start=stage).cpu().numpy()
            interventions.append(dict(obs_id=obs,kind=kind,seed=seed,arm=arm,pred=int(score.argmax()),y=y,margin=float(margin(score,np.array([y]))[0])))
        selected.append(dict(**case,obs_id=obs,y=y,before=int(before['score'][k].argmax()),after=int(after['score'][k].argmax()),
            image_vv=table.iloc[idx].image_vv,image_vh=table.iloc[idx].image_vh,panel=filename))
        del model
    old.write(panels/'cases.json',selected)
    pd.DataFrame(interventions).to_csv(HERE/'case_interventions.csv',index=False)
    print(full.groupby(['stage','gate'])[['OA','BA','validation_j']].mean().to_string(),flush=True)
    print(pd.DataFrame(paired).to_string(index=False),flush=True)
    print('SOURCE SELECTED',chosen,flush=True)


if __name__ == '__main__':
    main()
