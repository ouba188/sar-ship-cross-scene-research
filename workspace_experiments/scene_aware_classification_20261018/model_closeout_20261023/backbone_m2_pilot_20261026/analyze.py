"""Paired outcomes and native-image evidence for the fixed M2 pilot."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from run import HERE, WEIGHT, SEEDS, MODES, load, batch, predict, write, RegionResNet50


def restored(mode, seed):
    model = RegionResNet50(WEIGHT, mode=mode).cuda().eval()
    ck = torch.load(HERE/'training'/f'{mode}_{seed}'/'selected.pt', weights_only=True)
    model.load_state_dict(ck['state'], strict=False)
    return model


def result(mode, seed):
    with np.load(HERE/'training'/f'{mode}_{seed}'/'evaluation.npz') as z:
        return {k: z[k].copy() for k in z.files}


def margin(score, y):
    other = score.copy()
    other[np.arange(len(y)), y] = -np.inf
    return score[np.arange(len(y)), y] - other.max(1)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    table, arrays = load()
    images = np.load(HERE/'cache/images.npy', mmap_mode='r')
    out = HERE/'analysis'
    out.mkdir(exist_ok=True)
    base = {s: result('none', s) for s in SEEDS}
    ids, y = base[17]['ids'], base[17]['y']
    frame = table.iloc[ids].reset_index(drop=True)
    summary, paired, perclass, diagnostics, predictions = [], [], [], [], []
    metrics = pd.read_csv(HERE/'metrics.csv')
    metrics.groupby('mode')[['OA', 'BA', 'NLL', 'Brier']].agg(['mean', 'std']).to_csv(HERE/'summary.csv')
    for mode in ('merged', 'regional'):
        changes = []
        for seed in SEEDS:
            data, old = result(mode, seed), base[seed]
            correct, oldcorrect = data['score'].argmax(1) == y, old['score'].argmax(1) == y
            changes.append(correct.astype(float)-oldcorrect)
            paired.append(dict(mode=mode, seed=seed, rescued=int((correct & ~oldcorrect).sum()),
                harmed=int((~correct & oldcorrect).sum()), delta_oa=float(correct.mean()-oldcorrect.mean())))
            for c in range(8):
                use = y == c
                perclass.append(dict(mode=mode, seed=seed, class_id=c, n=int(use.sum()),
                    recall=float(correct[use].mean()), delta_recall=float((correct[use]-oldcorrect[use].astype(float)).mean())))
            model = restored(mode, seed)
            model.stages = ()
            bypass = predict(model, arrays, ids)
            model.stages = (2,)
            take = ids[:48]
            x, masks, valid, avail = batch(arrays, take)
            for p in range(2):
                m = masks[:, p]
                if mode == 'merged':
                    m = m.sum(1, keepdim=True).clamp(0, 1).expand(-1, 3, -1, -1)
                _, d = model.blocks[f'{p}_2'](x[:, p], m, valid[:, p], return_details=True)
                active = m.sum(1) > 0
                ratio = d['update'].norm(dim=1)[active].mean()/x[:, p].norm(dim=1)[active].mean()
                diagnostics.append(dict(mode=mode, seed=seed, pol=p,
                    gain=float(model.blocks[f'{p}_2'].gain), relative_update=float(ratio),
                    update_norm=float(d['update'].norm()),
                    bypass_score_max=float(abs(data['score']-bypass).max()),
                    bypass_prediction_changes=int((data['score'].argmax(1) != bypass.argmax(1)).sum()),
                    bypass_net_rescues=int(((data['score'].argmax(1) == y).astype(int)-(bypass.argmax(1) == y)).sum())))
            for i in range(len(ids)):
                predictions.append(dict(mode=mode, seed=seed, obs_id=int(frame.iloc[i].obs_id),
                    y=int(y[i]), baseline=int(old['score'][i].argmax()), prediction=int(data['score'][i].argmax()),
                    margin_before=float(margin(old['score'][i:i+1], y[i:i+1])[0]),
                    margin_after=float(margin(data['score'][i:i+1], y[i:i+1])[0])))
            del model
        change = np.mean(changes, axis=0)
        groups = [np.flatnonzero(frame.pass_group.eq(g)) for g in frame.pass_group.unique()]
        rng = np.random.default_rng(171)
        samples = []
        for _ in range(2000):
            selected = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            samples.append(change[selected].mean())
        summary.append(dict(mode=mode, delta_oa=float(change.mean()),
            pass_bootstrap_low=float(np.quantile(samples, .025)), pass_bootstrap_high=float(np.quantile(samples, .975))))
    pd.DataFrame(paired).to_csv(HERE/'paired.csv', index=False)
    pd.DataFrame(perclass).to_csv(HERE/'by_class.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(HERE/'diagnostics.csv', index=False)
    pd.DataFrame(summary).to_csv(HERE/'uncertainty.csv', index=False)
    pd.DataFrame(predictions).to_csv(HERE/'predictions.csv', index=False)

    # Predeclared outcome categories; largest margin difference within each.
    candidates = []
    for seed in SEEDS:
        old, new = base[seed]['score'], result('regional', seed)['score']
        cb, cn = old.argmax(1) == y, new.argmax(1) == y
        delta = margin(new, y)-margin(old, y)
        for name, flag in [('rescued', ~cb & cn), ('harmed', cb & ~cn), ('CC', cb & cn), ('WW', ~cb & ~cn)]:
            pool = np.flatnonzero(flag)
            if len(pool):
                k = pool[np.argmax(abs(delta[pool]))]
                candidates.append((name, seed, int(k), float(abs(delta[k]))))
    cases = []
    for name in ('rescued', 'harmed', 'CC', 'WW'):
        pool = [r for r in candidates if r[0] == name]
        if not pool:
            continue
        _, seed, k, _ = max(pool, key=lambda r: r[3])
        idx = int(ids[k])
        row = frame.iloc[k]
        model = restored('regional', seed)
        x, masks, valid, avail = batch(arrays, [idx])
        old, new = base[seed]['score'][k], result('regional', seed)['score'][k]
        fig, axes = plt.subplots(2, 5, figsize=(14, 7.5), constrained_layout=True)
        stats = []
        for p, pol in enumerate(('VV', 'VH')):
            _, d = model.blocks[f'{p}_2'](x[:, p], masks[:, p], valid[:, p], return_details=True)
            img = images[idx, p, 0]
            axes[p, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[p, 0].set_title(f'{pol} native crop/pad')
            update = d['update'][0].norm(dim=0).cpu().numpy()
            axes[p, 1].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[p, 1].imshow(update, extent=(0,160,160,0), cmap='magma', alpha=.65,
                              interpolation='nearest', vmin=0)
            axes[p, 1].set_title(f'Feature update, max={update.max():.5f}')
            for r, label in enumerate(('body', 'end', 'side')):
                weights = d['weights'][0, r].cpu().numpy()
                axes[p, r+2].imshow(img, cmap='gray', vmin=0, vmax=255)
                axes[p, r+2].imshow(weights, extent=(0,160,160,0), cmap='viridis', vmin=0, vmax=1, alpha=.6,
                                    interpolation='nearest')
                axes[p, r+2].set_title(label+' feedback weight')
            stats.append(dict(pol=pol, available=bool(avail[0, p]), update_max=float(update.max()),
                              region_mass=d['mass'][0].cpu().tolist()))
        for a in axes.ravel():
            a.axis('off')
        fig.suptitle(f'{name}: obs {row.obs_id}, seed {seed}, true {y[k]}, pred {old.argmax()} -> {new.argmax()}\n'
                     f'true margin {margin(old[None],y[k:k+1])[0]:.4f} -> {margin(new[None],y[k:k+1])[0]:.4f}')
        filename = f'{name}_{row.obs_id}_{seed}.png'
        fig.savefig(out/filename, dpi=130)
        plt.close(fig)
        cases.append(dict(kind=name, obs_id=int(row.obs_id), seed=seed, true=int(y[k]),
                          before=int(old.argmax()), after=int(new.argmax()),
                          image_vv=row.image_vv, image_vh=row.image_vh, panel=filename,
                          old_score=old.tolist(), new_score=new.tolist(), details=stats))
        del model
    write(out/'cases.json', cases)
    write(HERE/'analysis_complete.json', dict(case_count=len(cases), evaluated=485,
           note='Feature-cell feedback, not physical scattering attribution. No reference bank in this pilot.'))
    print(pd.read_csv(HERE/'metrics.csv').to_string(index=False), flush=True)
    print(pd.DataFrame(paired).to_string(index=False), flush=True)
    print(pd.DataFrame(diagnostics).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
