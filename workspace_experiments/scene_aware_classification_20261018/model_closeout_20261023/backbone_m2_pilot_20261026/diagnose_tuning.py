"""Source-only tuning diagnosis, including native validation examples."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from run import HERE, WEIGHT, SEEDS, load, batch, predict, write, RegionResNet50
from tune import OUT, CONFIGS
from analyze import restored, margin


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    table, arrays = load()
    ids = np.flatnonzero(table.role.eq('fit_validation'))
    y = table.iloc[ids].class_id.to_numpy(int)
    rows, scores = [], {}
    for config in ('original', *CONFIGS):
        for seed in SEEDS:
            if config == 'original':
                model = restored('regional', seed)
            else:
                model = RegionResNet50(WEIGHT, mode='regional', temperature=CONFIGS[config]['temperature']).cuda().eval()
                ck = torch.load(OUT/config/f'regional_{seed}'/'selected.pt', weights_only=True)
                model.load_state_dict(ck['state'], strict=False)
            scores[config, seed] = predict(model, arrays, ids)
            values = [[], []]
            for first in range(0, len(ids), 32):
                x, masks, valid, avail = batch(arrays, ids[first:first+32])
                for p in range(2):
                    _, d = model.blocks[f'{p}_2'](x[:, p], masks[:, p], valid[:, p], return_details=True)
                    occupancy = masks[:, p]/masks[:, p].sum(1, keepdim=True).clamp_min(1)
                    use = occupancy > .01
                    values[p].append((d['weights'][use]/occupancy[use]).cpu().numpy())
            for p in range(2):
                g = np.concatenate(values[p])
                rows.append(dict(config=config, seed=seed, pol=('VV', 'VH')[p],
                    gate_mean=float(g.mean()), near_closed=float((g < .01).mean()), near_open=float((g > .99).mean())))
            del model
    pd.DataFrame(rows).to_csv(OUT/'source_gate_diagnostics.csv', index=False)
    grid = pd.read_csv(OUT/'source_grid.csv')
    candidate = grid[grid.config.ne('original')].groupby('config').validation_j.mean().idxmax()
    seed = 17
    old, new = scores['original', seed], scores[candidate, seed]
    cb, cn = old.argmax(1) == y, new.argmax(1) == y
    delta = margin(new, y)-margin(old, y)
    images = np.load(HERE/'cache/images.npy', mmap_mode='r')
    cases = []
    dest = OUT/'source_cases'
    dest.mkdir(exist_ok=True)
    for kind, flag in [('rescued', ~cb & cn), ('harmed', cb & ~cn), ('CC', cb & cn), ('WW', ~cb & ~cn)]:
        pool = np.flatnonzero(flag)
        if not len(pool):
            continue
        k = int(pool[np.argmax(abs(delta[pool]))])
        idx = int(ids[k])
        fig, ax = plt.subplots(1, 2, figsize=(7, 4), constrained_layout=True)
        for p, pol in enumerate(('VV', 'VH')):
            ax[p].imshow(images[idx, p, 0], cmap='gray', vmin=0, vmax=255)
            ax[p].set_title(pol + (' unavailable' if not arrays['available'][idx, p] else ''))
            ax[p].axis('off')
        obs = int(table.iloc[idx].obs_id)
        fig.suptitle(f'SOURCE validation {kind}: {obs}, {candidate}, seed17\n'
                     f'true {y[k]}, pred {old[k].argmax()} -> {new[k].argmax()}, '
                     f'margin {margin(old,y)[k]:.3f} -> {margin(new,y)[k]:.3f}')
        name = f'{kind}_{obs}.png'
        fig.savefig(dest/name, dpi=130)
        plt.close(fig)
        cases.append(dict(obs_id=obs, kind=kind, config=candidate, seed=17,
            true=int(y[k]), before=int(old[k].argmax()), after=int(new[k].argmax()),
            old_margin=float(margin(old,y)[k]), new_margin=float(margin(new,y)[k]),
            image_vv=table.iloc[idx].image_vv, image_vh=table.iloc[idx].image_vh, panel=name))
    write(dest/'cases.json', cases)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(json.dumps(cases, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
