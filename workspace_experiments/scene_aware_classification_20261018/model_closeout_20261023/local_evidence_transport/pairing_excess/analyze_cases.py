"""Native chips and exact paired-versus-repaired evidence for frozen fits."""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run as experiment

HERE = Path(__file__).resolve().parent
OUT = HERE/'analysis'


def main():
    OUT.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(OUT/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from extract_tokens import geo, slots
    table, x, mass, bank = experiment.edge.support.load()
    y = table.class_id.to_numpy(int)
    receipts = pd.read_csv(HERE/'training_receipts.csv')
    active = receipts[receipts.arm.eq('excess') & receipts.selected_step.gt(0)].seed.to_list()
    seed = int(active[0]) if active else 17
    pred = np.load(HERE/f'predictions_{seed}.npz')
    features = np.load(HERE/f'features_{seed}.npz')
    weights = np.load(HERE/'weights.npz')
    scales = pd.read_csv(HERE/'scales.csv')
    divisor = scales[scales.arm.eq('excess')].sort_values('feature').divisor.to_numpy()
    reader = experiment.Reader(x, mass, bank, 'excess', divisor, device='cpu', dtype=torch.float64)
    W = torch.tensor(weights[f'excess_{seed}_W'], dtype=torch.float64)
    coef = weights[f'excess_{seed}_a']
    ev = np.flatnonzero(table.role.eq('evaluation'))
    old, new = pred['anchor'][ev].argmax(-1) == y[ev], pred['excess'][ev].argmax(-1) == y[ev]
    cases, margins, contributions, references = [], [], [], []
    for name, mask in [('rescued', ~old & new), ('harmed', old & ~new), ('CC', old & new), ('WW', ~old & ~new)]:
        ids = ev[mask]
        if not len(ids):
            continue
        i = int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy())])
        c = int(y[i])
        score = pred['excess'][i] if name == 'harmed' else pred['anchor'][i]
        r = int(np.argmax(np.where(np.arange(8) == c, -np.inf, score)))
        with torch.inference_mode():
            identity = reader.full_scores(np.array([i]), torch.eye(32, dtype=torch.float64), True)
            learned = reader.full_scores(np.array([i]), W, True)
        z = (learned[0]-identity[0])[0].numpy()
        parts = coef*(z[c]-z[r])
        replay = float(parts.sum())
        observed = float(pred['excess'][i, c]-pred['excess'][i, r]-pred['anchor'][i, c]+pred['anchor'][i, r])
        assert abs(replay-observed) < 2e-5, (i, replay, observed)
        group = int(np.argmax(np.abs(parts)))
        ref_ids = {}
        for cl in (c, r):
            refdiff = learned[3][group]['reference_score'][0, cl]-identity[3][group]['reference_score'][0, cl]
            valid = learned[3][group]['reference_valid'][cl]
            positions = np.flatnonzero(valid)
            chosen = positions[np.argsort(-np.abs(refdiff[positions]), kind='stable')[:2]]
            ref_ids[cl] = chosen.tolist()
        ca = dict(i=i, obs_id=int(table.iloc[i].obs_id), outcome=name, seed=seed, true=c, rival=r, group=group,
                  count=int(mask.sum()), references=ref_ids, margin_increment=replay, replay_error=abs(replay-observed))
        cases.append(ca)
        for g, label in enumerate(experiment.edge.GROUPS):
            contributions.append(dict(obs_id=ca['obs_id'], group=label, increment=float(parts[g])))
            for cl in (c, r):
                for j in range(8):
                    refs = learned[3][g]
                    references.append(dict(obs_id=ca['obs_id'], group=label, class_id=cl,
                        reference_obs=int(table.iloc[bank[cl, j]].obs_id), valid=bool(refs['reference_valid'][cl, j]),
                        identity=float(identity[3][g]['reference_score'][0, cl, j]),
                        learned=float(refs['reference_score'][0, cl, j]),
                        difference=float(refs['reference_score'][0, cl, j]-identity[3][g]['reference_score'][0, cl, j])))
        for arm in ('anchor', 'joint', 'covariance', 'excess', 'excess_shuffle101', 'excess_shuffle307'):
            v = pred[arm][i]
            margins.append(dict(obs_id=ca['obs_id'], outcome=name, arm=arm, seed=seed, true=c, rival=r,
                                prediction=int(v.argmax()), margin=float(v[c]-v[r])))
        # The visible matrix is the learned change. Row contrast reproduces the local excess change.
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        qi = np.flatnonzero(reader.legal[i, group].numpy())
        for ax, cl in zip(axes, (c, r)):
            j = ref_ids[cl][0]
            C = learned[3][group]['C'][0, cl, j]-identity[3][group]['C'][0, cl, j]
            C = C[np.ix_(qi, qi)]
            vmax = max(float(np.abs(C).max()), 1e-8)
            im = ax.imshow(C, cmap='coolwarm', vmin=-vmax, vmax=vmax)
            ax.set(title=f'class {cl}; ref obs {int(table.iloc[bank[cl,j]].obs_id)}\nmean diag-minus-row = {np.mean(C.diagonal()-C.mean(1)):+.5f}',
                   xlabel='query neighborhood slot k', ylabel='query body slot i', xticks=np.arange(len(qi)), yticks=np.arange(len(qi)), xticklabels=qi, yticklabels=qi)
            for a in range(len(qi)):
                ax.scatter(a, a, facecolors='none', edgecolors='black', s=100)
            fig.colorbar(im, ax=ax, shrink=.75)
        fig.suptitle(f"obs {ca['obs_id']}, {name}, {experiment.edge.GROUPS[group]} | C(W)-C(I), not physical attribution")
        fig.tight_layout()
        fig.savefig(OUT/f"matrix_{name}_{ca['obs_id']}.png", dpi=140)
        plt.close(fig)
    wanted = {int(table.iloc[ix].obs_id) for ca in cases for ix in [ca['i'], *[bank[cl,j] for cl in (ca['true'],ca['rival']) for j in ca['references'][cl]]]}
    snaps = {int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    tokens = np.load(HERE.parent/'tokens.npz')
    image_records = []
    for ca in cases:
        views = [(ca['i'], 'QUERY')]+[(bank[cl,j], f'{role} REF') for cl,role in ((ca['true'],'TRUE'),(ca['rival'],'RIVAL')) for j in ca['references'][cl]]
        fig, axes = plt.subplots(2, len(views), figsize=(4*len(views), 9), squeeze=False)
        for p, pol in enumerate(('vv', 'vh')):
            for column,(ix,label) in enumerate(views):
                pix, fp, geometry, place = geo.native(table, snaps, int(ix), p)
                ax = axes[p,column]
                ax.imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                if tokens['valid'][ix,p]:
                    masks = slots(geometry, fp)
                    for (lo,hi),color in zip(((0,8),(8,16),(16,20)),('#24a878','#2c84c6','#d39222')):
                        region = masks[lo:hi].sum(0)
                        if region.min() < .5 < region.max():
                            ax.contour(region, levels=[.5], colors=[color], linewidths=.7)
                row = table.iloc[ix]
                ax.set(title=f'{pol.upper()} {label} obs {int(row.obs_id)}\n{row.class_name}; {row.port}\nvalid={int(tokens["valid"][ix,p])}',
                       xlim=(-.5,159.5), ylim=(159.5,-.5), xticks=[0,80,159], yticks=[0,80,159])
                image_records.append(dict(query_obs=ca['obs_id'], obs_id=int(row.obs_id), pol=pol, path=row['image_'+pol], placement=place))
        current = [v for v in margins if v['obs_id'] == ca['obs_id']]
        fig.suptitle(f"Pairing excess: {ca['outcome']} obs {ca['obs_id']} | seed {seed} | group n={ca['count']}\nReferences: largest absolute W-I relation change, not attention or independent trials", fontsize=12)
        text = '; '.join(f"{v['arm']}: {v['margin']:+.3f}" for v in current)
        fig.text(.02,.015,text+'\nNative160 crop/pad, NO resize; layer1 RF35. Continuous margins compare true with fixed rival.',fontsize=9)
        fig.subplots_adjust(top=.84,bottom=.12,wspace=.18,hspace=.35)
        fig.savefig(OUT/f"case_{ca['outcome']}_{ca['obs_id']}.png",dpi=130)
        plt.close(fig)
    pd.DataFrame(margins).to_csv(OUT/'case_margins.csv', index=False)
    pd.DataFrame(contributions).to_csv(OUT/'group_contributions.csv', index=False)
    pd.DataFrame(references).to_csv(OUT/'reference_contributions.csv', index=False)
    (OUT/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (OUT/'image_records.json').write_text(json.dumps(image_records,indent=2),encoding='utf-8')
    print(pd.DataFrame(margins).to_string(index=False))


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
