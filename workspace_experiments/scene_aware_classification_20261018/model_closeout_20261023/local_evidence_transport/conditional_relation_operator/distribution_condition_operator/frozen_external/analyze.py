"""Native query/reference review of frozen output-kernel generalization."""
import os
import sys
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('external_kernel_run', HERE/'run.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
spec = importlib.util.spec_from_file_location('output_channel_trace', HERE.parent/'analyze_independent.py')
tracer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracer)


def main():
    arm = 'excess' if '--excess' in sys.argv else 'independent'
    if arm == 'excess':
        spec = importlib.util.spec_from_file_location('external_conditional_trace', HERE.parent/'analyze.py')
        ct = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ct)
        def trace(reader, i, mb, mn):
            raw, values, detail, error = ct.trace(reader, i, mb, mn)
            return raw, values[..., None], detail, error
    else:
        trace = tracer.trace
    out = HERE/('excess_analysis' if arm == 'excess' else 'analysis')
    out.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from extract_tokens import geo, slots
    source, target, x, mass, bank, idx, tr, va, te, ids = ev.load()
    table = pd.concat([source, target], ignore_index=True)
    seed = 43 if arm == 'excess' else 17
    saved = np.load(HERE/f'predictions_{seed}.npz')
    labels = target.class_id.to_numpy(int)
    old, new, simple = [saved[k] for k in ('full', arm, 'old_independent')]
    ok, nk, sk = [z.argmax(-1) == labels for z in (old, new, simple)]
    w = np.load(HERE.parent/'weights.npz')
    mb, mn, a = [w[f'{arm}_{seed}_{k}'] for k in ('Mb', 'Mn', 'a')]
    scales = pd.read_csv(HERE.parent/'scales.csv')
    scale = scales[scales.arm.eq(arm)].sort_values('feature').divisor.to_numpy()
    reader = ev.run.Reader(x, mass, bank, arm, device='cpu', dtype=torch.float64)
    cases, margins, channels, refs = [], [], [], []
    definitions = [('rescued', ~ok & nk), ('harmed', ok & ~nk), ('CC', ok & nk), ('WW', ~ok & ~nk),
                   ('better_than_simple', ~sk & nk), ('worse_than_simple', sk & ~nk)]
    for name, mask in definitions:
        options = np.flatnonzero(mask)
        if not len(options):
            continue
        loc = int(options[np.argmin(target.iloc[options].obs_id.to_numpy())])
        i, c = int(ids[loc]), int(labels[loc])
        score = new[loc] if name in ('harmed', 'worse_than_simple') else (simple[loc] if name == 'better_than_simple' else old[loc])
        rival = int(np.argmax(np.where(np.arange(8) == c, -np.inf, score)))
        raw0, v0, d0, e0 = trace(reader, i, np.eye(32), np.eye(32))
        raw1, v1, d1, e1 = trace(reader, i, mb, mn)
        contribution = (v1-v0).sum((2, 3))
        channel = a[:, None]*(contribution[:, c]-contribution[:, rival])/scale[:, None]
        error = abs(float(channel.sum()-(new[loc, c]-new[loc, rival]-old[loc, c]+old[loc, rival])))
        assert error < 4e-5, error
        g = int(abs(channel.sum(-1)).argmax())
        picked = []
        for cl in (c, rival):
            value = a[g]*(v1[g, cl]-v0[g, cl]).sum((-1, -2))/scale[g]
            j = int(np.argmax(np.where(d1['rv'][g, cl], abs(value), -np.inf)))
            picked.append(int(bank[cl, j]))
            for r in np.flatnonzero(d1['rv'][g, cl]):
                refs.append(dict(case=name, obs_id=int(table.iloc[i].obs_id), class_id=cl,
                    ref_obs=int(table.iloc[bank[cl, r]].obs_id), group=ev.exp.edge.GROUPS[g], increment=float(value[r])))
        cases.append(dict(case=name, i=i, loc=loc, obs_id=int(table.iloc[i].obs_id), n=int(mask.sum()),
            true=c, rival=rival, refs=picked, group=g, seed=seed, replay_error=error, allocation_error=max(e0, e1)))
        for j, group in enumerate(ev.exp.edge.GROUPS):
            row = dict(case=name, obs_id=int(table.iloc[i].obs_id), group=group, total=float(channel[j].sum()))
            if arm == 'independent':
                row.update(body=float(channel[j, 0]), neighbor=float(channel[j, 1]))
            channels.append(row)
        for method in ('full', 'independent', 'excess', 'old_independent', 'old_body_match', 'old_pooled'):
            value = saved[method][loc]
            margins.append(dict(case=name, obs_id=int(table.iloc[i].obs_id), arm=method,
                prediction=int(value.argmax()), margin=float(value[c]-value[rival])))
    wanted = {int(table.iloc[i].obs_id) for ca in cases for i in [ca['i'], *ca['refs']]}
    snapshots = {int(s['obs_id']): s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    valid = np.concatenate([np.load(ev.ROOT/'tokens.npz')['valid'], np.load(ev.EXT/'tokens.npz')['valid']])
    provenance = []
    for ca in cases:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
        for col, (i, label) in enumerate(zip([ca['i'], *ca['refs']], ('QUERY', 'TRUE REF', 'RIVAL REF'))):
            for p, pol in enumerate(('vv', 'vh')):
                pixels, fp, geometry, place = geo.native(table, snapshots, i, p)
                ax = axes[p, col]
                ax.imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                if valid[i, p]:
                    masks = slots(geometry, fp)
                    for (lo, hi), color in zip(((0,8), (8,16), (16,20)), ('#24a878', '#2c84c6', '#d39222')):
                        region = masks[lo:hi].sum(0)
                        if region.min() < .5 < region.max():
                            ax.contour(region, levels=[.5], colors=[color], linewidths=.7)
                row = table.iloc[i]
                ax.set(title=f"{pol.upper()} {label} {int(row.obs_id)}\n{row.class_name}\n{row.port}; valid={int(valid[i,p])}",
                    xlim=(-.5,159.5), ylim=(159.5,-.5), xticks=[0,80,159], yticks=[0,80,159])
                provenance.append(dict(case=ca['case'], query_obs=ca['obs_id'], obs_id=int(row.obs_id),
                    pol=pol, path=row['image_'+pol], placement=place, valid=bool(valid[i,p])))
        fig.suptitle(f"Frozen external {arm} | {ca['case']} {ca['obs_id']}; n={ca['n']}; seed{seed}\nActual contributions in {ev.exp.edge.GROUPS[ca['group']]}")
        text = '; '.join(v['arm']+f": pred{v['prediction']}, margin{v['margin']:+.3f}" for v in margins if v['case']==ca['case'])
        fig.text(.02,.02,text+'\nNative160 crop/pad; no resize. Contribution allocation, not causal ablation.',fontsize=7)
        fig.subplots_adjust(top=.83,bottom=.13,wspace=.25,hspace=.5)
        fig.savefig(out/f"case_{ca['case']}_{ca['obs_id']}.png",dpi=130)
        plt.close(fig)
    for name, values in [('channels',channels),('margins',margins),('references',refs)]:
        pd.DataFrame(values).to_csv(out/f'{name}.csv',index=False)
    (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (out/'image_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(pd.DataFrame(margins).to_string(index=False))
    print(pd.DataFrame(channels).to_string(index=False))


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
