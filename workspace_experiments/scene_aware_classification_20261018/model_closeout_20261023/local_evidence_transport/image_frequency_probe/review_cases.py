"""Review raw query, perturbation, and both actual references of competing classes."""
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run


def main():
    f = pd.read_csv(HERE/'prediction_effects.csv')
    part = f[f.main_sample & f.seed.eq(17) & f.kind.eq('attenuate')].copy()
    rescues = f[f.main_sample & f.kind.eq('attenuate') & f.rescue.eq(1)]
    if len(rescues):
        rescues = rescues[rescues.seed.eq(rescues.seed.min())]
    definitions = [('rescue', rescues, 'max'), ('harm', part[part.harm.eq(1)], 'min'),
        ('CC', part[part.before_correct & part.after_correct], 'abs'),
        ('WW', part[~part.before_correct & ~part.after_correct], 'abs')]
    picked = []
    for name, pool, direction in definitions:
        if len(pool):
            key = pool.delta_margin.abs().idxmax() if direction == 'abs' else (pool.delta_margin.idxmax() if direction == 'max' else pool.delta_margin.idxmin())
            row = pool.loc[key].to_dict(); row['case'] = name; picked.append(row)
    for obs in (135, 161):
        pool = f[f.obs_id.eq(obs) & f.seed.eq(17) & f.pol.eq(0) & f.region.eq('neighbor') & f.band.eq('coarse') & f.kind.eq('attenuate')]
        if len(pool):
            row = pool.iloc[0].to_dict(); row['case'] = f'prior_{obs}'; picked.append(row)
    # Opposite-polarization coarse-neighbor signal gets its own largest-effect example.
    pool = part[part.pol.eq(1) & part.region.eq('neighbor') & part.band.eq('coarse')]
    row = pool.loc[pool.delta_margin.idxmax()].to_dict(); row['case'] = 'VH_coarse_positive'; picked.append(row)
    pool = part[part.pol.eq(0) & part.region.eq('neighbor') & part.band.eq('coarse')]
    row = pool.loc[pool.delta_margin.idxmin()].to_dict(); row['case'] = 'VV_coarse_negative'; picked.append(row)
    table = pd.read_csv(run.UP/'table.csv', dtype={'mmsi': str})
    snapshots = {int(s['obs_id']): s for s in json.loads((run.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs = np.load(HERE/'reference_ids.npz')['refs']; valid = np.load(run.ROOT/'tokens.npz')['valid']
    z = np.load(HERE/'results.npz'); iv = pd.read_csv(HERE/'interventions.csv')
    out = HERE/'analysis'; out.mkdir(exist_ok=True); metadata = []
    for case in picked:
        i = int(case['table_index']); p0 = int(case['pol']); c = int(case['class_id']); rival = int(case['fixed_rival']); j = int(case['local_index'])
        sr = refs[j, [c, rival]].ravel()
        fig, axes = plt.subplots(2, 7, figsize=(19, 6.7), squeeze=False)
        provenance = []
        for p in range(2):
            pix, fp, g, placement, bands, regions = run.load_view(table, snapshots, i, p)
            alt = pix.copy()
            if p == p0:
                b = run.BANDS.index(case['band']); r = 0 if case['region'] == 'body' else 1
                alt = run.perturbations(pix, placement, bands[b], regions[r], int(case['obs_id']), p, b)[0][1]
            for col, image in enumerate((pix, alt)):
                axes[p,col].imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                region = run.geo.soft_regions(g)[0]*fp
                if region.max() > .5:
                    axes[p,col].contour(region, levels=[.5], colors=['#20a780'], linewidths=.6)
            d = alt-pix; scale = max(float(abs(d).max()), .005)
            axes[p,2].imshow(d, cmap='RdBu_r', vmin=-scale, vmax=scale, interpolation='nearest')
            axes[p,0].set_title(f'{("VV","VH")[p]} query {case["obs_id"]}\n{table.iloc[i].class_name}', fontsize=8)
            axes[p,1].set_title('attenuated' if p == p0 else 'unchanged', fontsize=8)
            axes[p,2].set_title(f'actual difference\nrange +/-{scale:.3f}', fontsize=8)
            provenance.append(dict(obs_id=int(table.iloc[i].obs_id), pol=p, path=table.iloc[i]['image_'+('vv','vh')[p]]))
            for col, ri in enumerate(sr, 3):
                raw, rfp, rg, _ = run.geo.native(table, snapshots, int(ri), p)
                axes[p,col].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                if rg is not None and valid[ri,p]:
                    region = run.geo.soft_regions(rg)[0]*rfp
                    if region.max() > .5:
                        axes[p,col].contour(region, levels=[.5], colors=['#20a780'], linewidths=.6)
                typ = 'TRUE' if col < 5 else 'RIVAL'
                axes[p,col].set_title(f'{typ} {int(table.iloc[ri].obs_id)} valid={int(valid[ri,p])}\n{table.iloc[ri].class_name}', fontsize=8)
                provenance.append(dict(obs_id=int(table.iloc[ri].obs_id), pol=p, path=table.iloc[ri]['image_'+('vv','vh')[p]]))
        for ax in axes.ravel():
            ax.set(xticks=[], yticks=[])
        fig.suptitle(f"SOURCE seed{case['seed']} | {case['case']} | {('VV','VH')[p0]} {case['region']} {case['band']}\n"
            f"prediction {case['before']} -> {case['after']}; true={c}; margin {case['margin_before']:+.3f} -> {case['margin_after']:+.3f}; "
            f"true score change {case['true_score_change']:+.3f}, fixed rival {case['rival_score_change']:+.3f}", fontsize=11)
        fig.text(.02,.025,'Native160 crop/pad, no resize; same display range; original references frozen. Cases chosen for explanation, not estimated prevalence.', fontsize=9)
        fig.subplots_adjust(top=.83,bottom=.09,wspace=.12,hspace=.18)
        fig.savefig(out/f"{case['case']}_{case['obs_id']}.png",dpi=130); plt.close(fig)
        index = int(iv.index[iv.obs_id.eq(case['obs_id']) & iv.variant.eq(case['variant'])][0])
        before = z['local'][z['rawloc'][j]]; after = z['local'][index]
        case['true_local_score_change'] = (after[c,...,0]-before[c,...,0]).tolist()
        case['rival_local_score_change'] = (after[rival,...,0]-before[rival,...,0]).tolist()
        case['reference_obs'] = table.iloc[sr].obs_id.astype(int).tolist()
        case['provenance'] = provenance
        metadata.append(case)
    (out/'cases.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(pd.DataFrame(metadata)[['case','obs_id','pol','region','band','before','after','delta_margin','true_score_change','rival_score_change']].to_string(index=False))


if __name__ == '__main__':
    main()
