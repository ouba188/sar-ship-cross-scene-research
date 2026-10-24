"""Bounded sensitivity check of the observed coarse-neighbor dependence."""
import json
import importlib.util
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, distance_transform_edt
import run

HERE = run.HERE


def main():
    tick = time.perf_counter(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    table = pd.read_csv(run.UP/'table.csv', dtype={'mmsi': str})
    cohort = pd.read_csv(HERE/'cohort.csv', dtype={'mmsi': str})
    ids = cohort.table_index.to_numpy(int)
    z = np.load(run.ROOT/'tokens.npz'); pr = np.load(run.ROOT/'projection_layer1.npz')
    x = run.unit((z['layer1']-pr['mean'])@pr['components'].T).astype(np.float32)
    mass = z['mass']; x[mass <= 0] = 0
    refs = np.load(HERE/'reference_ids.npz')['refs']
    snapshots = {int(s['obs_id']): s for s in json.loads((run.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    extractor = run.Extractor(pr); rows = []; descs = []
    for j, i in enumerate(ids):
        for p in range(2):
            pix, fp, g, placement, bands, regions = run.load_view(table, snapshots, i, p)
            original_mask = regions[1]
            distance = distance_transform_edt(original_mask > 0)
            t = np.clip((distance-.5)/2, 0, 1)
            interior_mask = original_mask*t*t*(3-2*t)
            with Image.open(table.iloc[i]['image_'+('vv', 'vh')[p]]) as im:
                raw = np.array(im.convert('L'), dtype=np.float32)/255
            nearest_band = gaussian_filter(raw, 4, mode='nearest')-gaussian_filter(raw, 12, mode='nearest')
            images = []; metadata = []
            for arm, field, mask in [('interior', bands[2], interior_mask), ('nearest_boundary', nearest_band, original_mask)]:
                for kind, image, info in run.perturbations(pix, placement, field, mask, int(table.iloc[i].obs_id), p, 2):
                    images.append(image)
                    metadata.append(dict(local_index=j, table_index=int(i), obs_id=int(table.iloc[i].obs_id), class_id=int(table.iloc[i].class_id),
                        mmsi=str(table.iloc[i].mmsi), pass_group=str(table.iloc[i].pass_group), main_sample=bool(cohort.iloc[j].main_sample),
                        variant=f'{p}_{arm}_{kind}', pol=p, region='neighbor', band='coarse', boundary_arm=arm, kind=kind,
                        mask_mass=float(mask.sum()), mask_mass_fraction=float(mask.sum()/max(original_mask.sum(), 1)), **info))
            desc = extractor(images, [run.slots(g, fp)]*len(images))
            for k, meta in enumerate(metadata):
                item = x[i].copy(); item[p] = desc[k]
                descs.append(item); rows.append(meta)
        if j%16 == 0:
            print('BOUNDARY', j+1, '/', len(ids), round(time.perf_counter()-tick, 1), flush=True)
    f = pd.DataFrame(rows); owner = f.local_index.to_numpy(int)
    local = run.independent(np.stack(descs), mass[ids[owner]], x[refs[owner]], mass[refs[owner]])
    old = np.load(HERE/'results.npz'); original = pd.read_csv(HERE/'interventions.csv')
    scores = []
    for seed in run.SEEDS:
        base = np.load(run.UP/f'strategies_seed{seed}.npz')['logits'][:, 2].astype(np.float32)
        ck = torch.load(run.ROOT/'models'/f'layer1_independent_{seed}.pt', map_location='cpu', weights_only=False)
        scores.append(run.head_scores(local, base[ids[owner]], ck))
    np.savez_compressed(HERE/'boundary_results.npz', scores=np.stack(scores), local=local)
    f.to_csv(HERE/'boundary_interventions.csv', index=False)
    spec = importlib.util.spec_from_file_location('frequency_analysis', HERE/'analyze.py')
    analysis = importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)
    margin, bootstrap = analysis.margin, analysis.bootstrap
    effects = []; summaries = []
    for si, seed in enumerate(run.SEEDS):
        raw = old['scores'][si, old['rawloc']][owner]
        d = f.copy(); d['seed'] = seed
        d['delta_margin'] = margin(scores[si], f.class_id.to_numpy())-margin(raw, f.class_id.to_numpy())
        d['correct_delta'] = (scores[si].argmax(-1) == f.class_id).astype(float)-(raw.argmax(-1) == f.class_id)
        effects.append(d)
    ef = pd.concat(effects); ef.to_csv(HERE/'boundary_effects.csv', index=False)
    for (arm, p), part in ef[ef.main_sample].groupby(['boundary_arm', 'pol']):
        real = part[part.kind.eq('attenuate')].groupby(['obs_id','pass_group']).mean(numeric_only=True).reset_index()
        control = part[part.kind.ne('attenuate')].groupby('obs_id').mean(numeric_only=True)
        real['diff_vs_control'] = real.delta_margin.to_numpy()-control.loc[real.obs_id, 'delta_margin'].to_numpy()
        lo, hi = bootstrap(real)
        summaries.append(dict(arm=arm, pol=p, n=len(real), margin_change=real.delta_margin.mean(),
            control_margin=control.delta_margin.mean(), difference=real.diff_vs_control.mean(), cluster_low=lo, cluster_high=hi,
            OA_delta_pp=100*real.correct_delta.mean(), control_OA_delta_pp=100*control.correct_delta.mean(),
            mask_fraction=real.mask_mass_fraction.mean(), mean_l2=real.l2.mean()))
    out = pd.DataFrame(summaries); out.to_csv(HERE/'boundary_summary.csv', index=False)
    (HERE/'boundary_receipt.json').write_text(json.dumps(dict(seconds=time.perf_counter()-tick, new_fits=0,
        rationale='coarse neighbor signal only; inspect mask-edge dependence and native boundary assumption',
        note='interior reduces dose and footprint; no claim of isolated edge removal at constant energy'), indent=2), encoding='utf-8')
    print(out.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
