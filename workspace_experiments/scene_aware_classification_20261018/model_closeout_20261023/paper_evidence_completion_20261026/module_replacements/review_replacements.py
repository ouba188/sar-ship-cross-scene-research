"""Post-hoc paired summaries and native-chip evidence, with no training."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplcache')
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_replacements as run

CLASSES = ('bulk', 'general', 'chemical', 'container')


def margin(s, y):
    return s[np.arange(len(y)), y] - np.where(np.arange(4)[None] == y[:, None], -np.inf, s).max(1)


def case_components(out, arm, base, sums, index, true, rival):
    ck = torch.load(out / 'models' / f'{arm}_seed17.pt', map_location='cpu', weights_only=False)
    model = run.front.upstream.AttentionHead() if arm.startswith('m1_') else run.front.upstream.RegionHead()
    model.load_state_dict(ck['state'])
    one = {k: v[index:index + 1] for k, v in sums.items()}
    raw = run.raw_features(arm, base[index:index + 1], one)
    x = torch.tensor(run.transform(raw, ck))
    with torch.inference_mode():
        if arm.startswith('m1_'):
            mean = model.mean(x[..., [0, 1, 2, 7, 8, 13, 14]])[0, :, 0].numpy()
            local = model.local(x[..., [0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]])[0, :, 0].numpy()
            return dict(mean_branch_pair=float(mean[true] - mean[rival]),
                        local_branch_pair=float(local[true] - local[rival]))
        delta = model(x)[0].numpy()
        return dict(region_head_pair=float(delta[true] - delta[rival]))


def main():
    replay, cases = [], []
    outgallery = HERE / 'rawcases'
    outgallery.mkdir(exist_ok=True)
    for source, target in (('oss1', 'oss2'), ('oss2', 'oss1')):
        direction = f'{source}-to-{target}'
        print('Review direction', direction, flush=True)
        out = HERE / direction
        for seed in run.SEEDS:
            for side in ('source', 'cross'):
                with np.load(out / f'seed{seed}/{side}_predictions.npz') as z:
                    for refit, original in (('m1_both_refit', 'original_m1'), ('m2_identity_refit', 'original_s12')):
                        a, b = z[refit], z[original]
                        replay.append(dict(direction=direction, seed=seed, side=side, arm=refit,
                            max_abs_score_difference=float(abs(a - b).max()),
                            changed_predictions=int((a.argmax(1) != b.argmax(1)).sum())))
        table = pd.read_csv(run.TRANSFER / direction / 'target_table.csv', dtype={'mmsi': str})
        print('Table and replay read', direction, flush=True)
        sums = run.front.aligned(run.TRANSFER / direction / 'summaries/summaries.npz', table)
        with np.load(out / 'seed17/cross_predictions.npz') as z:
            scores = {k: z[k].copy() for k in ('frontend_base', 'original_m1', *run.ARMS)}
        targetroot = run.ROOT / f'{target}_m123_replication_20261026'
        run.front.extract.VIEW_TABLE = targetroot / 'view_inventory.csv'
        run.front.extract.RADAR_TABLE = (targetroot / 'geometry.csv' if (targetroot / 'geometry.csv').exists()
                                        else Path('E:/临时会话/oss1_joint_geometry_20261017/geometry.csv'))
        y = table.class_id.to_numpy(int)
        for candidate, reference in run.PAIRS[:2]:
            print('Cases', direction, candidate, flush=True)
            a, b = scores[candidate], scores[reference]
            ac, bc = a.argmax(1) == y, b.argmax(1) == y
            delta = margin(a, y) - margin(b, y)
            picked = []
            for status, mask in (('rescue', ac & ~bc), ('harm', ~ac & bc), ('CC', ac & bc), ('WW', ~ac & ~bc)):
                idx = np.flatnonzero(mask)
                if len(idx):
                    i = int(idx[np.argmax(abs(delta[idx]))])
                    picked.append((status, i))
            records = run.front.extract.prepare_inventory(table.iloc[[i for _, i in picked]])
            print('Inventory read', [int(table.iloc[i].obs_id) for _, i in picked], flush=True)
            prepared = [run.front.extract.prepare(record) for record in records]
            extent = max(max(v['raw'].shape) for pair in prepared for v in pair if v is not None)
            fig, axes = plt.subplots(2, len(picked), figsize=(13, 7), squeeze=False)
            for col, ((status, i), record, views) in enumerate(zip(picked, records, prepared)):
                true = int(y[i])
                rival = int(np.where(np.arange(4) == true, -np.inf, b[i]).argmax())
                base = scores['frontend_base' if candidate.startswith('m1_') else 'original_m1']
                comps = {arm: case_components(out, arm, base, sums, i, true, rival) for arm in (reference, candidate)}
                residual_change = sum(comps[candidate].values()) - sum(comps[reference].values())
                actual = float((a[i, true] - a[i, rival]) - (b[i, true] - b[i, rival]))
                if abs(residual_change - actual) > 2e-5:
                    raise ValueError('Case fixed-pair branch accounting failed')
                provenance = []
                for p, v in enumerate(views):
                    ax = axes[p, col]
                    if v is None:
                        ax.text(.5, .5, 'Unavailable', ha='center')
                        continue
                    ax.imshow(v['raw'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                    ax.set_xlim(-.5, extent - .5)
                    ax.set_ylim(extent - .5, -.5)
                    ax.set_title(f"{status} obs{table.iloc[i].obs_id} {('VV', 'VH')[p]}\n"
                                 f"native {v['raw'].shape}, target={v['valid_target']}", fontsize=9)
                    ax.set(xticks=[], yticks=[])
                    provenance.append(dict(pol=('VV', 'VH')[p], path=record[('vv', 'vh')[p]]['image'],
                        native_shape=list(v['raw'].shape), placement=v['placement']))
                axes[1, col].set_xlabel(f"true {CLASSES[true]}\n{CLASSES[b[i].argmax()]} -> {CLASSES[a[i].argmax()]}\n"
                    f"margin {margin(b[i:i+1], y[i:i+1])[0]:+.3f} -> {margin(a[i:i+1], y[i:i+1])[0]:+.3f}", fontsize=9)
                cases.append(dict(direction=direction, seed=17, candidate=candidate, reference=reference,
                    status=status, obs_id=int(table.iloc[i].obs_id), sample_id=table.iloc[i].sample_id,
                    mmsi=str(table.iloc[i].mmsi), pass_group=table.iloc[i].pass_group,
                    true=true, rival=rival, before_prediction=int(b[i].argmax()), after_prediction=int(a[i].argmax()),
                    before_margin=float(margin(b[i:i+1], y[i:i+1])[0]), after_margin=float(margin(a[i:i+1], y[i:i+1])[0]),
                    fixed_pair_change=actual, branch_components=comps,
                    valid=sums['valid'][i].tolist(), quality=sums['quality'][i].tolist(),
                    mean3=sums['mean3'][i, :, [true, rival]].tolist(), hb3=sums['hb3'][i, :, [true, rival]].tolist(),
                    summary1=sums['summary1'][i, :, :, [true, rival]].tolist(), provenance=provenance,
                    panel=f'{direction}_{candidate}.png'))
            fig.suptitle(f'{direction} seed17: {reference} -> {candidate}\nNative query pixels only; fixed weights/refit scores; no M3 references', fontsize=12)
            fig.tight_layout(rect=(0, .02, 1, .92))
            fig.savefig(outgallery / f'{direction}_{candidate}.png', dpi=130)
            plt.close(fig)
            print('Panel saved', direction, candidate, flush=True)
        pd.DataFrame(replay).to_csv(HERE / 'identity_replay.csv', index=False)
        run.write_json(outgallery / 'cases.json', cases)
    pd.DataFrame(replay).to_csv(HERE / 'identity_replay.csv', index=False)
    run.write_json(outgallery / 'cases.json', cases)
    run.write_json(outgallery / 'receipt.json', dict(seed=17, cases=len(cases), panels=4,
        selection='each outcome group maximum absolute true-vs-best-other margin change; fixed seed17',
        native_pixels=True, interpolation='nearest', display_range=[0, 255], references_used=False,
        mechanism_scope='joint summary/input and head-refit change, conditional on frozen prior stages',
        visual_review_pending=True))
    print(pd.DataFrame(replay).to_string(index=False))
    print(pd.DataFrame([{k: a[k] for k in ('direction', 'candidate', 'status', 'obs_id', 'before_margin', 'after_margin', 'fixed_pair_change')} for a in cases]).to_string(index=False))


if __name__ == '__main__':
    main()
