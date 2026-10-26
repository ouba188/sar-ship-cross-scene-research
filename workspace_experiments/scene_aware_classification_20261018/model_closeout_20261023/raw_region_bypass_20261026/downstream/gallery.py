"""Adapted from strong_base_refit/gallery.py; same native chips and references."""
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplcache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT, REPAIR, TRANSFER, EVIDENCE, load, write, CLASSES, DIRECTIONS
import extract_frontend as extract


def main():
    assert (HERE / 'evaluation_complete.json').exists()
    cases, counts = [], []
    image_dir = HERE / 'gallery'
    image_dir.mkdir(exist_ok=True)
    for source, target in DIRECTIONS:
        direction = f'{source}-to-{target}'
        table = pd.read_csv(REPAIR / direction / 'table.csv', dtype={'mmsi': str})
        cohort = pd.read_csv(TRANSFER / direction / 'target_table.csv', dtype={'mmsi': str})
        y = cohort.class_id.to_numpy(int)
        z = load(HERE / 'anchored_raw_summary' / direction / 'seed17/cross_predictions.npz')
        orig = load(REPAIR / direction / 'seed17/predictions.npz')
        raw = load(EVIDENCE / f'classical_fit_only/{source}/cross_predictions.npz')
        for obj in (z, orig, raw):
            np.testing.assert_array_equal(obj['obs_id'], cohort.obs_id)
        before, after, rawscore = z['bare'], z['frequency'], raw['regional_mlp_seed17']
        rival = np.where(np.arange(4)[None] == y[:, None], -np.inf, before).argmax(1)
        ids = np.arange(len(y))
        mb = before[ids, y] - before[ids, rival]
        ma = after[ids, y] - after[ids, rival]
        mr = rawscore[ids, y] - rawscore[ids, rival]
        a, b = before.argmax(1) == y, after.argmax(1) == y
        masks = dict(rescue=b & ~a, harm=a & ~b, CC=a & b, WW=~a & ~b)
        counts.append(dict(direction=direction, seed=17, **{k: int(v.sum()) for k, v in masks.items()}))
        # Primary rank: M3 margin movement; tied zero changes use bare-vs-rawMLP movement.
        ranked = {status: sorted(np.flatnonzero(mask).tolist(),
                  key=lambda i: (-abs(float(ma[i]-mb[i])), -abs(float(mb[i]-mr[i])), i))
                  for status, mask in masks.items()}
        selected = [(status, pool.pop(0)) for status, pool in ranked.items() if pool]
        while len(selected) < 4:
            for status in ('CC', 'WW'):
                if ranked[status] and len(selected) < 4:
                    selected.append((status, ranked[status].pop(0)))
        picks = []
        for number, (status, i) in enumerate(selected, 1):
            true, other = int(y[i]), int(rival[i])
            refs = orig['refs'][i, [true, other]].astype(int)
            rows = [int(orig['query_rows'][i]), *refs.ravel().tolist()]
            centered_before = before[i]-before[i].mean()
            centered_raw = rawscore[i]-rawscore[i].mean()
            picks.append(dict(direction=direction, seed=17, status=status,
                candidate='anchored_raw_summary', target_index=i, obs_id=int(cohort.iloc[i].obs_id),
                mmsi=str(cohort.iloc[i].mmsi), pass_group=str(cohort.iloc[i].pass_group),
                true=true, rival=other, before_prediction=int(before[i].argmax()), after_prediction=int(after[i].argmax()),
                rawMLP_prediction=int(rawscore[i].argmax()), rawMLP_margin=float(mr[i]),
                before_margin=float(mb[i]), after_margin=float(ma[i]),
                joint_margin=float(z['joint'][i, true]-z['joint'][i, other]),
                B_margin=float(z['B_fixed'][i, true]-z['B_fixed'][i, other]),
                rawMLP_scores=rawscore[i].tolist(), before_scores=before[i].tolist(), after_scores=after[i].tolist(),
                bare_minus_rawMLP_centered=(centered_before-centered_raw).tolist(),
                bare_minus_rawMLP_true_vs_each=((before[i,true]-before[i])-(rawscore[i,true]-rawscore[i])).tolist(),
                final_minus_bare=(after[i]-before[i]).tolist(),
                polar_weight=z['weight_frequency'][i].tolist(), delta_vv=z['delta_vv'][i].tolist(),
                delta_vh=z['delta_vh'][i].tolist(), available=z['available'][i].tolist(),
                refs_source_rows=refs.tolist(), refs_obs_id=table.iloc[refs.ravel()].original_obs_id.to_numpy(int).reshape(2,2).tolist(),
                rows=rows, image=f'{source}_{number}_{status}_{int(cohort.iloc[i].obs_id)}.png'))
        needed = sorted({r for p in picks for r in p['rows']})
        records = {}
        for side, dataset in [('source', source), ('target', target)]:
            root = ROOT / f'{dataset}_m123_replication_20261026'
            part = table.loc[[i for i in needed if table.iloc[i].dataset_side == side]].copy()
            part['obs_id'] = part.original_obs_id.astype(int)
            extract.VIEW_TABLE = root / 'view_inventory.csv'
            extract.RADAR_TABLE = root / 'geometry.csv' if (root / 'geometry.csv').exists() else Path('E:/临时会话/oss1_joint_geometry_20261017/geometry.csv')
            records.update(zip(part.index, extract.prepare_inventory(part)))
        prepared = {i: extract.prepare(record) for i, record in records.items()}
        for p in picks:
            fig, axes = plt.subplots(2, 5, figsize=(15, 6.7))
            provenance = []
            for col, (row, name) in enumerate(zip(p['rows'], ['QUERY', 'TRUE #1', 'TRUE #2', 'RIVAL #1', 'RIVAL #2'])):
                for pol, view in enumerate(prepared[row]):
                    ax = axes[pol, col]
                    if view is not None:
                        ax.imshow(view['pixels'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                        if view['masks'][0].any():
                            ax.contour(view['masks'][0], levels=[.5], colors=['#19a580'], linewidths=.65)
                        provenance.append(dict(name=name, pol=('VV', 'VH')[pol], row=row,
                            obs_id=int(table.iloc[row].original_obs_id),
                            image=records[row][('vv', 'vh')[pol]]['image'], native_shape=list(view['raw'].shape),
                            placement=view['placement'], valid_target=bool(view['valid_target'])))
                    else:
                        ax.text(.5, .5, 'Unavailable', ha='center')
                    ax.set_title(f"{('VV','VH')[pol]} {name} obs{int(table.iloc[row].original_obs_id)}\n{CLASSES[int(table.iloc[row].class_id)]}", fontsize=8)
                    ax.set(xticks=[], yticks=[])
            fig.suptitle(f"{direction} anchored bare -> fixed-matcher M3; {p['status']}; seed17\n"
                f"true {CLASSES[p['true']]}: {CLASSES[p['before_prediction']]} -> {CLASSES[p['after_prediction']]}\n"
                f"fixed true/rival margin rawMLP {p['rawMLP_margin']:+.3f} / bare {p['before_margin']:+.3f} / B {p['B_margin']:+.3f} / final {p['after_margin']:+.3f}", fontsize=10)
            fig.text(.02, .025, 'Native160 crop/pad, no geometric resizing; VV/VH separately. Green: accepted OBB.\nActual frozen source references. Visible resemblance does not establish physical causation.', fontsize=8)
            fig.subplots_adjust(top=.78, bottom=.12, hspace=.28, wspace=.08)
            fig.savefig(image_dir / p['image'], dpi=115)
            plt.close(fig)
            p['provenance'] = provenance
            cases.append(p)
    write(HERE / 'cases.json', cases)
    write(HERE / 'gallery_receipt.json', dict(panels=len(cases), seed=17,
        selection='one per nonempty rescue/harm/CC/WW; fill absent flip categories with extra CC/WW; max absolute M3 margin change, then bare-vs-historical-rawMLP change',
        counts=counts, actual_frozen_references=True, visual_review_pending=True,
        adapted_from=str(ROOT / 'readout_bottleneck_20261026/strong_base_refit/gallery.py')))
    print('GALLERY_SAVED', len(cases), flush=True)


if __name__ == '__main__':
    main()
