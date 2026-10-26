"""Read saved frozen reference/intervention predictions and render 16 cases only."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplcache')
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EVIDENCE = HERE.parent
ROOT = EVIDENCE.parent
TRANSFER = ROOT / 'public_m123_cross_dataset_20261026'
REPAIR = TRANSFER / 'normalization_repair_20261026'
sys.path.insert(0, str(ROOT / 'oss1_m123_replication_20261026'))
import extract_frontend as extract
CLASSES = ('bulk', 'general', 'chemical', 'container')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    cases, summaries = [], []
    for source, target, intervention in (('oss1', 'oss2', 'no_end'), ('oss2', 'oss1', 'no_side')):
        direction = f'{source}-to-{target}'
        merged = pd.read_csv(REPAIR / direction / 'table.csv', dtype={'mmsi': str})
        cohort = pd.read_csv(REPAIR / direction / 'target_table.csv', dtype={'mmsi': str})
        y = cohort.class_id.to_numpy(int)
        with np.load(REPAIR / direction / 'seed17/predictions.npz') as z:
            np.testing.assert_array_equal(z['obs_id'], cohort.obs_id)
            original = z['frequency'].copy()
            original_refs, query_rows = z['refs'].copy(), z['query_rows'].copy()
            local, weights = z['local'].copy(), z['weight_frequency'].copy()
        with np.load(EVIDENCE / 'reference_sensitivity' / source / 'draw101/seed17.npz') as z:
            n = len(z['within_obs_id'])
            np.testing.assert_array_equal(z['cross_obs_id'], cohort.obs_id)
            alternate, alternate_refs = z['frequency'][n:].copy(), z['refs'][n:].copy()
        with np.load(EVIDENCE / 'm3_interventions' / source / 'seed17.npz') as z:
            n = len(z['within_obs_id'])
            np.testing.assert_array_equal(z['cross_obs_id'], cohort.obs_id)
            np.testing.assert_array_equal(z['original'][n:], original)
            intervened = z[intervention][n:].copy()
        # Table source rows are exactly the original source stack rows; refs use them.
        source_table = pd.read_csv(ROOT / f'{source}_m123_replication_20261026/stack_table.csv')
        np.testing.assert_array_equal(merged.loc[merged.dataset_side.eq('source'), 'original_obs_id'], source_table.obs_id)
        rival = np.where(np.arange(4)[None] == y[:, None], -np.inf, original).argmax(1)
        ix = np.arange(len(y))
        before_fixed = original[ix, y] - original[ix, rival]
        picks = []
        for family, arm, score, refs in (('reference', 'draw101', alternate, alternate_refs),
                                         ('mechanism', intervention, intervened, original_refs)):
            after_fixed = score[ix, y] - score[ix, rival]
            change = after_fixed - before_fixed
            before_ok, after_ok = original.argmax(1) == y, score.argmax(1) == y
            oa = float(after_ok.mean() - before_ok.mean()) * 100
            ba = float(np.mean([after_ok[y == c].mean() - before_ok[y == c].mean() for c in range(4)])) * 100
            summaries.append(dict(source=source, direction=direction, family=family, arm=arm, seed=17, n=len(y),
                ships=cohort.mmsi.nunique(), passes=cohort.pass_group.nunique(),
                rescued=int((after_ok & ~before_ok).sum()), harmed=int((~after_ok & before_ok).sum()),
                changed=int((score.argmax(1) != original.argmax(1)).sum()), OA_delta_pp=oa, BA_delta_pp=ba,
                max_fixed_pair_change=float(abs(change).max())))
            for status, mask in (('rescue', after_ok & ~before_ok), ('harm', ~after_ok & before_ok),
                                  ('CC', after_ok & before_ok), ('WW', ~after_ok & ~before_ok)):
                pool = np.flatnonzero(mask)
                if not len(pool):
                    raise ValueError(f'No requested outcome group: {source}/{family}/{status}')
                i = int(pool[np.argmax(abs(change[pool]))])
                true, other = int(y[i]), int(rival[i])
                old = original_refs[i, [true, other]]
                new = refs[i, [true, other]]
                if family == 'reference':
                    rows = [int(query_rows[i]), int(old[0, 0]), int(old[1, 0]), int(new[0, 0]), int(new[1, 0])]
                    names = ['QUERY', 'OLD TRUE #1', 'OLD RIVAL #1', 'NEW TRUE #1', 'NEW RIVAL #1']
                else:
                    rows = [int(query_rows[i]), *old.ravel().astype(int).tolist()]
                    names = ['QUERY', 'TRUE #1', 'TRUE #2', 'RIVAL #1', 'RIVAL #2']
                picks.append(dict(source=source, direction=direction, family=family, arm=arm, seed=17,
                    status=status, obs_id=int(cohort.iloc[i].obs_id), sample_id=cohort.iloc[i].sample_id,
                    true=true, rival=other, before_prediction=int(original[i].argmax()), after_prediction=int(score[i].argmax()),
                    before_fixed_margin=float(before_fixed[i]), after_fixed_margin=float(after_fixed[i]),
                    fixed_pair_change=float(change[i]),
                    before_scores=original[i].tolist(), after_scores=score[i].tolist(),
                    before_top_margin=float(before_fixed[i]),
                    after_top_margin=float(score[i, true] - np.where(np.arange(4) == true, -np.inf, score[i]).max()),
                    old_refs_source_rows=old.tolist(), new_refs_source_rows=new.tolist(),
                    old_refs_obs_id=merged.iloc[old.ravel()].original_obs_id.to_numpy(int).reshape(2, 2).tolist(),
                    new_refs_obs_id=merged.iloc[new.ravel()].original_obs_id.to_numpy(int).reshape(2, 2).tolist(),
                    old_polar_weights=weights[i].tolist(), old_local_true_rival=local[i, [true, other]].tolist(),
                    display_rows=rows, display_names=names,
                    image=f'{source}_{family}_{arm}_{status}_{int(cohort.iloc[i].obs_id)}.png'))
        needed = sorted({row for a in picks for row in a['display_rows']})
        records = {}
        for side, dataset in (('source', source), ('target', target)):
            root = ROOT / f'{dataset}_m123_replication_20261026'
            part = merged.loc[[j for j in needed if merged.iloc[j].dataset_side == side]].copy()
            part['obs_id'] = part.original_obs_id.astype(int)
            extract.VIEW_TABLE = root / 'view_inventory.csv'
            extract.RADAR_TABLE = root / 'geometry.csv' if (root / 'geometry.csv').exists() else Path('E:/临时会话/oss1_joint_geometry_20261017/geometry.csv')
            records.update(zip(part.index, extract.prepare_inventory(part)))
        prepared = {i: extract.prepare(r) for i, r in records.items()}
        for a in picks:
            fig, axes = plt.subplots(2, 5, figsize=(15, 6.7))
            provenance = []
            for col, (row, name) in enumerate(zip(a['display_rows'], a['display_names'])):
                for p, view in enumerate(prepared[row]):
                    ax = axes[p, col]
                    if view is not None:
                        ax.imshow(view['pixels'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                        if view['masks'][0].any():
                            ax.contour(view['masks'][0], levels=[.5], colors=['#19a580'], linewidths=.65)
                        provenance.append(dict(display=name, row=row, obs_id=int(merged.iloc[row].original_obs_id),
                            pol=('VV', 'VH')[p], image=records[row][('vv', 'vh')[p]]['image'],
                            native_shape=list(view['raw'].shape), placement=view['placement'],
                            target_valid=bool(view['valid_target'])))
                    else:
                        ax.text(.5, .5, 'Unavailable', ha='center')
                    ax.set_title(f"{('VV', 'VH')[p]} {name} obs{int(merged.iloc[row].original_obs_id)}\n{CLASSES[int(merged.iloc[row].class_id)]}", fontsize=8)
                    ax.set(xticks=[], yticks=[])
            fig.suptitle(f"{a['direction']} {a['family']}/{a['arm']} {a['status']} seed17\n"
                f"true={CLASSES[a['true']]} rival={CLASSES[a['rival']]}; {CLASSES[a['before_prediction']]} -> {CLASSES[a['after_prediction']]}\n"
                f"fixed margin {a['before_fixed_margin']:+.4f} -> {a['after_fixed_margin']:+.4f} (change {a['fixed_pair_change']:+.4f})", fontsize=11)
            footer = ('Frozen bank replacement; first retrieved reference per class shown; both references recorded in JSON.'
                      if a['family'] == 'reference' else
                      'Matching-summary intervention; original pixels and actual references unchanged. Not a pixel/physical intervention.')
            fig.text(.02, .025, footer + '\nNative160 crop/pad; VV/VH separate; fixed 0-255 display; green = detected target.', fontsize=8)
            fig.subplots_adjust(top=.79, bottom=.12, hspace=.28, wspace=.08)
            fig.savefig(HERE / a['image'], dpi=115)
            plt.close(fig)
            a['provenance'] = provenance
            cases.append(a)
            print('SAVED', a['image'], flush=True)
    pd.DataFrame(summaries).to_csv(HERE / 'summary.csv', index=False)
    write_json(HERE / 'cases.json', cases)
    write_json(HERE / 'receipt.json', dict(panels=len(cases), seed=17, cross_only=True, trained=False,
        selection='per outcome group max absolute change in true versus original strongest nontrue class fixed margin; table-order ties',
        reference='draw101 only; frozen bank replacement, no retraining',
        mechanism={'oss1': 'no_end', 'oss2': 'no_side'}, visual_review_pending=True))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
