"""Eight paired native VV/VH queries with actual frozen source references."""
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT, REPAIR, TRANSFER, load, write
import extract_frontend as extract
CLASSES = ('bulk', 'general', 'chemical', 'container')


def main():
    cases = []
    for source, target in [('oss1', 'oss2'), ('oss2', 'oss1')]:
        direction = f'{source}-to-{target}'
        table = pd.read_csv(REPAIR/direction/'table.csv', dtype={'mmsi': str})
        cohort = pd.read_csv(TRANSFER/direction/'target_table.csv', dtype={'mmsi': str})
        y = cohort.class_id.to_numpy(int)
        z = load(HERE/direction/'seed17/strong/cross_predictions.npz')
        orig = load(REPAIR/direction/'seed17/predictions.npz')
        np.testing.assert_array_equal(z['obs_id'], cohort.obs_id)
        np.testing.assert_array_equal(orig['obs_id'], cohort.obs_id)
        before, after = z['bare'], z['frequency']
        rival = np.where(np.arange(4)[None]==y[:, None], -np.inf, before).argmax(1)
        ids = np.arange(len(y))
        mb = before[ids, y]-before[ids, rival]
        ma = after[ids, y]-after[ids, rival]
        a, b = before.argmax(1)==y, after.argmax(1)==y
        picks = []
        for status, mask in [('rescue', b&~a), ('harm', a&~b), ('CC', a&b), ('WW', ~a&~b)]:
            pool = np.flatnonzero(mask)
            if not len(pool):
                raise ValueError(f'No {direction}/{status} for predeclared seed17')
            i = int(pool[np.argmax(abs((ma-mb)[pool]))])
            true, other = int(y[i]), int(rival[i])
            refs = orig['refs'][i, [true, other]].astype(int)
            rows = [int(orig['query_rows'][i]), *refs.ravel().tolist()]
            picks.append(dict(direction=direction, seed=17, status=status, obs_id=int(cohort.iloc[i].obs_id),
                mmsi=str(cohort.iloc[i].mmsi), pass_group=str(cohort.iloc[i].pass_group), true=true, rival=other,
                before_prediction=int(before[i].argmax()), after_prediction=int(after[i].argmax()),
                before_margin=float(mb[i]), after_margin=float(ma[i]),
                M2_margin=float(z['M2'][i,true]-z['M2'][i,other]),
                B_margin=float(z['B_fixed'][i,true]-z['B_fixed'][i,other]),
                before_scores=before[i].tolist(), after_scores=after[i].tolist(),
                polar_weight=z['weight_frequency'][i].tolist(),
                delta_vv=z['delta_vv'][i].tolist(), delta_vh=z['delta_vh'][i].tolist(),
                refs_source_rows=refs.tolist(), refs_obs_id=table.iloc[refs.ravel()].original_obs_id.to_numpy(int).reshape(2,2).tolist(),
                rows=rows, image=f'{source}_{status}_{int(cohort.iloc[i].obs_id)}.png'))
        needed = sorted({r for p in picks for r in p['rows']})
        records = {}
        for side, dataset in [('source', source), ('target', target)]:
            root = ROOT/f'{dataset}_m123_replication_20261026'
            part = table.loc[[i for i in needed if table.iloc[i].dataset_side==side]].copy()
            part['obs_id'] = part.original_obs_id.astype(int)
            extract.VIEW_TABLE = root/'view_inventory.csv'
            extract.RADAR_TABLE = root/'geometry.csv' if (root/'geometry.csv').exists() else Path('E:/临时会话/oss1_joint_geometry_20261017/geometry.csv')
            records.update(zip(part.index, extract.prepare_inventory(part)))
        prepared = {i: extract.prepare(r) for i,r in records.items()}
        for p in picks:
            fig, axes = plt.subplots(2,5,figsize=(15,6.7))
            provenance=[]
            for col,(row,name) in enumerate(zip(p['rows'],['QUERY','TRUE #1','TRUE #2','RIVAL #1','RIVAL #2'])):
                for pol,view in enumerate(prepared[row]):
                    ax=axes[pol,col]
                    if view is not None:
                        ax.imshow(view['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                        if view['masks'][0].any():
                            ax.contour(view['masks'][0],levels=[.5],colors=['#19a580'],linewidths=.65)
                        provenance.append(dict(name=name,pol=('VV','VH')[pol],row=row,
                            image=records[row][('vv','vh')[pol]]['image'], native_shape=list(view['raw'].shape),
                            placement=view['placement'],valid_target=bool(view['valid_target'])))
                    else:
                        ax.text(.5,.5,'Unavailable',ha='center')
                    ax.set_title(f"{('VV','VH')[pol]} {name} obs{int(table.iloc[row].original_obs_id)}\n{CLASSES[int(table.iloc[row].class_id)]}",fontsize=8)
                    ax.set(xticks=[],yticks=[])
            fig.suptitle(f"{direction} strong base -> M2 + fixed-matcher M3; {p['status']}; seed17\n"
                f"true {CLASSES[p['true']]}: {CLASSES[p['before_prediction']]} -> {CLASSES[p['after_prediction']]}\n"
                f"fixed margin bare {p['before_margin']:+.3f} / M2 {p['M2_margin']:+.3f} / B {p['B_margin']:+.3f} / frequency {p['after_margin']:+.3f}",fontsize=10)
            fig.text(.02,.025,'Native160 crop/pad, no geometric resizing; VV/VH separately. Green: accepted OBB.\nActual retrieved references; apparent shape overlap is not a physical causal explanation.',fontsize=8)
            fig.subplots_adjust(top=.78,bottom=.12,hspace=.28,wspace=.08)
            fig.savefig(HERE/p['image'],dpi=115)
            plt.close(fig)
            p['provenance']=provenance
            cases.append(p)
    write(HERE/'cases.json',cases)
    write(HERE/'gallery_receipt.json',dict(panels=len(cases),seed=17,
        selection='each cross direction, max absolute fixed true/rival margin change per rescue/harm/CC/WW',
        visual_review_pending=True, actual_frozen_references=True))


if __name__=='__main__':
    main()
