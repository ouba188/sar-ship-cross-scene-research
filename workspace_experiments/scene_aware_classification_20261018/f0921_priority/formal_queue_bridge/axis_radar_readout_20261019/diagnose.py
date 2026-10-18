"""Measure whether geometry actually changes usable relations; no extra tuning."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_axis as ev

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent


def main():
    features=dict(np.load(PARENT/'features.npz'))
    d=dict(np.load(HERE/'direction_features.npz'))
    axes=pd.read_csv(HERE/'axes.csv')
    geo=pd.read_csv(HERE/'geometry.csv')
    meta=pd.read_csv(PARENT/'cohort.csv',dtype={'date':str})
    tasks=json.loads((PARENT/'episodes.json').read_text())
    labels=meta.label.to_numpy()
    items=[]
    with threadpool_limits(limits=1):
        for task in tasks:
            q=np.array(task['query'])
            for draw,rows in enumerate(task['supports']):
                s=np.array(rows)
                local=ev.kernels(features,d,axes,s)
                for arm,comp in [('ship_soft','image_soft'),('ship_radar','ship_image'),('ship_soft','fg_bg_global')]:
                    ka,kb=local[arm][np.ix_(q,s)],local[comp][np.ix_(q,s)]
                    same=labels[q,None]==labels[s][None]
                    for group,mask in [('all',np.ones_like(same)),('both_radar_known',d['radar_valid'][q,None]&d['radar_valid'][s][None])]:
                        if not mask.any():
                            continue
                        delta=ka-kb
                        items.append(dict(fold=task['fold'],budget=task['budget'],draw=draw,arm=arm,comparator=comp,group=group,
                            pairs=int(mask.sum()),mean_abs_local_delta=float(np.abs(delta[mask]).mean()),max_abs_local_delta=float(np.abs(delta[mask]).max()),
                            correlation=float(np.corrcoef(ka[mask],kb[mask])[0,1]),
                            same_delta=float(delta[mask&same].mean()) if (mask&same).any() else np.nan,
                            wrong_delta=float(delta[mask&~same].mean()) if (mask&~same).any() else np.nan,
                            support_radar_available=int(d['radar_valid'][s].sum()),query_radar_available=int(d['radar_valid'][q].sum())))
    pd.DataFrame(items).to_csv(HERE/'kernel_diagnostics.csv',index=False)
    rows=[]
    for date,index in meta.groupby('date').groups.items():
        index=np.array(list(index))
        good=index[d['radar_valid'][index]]
        radar=geo.iloc[good].radar_angle_image_rad.to_numpy()
        ship=axes.iloc[good].ship_axis_rad.to_numpy()
        # Axial angles are modulo pi; report signed distance from image horizontal.
        horizontal=np.degrees(.5*np.angle(np.exp(2j*radar)))
        relative=np.degrees(.5*np.arccos(np.clip(np.cos(2*(ship-radar)),-1,1)))
        rows.append(dict(date=date,rows=len(index),geometry_valid=len(good),
            radar_image_min_deg=horizontal.min() if len(good) else np.nan,
            radar_image_max_deg=horizontal.max() if len(good) else np.nan,
            ship_radar_relative_min_deg=relative.min() if len(good) else np.nan,
            ship_radar_relative_max_deg=relative.max() if len(good) else np.nan,
            incidence_min=geo.iloc[good].incidence_deg.min(),incidence_max=geo.iloc[good].incidence_deg.max(),
            axis_reliability_mean=axes.iloc[index].axis_reliability.mean()))
    pd.DataFrame(rows).to_csv(HERE/'geometry_by_date.csv',index=False)
    p=pd.read_csv(HERE/'predictions.csv',dtype={'fold':str})
    a=p[p.arm.eq('ship_radar')].set_index(ev.KEYS).sort_index()
    b=p[p.arm.eq('ship_image')].set_index(ev.KEYS).sort_index()
    values=np.abs(a[ev.roi.PROBS].to_numpy()-b[ev.roi.PROBS].to_numpy())
    receipt=dict(radar_vs_matched_image_probability_mean_abs=float(values.mean()),
        radar_vs_matched_image_probability_max_abs=float(values.max()),
        changed_top1=int((a.prediction!=b.prediction).sum()),
        axis_reliability_quantiles=axes.axis_reliability.quantile([0,.25,.5,.75,1]).to_dict(),
        both_pol_angle_gap_quantiles=axes.polarization_axis_gap_deg.quantile([.5,.9,1]).to_dict(),
        role_means_reused=True,new_training=False)
    (HERE/'diagnostic_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(items).groupby(['arm','comparator','group'])[['mean_abs_local_delta','correlation']].mean().to_string())
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
