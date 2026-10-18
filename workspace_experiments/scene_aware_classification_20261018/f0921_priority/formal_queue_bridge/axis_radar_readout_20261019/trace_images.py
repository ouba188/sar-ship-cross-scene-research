"""Render actual largest signed-path supports, not only nearest neighbors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import inspect_cases as inspection

HERE=Path(__file__).resolve().parent


def main():
    meta=pd.read_csv(HERE.parent/'cohort.csv').set_index('row',drop=False)
    axes=pd.read_csv(HERE/'axes.csv').set_index('row',drop=False)
    geo=pd.read_csv(HERE/'geometry.csv').set_index('row',drop=False)
    selections=[('',73,'fg_bg_global'),('condition_',175,'availability_control'),('condition_',102,'fg_bg_global')]
    out=HERE/'case_review/signed_paths';out.mkdir(exist_ok=True)
    records=[]
    for prefix,query,comp in selections:
        events=pd.read_csv(HERE/(prefix+'selected_cases.csv'),dtype={'fold':str})
        event=events[events.row.eq(query)&events.comparator.eq(comp)].iloc[0]
        contrib=pd.read_csv(HERE/(prefix+'case_support_contributions.csv'),dtype={'fold':str})
        part=contrib[contrib.row.eq(query)&contrib.comparator.eq(comp)&contrib.fold.eq(event.fold)&contrib.budget.eq(event.budget)&contrib.draw.eq(event.draw)]
        ordered=part.sort_values('total')
        selected=pd.concat([ordered.head(2),ordered.tail(2)]).drop_duplicates('support_row')
        ids=[query]+selected.support_row.astype(int).tolist()
        inspection.axis_panel(ids,meta,axes,geo,out/f'row{query}_{comp}_axes.png')
        for i in ids:
            native=out/f'row{query}_{comp}_member{i}_native.png'
            inspection.review.native_query(meta.loc[i],native)
            g,a=geo.loc[i],axes.loc[i]
            entry=dict(query=query,comparator=comp,fold=event.fold,budget=int(event.budget),draw=int(event.draw),
                member=i,label=int(meta.loc[i].label),native=str(native),radar_valid=bool(g.geometry_valid),
                relative_axis_deg=float(np.degrees((a.ship_axis_rad-g.radar_angle_image_rad)%np.pi)) if g.geometry_valid else None,
                incidence_deg=float(g.incidence_deg) if g.geometry_valid else None)
            if i!=query:
                entry.update(selected[selected.support_row.eq(i)].iloc[0].to_dict())
            records.append(entry)
    pd.DataFrame(records).to_csv(out/'trace_image_index.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':
    main()
