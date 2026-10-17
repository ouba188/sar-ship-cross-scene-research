"""Count restored information actually reached by each task's two roles."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'geometry_targeted_fix_20261016'))
from recover_geometry import fixed_tasks, masks


def main():
    old=pd.read_csv(HERE.parent/'geometry_targeted_fix_20261016/recovered_geometry.csv',low_memory=False)
    new=pd.read_csv(HERE.parent/'geometry_source_recovery_20261016/recovered_geometry.csv',low_memory=False)
    cols=['ship_axis_rad','radar_axis_rad','incidence_rad','ship_valid','radar_valid','incidence_valid']
    changed=(old[cols].fillna(0)!=new[cols].fillna(0)).any(axis=1).to_numpy()
    rows=[]
    for pool,split,meta,tasks in fixed_tasks():
        for origin in sorted({t['origin'] for t in tasks}):
            ts=[t for t in tasks if t['origin']==origin]
            for family,criterion in [('strict','triple'),('optional','ship_required')]:
                active={}; counts={}
                for version,frame in [('old',old),('new',new)]:
                    mask=masks(frame)[criterion]
                    active[version]={'s':set(),'q':set()}; counts[version]=0
                    for t in ts:
                        s=meta.iloc[t['s']].global_row.to_numpy(int)
                        q=meta.iloc[t['q']].global_row.to_numpy(int)
                        if mask[s].sum()<2 or not mask[q].any():
                            continue
                        counts[version]+=1
                        active[version]['s'].update(s[mask[s]])
                        active[version]['q'].update(q[mask[q]])
                for role in ('s','q'):
                    a,b=active['old'][role],active['new'][role]
                    ids=sorted(b)
                    rows.append(dict(pool=pool,split=split,origin=origin,family=family,role=role,
                        old_tasks=counts['old'],new_tasks=counts['new'],old_active=len(a),new_active=len(b),
                        new_active_rows=len(b-a),changed_fields_active=int(changed[ids].sum()),
                        active_ships=int(meta.loc[meta.global_row.isin(ids),'mmsi'].nunique())))
    out=HERE/'diagnostics'; out.mkdir(exist_ok=True)
    result=pd.DataFrame(rows)
    result.to_csv(out/'data_exposure.csv',index=False)
    print(result.loc[result.changed_fields_active.gt(0)].to_string(index=False))


if __name__=='__main__':
    main()
