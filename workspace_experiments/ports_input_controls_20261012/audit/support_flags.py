"""Locate geometry flags in existing supports; do not solve or change them."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=Path('E:/Hermes/paper_tracking/sensitivity')
flags=pd.read_csv(OUT/'primary_geometry_flags.csv').fillna('')
records=[]
for dataset,task in [('self24ports','Los Angeles'),('self24ports','Port Klang'),('self24ports','Santos'),('oss1','oss1'),('oss2','oss2')]:
    if dataset=='self24ports':
        path=ROOT/f'learned_frequency_residual_20261009/tasks/self24ports/{task}/target.npz'
        with np.load(path,allow_pickle=False) as z:
            ids=z['ids'].astype(str)
            supports=z['support']
        subset=flags[(flags.dataset==dataset)&(flags.port==task)]
    else:
        path=ROOT/f'opensarship{dataset[-1]}_transfer_20261007/policies.npz'
        with np.load(path,allow_pickle=False) as z:
            ids=z['sample_ids'].astype(str)
            supports=z['support_rows']
        subset=flags[flags.dataset==dataset]
    assert len(supports)==5 and supports.shape[1]==64
    for sample_id,g in subset.groupby('sample_id'):
        matches=np.flatnonzero(ids==sample_id)
        assert len(matches)==1
        pos=int(matches[0])
        unknown=dataset!='self24ports' and (not g.xml_size_matches_header.astype(bool).all() or not g.endpoints_inside_original.astype(bool).all())
        for draw,support in enumerate(supports):
            records.append(dict(dataset=dataset,task=task,sample_id=sample_id,target_local_row=pos,
                class_name=g.class_name.iloc[0],risk_kind='indeterminate_original_geometry' if unknown else 'new_crop_proxy_crossing',
                draw=draw,in_support=bool(pos in support),support_source=str(path)))
table=pd.DataFrame(records)
table.to_csv(OUT/'support_geometry_flags.csv',index=False,encoding='utf-8-sig')
print(table.groupby(['dataset','task','risk_kind']).agg(samples=('sample_id','nunique'),support_occurrences=('in_support','sum')).to_string())
