"""Inspect the second direct rescue and the late-versus-early aggregation harm."""
import analyze_hybrid as a
import json
import numpy as np
import pandas as pd
import torch
tr=a.tr


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=tr.Data();reader=dict(np.load(tr.HERE/'reader.npz'))
    on=np.load(a.HERE/'results/axis_role_43.npz')['score']
    cache=np.load(a.HERE/'results/switches.npz')
    geo=tr.load('hybrid_extra_geo',tr.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    up=pd.read_csv(tr.UP/'table.csv');valid=np.load(tr.UP/'summaries.npz')['valid'];refs=np.load(tr.LOCAL/'scores_layer1.npz')['refs']
    records=[]
    for obs,status,variant in [(33528,'rescue','off'),(25781,'harm','early')]:
        j=int(np.flatnonzero(data.table.obs_id.to_numpy()==obs)[0]);k=int(np.flatnonzero(data.te==j)[0])
        case=dict(seed=43,obs_id=obs,index=j,eval_index=k,case=status,comparison='conv_off' if variant=='off' else 'early_mean',
                  selection='All remaining direct rescue and only direct late-vs-early harm; posthoc diagnostic, not extra independent cohort')
        before=cache[f'43_axis_role_{variant}'][k];after=on[j]
        path,provenance,true,rival=a.render(case,data,before,after,geo,up,snapshots,valid,refs)
        records.append(dict(**case,true=true,rival=rival,before=before,after=after,
            old_full=data.baselines[43][j],conv_off=cache['43_axis_role_off'][k],early=cache['43_axis_role_early'][k],
            diagnostics=a.case_diagnostics(case,data,reader,rival),figure=path,provenance=provenance))
    a.write_json(a.HERE/'analysis/extra_cases.json',records)
    print('Extra cases rendered:',[r['obs_id'] for r in records])


if __name__=='__main__':main()
