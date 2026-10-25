"""Bounded review of all five new errors and the continued-old control rescue."""
import analyze as a
import json
import numpy as np
import pandas as pd
import torch


def margin_coverage():
    run=a.run;here=a.HERE
    table=pd.read_csv(run.NATIVE/'table.csv')
    old=np.load(here/'results/old_full_43.npz')['score']
    new=np.load(here/'results/typed_both_43.npz')['score']
    rows=[];flips=[]
    for role in ('utility_train','selection','evaluation'):
        ii=np.flatnonzero(table.role.eq(role));y=table.iloc[ii].class_id.to_numpy(int)
        m0,m1=a.margin(old[ii],y),a.margin(new[ii],y)
        for c in range(8):
            keep=y==c;wrong=keep&(m0<0);rescue=wrong&(m1>0);harm=keep&(m0>0)&(m1<0)
            row=dict(role=role,class_id=c,n=int(keep.sum()),old_wrong=int(wrong.sum()),
                rescued=int(rescue.sum()),harmed=int(harm.sum()),
                improved_old_wrong=int((wrong&(m1>m0)).sum()),
                improved_still_wrong=int((wrong&(m1>m0)&(m1<0)).sum()),
                mean_margin_change=float((m1-m0)[keep].mean()))
            for cut in (.05,.1,.2):
                row[f'old_wrong_margin_above_minus_{cut}']=int((wrong&(m0>=-cut)).sum())
            rows.append(row)
            for k in np.flatnonzero(rescue|harm):
                r=table.iloc[ii[k]]
                flips.append(dict(role=role,class_id=c,obs_id=int(r.obs_id),pass_group=r.pass_group,
                    outcome='rescue' if rescue[k] else 'harm',old_margin=float(m0[k]),
                    new_margin=float(m1[k]),margin_change=float(m1[k]-m0[k])))
    pd.DataFrame(rows).to_csv(here/'margin_coverage.csv',index=False)
    pd.DataFrame(flips).to_csv(here/'source_target_flips.csv',index=False)
    print(pd.DataFrame(rows).query('class_id==3').to_string(index=False),flush=True)
    print(pd.DataFrame(flips).query("role=='selection' and class_id==3").to_string(index=False),flush=True)


def main():
    run=a.run;here=a.HERE
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=run.Data();ids=data.te;y=data.table.iloc[ids].class_id.to_numpy(int)
    old=np.load(here/'results/old_full_43.npz')['score']
    new=np.load(here/'results/typed_both_43.npz')['score']
    sw=np.load(here/'results/switches.npz')
    harm=np.flatnonzero((old[ids].argmax(1)==y)&(new[ids].argmax(1)!=y))
    assert len(harm)==5
    reader=run.Reader(data,43);model,_=a.restored(reader,'typed_both',43)
    geo=run.load('anchored_error_geo',run.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up=pd.read_csv(run.BASE/'regional_utility_pilot/upstream/table.csv')
    valid=np.load(run.BASE/'regional_utility_pilot/upstream/summaries.npz')['valid']
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs=np.load(run.LOCAL/'scores_layer1.npz')['refs']
    records=[];rows=[]
    for k in harm:
        j=int(ids[k]);row=data.table.iloc[j]
        case=dict(arm='typed_both',seed=43,index=j,obs_id=int(row.obs_id),case='harm')
        path,prov,true,rival=a.panel(case,data,old[j],new[j],geo,up,valid,snapshots,refs)
        diagnostics=a.diagnose_case(data,reader,model,case,true,rival)
        record=dict(**case,true=true,rival=rival,before=old[j],after=new[j],
                    diagnostics=diagnostics,figure=str(path),provenance=prov)
        records.append(record)
        rows.append(dict(obs_id=int(row.obs_id),class_id=true,class_name=row.class_name,
            mmsi=row.mmsi,port=row.port,pass_group=row.pass_group,
            old_margin=float(old[j,true]-old[j,rival]),new_margin=float(new[j,true]-new[j,rival]),
            old_prediction=int(old[j].argmax()),new_prediction=int(new[j].argmax()),
            transport_only_prediction=int(sw['43_transport'][k].argmax()),
            score_only_prediction=int(sw['43_score'][k].argmax())))
    pd.DataFrame(rows).to_csv(here/'error_trace.csv',index=False)
    # Include the genuine control rescue without attributing it to typed evidence.
    old29=np.load(here/'results/old_full_29.npz')['score']
    continued=np.load(here/'results/old_continue_29.npz')['score']
    rescued=np.flatnonzero((old29[ids].argmax(1)!=y)&(continued[ids].argmax(1)==y))
    for k in rescued:
        j=int(ids[k]);row=data.table.iloc[j]
        case=dict(arm='old_continue',seed=29,index=j,obs_id=int(row.obs_id),case='control_rescue')
        path,prov,true,rival=a.panel(case,data,old29[j],continued[j],geo,up,valid,snapshots,refs)
        records.append(dict(**case,true=true,rival=rival,before=old29[j],after=continued[j],
                            figure=str(path),provenance=prov,diagnostics=None))
    run.write_json(here/'analysis/error_cases.json',records)
    biases=[]
    for role,ii in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
        yy=data.table.iloc[ii].class_id.to_numpy(int);delta=new[ii]-old[ii]
        for c in range(8):
            take=yy==c
            biases.append(dict(role=role,class_id=c,
                mean_candidate_score_change=float(delta[:,c].mean()),
                class_balanced_candidate_score_change=float(np.mean([delta[yy==k,c].mean() for k in range(8)])),
                true_class_margin_change=float((a.margin(new[ii][take],yy[take])-a.margin(old[ii][take],yy[take])).mean()),
                **a.effect(new[ii][take],old[ii][take],yy[take])))
    pd.DataFrame(biases).to_csv(here/'class_score_changes.csv',index=False)
    margin_coverage()
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('control rescues',len(rescued),flush=True)


if __name__=='__main__':main()
