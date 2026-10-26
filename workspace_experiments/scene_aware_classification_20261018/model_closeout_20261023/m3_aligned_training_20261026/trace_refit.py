"""Explain the seed43 OA/BA tradeoff; posthoc cases, never model selection."""
import analyze as a
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
run,tr,HERE=a.run,a.tr,a.HERE


def main():
    out=HERE/'analysis'
    folder=HERE/'oss2-to-oss1/seed43'
    table=tr.read_table(tr.HERE/'oss2-to-oss1/target_table.csv',labels=True)
    y=table.class_id.to_numpy(int)
    z=tr.load_npz(folder/'cross_predictions.npz')
    orig=tr.load_npz(run.REPAIR/'oss2-to-oss1/seed43/predictions.npz')
    decision=run.FrozenDecision(folder,z,z['bare'],'cpu')
    with torch.no_grad():
        counter=np.concatenate([decision(torch.tensor(orig['local'][k:k+48]),np.arange(k,min(k+48,len(y)))).numpy() for k in range(0,len(y),48)])
    metrics=[]
    for name,score in (('old',z['old']),('metric_only',z['metric_only']),('old_local_new_readout',counter),('frequency',z['frequency'])):
        metrics.append(dict(arm=name,**run.m3.metrics(y,score)))
    pd.DataFrame(metrics).to_csv(out/'seed43_readout_cross.csv',index=False)
    before,after=z['old'],z['frequency']
    ac,bc=after.argmax(1)==y,before.argmax(1)==y
    combined=pd.read_csv(tr.HERE/'oss2-to-oss1/table.csv',dtype={'mmsi':str})
    records=a.input_records(combined,tr.read_json(tr.HERE/'oss2-to-oss1/protocol.json'))
    cases=[]
    for status,mask in (('bulk_rescue',(y==0)&ac&~bc),('general_harm',(y==1)&bc&~ac)):
        ix=np.flatnonzero(mask)
        j=int(ix[np.argmin(table.iloc[ix].obs_id.to_numpy())])
        true=int(y[j]); rival=int(before[j].argmax() if ac[j] else after[j].argmax())
        margin=lambda s:float(s[true]-s[rival])
        refs=z['refs'][j,[true,rival]].astype(int).ravel()
        rows=[int(orig['query_rows'][j]),*refs.tolist()]
        fig,axes=plt.subplots(2,5,figsize=(15,6.7))
        provenance=[]
        for col,row in enumerate(rows):
            record=records[row]
            for p,v in enumerate(tr.extract.prepare(record)):
                ax=axes[p,col]
                if v is not None:
                    ax.imshow(v['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    if v['masks'][0].any():ax.contour(v['masks'][0],levels=[.5],colors=['#20a780'],linewidths=.6)
                    provenance.append(dict(pol=('VV','VH')[p],path=record[('vv','vh')[p]]['image'],placement=v['placement']))
                ax.set_title(f'{("VV","VH")[p]} {("QUERY","TRUE REF1","TRUE REF2","RIVAL REF1","RIVAL REF2")[col]}\nobs {combined.iloc[row].original_obs_id}',fontsize=9)
                ax.set(xticks=[],yticks=[])
        mb,ma,mc=margin(before[j]),margin(after[j]),margin(counter[j])
        filename=f'seed43_{status}_{int(table.iloc[j].obs_id)}.png'
        fig.suptitle(f'OSS2->OSS1 seed43 {status}, old -> refit\n{a.CLASSES[before[j].argmax()]} -> {a.CLASSES[after[j].argmax()]}; margin {mb:+.4f} -> {ma:+.4f}')
        fig.text(.02,.02,f'Native160, unchanged actual references. Local effect {ma-mc:+.4f}; readout effect {mc-mb:+.4f}. Posthoc representative case.',fontsize=8)
        fig.subplots_adjust(top=.80,bottom=.09,hspace=.2,wspace=.1)
        fig.savefig(out/filename,dpi=115);plt.close(fig)
        cases.append(dict(seed=43,status=status,obs_id=int(table.iloc[j].obs_id),true=true,rival=rival,
                          before_margin=mb,after_margin=ma,metric_only_margin=margin(z['metric_only'][j]),
                          local_effect=ma-mc,readout_effect=mc-mb,new_weights=z['weight_frequency'][j].tolist(),
                          image=filename,provenance=provenance))
    tr.write_json(out/'seed43_cases.json',cases)
    print(pd.DataFrame(metrics).to_string(index=False))
    print([{k:v for k,v in c.items() if k!='provenance'} for c in cases])


if __name__=='__main__':main()
