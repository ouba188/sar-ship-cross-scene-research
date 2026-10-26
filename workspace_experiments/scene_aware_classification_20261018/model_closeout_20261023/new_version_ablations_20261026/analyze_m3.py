"""Matched operator contrasts and actual-reference native-chip evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import analyze_readout as common
run, tr = common.run, common.tr
HERE = Path(__file__).resolve().parent
DEST = HERE/'m3'
ARMS = ('both','learned_fixed','cosine_soft','cosine_fixed')


def main():
    assert (DEST/'evaluation_complete.json').exists()
    out=DEST/'analysis'
    out.mkdir(exist_ok=True)
    intervals, cases, classrows, activity=[],[],[],[]
    for source,target in (('oss1','oss2'),('oss2','oss1')):
        direction=f'{source}-to-{target}'
        root=tr.HERE/direction
        tab=tr.read_table(root/'target_table.csv',labels=True)
        y=tab.class_id.to_numpy(int)
        data={arm:{s:tr.load_npz(DEST/direction/f'seed{s}'/arm/'cross_predictions.npz') for s in run.SEEDS} for arm in ARMS}
        for arm in ARMS:
            for seed in run.SEEDS:
                z, control=data[arm][seed],data['both'][seed]
                activity.append(dict(source=source,seed=seed,arm=arm,
                                     local_max_vs_both=float(abs(z['local']-control['local']).max()),
                                     final_max_vs_both=float(abs(z['frequency']-control['frequency']).max()),
                                     prediction_flips_vs_both=int((z['frequency'].argmax(1)!=control['frequency'].argmax(1)).sum())))
        comparisons=[(arm,'frequency','both','frequency') for arm in ARMS if arm!='both']
        comparisons += [(arm,'B_fixed',arm,'spatial') for arm in ARMS]
        comparisons += [(arm,'spatial',arm,'frequency') for arm in ARMS]
        comparisons += [('both','bare','both','frequency')]
        for ba,be,aa,ae in comparisons:
            effects=[]
            for s in run.SEEDS:
                b,a=data[ba][s][be],data[aa][s][ae]
                d=(a.argmax(1)==y).astype(float)-(b.argmax(1)==y)
                effects.append(d)
                for c in range(4):
                    mask=y==c
                    classrows.append(dict(source=source,before_arm=ba,before=be,after_arm=aa,after=ae,seed=s,
                                          class_name=common.CLASSES[c],n=int(mask.sum()),recall_pp=100*d[mask].mean(),
                                          rescue=int((d[mask]>0).sum()),harm=int((d[mask]<0).sum())))
            intervals.append(dict(source=source,before_arm=ba,before=be,after_arm=aa,after=ae,**common.interval(tab,effects)))
        combined=pd.read_csv(root/'table.csv',dtype={'mmsi':str})
        records=common.input_records(combined,tr.read_json(root/'protocol.json'))
        mapping=tr.load_npz(run.aligned.REPAIR/direction/'seed17/predictions.npz')
        for ba in ('cosine_soft','learned_fixed'):
            old,new=data[ba][17],data['both'][17]
            np.testing.assert_array_equal(old['refs'],new['refs'])
            np.testing.assert_array_equal(old['bare'],new['bare'])
            b,a=old['frequency'],new['frequency']
            bc,ac=b.argmax(1)==y,a.argmax(1)==y
            picks=[]
            for name,mask in dict(rescue=ac&~bc,harm=bc&~ac,CC=ac&bc,WW=~ac&~bc).items():
                ids=np.flatnonzero(mask)
                if len(ids):picks.append((name,int(ids[np.argmin(tab.iloc[ids].obs_id.to_numpy())])))
            fig,ax=plt.subplots(2,len(picks),figsize=(4*len(picks),7),squeeze=False)
            filename=f'{source}_{ba}_to_both.png'
            for col,(status,j) in enumerate(picks):
                true=int(y[j]);rival=int(b[j].argmax() if ac[j] and not bc[j] else a[j].argmax())
                if rival==true:rival=int(np.where(np.arange(4)==true,-np.inf,b[j]).argmax())
                rec=records[int(mapping['query_rows'][j])]
                prov=[]
                for p in range(2):
                    prov.append(common.display(ax[p,col],rec,p))
                    ax[p,col].set_title(f'{status} obs{int(tab.iloc[j].obs_id)} {common.CLASSES[true]} {("VV","VH")[p]}\n{common.CLASSES[b[j].argmax()]} -> {common.CLASSES[a[j].argmax()]}',fontsize=9)
                    ax[p,col].set(xticks=[],yticks=[])
                def contribution(z):
                    return (np.array([z['delta_vv'][j,true]-z['delta_vv'][j,rival],
                                      z['delta_vh'][j,true]-z['delta_vh'][j,rival]])*z['weight_frequency'][j]).tolist()
                cb,ca=contribution(old),contribution(new)
                margin=lambda score:float(score[j,true]-score[j,rival])
                np.testing.assert_allclose(sum(ca)-sum(cb),margin(a)-margin(b),atol=2e-5,rtol=1e-5)
                entry=dict(source=source,seed=17,before=ba,after='both',status=status,obs_id=int(tab.iloc[j].obs_id),
                           true=true,rival=rival,before_pred=int(b[j].argmax()),after_pred=int(a[j].argmax()),
                           before_margin=margin(b),after_margin=margin(a),base_margin=margin(new['bare']),
                           before_polar_contribution=cb,after_polar_contribution=ca,
                           before_weights=old['weight_frequency'][j].tolist(),after_weights=new['weight_frequency'][j].tolist(),
                           before_local=old['local'][j,[true,rival]].tolist(),after_local=new['local'][j,[true,rival]].tolist(),
                           image=filename,provenance=prov)
                if status in ('rescue','harm'):
                    refs=new['refs'][j,[true,rival]].astype(int).ravel()
                    rows=[int(mapping['query_rows'][j]),*refs.tolist()]
                    rf,ra=plt.subplots(2,5,figsize=(15,6.5))
                    rp=[]
                    for c,row in enumerate(rows):
                        for p in range(2):
                            rp.append(common.display(ra[p,c],records[row],p))
                            ra[p,c].set_title(f'{("QUERY","TRUE1","TRUE2","RIVAL1","RIVAL2")[c]} {("VV","VH")[p]}',fontsize=9)
                            ra[p,c].set(xticks=[],yticks=[])
                    rn=f'{source}_{ba}_{status}_{entry["obs_id"]}_refs.png'
                    rf.suptitle(f'{direction} seed17 {ba} -> both, {status}; identical references')
                    rf.subplots_adjust(top=.88,bottom=.04,hspace=.25,wspace=.08)
                    rf.savefig(out/rn,dpi=110);plt.close(rf)
                    entry.update(reference_image=rn,reference_provenance=rp,reference_rows=refs.tolist())
                cases.append(entry)
            fig.suptitle(f'{direction} seed17 {ba} -> both; paired refit, common base')
            fig.text(.02,.02,'Native160 crop/pad, fixed grayscale; minimum obsID per state. No physical causality attribution.',fontsize=8)
            fig.subplots_adjust(top=.88,bottom=.08,hspace=.2,wspace=.1)
            fig.savefig(out/filename,dpi=115);plt.close(fig)
    pd.DataFrame(intervals).to_csv(out/'uncertainty.csv',index=False)
    pd.DataFrame(classrows).to_csv(out/'class_effects.csv',index=False)
    pd.DataFrame(activity).to_csv(out/'operator_activity.csv',index=False)
    tr.write_json(out/'cases.json',cases)
    tr.write_json(out/'receipt.json',dict(native_review_pending=True,case_seed=17,positions=len(cases),actual_references=True))
    print(pd.DataFrame(intervals).to_string(index=False),flush=True)


if __name__=='__main__':main()
