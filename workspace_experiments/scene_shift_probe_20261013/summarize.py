"""Read saved predictions independently; summarize matched errors and scope."""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Query scenes need not contain all eight output classes; retain one notice.
warnings.filterwarnings('once', message='y_pred contains classes not in y_true', category=UserWarning)

OUT=Path(__file__).resolve().parent
episodes=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
metrics=pd.read_csv(OUT/'paired_metrics.csv').set_index(['port','acquisition','budget','draw','arm','mode'])
changes=[]; counts=[]; checked=0
query_rows={16:set(),64:set()}; query_ships={16:set(),64:set()}
for ep in episodes:
    with np.load(OUT/'predictions'/ep['port']/f"{ep['acquisition']}_B{ep['budget']}.npz") as z:
        probs=z['probabilities']; y=z['labels']; supports=z['supports']; q=z['query_local']
        np.testing.assert_allclose(probs.sum(-1),1,atol=1e-12)
        assert not set(q)&set(supports.ravel())
        query_rows[ep['budget']].update(z['query_rows'].tolist())
        query_ships[ep['budget']].update(z['mmsi'].tolist())
        pred=probs.argmax(-1)
        counts.append(dict(port=ep['port'],acquisition=ep['acquisition'],budget=ep['budget'],
            query_n=len(y),query_classes=len(np.unique(y)),supported_classes=int((np.array(ep['quota'])>0).sum()),
            unsupported_query_n=int(sum(ep['quota'][int(c)]==0 for c in y)),
            same_pool=ep['same_pool'],other_pool=ep['other_pool']))
        for draw in range(5):
            for ai,arm in enumerate(['same','other']):
                for mi,mode in enumerate(['base','full']):
                    got=[accuracy_score(y,pred[draw,ai,mi])*100,balanced_accuracy_score(y,pred[draw,ai,mi])*100]
                    old=metrics.loc[(ep['port'],ep['acquisition'],ep['budget'],draw,arm,mode),['oa','ba']].to_numpy(float)
                    np.testing.assert_allclose(got,old,atol=1e-10,rtol=0)
                    checked+=1
            for mi,mode in enumerate(['base','full']):
                same=pred[draw,0,mi]==y; other=pred[draw,1,mi]==y
                for c in np.unique(y):
                    ix=y==c
                    changes.append(dict(port=ep['port'],acquisition=ep['acquisition'],budget=ep['budget'],draw=draw,
                        mode=mode,class_id=int(c),n=int(ix.sum()),support_n=ep['quota'][int(c)],
                        lost=int((same&~other&ix).sum()),rescued=int((~same&other&ix).sum()),
                        lost_rate=100*float(np.mean((same&~other)[ix])),
                        rescued_rate=100*float(np.mean((~same&other)[ix]))))
pd.DataFrame(counts).to_csv(OUT/'query_coverage.csv',index=False)
ch=pd.DataFrame(changes)
ch.to_csv(OUT/'error_transitions.csv',index=False)
ch['delta_recall']=ch.rescued_rate-ch.lost_rate
cs=ch.groupby(['budget','mode','port','acquisition','class_id'])[['lost_rate','rescued_rate','delta_recall']].mean().reset_index()
cs=cs.groupby(['budget','mode','port','class_id'])[['lost_rate','rescued_rate','delta_recall']].mean().reset_index()
summary=cs.groupby(['budget','mode','class_id']).agg(ports=('port','nunique'),lost_rate=('lost_rate','mean'),
    rescued_rate=('rescued_rate','mean'),delta_recall=('delta_recall','mean')).reset_index()
summary.to_csv(OUT/'class_effects.csv',index=False)

coverage=pd.read_csv(OUT/'coverage.csv')
effects=pd.read_csv(OUT/'port_effects.csv')
effects=effects.merge(coverage[['port','acquisitions_shared_other_ports']],on='port',validate='many_to_one')
sensitivity=effects[effects.acquisitions_shared_other_ports==0].groupby(['budget','mode']).agg(
    ports=('port','nunique'),delta_oa=('delta_oa','mean'),delta_ba=('delta_ba','mean')).reset_index()
sensitivity.to_csv(OUT/'unshared_acquisition_ports.csv',index=False)

# External cache is descriptive only: OSS2 does not supply actual port IDs here.
external=Path('E:/Hermes/paper_tracking/sensitivity/backbone_ablation_20261010/targets/oss2/oss2/predictions.npz')
md=pd.read_csv(OUT.parent/'ports_root_cause_20261013/dataset_audit/public_metadata.csv',dtype={'identity':str})
md=md[md.dataset=='oss2'].set_index('sample_id')
ext=[]
with np.load(external) as z:
    md=md.loc[z['ids']]
    mmsi=md.identity.fillna('').to_numpy(); acq=md.datatake_key.to_numpy(); y=md.class_id.to_numpy()
    known=np.array([bool(s.strip()) and s.strip().lower() not in ['nan','none','unknown'] for s in mmsi])
    same=np.array([np.isin(acq,acq[s]) for s in z['support']])
    clean=np.array([m&known&~np.isin(mmsi,mmsi[s]) for m,s in zip(z['eval_masks'],z['support'])])
    switches=(clean&same).any(0)&(clean&~same).any(0)
    for mode in ['base','r50_late','dual_refit_late']:
        ok=z['predictions'][:,list(z['modes']).index(mode)]==y[None,:]
        for i in np.flatnonzero(switches):
            delta=100*(ok[clean[:,i]&~same[:,i],i].mean()-ok[clean[:,i]&same[:,i],i].mean())
            ext.append(dict(sample_id=z['ids'][i],mode=mode,class_id=int(y[i]),acquisition=acq[i],delta=delta))
pd.DataFrame(ext).to_csv(OUT/'oss2_archival_switchers.csv',index=False)
extdf=pd.DataFrame(ext)
exsum=extdf.groupby('mode').agg(n=('sample_id','size'),delta_oa=('delta','mean'))
exsum['delta_ba']=extdf.groupby(['mode','class_id']).delta.mean().groupby('mode').mean()
exsum.to_csv(OUT/'oss2_archival_summary.csv')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained')
for ax,budget in zip(axes,[16,64]):
    p=effects[(effects.budget==budget)&(effects['mode']=='full')].sort_values('delta_ba')
    ax.barh(p.port,p.delta_ba,color=['#bd4754' if v<0 else '#258f88' for v in p.delta_ba])
    ax.axvline(0,color='#454545',lw=.8)
    ax.set_title(f'B = {budget}: matched query vessels and class counts')
    ax.set_xlabel('Other-acquisition minus same-acquisition BA (pp)')
    ax.grid(axis='x',alpha=.15)
    ax.set_axisbelow(True)
fig.savefig(OUT/'scene_transfer_by_port.png',dpi=170)
fig.savefig(OUT/'scene_transfer_by_port.svg')
plt.close(fig)

receipt=dict(metric_rows_independently_recomputed=checked,query_support_index_overlap=0,
    query_unique_rows={str(k):len(v) for k,v in query_rows.items()},
    query_unique_vessels={str(k):len(v) for k,v in query_ships.items()},
    scene_budget_contexts=len(episodes),readout='frozen base + dual_refit expert, equal probability fusion',
    evaluation='BA over query-present classes; paired identical queries; not legacy eight-class overall BA',
    no_shared_acquisition_port_sensitivity=sensitivity.to_dict('records'),
    oss2_descriptive=exsum.reset_index().to_dict('records'))
(OUT/'VERIFIED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
print('CLASS EFFECTS\n',summary[summary['mode']=='full'].to_string(index=False))
