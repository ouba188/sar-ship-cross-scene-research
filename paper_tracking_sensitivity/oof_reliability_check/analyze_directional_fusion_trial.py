"""Separate fusion utility, calibration, class tradeoffs and routing effects."""

import json
import numpy as np
import pandas as pd

from run_directional_fusion_trial import OUT, SOURCE, MODES, LEARNED, archive


def analyze():
    frame=pd.read_csv(SOURCE/'manifest.csv',usecols=['class_id','port'])
    metrics=pd.read_csv(OUT/'metrics.csv')
    port=pd.read_csv(OUT/'port_means.csv')
    contrast=[]
    for stage in ['base','full']:
        part=port[(port.stage==stage)&(port.subset=='all')]
        ref=part[part['mode']=='c4'].set_index('port')
        for name in MODES:
            candidate=part[part['mode']==name].set_index('port')
            for measure in ['oa','ba']:
                delta=candidate[measure]-ref[measure]
                contrast.append(dict(stage=stage,mode=name,metric=measure,delta_pp=float(delta.mean()*100),
                    positive=int((delta>1e-12).sum()),negative=int((delta< -1e-12).sum()),ties=int((np.abs(delta)<=1e-12).sum())))
    pd.DataFrame(contrast).to_csv(OUT/'contrasts.csv',index=False)
    metrics[(metrics.subset=='all')].groupby(['draw','stage','mode'])[['oa','ba']].mean().to_csv(OUT/'draw_means.csv')
    # Each image is averaged over its own query draws before class aggregation.
    instances=[]; diagnostics=[]; parameters=[]
    for folder in sorted((OUT/'ports').iterdir()):
        p=archive(folder/'policies.npz'); rows=p['query_rows']; truth=frame.loc[rows,'class_id'].to_numpy()
        denominator=p['eval_mask'].sum(0); has_query=denominator>0
        old=p['base_scores'][:,list(MODES).index('original')].argmax(2)
        c4=p['base_scores'][:,list(MODES).index('c4')].argmax(2)
        source=archive(folder/'source_predictions.npz')
        models=json.loads((folder/'fusion_models.json').read_text())
        for name, model in models.items():
            diagnostics.append(dict(port=folder.name,mode=name,success=model['success'],nit=model['nit'],message=model['message'],
                training_logloss=model['training_logloss'],training_objective=model['training_objective']))
        for cl,a in enumerate(models['class_mix']['alpha']):
            parameters.append(dict(port=folder.name,class_id=cl,alpha=a,global_alpha=models['global_mix']['alpha']))
        for stage, predictions in [('base',p['base_scores'].argmax(3)),('full',p['predictions'])]:
            reference=predictions[:,list(MODES).index('c4')]
            for mi,name in enumerate(MODES):
                pred=predictions[:,mi]
                values=dict(correct=(pred==truth),rescue=(reference!=truth)&(pred==truth),harm=(reference==truth)&(pred!=truth),
                    old_rescue_potential=(c4!=truth)&(old==truth),c4_unique=(c4==truth)&(old!=truth),
                    neither_correct=(c4!=truth)&(old!=truth),disagreement=(c4!=old))
                averaged={key:(value*p['eval_mask']).sum(0)[has_query]/denominator[has_query] for key,value in values.items()}
                instances.append(pd.DataFrame(dict(manifest_row=rows[has_query],port=folder.name,stage=stage,mode=name,
                    class_id=truth[has_query],**averaged)))
    inst=pd.concat(instances,ignore_index=True)
    inst.to_csv(OUT/'instance_utility.csv',index=False)
    inst.groupby(['class_id','stage','mode']).agg(n=('manifest_row','size'),recall=('correct','mean'),
        rescue=('rescue','mean'),harm=('harm','mean'),old_rescue_potential=('old_rescue_potential','mean'),
        c4_unique=('c4_unique','mean'),neither_correct=('neither_correct','mean'),disagreement=('disagreement','mean')).to_csv(OUT/'class_utility.csv')
    inst.groupby(['port','class_id','stage','mode']).agg(n=('manifest_row','size'),recall=('correct','mean'),
        rescue=('rescue','mean'),harm=('harm','mean')).to_csv(OUT/'port_class_utility.csv')
    pd.DataFrame(diagnostics).to_csv(OUT/'solver_diagnostics.csv',index=False)
    pd.DataFrame(parameters).to_csv(OUT/'class_static_weights.csv',index=False)
    training=pd.read_csv(OUT/'source_fit_metrics.csv')
    training.groupby(['port','mode'])[['oa','ba']].mean().groupby('mode').mean().to_csv(OUT/'source_fit_means.csv')
    angle=pd.read_csv(OUT/'angle_predictions.csv')
    angle['angle_bin']=pd.cut(angle.relative_angle,[0,15,30,45,60,75,90.0001],right=False)
    angle=angle.groupby(['manifest_row','port','class_id','mode','angle_bin'],observed=True).correct.mean().reset_index()
    angle.groupby(['port','class_id','mode','angle_bin'],observed=True).agg(n=('manifest_row','size'),recall=('correct','mean')).to_csv(OUT/'port_class_angle.csv')
    angle.groupby(['class_id','mode','angle_bin'],observed=True).agg(n=('manifest_row','size'),recall=('correct','mean')).to_csv(OUT/'class_angle.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['half_mix',*LEARNED,'c4_gate_refit','c4_direction_gate']
    full=port[(port.stage=='full')&(port.subset=='all')].pivot(index='port',columns='mode',values='ba')
    delta=(full[names].subtract(full['c4'],axis=0))*100
    fig,ax=plt.subplots(figsize=(12,8),layout='constrained')
    limit=max(1.,float(np.max(np.abs(delta.to_numpy()))))
    im=ax.imshow(delta.to_numpy(),cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(np.arange(len(names)),names,rotation=35,ha='right',fontsize=8)
    ax.set_yticks(np.arange(len(delta)),delta.index,fontsize=8)
    ax.set_title('Full-model BA change versus C4 base + original expert (pp)')
    for i in range(len(delta)):
        for j in range(len(names)):
            ax.text(j,i,f'{delta.iloc[i,j]:+.1f}',ha='center',va='center',fontsize=7,
                color='white' if abs(delta.iloc[i,j])>.55*limit else 'black')
    fig.colorbar(im,ax=ax,label='BA change (percentage points)')
    fig.savefig(OUT/'port_ba_changes.png',dpi=150); plt.close(fig)
    print(pd.DataFrame(contrast).query("stage == 'full'").to_string(index=False),flush=True)
    print('SOLVERS',pd.DataFrame(diagnostics).groupby('mode').success.agg(['sum','size']).to_string(),flush=True)


if __name__=='__main__': analyze()
