"""Two fixed starts for source sigmoid optimization; no target-based choice."""

import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='2'
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from direction_score_fusion import quality_features, _standardize, _sigmoid_objective, predict_fusion, _loss_gradient

ROOT=Path('E:/Hermes/paper_tracking/sensitivity/directional_fusion_trial_20261009')
SOURCE=ROOT/'nested_geometry_gpu'
OUT=ROOT/'nested_geometry_multistart'

if __name__=='__main__':
    records=[]
    for folder in sorted((SOURCE/'ports').iterdir()):
        output=OUT/'ports'/folder.name; output.mkdir(parents=True,exist_ok=True)
        models=json.loads((folder/'fusion_models.json').read_text())
        with np.load(folder/'source_predictions.npz') as f: d={k:f[k] for k in f.files}
        weights=d['weights']/d['weights'].sum(); quality=quality_features(d['original'],d['c4'])
        for name in ['quality_mix','class_quality_mix','class_geometry_mix']:
            model=models[name]; shared=name=='quality_mix'; groups=1 if shared else 8
            features=np.c_[quality,d['context']] if name=='class_geometry_mix' else quality
            x=_standardize(features,model)
            fit=minimize(_sigmoid_objective,np.zeros(groups*(1+x.shape[1])),
                args=(x,d['c4'],d['original']-d['c4'],d['labels'],weights,model['logit_global'],shared),
                method='L-BFGS-B',jac=True,options=dict(maxiter=150,ftol=1e-10,gtol=1e-7))
            previous=model['training_objective']
            selected=bool(fit.success and fit.fun<previous-1e-9)
            if selected:
                model.update(intercepts=fit.x[:groups].tolist(),slopes=fit.x[groups:].reshape(groups,-1).tolist(),
                    success=bool(fit.success),status=int(fit.status),message=str(fit.message),nit=int(fit.nit),training_objective=float(fit.fun))
                s,_=predict_fusion(model,d['original'],d['c4'],d['context'])
                model['training_logloss']=_loss_gradient(s,d['labels'],weights)[0]
            model['initialization']='neutral' if selected else 'global_anchor'
            records.append(dict(port=folder.name,mode=name,anchor_objective=previous,neutral_objective=float(fit.fun),
                neutral_converged=bool(fit.success),neutral_selected=selected))
        (output/'fusion_models.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
        print(folder.name,[(r['mode'],r['neutral_selected']) for r in records[-3:]],flush=True)
    import pandas as pd
    pd.DataFrame(records).to_csv(OUT/'initialization_choices.csv',index=False)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(ports=23,fixed_starts=['global_anchor','neutral0.5'],
        selection='Lower regularized source CE among converged starts; difference must exceed1e-9',target_labels_used=0),indent=2),encoding='utf-8')
