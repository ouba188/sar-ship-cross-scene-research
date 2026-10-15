"""Source-only check: did near-zero gates arise just from initialization?"""

import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='2'
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from direction_score_fusion import quality_features, _standardize, _sigmoid_objective

ROOT=Path('E:/Hermes/paper_tracking/sensitivity/directional_fusion_trial_20261009/nested_geometry_gpu')

if __name__=='__main__':
    folder=ROOT/'ports/Callao'
    with np.load(folder/'source_predictions.npz') as f: d={k:f[k] for k in f.files}
    models=json.loads((folder/'fusion_models.json').read_text())
    weights=d['weights']/d['weights'].sum(); result=[]
    quality=quality_features(d['original'],d['c4'])
    for name in ['quality_mix','class_quality_mix','class_geometry_mix']:
        model=models[name]; shared=name=='quality_mix'; groups=1 if shared else 8
        features=np.c_[quality,d['context']] if name=='class_geometry_mix' else quality
        x=_standardize(features,model)
        fitted=minimize(_sigmoid_objective,np.zeros(groups*(1+x.shape[1])),
            args=(x,d['c4'],d['original']-d['c4'],d['labels'],weights,model['logit_global'],shared),
            method='L-BFGS-B',jac=True,options=dict(maxiter=150,ftol=1e-10,gtol=1e-7))
        alpha=expit(x@fitted.x[groups:].reshape(groups,-1).T+fitted.x[:groups])
        result.append(dict(mode=name,original_objective=model['training_objective'],neutral_start_objective=float(fitted.fun),
            success=bool(fitted.success),iterations=int(fitted.nit),neutral_start_mean_alpha=float(alpha.mean())))
    (ROOT/'INITIALIZATION_PROBE.json').write_text(json.dumps(dict(outer_port='Callao',target_labels_used=0,
        change='Start sigmoid gates at0.5; objective and anchor penalty unchanged',results=result),indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
