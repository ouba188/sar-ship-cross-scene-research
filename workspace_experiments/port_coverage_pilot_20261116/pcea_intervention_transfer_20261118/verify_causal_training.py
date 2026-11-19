"""Exercise the actual training/validation/save loop on a bounded source subset."""
from types import SimpleNamespace
from unittest.mock import patch
import json
import torch
import causal_run as run


def main():
    torch.set_num_threads(4)
    inputs,fit,inner=run.base.runtime('unified')
    fit=run.np.setdiff1d(fit,inputs.groups['references'])[:8]
    inner=inner[:4]
    assert not run.np.isin(inner,fit).any()
    args=SimpleNamespace(phase='train',arm='causal',seed=17,steps=2,batch=4,
        keep_ratio=.5,warmup=1,ramp=1,swap_weight=1.,bias_weight=.5,eval_every=2)
    output=run.OUT/'runner_check_local'
    with patch.object(run.base,'runtime',return_value=(inputs,fit,inner)),patch.object(run,'OUT',output):
        run.train(args)
    folder=output/'causal_s17'
    history=json.loads((folder/'training_log.json').read_text())
    assert history[0]['strength']==0 and history[1]['strength']==1
    model=run.make_model(.75)
    model.load_state_dict(torch.load(folder/'last.pt',map_location='cuda',weights_only=True),strict=True)
    assert model.disentangler.keep==128
    scores=run.predict(model,inputs,inner)
    assert run.np.isfinite(scores).all()
    result=dict(passed=True,training_queries=8,source_validation=4,optimizer_steps=2,
        warmup_then_joint=True,new_instance_capacity_restore=True,
        artifacts=['config.json','step_0.json','step_2.json','best.pt','last.pt','receipt.json'],
        not_a_performance_experiment=True,external_evaluation=False)
    assert all((folder/p).exists() for p in result['artifacts'])
    (run.OUT/'training_entry_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
