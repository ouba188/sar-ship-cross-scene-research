"""Prioritize the already registered aux=0.1 seed29 run; later full sweep reuses it."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
AUX=LAB/'oss2_disentangle_tuning_20261122'


def seed29_first_ready(aux, experiment):
    """Require completed 17/29 and an explicitly idle original experiment.

    An interrupted seed43 attempt must be archived and removed from the active
    state by the original scheduler owner. This entry point never clears it.
    """
    if (experiment/'.run_comparison.lock').exists():
        raise RuntimeError('Original run lock is present; seed29-first cannot run')
    state=aux.read(experiment/'state_local.json')
    def reject_active(value):
        if isinstance(value,dict):
            if value.get('status') in ('running','failed'):
                raise RuntimeError('Original state still contains running/failed status; inspect/archive the attempt first')
            for item in value.values(): reject_active(item)
        elif isinstance(value,list):
            for item in value: reject_active(item)
    reject_active(state)
    if state.get('current_step') is not None:
        raise RuntimeError('Original state still declares a current step')
    steps=state.get('steps',{})
    if any(entry.get('status') != 'succeeded' for entry in steps.values()):
        raise RuntimeError('Unfinished original step remains; archive/remove interrupted attempt before seed29-first')
    required=aux.STAGES+('baseline_base_validation','disentangle_base_validation','full_h_validation')
    for seed in (17,29):
        for stage in required:
            key=f'seed{seed}/{stage}'; entry=steps.get(key,{})
            if entry.get('status') != 'succeeded':
                raise RuntimeError(f'Seed29-first requires completed original step: {key}')
            suffix='.json' if stage.endswith('_validation') else '.pt'
            aux.checked(experiment/'runs'/f'seed{seed}'/(stage+suffix),entry['output_sha256'])


def check_ready(aux, first, *, per_step=False):
    if first:
        seed29_first_ready(aux,aux.EXPERIMENT)
    else:
        aux.original_ready(aux.EXPERIMENT,verify_artifacts=not per_step)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--seed29-first',action='store_true',
        help='Require finished original seeds 17/29 and an idle/clean original state; seed43 may be absent')
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('aux_scale',AUX/'run_aux_scale_ablation.py')
    aux=importlib.util.module_from_spec(spec); spec.loader.exec_module(aux)
    plan,code=aux.prepare()
    commands=list(aux.commands(AUX,aux.EXPERIMENT,'aux_0p1',29,'cuda'))
    if not args.run:
        print(json.dumps(dict(mode='preview',seed29_first=args.seed29_first,
                              steps=[key for key,_,_ in commands],test_access=False)))
        return
    check_ready(aux,args.seed29_first)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        SAR_PORT_PREPARE_CACHE=str(aux.EXPERIMENT/'prepared_cache_local'))
    env['PYTHONPATH']=str(code)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    with aux.exclusive(AUX):
        for key,command,output in commands:
            check_ready(aux,args.seed29_first,per_step=True)
            aux.execute(AUX,key,command,output,env,code)
        metrics=('accuracy','balanced_accuracy','macro_f1','confusion_matrix','count')
        rows={}
        for arm,prefix,directory,config in (
            ('baseline','baseline_',aux.EXPERIMENT/'runs/seed29',aux.EXPERIMENT/'configs/baseline_seed29.json'),
            ('original_aux1','disentangle_',aux.EXPERIMENT/'runs/seed29',aux.EXPERIMENT/'configs/disentangle_seed29.json'),
            ('pilot_aux0p1','',AUX/'runs_local/aux_0p1/seed29',AUX/'configs_local/aux_0p1_seed29.json')):
            measured=aux.read(directory/(prefix+'base_validation.json'))
            rows[arm]=dict(h0={k:measured[k] for k in metrics},
                stage1_original_head=aux.selected_stage1(aux.read(directory/(prefix+'stage1.history.json')),
                    aux.read(config),baseline=arm=='baseline'))
        result=dict(seed=29,status='pilot_completed',seed29_first=args.seed29_first,test_access=False,rows=rows,
            delta_h0_ba_vs_original_pp=100*(rows['pilot_aux0p1']['h0']['balanced_accuracy']-rows['original_aux1']['h0']['balanced_accuracy']),
            interpretation='Single-seed diagnostic of the predeclared four-loss candidate; no final model selection. Full three-seed sweep reuses these exact artifacts.',
            source_plan_sha256=aux.digest(AUX/'ablation_plan_local.json'))
        aux.write(ROOT/'aggregate_seed29_aux_pilot.json',result)
        state=aux.read(AUX/'state_local.json')
        state.update(status='pilot_completed',current_step=None,remaining='Other seeds and zero diagnostic remain in the original bounded sweep')
        aux.write(AUX/'state_local.json',state)
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
