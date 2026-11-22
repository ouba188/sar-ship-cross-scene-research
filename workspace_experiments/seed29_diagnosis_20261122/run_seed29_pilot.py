"""Prioritize the already registered aux=0.1 seed29 run; later full sweep reuses it."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
AUX=LAB/'oss2_disentangle_tuning_20261122'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('aux_scale',AUX/'run_aux_scale_ablation.py')
    aux=importlib.util.module_from_spec(spec); spec.loader.exec_module(aux)
    plan,code=aux.prepare()
    commands=list(aux.commands(AUX,aux.EXPERIMENT,'aux_0p1',29,'cuda'))
    if not args.run:
        print(json.dumps(dict(mode='preview',steps=[key for key,_,_ in commands],test_access=False)))
        return
    aux.original_ready(aux.EXPERIMENT)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        SAR_PORT_PREPARE_CACHE=str(aux.EXPERIMENT/'prepared_cache_local'))
    env['PYTHONPATH']=str(code)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    with aux.exclusive(AUX):
        for key,command,output in commands:
            aux.original_ready(aux.EXPERIMENT,verify_artifacts=False)
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
        result=dict(seed=29,status='pilot_completed',test_access=False,rows=rows,
            delta_h0_ba_vs_original_pp=100*(rows['pilot_aux0p1']['h0']['balanced_accuracy']-rows['original_aux1']['h0']['balanced_accuracy']),
            interpretation='Single-seed diagnostic of the predeclared four-loss candidate; no final model selection. Full three-seed sweep reuses these exact artifacts.',
            source_plan_sha256=aux.digest(AUX/'ablation_plan_local.json'))
        aux.write(ROOT/'aggregate_seed29_aux_pilot.json',result)
        state=aux.read(AUX/'state_local.json')
        state.update(status='pilot_completed',current_step=None,remaining='Other seeds and zero diagnostic remain in the original bounded sweep')
        aux.write(AUX/'state_local.json',state)
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
