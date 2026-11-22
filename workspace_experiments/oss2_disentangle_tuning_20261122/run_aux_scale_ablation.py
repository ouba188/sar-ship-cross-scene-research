"""Bounded validation-only auxiliary-loss ablation; default prepares, never trains.

0.1 retains the four objectives; 0 is a gated/paired-CE diagnostic, not complete
disentanglement. Original scale 1 and baseline results are reused, never edited.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent/'oss2_three_arm_experiment_20261122'
SEEDS = (17, 29, 43)
SCALES = {'aux_0p1': .1, 'aux_0': 0.}
STAGES = ('baseline_stage1', 'baseline_base', 'disentangle_stage1', 'disentangle_base', 'full_h')
LAUNCH = "import numpy, runpy; runpy.run_module('sar_port.two_stage.cli', run_name='__main__')"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.', suffix='.tmp',
                                     encoding='utf8', delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f'Pinned contents differ; refusing overwrite: {path}')
    else:
        write(path, value)


def checked(path, expected):
    if not Path(path).is_file() or digest(path) != expected:
        raise RuntimeError(f'Pinned input/artifact missing or changed: {path}')


def scaled_config(original, scale):
    if scale not in SCALES.values():
        raise ValueError('Only preregistered scales 0.1 and 0 are allowed')
    result = deepcopy(original)
    for name, expected in (('scene', 1.), ('invariance', 1.), ('leakage', .1)):
        if result['loss'][name] != expected:
            raise RuntimeError('Original auxiliary weights differ from the bounded protocol')
        result['loss'][name] = round(expected*scale, 12)
    return result


def verify_inputs(experiment):
    ledger = read(experiment/'ledger_local.json'); inputs = ledger['inputs']
    code = Path(inputs['code_root'])
    checked(experiment/'experiment_manifest_local.json', inputs['manifest']['sha256'])
    checked(experiment/'run_comparison.py', inputs['runner_sha256'])
    actual = {p.relative_to(code).as_posix() for p in (code/'sar_port').rglob('*.py')}
    if actual != set(inputs['code_sha256']):
        raise RuntimeError('Pinned model source file set changed')
    for name, expected in inputs['code_sha256'].items():
        checked(code/name, expected)
    if Path(inputs['python_executable']).resolve() != Path(sys.executable).resolve():
        raise RuntimeError('Use the original experiment Python executable')
    configs = {}; weights_checked = set()
    for seed in SEEDS:
        for arm in ('baseline', 'disentangle'):
            key = f'{arm}_seed{seed}'; entry = ledger['configs'][key]
            path = experiment/'configs'/(key+'.json')
            checked(path, entry['sha256']); config = read(path)
            if config['seed'] != seed or config.get('allow_incomplete_supervision', False):
                raise RuntimeError(f'Wrong seed/supervision: {key}')
            if config.get('encoder', {}).get('gate_enabled', True) != (arm == 'disentangle'):
                raise RuntimeError(f'Unexpected encoder arm: {key}')
            if (config.get('stage1_epochs'), config.get('base_epochs'), config.get('selection_metric')) != (20, 10, 'balanced_accuracy'):
                raise RuntimeError(f'Unexpected budget/selection metric: {key}')
            weight = entry['weights']; identity = (weight['path'], weight['sha256'])
            if identity not in weights_checked:
                checked(weight['path'], weight['sha256']); weights_checked.add(identity)
            configured = Path(config['encoder']['weights_path'])
            if not configured.is_absolute(): configured = code/configured
            if configured.resolve() != Path(weight['path']).resolve():
                raise RuntimeError('Configuration and pinned initialization weights differ')
            configs[key] = config
    return ledger, code, configs


def prepare(root=ROOT, experiment=EXPERIMENT):
    ledger, code, originals = verify_inputs(experiment)
    configs = {}
    for name, scale in SCALES.items():
        for seed in SEEDS:
            path = root/'configs_local'/f'{name}_seed{seed}.json'
            value = scaled_config(originals[f'disentangle_seed{seed}'], scale)
            immutable(path, value)
            configs[f'{name}_seed{seed}'] = {'path': str(path), 'sha256': digest(path)}
    plan = dict(version=1, source_inputs=ledger['inputs'], source_configs=ledger['configs'],
                script_sha256=digest(Path(__file__)), wrapper_sha256=digest(root.parent/'run_on_d.py'),
                experiment=str(experiment.resolve()), configs=configs, scales=SCALES, seeds=list(SEEDS),
                selection='Choose 0.1 only if its three-seed mean h0 validation BA exceeds scale 1; exact ties retain 1; zero is diagnostic only',
                original_results_reused=['baseline', 'disentangle_scale_1'],
                stage1_epochs=20, base_epochs=10, evaluation_split='validation', test_access=False)
    immutable(root/'ablation_plan_local.json', plan)
    return plan, code


def original_ready(experiment, verify_artifacts=True):
    if (experiment/'.run_comparison.lock').exists():
        raise RuntimeError('Original run lock is present; no ablation may run yet')
    state = read(experiment/'state_local.json'); required = []
    for seed in SEEDS:
        for stage in STAGES + ('baseline_base_validation', 'disentangle_base_validation', 'full_h_validation'):
            key = f'seed{seed}/{stage}'; entry = state.get('steps', {}).get(key, {})
            if entry.get('status') != 'succeeded': required.append(key); continue
            if verify_artifacts:
                extension = '.json' if stage.endswith('_validation') else '.pt'
                checked(experiment/'runs'/f'seed{seed}'/(stage+extension), entry['output_sha256'])
    if required:
        raise RuntimeError('All original training/validation must finish first: '+', '.join(required))


@contextmanager
def exclusive(root):
    path = root/'.run_aux_scale.lock'
    with path.open('x', encoding='utf8') as stream:
        json.dump({'pid': os.getpid(), 'started_at': now()}, stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def commands(root, experiment, name, seed, device):
    directory = root/'runs_local'/name/f'seed{seed}'
    config = root/'configs_local'/f'{name}_seed{seed}.json'
    common = [sys.executable, '-u', str(root.parent/'run_on_d.py'), '-u', '-c', LAUNCH]
    stage1, base = directory/'stage1.pt', directory/'base.pt'
    for stage, action, output, source in (
        ('stage1', 'train-stage1', stage1, None),
        ('base', 'train-base', base, stage1),
        ('base_validation', 'evaluate', directory/'base_validation.json', base)):
        command = common+[action, '--manifest', str(experiment/'experiment_manifest_local.json'),
                          '--config', str(config), '--output', str(output), '--device', device]
        if source: command += ['--checkpoint', str(source)]
        if action == 'evaluate': command += ['--split', 'validation']
        yield f'{name}/seed{seed}/{stage}', command, output


def execute(root, key, command, output, env, code):
    if '--split' in command and command[command.index('--split')+1] != 'validation':
        raise RuntimeError('This tool only permits validation evaluation')
    state_path = root/'state_local.json'
    state = read(state_path) if state_path.exists() else {'version': 1, 'steps': {}}
    entry = state['steps'].get(key)
    if entry:
        if entry['status'] != 'succeeded':
            raise RuntimeError(f'{key}: prior {entry["status"]}; preserved, automatic retry disabled')
        for path, expected in entry['artifacts'].items(): checked(path, expected)
        print('SKIP '+key, flush=True); return
    log = output.with_suffix('.log')
    history = output.with_suffix('.history.json') if output.suffix == '.pt' else None
    if any(p.exists() for p in (output, log, history) if p is not None):
        raise RuntimeError(f'Unregistered output/log/history exists for {key}; refusing overwrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(status='running', started_at=now(), command=command, output=str(output), log=str(log))
    state['steps'][key] = entry; state.update(status='running', current_step=key); write(state_path, state)
    try:
        print('START '+key, flush=True)
        with log.open('x', encoding='utf8') as stream:
            result = subprocess.run(command, cwd=code, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        entry['returncode'] = result.returncode
        if result.returncode: raise RuntimeError(f'{key} exited {result.returncode}; see {log}')
        artifacts = [output, log]+([history] if history else [])
        if any(not p.is_file() or p.stat().st_size == 0 for p in artifacts):
            raise RuntimeError(f'Missing/empty artifact for {key}')
        if history:
            expected_epochs = 20 if key.endswith('/stage1') else 10
            if len(read(history)) != expected_epochs: raise RuntimeError('Incomplete training history')
        else:
            metrics = read(output)
            if metrics.get('count') != 227 or not 0 <= metrics.get('balanced_accuracy', -1) <= 1:
                raise RuntimeError('Validation count/metrics differ from the locked protocol')
        entry.update(status='succeeded', finished_at=now(), artifacts={str(p): digest(p) for p in artifacts})
        state.update(current_step=None); write(state_path, state)
    except BaseException as error:
        entry.update(status='failed', finished_at=now(), error=f'{type(error).__name__}: {error}')
        state.update(status='failed'); write(state_path, state); raise


def selected_stage1(history, config, baseline=False):
    # Recover the exact eligible-epoch rule using the logged mean hard fraction:
    # monotonically increasing hard/ramp schedules mean an epoch with any hard
    # step is eligible once its final step is beyond both thresholds.
    last_step = -1; candidates = []
    for row in history:
        if baseline:
            candidates.append(row); continue
        steps = round(row['paired_samples']/row['losses']['paired_samples'])
        last_step += steps
        loss = config['loss']
        if last_step >= max(loss['hard_gate_after'], loss['warmup_steps']+loss['ramp_steps']-1):
            candidates.append(row)
    if not candidates: raise RuntimeError('No eligible Stage1 checkpoint in history')
    best = max(candidates, key=lambda r: r['selection_score'])
    return {'epoch': best['epoch'], **best['validation_metrics']}


def collect(root, experiment):
    rows = []
    for seed in SEEDS:
        for name in ('baseline', 'original_scale_1', *SCALES):
            original = name in ('baseline', 'original_scale_1')
            arm = 'baseline' if name == 'baseline' else 'disentangle'
            directory = experiment/'runs'/f'seed{seed}' if original else root/'runs_local'/name/f'seed{seed}'
            prefix = arm+'_' if original else ''
            config = read(experiment/'configs'/f'{arm}_seed{seed}.json') if original else read(root/'configs_local'/f'{name}_seed{seed}.json')
            h0 = read(directory/(prefix+'base_validation.json'))
            metrics = {k: h0[k] for k in ('accuracy','balanced_accuracy','macro_f1','confusion_matrix','count')}
            cm = metrics['confusion_matrix']
            metrics['recall_per_class'] = [cm[i][i]/sum(cm[i]) for i in range(len(cm))]
            stage1 = selected_stage1(read(directory/(prefix+'stage1.history.json')), config, name == 'baseline')
            base_history = read(directory/(prefix+'base.history.json'))
            best = max(base_history, key=lambda r:r['selection_score'])
            if abs(best['selection_score']-metrics['balanced_accuracy']) > 1e-12:
                raise RuntimeError('Saved h0 validation differs from best history')
            rows.append(dict(seed=seed, arm=name, role='diagnostic_not_complete_disentanglement' if name=='aux_0' else 'comparison',
                             stage1_original_head=stage1, h0_selected_epoch=best['epoch'], h0=metrics))
    lookup={(r['seed'],r['arm']):r for r in rows}
    comparisons={}
    for arm in ('original_scale_1', *SCALES):
        values=[lookup[(s,arm)]['h0']['balanced_accuracy'] for s in SEEDS]
        comparisons[arm]=dict(mean_h0_ba=statistics.mean(values), sample_std_h0_ba=statistics.stdev(values),
            paired_vs_baseline_pp=[100*(lookup[(s,arm)]['h0']['balanced_accuracy']-lookup[(s,'baseline')]['h0']['balanced_accuracy']) for s in SEEDS],
            paired_vs_scale1_pp=[100*(lookup[(s,arm)]['h0']['balanced_accuracy']-lookup[(s,'original_scale_1')]['h0']['balanced_accuracy']) for s in SEEDS])
    selected_scale = select_complete_scale(comparisons)
    result=dict(scope='Validation-only bounded auxiliary-loss study, not an unbiased final performance estimate',
                test_access=False, seeds=list(SEEDS), rows=rows, comparisons=comparisons,
                class_order=['bulk_carrier','general_cargo','product_chemical_tanker','container_ship'],
                four_loss_candidate='aux_0p1', diagnostic_only='aux_0', selected_complete_aux_scale=selected_scale,
                selection_rule='Three-seed mean h0 validation BA; exact tie retains scale 1; scale 0 is never eligible',
                next_h_training_automatic=False)
    write(root/'aggregate_aux_scale_validation.json', result)
    text=['# 辅助权重验证对照结果','','原 scale=1 与基线复用；0.1 是保留四损失的候选，0 仅为门控+成对 CE 诊断。没有读取测试预测，也不以诊断臂冒充完整解耦。','','|种子|分支|Stage1 原头 BA|h0 BA|','|---|---|---:|---:|']
    for r in rows: text.append(f"|{r['seed']}|{r['arm']}|{100*r['stage1_original_head']['balanced_accuracy']:.3f}|{100*r['h0']['balanced_accuracy']:.3f}|")
    delta=statistics.mean(comparisons['aux_0p1']['paired_vs_scale1_pp'])
    text+=['',f'0.1 候选相对原 scale=1 的三种子平均配对 h0 BA 差值为 {delta:+.3f} 个百分点。按预定规则记录完整四损失倍率 {selected_scale:g}；精确并列保留 1，诊断倍率 0 永不参与选择。逐种子变化、逐类召回及标准差见聚合 JSON；负结果保留。新的编码器需要配套重新训练 H，本工具不会自动启动该步骤。']
    (root/'AUX_SCALE_RESULTS_ZH.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    return result


def select_complete_scale(comparisons):
    return .1 if comparisons['aux_0p1']['mean_h0_ba'] > comparisons['original_scale_1']['mean_h0_ba'] else 1.


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--run',action='store_true')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    plan,code=prepare()
    if not args.run:
        print(json.dumps(dict(mode='prepare_only',configs=len(plan['configs']),planned_steps=18,
                              original_lock_present=(EXPERIMENT/'.run_comparison.lock').exists(),test_access=False)))
        return
    original_ready(EXPERIMENT)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',
               SAR_PORT_PREPARE_CACHE=str(EXPERIMENT/'prepared_cache_local'))
    env['PYTHONPATH']=str(code)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    with exclusive(ROOT):
        for name in SCALES:
            for seed in SEEDS:
                for key,command,output in commands(ROOT,EXPERIMENT,name,seed,args.device):
                    original_ready(EXPERIMENT,verify_artifacts=False)
                    execute(ROOT,key,command,output,env,code)
        collect(ROOT,EXPERIMENT)
        state=read(ROOT/'state_local.json'); state.update(status='completed',current_step=None,finished_at=now())
        write(ROOT/'state_local.json',state)


if __name__=='__main__':
    main()
