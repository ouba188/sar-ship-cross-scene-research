"""Evaluate one validation-selected gamma after all three final scans and training finish.

--dry-run performs standard-library validation and previews commands only.
The original training/checkpoint/config/ledger files are never written.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
EXPERIMENT = LAB/'oss2_three_arm_experiment_20261122'
OUTPUT = ROOT/'calibrated_local'
SEEDS = (17, 29, 43)
STAGES = ('baseline_stage1', 'baseline_base', 'disentangle_stage1', 'disentangle_base', 'full_h')
LAUNCH = "import numpy, runpy; runpy.run_module('sar_port.two_stage.cli', run_name='__main__')"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def unchanged_json(path, payload):
    if path.exists():
        if read_json(path) != payload:
            raise RuntimeError(f'Immutable provenance differs: {path}')
    else:
        atomic_json(path, payload)


def checked_hash(path, expected):
    if not Path(path).is_file() or digest(path) != expected:
        raise RuntimeError(f'Pinned file missing or changed: {path}')


def collect_again():
    path = ROOT/'collect_fusion_scan.py'
    spec = importlib.util.spec_from_file_location('selected_gamma_collector', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect(ROOT)[1]


def readiness(device):
    """No NumPy/Torch imports, test predictions, or test images in this function."""
    selection_path = ROOT/'selected_gamma.json'
    if not selection_path.is_file():
        raise RuntimeError('selected_gamma.json is missing: first finish all three final validation scans and collect them')
    selection = read_json(selection_path)
    if selection != collect_again():
        raise RuntimeError('Saved gamma selection differs from an exact collector recomputation')
    if (EXPERIMENT/'.run_comparison.lock').exists():
        raise RuntimeError('Original comparison still holds its run lock; wait until it finishes')
    ledger = read_json(EXPERIMENT/'ledger_local.json')
    state = read_json(EXPERIMENT/'state_local.json')
    inputs = ledger['inputs']; manifest = EXPERIMENT/'experiment_manifest_local.json'
    checked_hash(manifest, inputs['manifest']['sha256'])
    if selection['manifest_sha256'] != inputs['manifest']['sha256']:
        raise RuntimeError('Selection and original experiment manifest disagree')
    code_root = Path(inputs['code_root'])
    for relative, expected in inputs['code_sha256'].items():
        checked_hash(code_root/relative, expected)
    training, scans, configs, selected_epochs = {}, {}, {}, {}
    for seed in SEEDS:
        for stage in STAGES:
            key = f'seed{seed}/{stage}'; entry = state.get('steps', {}).get(key, {})
            if entry.get('status') != 'succeeded':
                raise RuntimeError(f'All original training must succeed before test access: {key}')
            checkpoint = EXPERIMENT/'runs'/f'seed{seed}'/(stage+'.pt')
            checked_hash(checkpoint, entry['output_sha256'])
            training[key] = entry['output_sha256']
        config = EXPERIMENT/'configs'/f'disentangle_seed{seed}.json'
        pinned_config = ledger['configs'][f'disentangle_seed{seed}']
        checked_hash(config, pinned_config['sha256']); configs[str(seed)] = pinned_config['sha256']
        if read_json(config).get('h_epochs') != 20:
            raise RuntimeError(f'seed{seed}: expected the completed original 20-epoch H budget')
        final_dir = ROOT/f'final_seed{seed}'
        scan_path = final_dir/'aggregate_validation_fusion_scan.json'
        scan = read_json(scan_path); plan = read_json(final_dir/'scan_plan.json')
        snapshot = read_json(final_dir/'snapshot_local.json')
        expected = training[f'seed{seed}/full_h']
        if scan['checkpoint_sha256'] != expected or snapshot['sha256'] != expected:
            raise RuntimeError(f'seed{seed}: final scan is not the completed training checkpoint')
        if plan['seed'] != seed or plan['manifest_sha256'] != inputs['manifest']['sha256'] or plan['code_sha256'] != inputs['code_sha256']:
            raise RuntimeError(f'seed{seed}: scan manifest/code/seed pin differs from training')
        if snapshot['selected_epoch'] != scan['checkpoint_selected_epoch']:
            raise RuntimeError(f'seed{seed}: selected epoch provenance differs')
        checked_hash(final_dir/'full_h_snapshot_local.pt', expected)
        checked_hash(final_dir/'validation_logits_evidence_local.npz', scan['cache_sha256'])
        scans[str(seed)] = digest(scan_path); selected_epochs[str(seed)] = scan['checkpoint_selected_epoch']
    return dict(version=1, selected_gamma=selection['selected_gamma'], evidence_temperature=.2,
        seeds=list(SEEDS), selected_gamma_sha256=digest(selection_path), final_scan_sha256=scans,
        collector_sha256=digest(ROOT/'collect_fusion_scan.py'), script_sha256=digest(Path(__file__)),
        manifest_sha256=inputs['manifest']['sha256'], code_root=str(code_root),
        code_sha256=inputs['code_sha256'], config_sha256=configs,
        original_training_output_sha256=training, selected_epochs=selected_epochs,
        python_executable=sys.executable, device=device,
        variant='h0_fallback' if selection['selected_gamma']==0 else 'validation_selected_H_fusion',
        calibration_change='Only h_config.gamma changes; selection provenance is stored in a sidecar.',
        selection_scope='Common gamma chosen on three seeds of the shared validation split; not independent confirmation.')


def checkpoint_path(seed):
    return OUTPUT/f'seed{seed}'/'full_h_selected_gamma.pt'


def commands(seed, device):
    wrapper = [sys.executable, str(LAB/'run_on_d.py'), '-u']
    materialize = wrapper+[str(Path(__file__).resolve()), '--materialize', str(seed), '--device', device]
    evaluate = wrapper+['-c', LAUNCH, 'evaluate', '--manifest', str(EXPERIMENT/'experiment_manifest_local.json'),
        '--config', str(EXPERIMENT/'configs'/f'disentangle_seed{seed}.json'),
        '--checkpoint', str(checkpoint_path(seed)), '--split', 'test', '--device', device,
        '--output', str(OUTPUT/f'seed{seed}'/'selected_gamma_test.json')]
    return materialize, evaluate


def equal_except_gamma(a, b, torch, path=()):
    if path == ('h_config', 'gamma'):
        return 0
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):
            raise RuntimeError(f'Calibration changed a tensor: {path}')
        return 1
    if type(a) is not type(b):
        raise RuntimeError(f'Calibration changed metadata type: {path}')
    if isinstance(a, dict):
        if a.keys()!=b.keys(): raise RuntimeError(f'Calibration changed metadata keys: {path}')
        return sum(equal_except_gamma(a[k],b[k],torch,path+(k,)) for k in a)
    if isinstance(a, (tuple,list)):
        if len(a)!=len(b): raise RuntimeError(f'Calibration changed metadata length: {path}')
        return sum(equal_except_gamma(x,y,torch,path+(i,)) for i,(x,y) in enumerate(zip(a,b)))
    if a!=b: raise RuntimeError(f'Calibration changed metadata: {path}')
    return 0


def materialize(seed, provenance):
    target = checkpoint_path(seed)
    receipt = target.parent/'checkpoint_provenance_local.json'
    temporary = target.with_suffix('.pt.materializing')
    if any(p.exists() for p in (target, receipt, temporary)):
        raise RuntimeError('Existing calibration artifact found; refusing overwrite or automatic retry')
    target.parent.mkdir(parents=True, exist_ok=True)
    source = EXPERIMENT/'runs'/f'seed{seed}'/'full_h.pt'
    expected = provenance['original_training_output_sha256'][f'seed{seed}/full_h']
    # Worker is reached only after readiness; parent process stays free of Torch.
    import numpy  # Must precede Torch on this Windows environment.
    import torch
    torch.set_num_threads(1)
    original = torch.load(source, map_location='cpu', weights_only=True)
    if original['stage']!='h' or original['h_training_config']['seed']!=seed:
        raise RuntimeError('Expected full H checkpoint for the selected seed')
    if original['h_config']['evidence_temperature']!=.2:
        raise RuntimeError('Only gamma may change; expected Te=.2')
    selected_epoch = [r for r in original['h_history'] if r['stage']=='match'][-1]['epoch']
    if selected_epoch!=provenance['selected_epochs'][str(seed)]:
        raise RuntimeError('Loaded checkpoint epoch differs from final scan')
    calibrated = dict(original, h_config=dict(original['h_config'], gamma=provenance['selected_gamma']))
    torch.save(calibrated, temporary)
    reread = torch.load(temporary, map_location='cpu', weights_only=True)
    tensors = equal_except_gamma(original, reread, torch)
    if reread['h_config']['gamma']!=provenance['selected_gamma']:
        raise RuntimeError('Serialized gamma differs from the locked selection')
    checked_hash(source, expected)
    os.replace(temporary, target)
    atomic_json(receipt, dict(seed=seed, created_at=now(), source_checkpoint_sha256=expected,
        calibrated_checkpoint_sha256=digest(target), selected_gamma=provenance['selected_gamma'],
        original_gamma=original['h_config']['gamma'], selected_epoch=selected_epoch,
        unchanged_tensor_count=tensors, all_other_metadata_exactly_equal=True,
        selection_provenance=provenance))


@contextmanager
def exclusive_run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = OUTPUT/'.evaluate_selected_gamma.lock'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), started_at=now()),stream)
    try:
        yield
    finally:
        lock.unlink()


def execute(state, key, command, output, log, env, provenance):
    entry = state['steps'].get(key)
    if entry is not None:
        if entry['status']!='succeeded':
            raise RuntimeError(f'{key}: {entry["status"]}; automatic retry/overwrite is disabled')
        checked_hash(output,entry['output_sha256'])
        if entry.get('sidecar'):
            checked_hash(entry['sidecar'],entry['sidecar_sha256'])
        print('SKIP verified '+key,flush=True)
        return
    if output.exists() or log.exists():
        raise RuntimeError(f'{key}: unregistered artifact/log; refusing overwrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(status='running', started_at=now(), command=command, output=str(output), log=str(log))
    state['steps'][key]=entry; atomic_json(OUTPUT/'state_local.json',state)
    try:
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=LAB,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'{key} exited {result.returncode}; inspect {log}')
        if output.suffix=='.json':
            payload=read_json(output)
            if payload.get('count',0)<1 or 'balanced_accuracy' not in payload:
                raise RuntimeError('Evaluation produced no valid metrics')
            payload.update(selected_gamma=provenance['selected_gamma'], variant=provenance['variant'],
                           selection_provenance_sha256=digest(OUTPUT/'provenance_local.json'))
            atomic_json(output,payload)
        else:
            sidecar=output.parent/'checkpoint_provenance_local.json'
            receipt=read_json(sidecar)
            if receipt['selection_provenance']!=provenance or receipt['calibrated_checkpoint_sha256']!=digest(output):
                raise RuntimeError('Calibration sidecar disagrees with the pinned provenance')
            entry.update(sidecar=str(sidecar),sidecar_sha256=digest(sidecar))
        entry.update(status='succeeded',finished_at=now(),output_sha256=digest(output))
        atomic_json(OUTPUT/'state_local.json',state)
    except BaseException as exc:
        entry.update(status='failed',finished_at=now(),error=f'{type(exc).__name__}: {exc}')
        atomic_json(OUTPUT/'state_local.json',state)
        raise


def verify_local_history(provenance):
    pinned=OUTPUT/'provenance_local.json'
    if pinned.exists() and read_json(pinned)!=provenance:
        raise RuntimeError('Existing evaluation provenance differs; preserve the previous run')
    state_path=OUTPUT/'state_local.json'
    if state_path.exists():
        if not pinned.exists(): raise RuntimeError('Local state has no pinned provenance')
        for key,entry in read_json(state_path).get('steps',{}).items():
            if entry['status']!='succeeded':
                raise RuntimeError(f'{key}: {entry["status"]}; automatic retry/overwrite is disabled')
            checked_hash(entry['output'],entry['output_sha256'])
            if entry.get('sidecar'):
                checked_hash(entry['sidecar'],entry['sidecar_sha256'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--materialize',type=int,choices=SEEDS,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    provenance=readiness(args.device)
    if args.materialize is None:
        verify_local_history(provenance)
    preview=[dict(seed=seed,materialize=commands(seed,args.device)[0],evaluate=commands(seed,args.device)[1]) for seed in SEEDS]
    if args.dry_run:
        print(json.dumps(dict(status='dry_run_ready',provenance=provenance,commands=preview),ensure_ascii=False,indent=2))
        return
    if args.materialize is not None:
        if read_json(OUTPUT/'provenance_local.json')!=provenance:
            raise RuntimeError('Materialization requires the parent-pinned provenance')
        materialize(args.materialize,provenance)
        return
    with exclusive_run():
        unchanged_json(OUTPUT/'provenance_local.json',provenance)
        state_path=OUTPUT/'state_local.json'
        state=read_json(state_path) if state_path.exists() else dict(version=1,created_at=now(),steps={})
        env=os.environ.copy()
        env.update(PYTHONPATH=provenance['code_root'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                   SAR_PORT_PREPARE_CACHE=str(EXPERIMENT/'prepared_cache_local'),PYTHONDONTWRITEBYTECODE='1')
        for seed in SEEDS:
            build,evaluate=commands(seed,args.device)
            directory=OUTPUT/f'seed{seed}'
            execute(state,f'seed{seed}/materialize',build,checkpoint_path(seed),directory/'materialize.log',env,provenance)
            execute(state,f'seed{seed}/test',evaluate,directory/'selected_gamma_test.json',directory/'selected_gamma_test.log',env,provenance)
        per_seed=[]
        for seed in SEEDS:
            path=OUTPUT/f'seed{seed}'/'selected_gamma_test.json'; result=read_json(path)
            per_seed.append(dict(seed=seed,output_sha256=digest(path),
                metrics={k:v for k,v in result.items() if k!='predictions'}))
        unchanged_json(OUTPUT/'selected_gamma_test.json',dict(status='completed',split='test',
            selected_gamma=provenance['selected_gamma'],variant=provenance['variant'],
            selection_provenance_sha256=digest(OUTPUT/'provenance_local.json'),per_seed=per_seed))
        print(json.dumps(dict(status='completed',selected_gamma=provenance['selected_gamma'],variant=provenance['variant'])))


if __name__=='__main__':
    try:
        main()
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        print(f'STOP: {exc}',file=sys.stderr)
        sys.exit(1)
