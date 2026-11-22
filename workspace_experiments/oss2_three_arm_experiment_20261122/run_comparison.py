"""Local OSS2 comparison runner. Keep manifests, ledgers, logs and checkpoints private.

Default: train only. Test access requires completed training for all three seeds.
Completed artifacts are verified and skipped; failed/interrupted steps never retry.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
DEFAULT_CODE_ROOT = Path('E:/临时会话/port_prototype_model_20261121')
LAUNCH = "import numpy, runpy; runpy.run_module('sar_port.two_stage.cli', run_name='__main__')"
TRAIN_STAGES = ('baseline_stage1', 'baseline_base', 'disentangle_stage1', 'disentangle_base', 'full_h')
EVALUATION_ARMS = {'baseline': 'baseline_base', 'disentangle': 'disentangle_base', 'full_h': 'full_h'}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
        temp = Path(stream.name)
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def exclusive_run(root):
    lock = root/'.run_comparison.lock'
    try:
        stream = lock.open('x', encoding='utf-8')
    except FileExistsError as exc:
        raise RuntimeError(f'Run lock exists: {lock}. Check its PID before manually resolving a stale lock.') from exc
    try:
        with stream:
            json.dump(dict(pid=os.getpid(), started_at=now()), stream)
        yield
    finally:
        lock.unlink(missing_ok=True)


def checkpoint(root, seed, stage):
    return root/'runs'/f'seed{seed}'/(stage+'.pt')


def config_path(root, seed, arm):
    return root/'configs'/f'{arm}_seed{seed}.json'


def pin_ledger(root, code_root, selected):
    manifest = root/'experiment_manifest_local.json'
    code = sorted((code_root/'sar_port').rglob('*.py'))
    if not (code_root/'sar_port'/'two_stage'/'cli.py').is_file() or not code:
        raise RuntimeError(f'Missing training CLI under {code_root}')
    common = dict(manifest=dict(path=str(manifest), sha256=digest(manifest)),
                  code_root=str(code_root),
                  code_sha256={p.relative_to(code_root).as_posix(): digest(p) for p in code},
                  runner_sha256=digest(Path(__file__)), python_executable=sys.executable,
                  required_test_seeds=list(SEEDS), thread_limits=dict(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'))
    path = root/'ledger_local.json'
    if path.exists():
        ledger = read_json(path)
        if ledger['inputs'] != common:
            raise RuntimeError('Pinned manifest/code/runner/runtime changed. Use a new experiment directory; no automatic retraining.')
    else:
        ledger = dict(version=1, created_at=now(), inputs=common, configs={})
    for seed in selected:
        for arm in ('baseline', 'disentangle'):
            path_cfg = config_path(root, seed, arm)
            config = read_json(path_cfg)
            if config.get('seed') != seed or config.get('allow_incomplete_supervision', False):
                raise RuntimeError(f'Seed mismatch or incomplete-supervision config: {path_cfg}')
            if config.get('encoder', {}).get('gate_enabled', True) != (arm == 'disentangle'):
                raise RuntimeError(f'Incorrect gate setting for {arm}: {path_cfg}')
            entry = dict(path=str(path_cfg), sha256=digest(path_cfg))
            weights = config.get('encoder', {}).get('weights_path')
            if weights:
                weight_path = Path(weights)
                if not weight_path.is_absolute(): weight_path = code_root/weight_path
                entry['weights'] = dict(path=str(weight_path.resolve()), sha256=digest(weight_path))
            key = f'{arm}_seed{seed}'
            if key in ledger['configs'] and ledger['configs'][key] != entry:
                raise RuntimeError(f'Pinned configuration/weights changed: {path_cfg}')
            ledger['configs'][key] = entry
    atomic_json(path, ledger)
    return ledger


def verified_success(state, key, output):
    record = state['steps'].get(key)
    if record is None:
        return False
    if record['status'] != 'succeeded':
        raise RuntimeError(f'{key} is {record["status"]}; inspect its log. Automatic retry is disabled.')
    if not output.is_file() or digest(output) != record['output_sha256']:
        raise RuntimeError(f'Completed artifact is missing or changed: {output}')
    return True


def execute_step(root, state, seed, stage, command, output, env, code_root):
    key = f'seed{seed}/{stage}'
    if verified_success(state, key, output):
        print(f'SKIP verified {key}', flush=True)
        return
    log = root/'runs'/f'seed{seed}'/(stage+'.log')
    if output.exists() or log.exists():
        raise RuntimeError(f'Unregistered existing output/log for {key}; refusing to overwrite.')
    log.parent.mkdir(parents=True, exist_ok=True)
    record = dict(status='running', started_at=now(), command=command,
                  output=str(output), log=str(log))
    state['steps'][key] = record
    state.update(status='running', current_step=key, updated_at=now())
    atomic_json(root/'state_local.json', state)
    print(f'START {key} | {log}', flush=True)
    try:
        with log.open('x', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=code_root, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        record['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'{key} exited {result.returncode}; see {log}')
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f'{key} produced no nonempty artifact; see {log}')
        if output.suffix == '.json':
            metrics = read_json(output)
            if metrics.get('count', 0) < 1 or 'balanced_accuracy' not in metrics:
                raise RuntimeError(f'{key} evaluation output lacks samples/metrics')
        record.update(status='succeeded', finished_at=now(), output_sha256=digest(output))
        state.update(current_step=None, updated_at=now())
        atomic_json(root/'state_local.json', state)
        print(f'DONE {key}', flush=True)
    except BaseException as exc:
        record.update(status='failed', finished_at=now(), error=f'{type(exc).__name__}: {exc}')
        state.update(status='failed', updated_at=now())
        atomic_json(root/'state_local.json', state)
        raise


def cli_command(root, seed, arm, action, output, device, source=None, split=None):
    command = [sys.executable, '-u', '-c', LAUNCH, action,
               '--manifest', str(root/'experiment_manifest_local.json'),
               '--config', str(config_path(root, seed, arm)), '--output', str(output), '--device', device]
    if source is not None: command += ['--checkpoint', str(source)]
    if split is not None: command += ['--split', split]
    return command


def train_seed(root, state, seed, device, env, code_root):
    plan = [('baseline_stage1', 'baseline', 'train-stage1', None),
            ('baseline_base', 'baseline', 'train-base', 'baseline_stage1'),
            ('disentangle_stage1', 'disentangle', 'train-stage1', None),
            ('disentangle_base', 'disentangle', 'train-base', 'disentangle_stage1'),
            ('full_h', 'disentangle', 'train-h', 'disentangle_base')]
    for stage, arm, action, source in plan:
        output = checkpoint(root, seed, stage)
        command = cli_command(root, seed, arm, action, output, device,
                              checkpoint(root, seed, source) if source else None)
        execute_step(root, state, seed, stage, command, output, env, code_root)
        if stage in EVALUATION_ARMS.values():
            eval_stage = stage+'_validation'
            result = output.with_name(eval_stage+'.json')
            command = cli_command(root, seed, arm, 'evaluate', result, device, output, 'validation')
            execute_step(root, state, seed, eval_stage, command, result, env, code_root)


def test_ready(root, state):
    missing = []
    for seed in SEEDS:
        for stage in TRAIN_STAGES:
            if not verified_success(state, f'seed{seed}/{stage}', checkpoint(root, seed, stage)):
                missing.append(f'seed{seed}/{stage}')
    if missing:
        raise RuntimeError('Test remains closed until all three seeds finish training: '+', '.join(missing))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train-only', action='store_true', help='Train plus validation; default. Never evaluates test.')
    mode.add_argument('--evaluate-test', action='store_true', help='Evaluate only, after training all 17/29/43 seeds.')
    parser.add_argument('--seed', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--code-root', type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    selected = list(dict.fromkeys(args.seed))
    code_root = args.code_root.resolve()
    env = os.environ.copy()
    env['SAR_PORT_PREPARE_CACHE'] = str(ROOT/'prepared_cache_local')
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(code_root)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    with exclusive_run(ROOT):
        pin_ledger(ROOT, code_root, SEEDS if args.evaluate_test else selected)
        state_path = ROOT/'state_local.json'
        state = read_json(state_path) if state_path.exists() else dict(version=1, steps={}, created_at=now())
        if args.evaluate_test:
            test_ready(ROOT, state)
            for seed in selected:
                for arm, stage in EVALUATION_ARMS.items():
                    source = checkpoint(ROOT, seed, stage)
                    output = source.with_name(arm+'_test.json')
                    config_arm = 'baseline' if arm == 'baseline' else 'disentangle'
                    command = cli_command(ROOT, seed, config_arm, 'evaluate', output, args.device, source, 'test')
                    execute_step(ROOT, state, seed, arm+'_test', command, output, env, code_root)
        else:
            for seed in selected: train_seed(ROOT, state, seed, args.device, env, code_root)
        state.update(status='completed_requested_steps', current_step=None, updated_at=now(),
                     last_mode='evaluate-test' if args.evaluate_test else 'train-only', last_seeds=selected)
        atomic_json(state_path, state)
        print(f'Completed requested steps. State: {state_path}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        sys.exit(1)
