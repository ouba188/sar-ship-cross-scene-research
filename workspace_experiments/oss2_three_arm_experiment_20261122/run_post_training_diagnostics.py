"""Wait for the comparison, then run serial diagnostics and bounded auxiliary-loss ablations.

The existing comparison is never relaunched. No test command is included. Run through run_on_d.py.
The status file records completed work, so a successful phase is not repeated.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
TUNING = LAB/'oss2_h_tuning_20261122'
OUTPUT = ROOT/'post_training_diagnostics_local'
SEEDS = (17, 29, 43)
TRAIN_STAGES = ('baseline_stage1', 'baseline_base', 'disentangle_stage1', 'disentangle_base', 'full_h')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(state):
    state['updated_at'] = now()
    path = OUTPUT/'state.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    temp.replace(path)


def process_alive(pid):
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    code = ctypes.c_ulong()
    try:
        if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
            raise OSError('Cannot inspect the training process')
        return code.value == 259
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def available_gib():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (key, ctypes.c_ulonglong) for key in ('total', 'available', 'page_total',
              'page_available', 'virtual_total', 'virtual_available', 'extended')]
    value = Memory()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError('Cannot read physical memory availability')
    return value.available/2**30


def plan():
    commands = [('training_integrity', [str(ROOT/'verify_finished_comparison.py')]),
                ('collect_validation', [str(ROOT/'collect_results.py'), '--split', 'validation'])]
    commands += [('branches_seed29', [str(ROOT/'branch_diagnostics.py'), '--seed', '29',
        '--output-dir', str(LAB/'oss2_disentangle_tuning_20261122/seed29'), '--device', 'cuda']),
        ('seed29_auxiliary_pilot', [str(LAB/'seed29_diagnosis_20261122/run_seed29_pilot.py'), '--run'])]
    commands += [(f'fusion_seed{seed}', [str(TUNING/'scan_validation.py'), '--seed', str(seed),
                 '--output-dir', str(TUNING/f'final_seed{seed}')]) for seed in SEEDS]
    commands += [('collect_fusion', [str(TUNING/'collect_fusion_scan.py')])]
    for seed in SEEDS:
        if seed == 29:
            continue
        output = LAB/'oss2_disentangle_tuning_20261122'/f'seed{seed}'
        commands.append((f'branches_seed{seed}', [str(ROOT/'branch_diagnostics.py'),
            '--seed', str(seed), '--output-dir', str(output), '--device', 'cuda']))
    disentangle = LAB/'oss2_disentangle_tuning_20261122'
    commands += [('head_reuse', [str(disentangle/'analyze_head_reuse.py')]),
                 ('auxiliary_scale_ablation', [str(disentangle/'run_aux_scale_ablation.py'), '--run'])]
    return commands


def wait_for_training(state):
    while True:
        training = read(ROOT/'state_local.json')
        if training.get('status') == 'failed':
            raise RuntimeError('Original training failed; inspect its recorded log before continuing')
        ready = all(training.get('steps', {}).get(f'seed{seed}/{stage}', {}).get('status') == 'succeeded'
                    for seed in SEEDS for stage in TRAIN_STAGES)
        lock = ROOT/'.run_comparison.lock'
        if ready and not lock.exists():
            return
        if not lock.exists() or not process_alive(read(lock)['pid']):
            raise RuntimeError('Training is incomplete and no live comparison runner owns it')
        state.update(status='waiting_for_training', current_training_step=training.get('current_step'))
        save(state)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    commands = plan()
    if args.dry_run:
        print(json.dumps(dict(test_access=False, commands=commands), ensure_ascii=False, indent=2))
        return
    OUTPUT.mkdir(exist_ok=True)
    lock = OUTPUT/'.queue.lock'
    with lock.open('x', encoding='utf8') as stream:
        json.dump(dict(pid=os.getpid(), started_at=now()), stream)
    state_path = OUTPUT/'state.json'
    state = read(state_path) if state_path.exists() else dict(started_at=now(), steps={}, test_access=False)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    try:
        if any(step['status'] != 'succeeded' for step in state['steps'].values()):
            raise RuntimeError('An earlier diagnostic failed or was interrupted; inspect it before retrying')
        wait_for_training(state)
        for name, arguments in commands:
            if name in state['steps']:
                continue
            while available_gib() < 2.0:
                state.update(status='waiting_for_memory', next_step=name, available_gib=available_gib())
                save(state)
                time.sleep(30)
            log = OUTPUT/(name+'.log')
            command = [sys.executable, str(LAB/'run_on_d.py'), '-u', *arguments]
            state['steps'][name] = dict(status='running', started_at=now(), command=command, log=str(log))
            state.update(status='running', current_step=name)
            save(state)
            print('START '+name, flush=True)
            with log.open('x', encoding='utf8') as stream:
                result = subprocess.run(command, cwd=LAB, env=env, stdout=stream, stderr=subprocess.STDOUT)
            state['steps'][name].update(status='succeeded' if result.returncode == 0 else 'failed',
                                       finished_at=now(), returncode=result.returncode)
            save(state)
            if result.returncode:
                raise RuntimeError(f'{name} exited {result.returncode}; inspect {log}')
            print('DONE '+name, flush=True)
        state.update(status='completed', current_step=None,
            next_action='Review validation findings and targeted training changes before opening historical test.')
        save(state)
        print('Diagnostics and bounded auxiliary-loss ablations completed. Test remains unopened.', flush=True)
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save(state)
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
