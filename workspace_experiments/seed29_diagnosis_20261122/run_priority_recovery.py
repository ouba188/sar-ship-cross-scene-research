"""Run seed29 recovery first, then resume the unchanged original comparison and diagnostics."""
from datetime import datetime, timezone
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
EXP=LAB/'oss2_three_arm_experiment_20261122'
QUEUE=EXP/'post_training_diagnostics_local'


def now(): return datetime.now(timezone.utc).isoformat()


def memory_budget():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[
            (key,ctypes.c_ulonglong) for key in ('total','available','page_total',
                'page_available','virtual_total','virtual_available','extended')]
    value=Memory();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError('Cannot read system memory budget')
    return dict(physical_available_gib=value.available/2**30,
                commit_available_gib=value.page_available/2**30)


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    os.replace(temp,path)


def require_idle():
    """Read-only guard against an orphaned CLI from the yielded scheduler."""
    if (EXP/'.run_comparison.lock').exists() or (QUEUE/'.queue.lock').exists():
        raise RuntimeError('Original scheduler and general diagnostic queue must yield first')
    manifest=(EXP/'experiment_manifest_local.json').as_posix().replace("'","''")
    script=("$ErrorActionPreference='Stop'; $matches=@(Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | "
            "Where-Object { $_.CommandLine -and $_.CommandLine.Contains('sar_port.two_stage.cli') -and "
            "$_.CommandLine.Replace('\\','/').IndexOf('"+manifest+"',[StringComparison]::OrdinalIgnoreCase) -ge 0 } | "
            "Select-Object ProcessId,ParentProcessId); ConvertTo-Json -InputObject $matches -Compress")
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],
        capture_output=True,text=True,encoding='utf8',check=True)
    active=json.loads(result.stdout.strip() or '[]')
    if active:
        raise RuntimeError(f'Original experiment CLI processes still exist; no process was killed: {active}')


def original_receipts():
    state=json.loads((EXP/'state_local.json').read_text(encoding='utf-8-sig'))
    stages=('baseline_stage1','baseline_base','disentangle_stage1','disentangle_base','full_h',
            'baseline_base_validation','disentangle_base_validation','full_h_validation')
    receipts={}
    for seed in (17,29,43):
        for stage in stages:
            entry=state.get('steps',{}).get(f'seed{seed}/{stage}',{})
            suffix='.json' if stage.endswith('_validation') else '.pt'
            path=EXP/'runs'/f'seed{seed}'/(stage+suffix)
            if entry.get('status')!='succeeded' or not path.is_file() or digest(path)!=entry['output_sha256']:
                raise RuntimeError(f'Original resume lacks a verified completed artifact: seed{seed}/{stage}')
            receipts[str(path)]=entry['output_sha256']
    return receipts


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def run_phase(name,arguments,local,state_path,env,queue):
    state=json.loads(state_path.read_text(encoding='utf8'))
    phases=state.setdefault('priority_phases',{})
    prior=phases.get(name)
    if prior:
        if prior['status']!='succeeded':
            raise RuntimeError(f'Priority phase {name} is {prior["status"]}; automatic retry disabled')
        for path,expected in prior.get('artifacts',{}).items():
            if not Path(path).is_file() or digest(path)!=expected:raise RuntimeError('Completed phase artifact changed: '+path)
        if name=='remaining_diagnostics' and any(state['steps'].get(k,{}).get('status')!='succeeded' for k,_ in queue.plan()):
            raise RuntimeError('Remaining diagnostics completion record disagrees with current steps')
        print('SKIP verified phase '+name,flush=True);return
    require_idle()
    log=local/(name+'.log')
    if log.exists():raise RuntimeError('Unregistered phase log exists; inspect without overwriting: '+str(log))
    command=[sys.executable,str(LAB/'run_on_d.py'),'-u',*arguments]
    phases[name]=dict(status='running',started_at=now(),command=command,log=str(log))
    state.update(status='priority_phase_running',current_step=name);write(state_path,state)
    try:
        print('START phase '+name,flush=True)
        with log.open('x',encoding='utf8') as stream:
            result=subprocess.run(command,cwd=LAB,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}; inspect {log}')
        artifacts=original_receipts() if name=='resume_original_comparison' else {}
        # The remaining queue updates this same state: reload, never overwrite its steps.
        state=json.loads(state_path.read_text(encoding='utf8'))
        if name=='remaining_diagnostics' and any(state['steps'].get(k,{}).get('status')!='succeeded' for k,_ in queue.plan()):
            raise RuntimeError('Remaining diagnostic queue returned without completing its plan')
        state['priority_phases'][name].update(status='succeeded',returncode=0,finished_at=now(),artifacts=artifacts)
        state.update(current_step=None);write(state_path,state)
        print('DONE phase '+name,flush=True)
    except BaseException as exc:
        state=json.loads(state_path.read_text(encoding='utf8'))
        state['priority_phases'][name].update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=now())
        write(state_path,state);raise


def main():
    spec=importlib.util.spec_from_file_location('diagnostic_queue',EXP/'run_post_training_diagnostics.py')
    queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    require_idle()
    QUEUE.mkdir(exist_ok=True)
    local=ROOT/'priority_local';local.mkdir(exist_ok=True)
    lock=local/'run.lock'
    with lock.open('x',encoding='utf8') as stream: json.dump(dict(pid=os.getpid(),started_at=now()),stream)
    state_path=QUEUE/'state.json'
    state=json.loads(state_path.read_text(encoding='utf8')) if state_path.exists() else dict(started_at=now(),steps={},test_access=False)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    steps=[('branches_seed29',[str(EXP/'branch_diagnostics.py'),'--seed','29',
        '--output-dir',str(LAB/'oss2_disentangle_tuning_20261122/seed29'),'--device','cuda']),
        ('seed29_auxiliary_pilot',[str(ROOT/'run_seed29_pilot.py'),'--run','--seed29-first'])]
    try:
        for name,args in steps:
            if name in state['steps']:
                prior=state['steps'][name]
                if prior['status']!='succeeded' or hashlib.sha256(Path(prior['artifact']).read_bytes()).hexdigest()!=prior['artifact_sha256']:
                    raise RuntimeError('Priority step is incomplete or changed; inspect instead of rerunning')
                print('SKIP verified '+name,flush=True)
                continue
            # Windows expands other working sets when the old job exits. Free
            # physical RAM alone can indefinitely block a serialized GPU job.
            # Keep 8 GiB system commit headroom (measured prior peak <5 GiB);
            # paging may slow execution, but do not alter model/batch numerics.
            budget=memory_budget()
            while budget['commit_available_gib']<8:
                state.update(status='priority_waiting_for_memory',current_step=name,
                    memory_budget=budget,minimum_commit_gib=8,updated_at=now());write(state_path,state)
                time.sleep(30)
                budget=memory_budget()
            require_idle()
            log=local/(name+'.log')
            cmd=[sys.executable,str(LAB/'run_on_d.py'),'-u',*args]
            state['steps'][name]=dict(status='running',started_at=now(),command=cmd,log=str(log),startup_memory=budget)
            state.update(status='priority_running',current_step=name);write(state_path,state)
            print('START '+name,flush=True)
            with log.open('x',encoding='utf8') as stream:
                result=subprocess.run(cmd,cwd=LAB,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                state['steps'][name].update(status='failed',returncode=result.returncode,finished_at=now())
                write(state_path,state)
                raise RuntimeError(f'{name} failed; inspect {log}')
            artifact=(LAB/'oss2_disentangle_tuning_20261122/seed29/aggregate_branch_diagnostics.json'
                if name=='branches_seed29' else ROOT/'aggregate_seed29_aux_pilot.json')
            state['steps'][name].update(status='succeeded',returncode=0,finished_at=now(),
                artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
            write(state_path,state);print('DONE '+name,flush=True)
        run_phase('resume_original_comparison',[str(EXP/'run_comparison.py'),'--train-only'],local,state_path,env,queue)
        run_phase('remaining_diagnostics',[str(EXP/'run_post_training_diagnostics.py')],local,state_path,env,queue)
        state=json.loads(state_path.read_text(encoding='utf8'))
        state.update(status='completed',current_step=None,priority_completed_at=now());write(state_path,state)
    except BaseException as exc:
        if state_path.exists(): state=json.loads(state_path.read_text(encoding='utf8'))
        state.update(status='priority_failed',error=f'{type(exc).__name__}: {exc}',updated_at=now());write(state_path,state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
