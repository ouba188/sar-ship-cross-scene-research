"""Continue the interrupted run once, recording each subprocess exit code."""
from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
RUN=ROOT/f'resume_{stamp}'
RUN.mkdir()
state=dict(started=datetime.now().astimezone().isoformat(),supervisor_pid=os.getpid(),
           run_directory=str(RUN),state='starting',stages=[],formal_model_changed=False)
def save():
    state['updated']=datetime.now().astimezone().isoformat()
    text=json.dumps(state,ensure_ascii=False,indent=2)
    (RUN/'status.json').write_text(text,encoding='utf-8')
    (ROOT/'resume_queue_status.json').write_text(text,encoding='utf-8')

try:
    for name,args in [('training',['train.py','train']),('evaluation',['train.py','evaluate']),
                      ('polarization_recovery',['evaluate_recovery.py']),('analysis',['analyze.py'])]:
        log=RUN/f'{name}.log'
        entry=dict(stage=name,started=datetime.now().astimezone().isoformat(),log=str(log))
        state['stage']=name;state['state']='running';state['stages'].append(entry)
        with log.open('wb',buffering=0) as out:
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/args[0]),*args[1:]],
                                     cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
            entry['pid']=process.pid;save()
            code=process.wait()
        entry.update(exit_code=code,exit_code_hex=hex(code & 0xffffffff),finished=datetime.now().astimezone().isoformat())
        save()
        if code:
            raise RuntimeError(f'{name} exited {code} ({code & 0xffffffff:#x}); see {log}')
    state['state']='complete_pending_visual_review'
    state['detail']='All computation complete; final results and chip gallery still require review.'
    save()
except BaseException:
    state['state']='failed';state['error']=traceback.format_exc();save()
    raise
