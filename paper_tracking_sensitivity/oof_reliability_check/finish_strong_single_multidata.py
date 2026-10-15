"""Finish the already authorized fixed comparison and its dependent reports."""

import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT/'strong_single_multidata_20261010'
GPU_PY = ROOT.parent/'envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe'
CPU_PY = Path('D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe')


def main():
    if not (ROOT/'strong_single_sarclip_20261010/f0921_EXTRACTED.json').exists():
        raise RuntimeError('Finish the in-flight F SARCLIP extraction before starting this queue')
    jobs = [
        (CPU_PY, 'strong_single_screen.py', 'predict', '--dataset', 'f0921', '--arm', 'sarclip'),
        (CPU_PY, 'strong_single_screen.py', 'report'),
        (GPU_PY, 'strong_single_multidata.py'),
        (GPU_PY, 'strong_single_multidata_crossover.py'),
        (GPU_PY, 'strong_single_multidata_explain.py'),
        (CPU_PY, 'strong_single_commonhard_expanded.py'),
        (GPU_PY, 'strong_single_error_modes.py'),
        (CPU_PY, 'strong_single_multidata_cases.py'),
        (GPU_PY, 'report_strong_single_multidata.py'),
    ]
    history = []
    for job in jobs:
        started = time.perf_counter()
        entry = dict(command=[str(x) for x in job], status='running')
        history.append(entry)
        (OUT/'FINISH_QUEUE.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print('START', *job[1:], flush=True)
        result = subprocess.run([str(job[0]), '-u', *job[1:]], cwd=HERE)
        entry.update(status='complete' if result.returncode == 0 else 'failed',
                     exit_code=result.returncode, seconds=time.perf_counter()-started)
        (OUT/'FINISH_QUEUE.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if result.returncode:
            raise SystemExit(result.returncode)
        print('DONE', *job[1:], flush=True)


if __name__ == '__main__':
    main()
