"""Launch experimental Python commands with all transient caches on D."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
env = os.environ.copy()
paths = dict(TEMP=ROOT/'temp', TMP=ROOT/'temp', TORCH_HOME=ROOT/'torch_cache',
             CUDA_CACHE_PATH=ROOT/'cuda_cache', MPLCONFIGDIR=ROOT/'matplotlib_cache',
             PYTHONPYCACHEPREFIX=ROOT/'python_cache')
for key,path in paths.items():
    path.mkdir(parents=True,exist_ok=True)
    env[key] = str(path)
if len(sys.argv) < 2:
    raise SystemExit('Usage: python run_on_d.py SCRIPT [ARGS...]')
result = subprocess.run([sys.executable,*sys.argv[1:]],cwd=ROOT,env=env)
raise SystemExit(result.returncode)
