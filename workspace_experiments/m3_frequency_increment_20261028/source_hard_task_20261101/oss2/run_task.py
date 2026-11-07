"""Serial bounded queue, four fits only."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
for arm in ('none','original','aggregate','competitive'):
    subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'train_task.py'),'--arm',arm],check=True)
