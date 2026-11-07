"""Two new paired seeds, no configuration search or evaluation expansion."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
for seed in (29,43):
    for arm in ('none','aggregate','competitive'):
        subprocess.run([sys.executable,'-B','-X','utf8',str(root/'train_task.py'),
                        '--arm',arm,'--seed',str(seed)],check=True)
