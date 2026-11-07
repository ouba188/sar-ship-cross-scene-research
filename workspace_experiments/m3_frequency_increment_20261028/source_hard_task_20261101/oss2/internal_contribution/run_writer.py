from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
command=[sys.executable,'-B','-X','utf8',str(ROOT/'reference_writer.py')]
subprocess.run(command+['calibrate'],check=True)
for seed in (17,29,43):
    subprocess.run(command+['--arm','aggregate','--seed',str(seed),
                            '--output-root',str(ROOT/'reference_message')],check=True)
