"""Six paired fits, no sweep or final-test access."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
for seed in (17, 29, 43):
    for variant in ('shared_metric', 'no_competition'):
        subprocess.run([sys.executable, '-B', '-X', 'utf8', str(ROOT/'ablate.py'),
                        variant, '--arm', 'aggregate', '--seed', str(seed),
                        '--output-root', str(ROOT/variant)], check=True)
