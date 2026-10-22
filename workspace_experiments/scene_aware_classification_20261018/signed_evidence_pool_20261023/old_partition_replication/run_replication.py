"""Exact matched jointCE/J protocol on the historical acquisition partition."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'source_utility_alignment/matched_controls'))
import run_controls as controls


if __name__ == '__main__':
    controls.HERE = HERE
    controls.parent.NEW = HERE
    controls.parent.CACHE = HERE
    controls.ARMS = ['signed_only', 'score_only', 'conditional_std', 'linear_distribution']
    controls.main()
