"""One posthoc training-composition control, retaining the initial run."""
from pathlib import Path
import run_pilot as run

if __name__ == '__main__':
    run.OUT = Path(__file__).resolve().parent/'matched_training'
    run.OUT.mkdir(exist_ok=True)
    run.TRAIN_SAMPLING = 'matched_scene_counts'
    with run.threadpool_limits(limits=2):
        run.main()
