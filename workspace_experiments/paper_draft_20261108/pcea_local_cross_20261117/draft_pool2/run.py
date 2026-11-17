"""Matched pilot; no teacher/top2 routing and no polarization-input changes."""
from pathlib import Path
import argparse
import importlib.util
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('local_cross_runner', ROOT.parent/'pcea_conditional_modulation_20261117/run.py')
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
runner.ROOT = ROOT
runner.ARMS = ('independent', 'cross_mean', 'cross_local')
runner.CONFIGS = {k: runner.CONFIGS[k] for k in ('base', 'slow')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('smoke', 'tune', 'confirm', 'evaluate'), default='tune')
    parser.add_argument('--arms', nargs='+', choices=runner.ARMS, default=list(runner.ARMS))
    args = parser.parse_args()
    model, inputs = runner.runtime()
    if args.phase == 'smoke':
        for arm in args.arms:
            print(runner.test_gradients(model, inputs, arm), flush=True)
    elif args.phase == 'tune':
        runner.tune(model, inputs, args.arms)
    elif args.phase == 'confirm':
        runner.confirm(model, inputs, args.arms)
    else:
        runner.evaluate(model, inputs, args.arms)


if __name__ == '__main__':
    main()
