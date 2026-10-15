"""Bounded standalone/pair trial; reuse matched independent control artifacts."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import shutil
import numpy as np
from prepare_learned_frequency import ROOT, OUT as EXP, load_npz

OUT = ROOT/'frequency_standalone_20261009'
PREVIOUS = ROOT/'frequency_specialists_20261009'
FAMILIES = ('old', 'independent', 'standalone', 'pair')


def configure_run():
    import frequency_specialist_run as run
    run.OUT, run.FAMILIES = OUT, FAMILIES
    run.MODES = ('learned', *(f'{f}_{b}' for f in FAMILIES for b in ('equal', *run.BANDS)),
                 *(f'{f}_support_{s}' for f in FAMILIES for s in ('static', 'local')), 'source_selected')
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['encode', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    if args.stage in ('encode', 'all'):
        import torch
        import frequency_specialist_train as train
        from frequency_standalone_core import auxiliary_objective
        train.OUT, train.ARMS = OUT, FAMILIES[1:]
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        maps = np.load(EXP/'maps/self24ports/maps.npy', mmap_mode='r')
        for folder in folders:
            info, dst = train.task_paths(folder)
            previous = PREVIOUS/'tasks'/info['dataset']/info['task']
            for name in ('independent.pt', 'independent_training.json', 'independent_embedding.npz'):
                if not (dst/name).exists():
                    shutil.copyfile(previous/name, dst/name)
            train.train_task(folder, maps, arms=('standalone', 'pair'), auxiliary=auxiliary_objective)
            train.encode_source(folder, maps)
            train.encode_target(folder)
    if args.stage in ('evaluate', 'all'):
        run = configure_run()
        for folder in folders:
            run.source_validate(folder)
        for folder in folders:
            run.target_predict(folder)
            info = json.loads((folder/'PREPARED.json').read_text())
            suffix = ('targets', info['dataset'], info['task'], 'policies.npz')
            current, previous = load_npz(OUT.joinpath(*suffix)), load_npz(PREVIOUS.joinpath(*suffix))
            for mode in [m for m in current['modes'] if m.startswith('independent_')]:
                now, old = list(current['modes']).index(mode), list(previous['modes']).index(mode)
                for key in ('base_predictions', 'predictions', 'frozen_predictions', 'selected'):
                    np.testing.assert_array_equal(current[key][:, now], previous[key][:, old])
        from frequency_standalone_report import main as report
        report()


if __name__ == '__main__':
    main()
