"""One matched preservation arm; controls are reused, never retrained or retuned."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import shutil
import numpy as np
from prepare_learned_frequency import ROOT, OUT as EXP, load_npz

OUT = ROOT/'frequency_preserve_20261010'
PREVIOUS = ROOT/'frequency_standalone_20261009'
FAMILIES = ('old', 'independent', 'standalone', 'preserved')


def configure_run():
    import frequency_specialist_run as run
    run.OUT, run.FAMILIES = OUT, FAMILIES
    run.MODES = ('learned', *(f'{f}_{b}' for f in FAMILIES for b in ('equal', *run.BANDS)),
        *(f'{f}_support_{s}' for f in FAMILIES for s in ('static', 'local')), 'source_selected')
    return run


def replay_controls(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    suffix = ('targets', info['dataset'], info['task'], 'policies.npz')
    current, previous = load_npz(OUT.joinpath(*suffix)), load_npz(PREVIOUS.joinpath(*suffix))
    for mode in [m for m in current['modes'] if m.startswith(('independent_', 'standalone_'))]:
        now, old = list(current['modes']).index(mode), list(previous['modes']).index(mode)
        for key in ('base_predictions', 'predictions', 'frozen_predictions', 'selected'):
            np.testing.assert_array_equal(current[key][:, now], previous[key][:, old])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['encode', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    if args.stage in ('encode', 'all'):
        import torch
        from frequency_preserve_train import train_and_encode_source, encode_target
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        maps = np.load(EXP/'maps/self24ports/maps.npy', mmap_mode='r')
        for folder in folders:
            info = json.loads((folder/'PREPARED.json').read_text())
            dst = OUT/'tasks'/info['dataset']/info['task']
            dst.mkdir(parents=True, exist_ok=True)
            previous = PREVIOUS/'tasks'/info['dataset']/info['task']
            for family in ('independent', 'standalone'):
                for suffix in ('.pt', '_embedding.npz', '_training.json'):
                    if not (dst/(family+suffix)).exists():
                        shutil.copyfile(previous/(family+suffix), dst/(family+suffix))
            train_and_encode_source(folder, maps)
            encode_target(folder)
    if args.stage in ('evaluate', 'all'):
        run = configure_run()
        for folder in folders:
            run.source_validate(folder)
            run.target_predict(folder)
            replay_controls(folder)
        from frequency_preserve_report import main as report
        report()


if __name__ == '__main__':
    main()
