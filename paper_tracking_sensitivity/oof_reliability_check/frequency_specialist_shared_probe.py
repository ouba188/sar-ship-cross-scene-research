"""Bounded source-only probe of standalone band evidence versus shared path."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
import numpy as np
import pandas as pd
from prepare_learned_frequency import ROOT, OUT as EXP, load_npz
from frequency_band_core import old_kernel, unit_bands, score_embedding, BANDS
from frequency_reliability_core import coefficients
from frequency_reliability_run import statistics
from frequency_specialist_run import OUT, FAMILIES


def main():
    folders = [EXP/'tasks/f0921', EXP/'tasks/oss2', EXP/'tasks/self24ports/Antwerp-Bruges']
    records, assignments = [], []
    for folder in folders:
        info = json.loads((folder/'PREPARED.json').read_text())
        src = OUT/'source'/info['dataset']/info['task']
        episodes = sorted((folder/'episodes').glob('*.npz'))
        for path in sorted(src.glob('episode_*.npz')):
            ei = int(path.stem.split('_')[-1])
            z = load_npz(path)
            ep = load_npz(episodes[ei])
            ix = z['indices']
            data = {k: ep[k][ix] for k in ('phi', 'prior', 'mean', 'conditions')}
            sy, y, support = ep['labels'][ix[:64]], ep['labels'][ix[64:]], np.arange(64)
            kernel, condition = old_kernel(data, support, info['eta'])
            shared = data['prior']+kernel@coefficients(kernel[support], sy, data['prior'][support])
            reference, _ = score_embedding(data, z['learned'].astype(float), support, sy, kernel, condition)
            oa0, ba0, _ = statistics(y, shared[64:].argmax(1))
            for family in FAMILIES:
                bands = unit_bands(z[family])
                full_band = []
                for b, name in enumerate(BANDS):
                    full, sim = score_embedding(data, bands[:, b], support, sy, kernel, condition)
                    full_band.append(full[64:])
                    gram = .25*sim
                    alone = gram@coefficients(gram[support], sy, np.zeros_like(data['prior'][support]))
                    oa, ba, _ = statistics(y, full[64:].argmax(1))
                    a_oa, a_ba, _ = statistics(y, alone[64:].argmax(1))
                    records.append(dict(dataset=info['dataset'], task=info['task'], episode=ei,
                        family=family, band=name, n=len(y), shared_oa=oa0, shared_ba=ba0,
                        full_band_oa=oa, full_band_ba=ba, standalone_oa=a_oa, standalone_ba=a_ba,
                        agreement_shared=float(np.mean(shared[64:].argmax(1) == full[64:].argmax(1)))))
                if family == 'old':
                    initial = np.stack(full_band, 1)
                    true = np.take_along_axis(initial, y[:, None, None], 2).squeeze(2)
                    other = initial.copy()
                    other[np.arange(len(y))[:, None], np.arange(4)[None], y[:, None]] = -np.inf
                    margin = true-other.max(2)
                    soft = np.exp((margin-margin.max(1, keepdims=True))/.1)
                    allocation = .5+2*soft/soft.sum(1, keepdims=True)
                    wrong = reference[64:].argmax(1) != y
                    can_rescue = (initial.argmax(2) == y[:, None]).any(1)
                    assignments.append(dict(dataset=info['dataset'], task=info['task'], episode=ei,
                        wrong=int(wrong.sum()), wrong_with_rescue=int(np.sum(wrong & can_rescue)),
                        wrong_without_rescue=int(np.sum(wrong & ~can_rescue)),
                        wrong_allocation_spread=float(np.mean(np.ptp(allocation[wrong], axis=1))) if wrong.any() else np.nan))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'shared_path_probe.csv', index=False)
    fields = ['shared_oa', 'shared_ba', 'full_band_oa', 'full_band_ba', 'standalone_oa', 'standalone_ba', 'agreement_shared']
    summary = frame.groupby(['dataset', 'task', 'family'])[fields].mean().reset_index()
    summary.to_csv(OUT/'shared_path_probe_summary.csv', index=False)
    assignment = pd.DataFrame(assignments)
    assignment.to_csv(OUT/'source_assignment_probe.csv', index=False)
    counts = assignment.groupby(['dataset', 'task'])[['wrong', 'wrong_with_rescue', 'wrong_without_rescue']].sum()
    counts['without_rescue_percent'] = 100*counts.wrong_without_rescue/counts.wrong
    counts.to_csv(OUT/'source_assignment_summary.csv')
    print(summary.to_string(index=False), flush=True)
    print(counts.to_string(), flush=True)


if __name__ == '__main__':
    main()
