"""Two missing OSS2 factorial corners, diagnostic only after the primary trial."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
import numpy as np
import pandas as pd

from prepare_learned_frequency import ROOT, OUT as EXP, DATASETS, load_npz
from unified_head_run import OUT, base_kernel, write, ba
from unified_head_core import combined
from unified_head_expert import ExpertContext

CORNERS = ((.5, 8.), (.25, 8.), (.5, 64.), (.25, 64.))


def main():
    folder = EXP/'tasks/oss2'
    info = json.loads((folder/'PREPARED.json').read_text())
    dst = OUT/'oss2_diagnosis'
    dst.mkdir(parents=True, exist_ok=True)
    data = load_npz(folder/'target.npz')
    if not (dst/'FROZEN.json').exists():
        write(dst/'PROTOCOL.json', dict(reason='Separate weight and regularization effects in source-selected OSS2 failure',
            corners=CORNERS, diagnostic_only=True, primary_results_preserved=True,
            no_new_model_selection=True, no_primary_replacement=True))
        context = ExpertContext(folder)
        embedding = load_npz(folder/'selective_embedding.npz')['embedding'].astype(float)
        old = load_npz(OUT/'targets/oss2/oss2/policies.npz')
        predictions = np.empty((5, 6, len(data['rows'])), np.int8)
        records = []
        for draw, support in enumerate(data['support']):
            sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
            pe, ke = context.target_components(data, support)
            pb, kb = data['prior'], base_kernel(data, embedding, support, info['eta'])
            for i, (ratio, strength) in enumerate(CORNERS):
                value, _ = combined(kb, ke, pb, pe, support, sy, ratio, strength)
                prior = (1-ratio)*pb+ratio*pe
                predictions[draw, i] = value.argmax(1)
                records.append(dict(draw=draw, ratio=ratio, strength=strength,
                    correction_rms=float(np.sqrt(np.mean((value[ev]-prior[ev])**2))),
                    departure_from_prior=100*float(np.mean(value[ev].argmax(1) != prior[ev].argmax(1)))))
            for i, ratio in enumerate((.5, .25)):
                predictions[draw, 4+i] = ((1-ratio)*pb+ratio*pe).argmax(1)
            for i, mode in [(0, 'unified_fixed'), (3, 'unified_selected')]:
                np.testing.assert_array_equal(predictions[draw, i], old['predictions'][draw, list(old['modes']).index(mode)])
        np.savez_compressed(dst/'policies.npz', predictions=predictions, eval_masks=data['eval_masks'],
                            rows=data['rows'], ids=data['ids'])
        pd.DataFrame(records).to_csv(dst/'correction_strength.csv', index=False)
        write(dst/'FROZEN.json', dict(target_query_labels_read=False, old_two_corners_exact=True,
                                     diagnostic_only=True, extra_fitted_corners=2))
    saved = load_npz(dst/'policies.npz')
    frame = pd.read_csv(ROOT/DATASETS['oss2'][0]/'target_manifest.csv',
                        usecols=['sample_id', 'class_id'], keep_default_na=False).iloc[saved['rows']]
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str), saved['ids'])
    truth = frame.class_id.to_numpy()
    records = []
    for draw, ev in enumerate(saved['eval_masks']):
        y = truth[ev]
        for i, (ratio, strength) in enumerate((*CORNERS, (.5, 0.), (.25, 0.))):
            pred = saved['predictions'][draw, i, ev]
            records.append(dict(draw=draw, ratio=ratio, strength=strength,
                                mode='adapted' if i < 4 else 'prior_only',
                                oa=100*np.mean(pred == y), ba=ba(y, pred)))
    metrics = pd.DataFrame(records)
    metrics.to_csv(dst/'metrics.csv', index=False)
    summary = metrics.groupby(['ratio', 'strength', 'mode'])[['oa', 'ba']].mean().reset_index()
    summary.to_csv(dst/'summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
