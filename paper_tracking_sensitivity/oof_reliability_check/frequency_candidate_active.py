"""Fixed source-gate active-band control, reusing all visual contrast caches."""

import json
import numpy as np

from frequency_candidate_run import OUT, EXP, CANDIDATES, source_fit, target_predict
from frequency_decision_run import report

OUTPUT = OUT/'active_band_control'


if __name__ == '__main__':
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    for folder in folders:
        history = json.loads((folder/'selective_training.json').read_text())
        # Existing source-only final-epoch last-batch gate, before its last update.
        band = int(np.argmax(history['epochs'][-1]['gate_mean']))
        source_fit(folder, output=OUTPUT,
            feature_indices={'pooled_band': [band], 'local_band': [band, band+4, band+8]})
    for folder in folders:
        target_predict(folder, output=OUTPUT)
    report(folders, output=OUTPUT, candidates=CANDIDATES)
