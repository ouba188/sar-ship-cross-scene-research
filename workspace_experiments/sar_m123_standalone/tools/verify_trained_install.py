"""Install retrained checkpoints into asset copies and verify deployment scores."""
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sar_m123.aligned import AlignedClassifier, FEATURES, load_npz
from export_aligned import install_training


def main():
    root = Path(__file__).resolve().parents[1]
    results = []
    for source in ('oss1', 'oss2'):
        destination = root/'training_outputs/deployed'/source
        if destination.exists():
            raise FileExistsError(f'Use a fresh deployment verification directory: {destination}')
        shutil.copytree(root/'assets_aligned'/source, destination)
        install_training(destination, root/'training_outputs'/source)
        model = AlignedClassifier(destination, 'cpu')
        bundle = root/'validation_local/aligned_training'/source
        data = load_npz(bundle/'features.npz')
        table = pd.read_csv(bundle/'table.csv', dtype={'mmsi':str})
        ids = np.flatnonzero(table.role.eq('evaluation'))[:16]
        features = dict(raw=data['raw'][ids], aux=data['aux'][ids],
                        local={k:data[k][ids] for k in FEATURES},
                        mmsi=table.mmsi.to_numpy()[ids])
        for seed in (17, 29, 43):
            scores, detail = model.score(features, seed)
            saved = load_npz(root/f'training_outputs/{source}/seed{seed}/predictions.npz')
            target = saved['frequency'][ids]
            np.testing.assert_allclose(scores, target, atol=2e-4, rtol=2e-5)
            np.testing.assert_array_equal(scores.argmax(1), target.argmax(1))
            np.testing.assert_array_equal(detail['matching']['reference_ids'], saved['refs'][ids])
            results.append(dict(source=source, seed=seed, n=len(ids),
                max_abs=float(abs(scores-target).max()), prediction_disagreements=0,
                references_identical=True))
    report = dict(status='passed', original_assets_modified=False, checks=results)
    (root/'validation_reports/aligned_trained_deployment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
