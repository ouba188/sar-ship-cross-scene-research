"""Paired verification of the exported adaptation loop on the 160 old cases."""
import json
from pathlib import Path

import numpy as np
import torch

from sar_m123.current.data import read_manifest
from sar_m123.current.model import load_model, predict_records, setup


def main():
    root = Path(__file__).resolve().parent
    setup()
    rows = read_manifest(root/'manifests/development.json')
    refs = read_manifest(root/'manifests/references.json')
    truth = np.array([r['label'] for r in rows])
    scores = {}
    for name, path in (('archived', 'weights/selected_s17.pt'),
                       ('rerun', 'verification/adapt_replay_s17/model.pt')):
        net = load_model(root/path, 'cuda')
        scores[name] = predict_records(net, rows, refs, 'cuda').numpy()
        del net
        torch.cuda.empty_cache()
    def metrics(score):
        z = torch.from_numpy(score)
        labels = torch.tensor(truth)
        correct = z.argmax(1).eq(labels).numpy()
        return dict(OA=float(correct.mean()*100),
            BA=float(np.mean([correct[truth==c].mean() for c in np.unique(truth)])*100),
            NLL=float(torch.nn.functional.cross_entropy(z, labels)),
            Brier=float((z.softmax(1)-torch.nn.functional.one_hot(labels, 8)).square().sum(1).mean()))
    changed = scores['archived'].argmax(1) != scores['rerun'].argmax(1)
    report = dict(n=len(rows), changed_predictions=int(changed.sum()),
        changed_ids=[r['id'] for r, changed in zip(rows, changed) if changed],
        max_score_difference=float(abs(scores['archived']-scores['rerun']).max()),
        metrics={name: metrics(s) for name, s in scores.items()},
        new_generalization_experiment=False)
    np.savez_compressed(root/'verification/adaptation_replay_scores.npz', y=truth, **scores)
    (root/'verification/adaptation_replay.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
