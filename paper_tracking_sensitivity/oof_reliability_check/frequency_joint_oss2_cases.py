"""Post-freeze OSS2 loss accounting and exact rare-class flip decomposition."""

import json
import numpy as np
import pandas as pd
import torch

from frequency_joint_run import OUT, EXP, ROOT, compare
from prepare_learned_frequency import DATASETS, load_npz
from frequency_reliability_core import kernels, coefficients


def main():
    folder = EXP/'tasks/oss2'
    info = json.loads((folder/'PREPARED.json').read_text())
    d = load_npz(folder/'target.npz')
    p = load_npz(OUT/'targets/oss2/oss2/policies.npz')
    y = pd.read_csv(ROOT/DATASETS['oss2'][0]/'target_manifest.csv').class_id.to_numpy()
    modes = list(p['modes'])
    embeddings = dict(learned=load_npz(folder/'selective_embedding.npz')['embedding'],
        **{m: load_npz(OUT/'tasks/oss2/oss2'/f'{m}_embedding.npz')['embedding'] for m in ('continued_ce', 'joint_kernel')})
    records, groups, cases = [], [], []
    for draw, support in enumerate(d['support']):
        sy, ev = d['support_labels'][draw], d['eval_masks'][draw]
        counts = np.bincount(sy, minlength=8)
        matrices, duals, scores = {}, {}, {}
        for mode, emb in embeddings.items():
            _, k = kernels(d, emb.astype(float), support, info['eta'])
            alpha = coefficients(k[support], sy, d['prior'][support])
            matrices[mode], duals[mode], scores[mode] = k, alpha, d['prior']+k@alpha
        old = p['base_predictions'][draw, modes.index('learned')]
        new = p['base_predictions'][draw, modes.index('joint_kernel')]
        rows = np.flatnonzero(ev & (old != new))
        logits = scores['learned'][rows].copy()
        rr, labels = np.arange(len(rows)), y[rows]
        logits[:, counts < 2] = -np.inf
        logits[rr, labels] = -np.inf
        negative = logits.argmax(1)
        valid = (counts[labels] >= 2) & np.isfinite(logits[rr, negative])
        margins = {}
        for mode, emb in embeddings.items():
            visual = compare(emb, d['conditions'], support, rows, sy)
            margins[mode] = visual[rr, labels]-visual[rr, negative]
        for i in np.flatnonzero(valid):
            records.append(dict(draw=draw, row=int(rows[i]), true_class=int(labels[i]), rival=int(negative[i]),
                **{m: float(v[i]) for m, v in margins.items()}))
        for reference in ('learned', 'continued_ce'):
            a, b = margins[reference][valid], margins['joint_kernel'][valid]
            _, inv, sizes = np.unique(labels[valid], return_inverse=True, return_counts=True)
            weight = 1./(len(sizes)*sizes[inv])
            categories = {'both_correct': (a > 0) & (b > 0), 'both_wrong': (a <= 0) & (b <= 0),
                'rescued': (a <= 0) & (b > 0), 'harmed': (a > 0) & (b <= 0)}
            for name, mask in categories.items():
                groups.append(dict(draw=draw, reference=reference, group=name, n=int(mask.sum()),
                    loss_delta=float(np.sum(weight[mask]*(np.logaddexp(0, -b[mask]/.1)-np.logaddexp(0, -a[mask]/.1)))),
                    pair_ba_delta=100*float(np.sum(weight[mask]*((b[mask] > 0).astype(float)-(a[mask] > 0)))),
                    old_margin=float(a[mask].mean()) if mask.any() else np.nan,
                    new_margin=float(b[mask].mean()) if mask.any() else np.nan))
        ce_pred = p['base_predictions'][draw, modes.index('continued_ce')]
        bad = np.flatnonzero(ev & (y == 7) & (ce_pred == y) & (new != y))
        for row in bad:
            truth, rival = int(y[row]), int(new[row])
            ka, kb = matrices['continued_ce'], matrices['joint_kernel']
            aa, ab = duals['continued_ce'], duals['joint_kernel']
            refit = ka[row]@(ab-aa)
            geometry = (kb[row]-ka[row])@ab
            before = scores['continued_ce'][row, truth]-scores['continued_ce'][row, rival]
            after = scores['joint_kernel'][row, truth]-scores['joint_kernel'][row, rival]
            r, g = float(refit[truth]-refit[rival]), float(geometry[truth]-geometry[rival])
            np.testing.assert_allclose(before+r+g, after, atol=1e-12)
            cases.append(dict(draw=draw, row=int(row), sample_id=str(d['ids'][row]), true_class=truth, rival=rival,
                class_eval_n=int(np.sum(ev & (y == truth))), class_support_n=int(counts[truth]),
                ce_margin=float(before), joint_margin=float(after), coefficient_refit=r, geometry_change=g,
                full_predictions={m: int(p['predictions'][draw, modes.index(m), row]) for m in modes},
                old_route=bool(p['selected'][draw, modes.index('continued_ce'), row]),
                joint_route=bool(p['selected'][draw, modes.index('joint_kernel'), row])))
    pd.DataFrame(records).to_csv(OUT/'oss2_comparison_margins.csv', index=False)
    table = pd.DataFrame(groups)
    table.to_csv(OUT/'oss2_loss_accounting.csv', index=False)
    (OUT/'oss2_rare_class_cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    print(table.groupby(['reference', 'group'])[['n', 'loss_delta', 'pair_ba_delta', 'old_margin', 'new_margin']].mean().to_string())
    print(json.dumps(cases, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
