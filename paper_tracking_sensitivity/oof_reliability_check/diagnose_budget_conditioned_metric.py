"""Read-only source fixed-selection diagnosis after the declared pilot."""

import numpy as np
import torch

from budget_conditioned_metric import training_loss, gate_policy
from train_budget_conditioned_metric import OUT, DEVICE, ARMS, read, write, load_episode, adapted


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    with torch.no_grad():
        for port in read(OUT / 'PROTOCOL.json')['ports']:
            folder = OUT / 'ports' / port
            with np.load(folder / 'gate.npz') as saved:
                gate = {k: torch.tensor(saved[k], dtype=torch.double, device=DEVICE) for k in saved.files}
            with np.load(folder / 'learned_weights.npz') as saved:
                theta = {arm: torch.tensor(saved[arm + '_theta'], device=DEVICE) for arm in ARMS}
            for descriptor in read(folder / 'PREPARED.json')['episodes']:
                ep = load_episode(descriptor['path'])
                original, query, labels = ep['expert'], ep['query'], ep['labels']
                old_selected, _ = gate_policy(ep['base'], original, gate, labels[ep['support']], query, ep['rows'])
                old_loss = float(training_loss(original, labels, query, old_selected, True))
                _, inverse, counts = torch.unique(labels[query], return_inverse=True, return_counts=True)
                weight = 1 / (len(counts) * counts[inverse].double())
                old_correct = original[query].argmax(1) == labels[query]
                for arm in ARMS:
                    scores = adapted(ep, theta[arm])
                    selected, _ = gate_policy(ep['base'], scores, gate, labels[ep['support']], query, ep['rows'])
                    new_correct = scores[query].argmax(1) == labels[query]
                    fixed_loss = float(training_loss(scores, labels, query, old_selected, True))
                    current_loss = float(training_loss(scores, labels, query, selected, True))
                    kept = old_selected[query]
                    records.append(dict(port=port, held_source=descriptor['port'], split=descriptor['split'], arm=arm,
                        original_objective=old_loss, fixed_old_selection_objective=fixed_loss,
                        current_selection_objective=current_loss,
                        score_learning_objective_delta=fixed_loss-old_loss,
                        selection_objective_delta=current_loss-fixed_loss,
                        fixed_selection_correct_ba_delta=float((weight[kept] *
                            (new_correct[kept].double()-old_correct[kept].double())).sum()),
                        fixed_selection_rescues=int((kept & ~old_correct & new_correct).sum()),
                        fixed_selection_harms=int((kept & old_correct & ~new_correct).sum()),
                        selection_retention=float((selected & old_selected).sum() / old_selected.sum().clamp(min=1)),
                        expert_changed=int((scores[query].argmax(1) != original[query].argmax(1)).sum()),
                        expert_score_delta_rms=float((scores[query]-original[query]).square().mean().sqrt())))
            print('FIXED-SET DIAGNOSED ' + port, flush=True)
    write(OUT / 'source_fixed_selection_diagnosis.json', records)


if __name__ == '__main__':
    main()
