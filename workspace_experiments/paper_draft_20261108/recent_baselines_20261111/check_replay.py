"""Validation-only numerical replay diagnosis, without changing any checkpoint."""
import json
import numpy as np
import torch
from train import ROOT, Inputs, build, predict, write


def main():
    inputs = Inputs()
    lr = json.loads((ROOT / 'selected.json').read_text(encoding='utf-8'))['lr']
    rows = []
    for seed in (17, 29, 43):
        folder = ROOT / f'cpinet_s{seed}_lr{lr:g}'
        net = build(seed)
        net.load_state_dict(torch.load(folder / 'best.pt', map_location='cpu', weights_only=True))
        saved = np.load(folder / 'validation.npz')['score']
        for mode in ('training_precision', 'full_float32'):
            torch.backends.cudnn.allow_tf32 = mode == 'training_precision'
            torch.set_float32_matmul_precision('high' if mode == 'training_precision' else 'highest')
            current = predict(net, inputs, inputs.groups['validation'])
            row = dict(seed=seed, mode=mode, max_log_score_error=float(abs(current-saved).max()),
                       mean_log_score_error=float(abs(current-saved).mean()),
                       changed_predictions=int((current.argmax(1) != saved.argmax(1)).sum()))
            rows.append(row)
            print(row, flush=True)
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        del net
    write(ROOT / 'numerical_replay.json', rows)


if __name__ == '__main__':
    main()
