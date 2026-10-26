"""Actual-source full/cached parity after nonzero adapter training."""
import json
import numpy as np
import torch
from experiment import HERE, CONFIGS, old, setup, load, restore


@torch.no_grad()
def main():
    setup()
    snapshots = {int(x['obs_id']): x for x in json.loads(old.SNAPSHOTS.read_text(encoding='utf-8'))}
    checks = []
    for stage, gate in CONFIGS:
        if gate == 'none':
            continue
        table, arrays = load(stage)
        ids = np.flatnonzero(table.role.eq('fit_validation'))[:16]
        items = [old.native(table.iloc[i],snapshots,arrays['available'][i]) for i in ids]
        im, masks, valid = [torch.tensor(np.stack([v[k] for v in items]),device='cuda') for k in range(3)]
        av = torch.tensor(arrays['available'][ids],device='cuda')
        for seed in old.SEEDS:
            model = restore(stage,gate,seed)
            full = model(im,masks,valid,av)
            fresh = model.forward_from(model.prefix(im,stop=stage),masks,valid,av,start=stage)
            cache = model.forward_from(*old.batch(arrays,ids),start=stage)
            full_error = float((fresh-full).abs().max())
            half_error = float((cache-full).abs().max())
            assert full_error < 1e-5
            checks.append(dict(stage=stage,gate=gate,seed=seed,full_float32_prefix_error=full_error,
                               cached_half_error=half_error,prediction_changes=int((cache.argmax(1)!=full.argmax(1)).sum())))
            del model
    old.write(HERE/'path_checks.json',checks)
    print(checks,flush=True)


if __name__ == '__main__':
    main()
