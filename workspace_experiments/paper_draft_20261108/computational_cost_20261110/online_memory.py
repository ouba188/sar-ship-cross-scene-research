"""Isolated online allocated memory: no original reference images kept on GPU."""
import gc
import json
import numpy as np
import pandas as pd
import benchmark as b

torch = b.torch


def cpu(data):
    return {k: v.cpu() for k, v in data.items()}


@torch.inference_mode()
def main():
    b.task.current.setup(b.DEVICE)
    b.task.tensor = b.resident_tensor
    torch.set_num_threads(2)
    results = []
    for classes in (4, 8):
        ids = json.loads((b.ROOT / f'inputs_{classes}class.json').read_text(encoding='utf-8'))
        if classes == 4:
            rows, _ = b.e.prev.records()
        else:
            path = b.PAPER.parents[0] / 'scene_aware_classification_20261018/model_closeout_20261023/spatial_m12_full_20261029/cohorts.json'
            groups = json.loads(path.read_text(encoding='utf-8'))
            rows = [row for group in groups.values() if isinstance(group, list) for row in group]
        lookup = {r['id']: r for r in rows}
        qrows = [lookup[i] for i in ids['query_ids']]
        rrows = [lookup[i] for i in ids['reference_ids']]
        qcpu, rcpu = cpu(b.prepare(qrows)), cpu(b.prepare(rrows))
        for arm in ('plain', 'se', 'full'):
            net = b.e.Network(arm, 17, b.PRETRAIN, classes)
            path = b.ChipPath(net).eval()
            checkpoint = b.STAGED / f'{arm}_s17_lr{0.3 if arm == "se" else 1}' / 'readout/final.pt'
            if classes == 4:
                state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
                net.load_state_dict(state, strict=True)
                del state
            b.e.configure(net, 3)
            path.eval().requires_grad_(False)
            ref = path.reference({k: v.to(b.DEVICE) for k, v in rcpu.items()}) if net.use3 else {}
            for batch in (1, 8):
                q = {k: v[:batch].to(b.DEVICE) for k, v in qcpu.items()}
                for _ in range(2):
                    path(q, qrows[:batch], ref, rrows)
                # The legacy forward stores this trace; it need not survive requests.
                net.readout_batch = None
                gc.collect()
                torch.cuda.synchronize()
                result = b.peak_memory(lambda: path(q, qrows[:batch], ref, rrows))
                result.update(classes=classes, arm=arm, batch=batch,
                              scope='resident model + current query inputs + reference descriptors; original reference images absent')
                results.append(result)
                net.readout_batch = None
                del q
            del net, path, ref
            gc.collect()
            torch.cuda.empty_cache()
    pd.DataFrame(results).to_csv(b.ROOT / 'online_memory.csv', index=False)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
