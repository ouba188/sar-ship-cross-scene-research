"""Frozen current M3 tails on their own purged source rows, no training."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from neural_complement import ComplementTail


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
PARENT = ROOT/'module_completion_20261008/ports_projection'
OUT = ROOT/'evidence_aware_handoff_20261008'
WS = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline')


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inventory = json.loads((ROOT/'oof_base_adapt_results_20261007/POLICIES_SAVED.json').read_text(encoding='utf-8'))
    prefix = np.load(ROOT/'neural_complement_pilot_20261007/source_layer3.npy', mmap_mode='r')
    checkpoint = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    model = ComplementTail(checkpoint['backbone']).cuda().eval()
    for item in inventory:
        port = item['port']
        folder = OUT/'source_features'/port
        folder.mkdir(parents=True, exist_ok=True)
        if (folder/'COMPLETE.json').exists():
            continue
        start = time.monotonic()
        with np.load(PARENT/port/'source_rows.npz') as saved:
            rows = saved['source_rows']
        model.layer4.load_state_dict(torch.load(PARENT/port/'joint/layer4.pt',
                                               map_location='cpu', weights_only=True))
        values = np.empty((len(rows), 512), dtype=np.float32)
        with torch.inference_mode():
            for lo in range(0, len(rows), 128):
                selected = rows[lo:lo+128]
                pixels = torch.from_numpy(np.array(prefix[selected])).cuda().float()
                values[lo:lo+len(selected)] = model.features(pixels).cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite frozen source features')
        np.save(folder/'features.npy', values)
        np.save(folder/'rows.npy', rows)
        receipt = dict(port=port, rows=len(rows), dimensions=512, seconds=time.monotonic()-start,
            frozen_tail=str(PARENT/port/'joint/layer4.pt'), training=False,
            source_encoder_access='Tail trained on these outer-source episodes; not inner-port-held encoder')
        (folder/'COMPLETE.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(f'SOURCE FEATURES {port}: {len(rows)} rows, {receipt["seconds"]:.1f}s', flush=True)
    print('SOURCE FEATURE EXTRACTION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
