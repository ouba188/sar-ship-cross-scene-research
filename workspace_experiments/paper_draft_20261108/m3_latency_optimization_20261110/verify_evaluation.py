"""Frozen-weight native-chip replay, not training or model selection."""
import json
import time
import numpy as np
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location('latency_pilot', Path(__file__).with_name('experiment.py'))
x = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x)

b, torch = x.b, x.torch


@torch.inference_mode()
def main():
    b.task.current.setup(b.DEVICE)
    b.task.tensor = b.resident_tensor
    torch.set_num_threads(2)
    rows, groups = b.e.prev.records()
    refs = [rows[i] for i in groups['references']]
    net = b.e.Network('full', 17, b.PRETRAIN, 4)
    path = b.ChipPath(net).eval()
    state = torch.load(b.STAGED / 'full_s17_lr1/readout/final.pt', map_location='cpu', weights_only=True, mmap=True)
    net.load_state_dict(state)
    del state
    b.e.configure(net, 3)
    path.eval().requires_grad_(False)
    ref = path.reference(b.prepare(refs))
    net.matcher = x.GraphMatcher(net.matcher)
    old = np.load(b.STAGED / 'full_s17_lr1/evaluation.npz')
    ids = groups['evaluation']
    assert np.array_equal(ids, old['ids'])
    scores, availability = [], []
    tick = time.perf_counter()
    for j, i in enumerate(ids):
        query = b.prepare([rows[i]])
        value = path(query, [rows[i]], ref, refs).cpu().numpy()[0]
        scores.append(value)
        availability.append(query['available'].cpu().numpy()[0])
        if (j + 1) % 63 == 0:
            print('VERIFIED', j + 1, 'of', len(ids), round(time.perf_counter() - tick, 1), flush=True)
    scores = np.array(scores)
    labels = np.array([rows[i]['label'] for i in ids])
    diff = abs(scores - old['score'])
    changes = np.flatnonzero(scores.argmax(1) != old['score'].argmax(1))
    av = np.array(availability)
    result = dict(n=len(ids), classes=4, seed=17, references=32,
        raw_chip_recomputed=True, query_cache=False, iterations=60,
        max_logit_error=float(diff.max()), mean_logit_error=float(diff.mean()),
        changed_predictions=len(changes),
        before=b.task.metrics(labels, old['score']), after=b.task.metrics(labels, scores),
        both_polarizations=int((av.sum(1) == 2).sum()),
        vv_only=int(((av[:, 0] == 1) & (av[:, 1] == 0)).sum()),
        vh_only=int(((av[:, 0] == 0) & (av[:, 1] == 1)).sum()),
        elapsed_including_io_seconds=time.perf_counter() - tick,
        source_predictions=str(b.STAGED / 'full_s17_lr1/evaluation.npz'),
        scope='numerical replay only; no accuracy improvement claimed')
    x.save('evaluation_parity.json', result)
    # Private sample-index predictions remain local.
    np.savez_compressed(x.ROOT / 'evaluation_predictions.npz', ids=ids, score=scores)
    assert result['max_logit_error'] < 1e-4 and not len(changes), result
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
