"""Paired public M3 deletion at a fixed training endpoint."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'final_oss2_replication_20261029'
sys.path.insert(0, str(PARENT))
import run


def metrics(score, y):
    import numpy as np

    pred = score.argmax(1)
    recall = [(pred[y == c] == c).mean() for c in range(4)]
    shifted = score - score.max(1, keepdims=True)
    prob = np.exp(shifted) / np.exp(shifted).sum(1, keepdims=True)
    nll = -np.log(prob[np.arange(len(y)), y].clip(1e-20)).mean()
    brier = ((prob - np.eye(4)[y]) ** 2).sum(1).mean()
    return dict(oa=float((pred == y).mean() * 100), ba=float(np.mean(recall) * 100),
                recall=[float(r * 100) for r in recall], nll=float(nll), brier=float(brier))


def main(stage):
    import numpy as np

    run.runtime('cuda')
    rows, cache = run.cohorts(), run.Cache('cuda')
    queries, refs = rows['evaluation'], rows['references']
    bank = cache.bank(refs)
    expected_ids = np.array([r['id'] for r in queries])
    expected_y = np.array([r['label'] for r in queries])
    records = []
    for seed in (17, 29, 43):
        path = HERE / 'training' / f'no_m3_s{seed}' / stage / 'model.pt'
        net = run.api.load(path, 'cuda')
        assert net.arm == 'no_m3' and not net.m3
        with run.torch.no_grad():
            off = np.concatenate([cache.score(net, queries[i:i + 16], refs, bank).cpu().numpy()
                                  for i in range(0, len(queries), 16)])
        native_error = run.native_error(net, cache, run.sample_rows(rows), refs, bank)
        del net
        baseline_file = PARENT / 'predictions' / f'evaluation_full_{stage}_s{seed}.npz'
        with np.load(baseline_file) as source:
            assert np.array_equal(source['ids'], expected_ids)
            assert np.array_equal(source['y'], expected_y)
            assert source['classes'].astype(str).tolist() == list(run.CLASSES)
            full = source['score'].copy()
        correct_full = full.argmax(1) == expected_y
        correct_off = off.argmax(1) == expected_y
        entry = dict(seed=seed, full=metrics(full, expected_y), no_m3=metrics(off, expected_y),
                     full_only=int((correct_full & ~correct_off).sum()),
                     no_m3_only=int((~correct_full & correct_off).sum()),
                     both_correct=int((correct_full & correct_off).sum()),
                     both_wrong=int((~correct_full & ~correct_off).sum()),
                     native_cache_error=native_error)
        records.append(entry)
        np.savez_compressed(HERE / f'no_m3_{stage}_s{seed}.npz', score=off, ids=expected_ids, y=expected_y)
        print('PAIRED', json.dumps(entry, ensure_ascii=False), flush=True)
        run.torch.cuda.empty_cache()
    means = {}
    for arm in ('full', 'no_m3'):
        means[arm] = {metric: float(np.mean([r[arm][metric] for r in records]))
                      for metric in ('oa', 'ba', 'nll', 'brier')}
    result = dict(dataset='OSS2', scope='evaluation378', endpoint=stage, seeds=[17, 29, 43],
                  records=records, mean=means,
                  delta={metric: means['full'][metric] - means['no_m3'][metric]
                         for metric in ('oa', 'ba', 'nll', 'brier')},
                  no_target_selection=True, training_initialization_shared=True)
    target = HERE / ('results.json' if stage == 'fit' else 'results_adapt.json')
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPLETE', json.dumps(result['delta']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('fit', 'adapt'), default='fit')
    main(parser.parse_args().stage)
