"""Inspect the existing source on/off flips without training or inference."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('routed_source_cases', ROOT/'analyze.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def main():
    context = a.metadata()
    info = context['inner']
    cache, candidates = {}, []
    for seed in a.SEEDS:
        for mode in ('on', 'off'):
            z = a.native.load_prediction(a.cache_path('inner', 'routed', seed, mode),
                                        info['ids'], info['rows'], require_label=True)
            cache['inner', 'routed', seed, mode] = dict(ids=info['ids'], label=info['label'], score=z)
        on, off = (cache['inner', 'routed', seed, mode] for mode in ('on', 'off'))
        delta = a.strength.margin(on['score'], on['label'])-a.strength.margin(off['score'], off['label'])
        for kind, mask in a.base.outcome_masks(on, off).items():
            if kind not in ('rescued', 'harmed'):
                continue
            for j in a.np.flatnonzero(mask):
                i = int(info['ids'][j])
                candidates.append(dict(split='inner', seed=seed, comparison='routed_vs_off', kind=kind,
                    before_arm='routed', before_mode='off', index=i, position=int(j),
                    corpus_global_row=info['rows'][i]['corpus_global_row'], priority=0,
                    magnitude=float(abs(delta[j]))))
    a.ROOT = ROOT/'source_case_review_local'
    a.ROOT.mkdir(exist_ok=True)
    coverage = a.gallery(candidates, cache, context)
    path = a.ROOT/'gallery/index.html'
    text = path.read_text(encoding='utf-8').replace('External diagnostic examples', 'Source development examples')
    path.write_text(text, encoding='utf-8')
    print(coverage)


if __name__ == '__main__':
    main()
