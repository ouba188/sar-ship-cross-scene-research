"""Four bounded native-chip cases for the matched OSS2 fit ablation."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'final_oss2_replication_20261029'
sys.path.insert(0, str(PARENT))
import analysis

IDS = ('pub2_opensarship2_000312', 'pub2_opensarship2_000694',
       'pub2_opensarship2_000145', 'pub2_opensarship2_000132')
STATES = ('full_only', 'no_m3_only', 'both_correct', 'both_wrong')


def main(stage):
    root = HERE / ('cases' if stage == 'fit' else 'cases_adapt')
    root.mkdir(exist_ok=True)
    rows = analysis.base.read_json(PARENT / 'manifests/evaluation.json')
    refs = analysis.base.read_json(PARENT / 'manifests/references.json')
    lookup = {r['id']: r for r in rows}
    native = analysis.full.module_at('oss2_native_m3_ablation', analysis.PACKAGE / 'sar_m123/input.py')
    scores = {}
    for seed in analysis.SEEDS:
        with np.load(HERE / f'no_m3_{stage}_s{seed}.npz') as source:
            scores['no_m3', seed] = dict(zip(source['ids'].astype(str), source['score']))
        with np.load(PARENT / 'predictions' / f'evaluation_full_{stage}_s{seed}.npz') as source:
            scores['full', seed] = dict(zip(source['ids'].astype(str), source['score']))
    if stage == 'adapt':
        candidates = {state: [] for state in STATES}
        for id_, query in lookup.items():
            truth = query['label']
            for seed in analysis.SEEDS:
                is_full = int(scores['full', seed][id_].argmax()) == truth
                is_off = int(scores['no_m3', seed][id_].argmax()) == truth
                state = ('both_correct' if is_full else 'no_m3_only') if is_off else (
                    'full_only' if is_full else 'both_wrong')
                candidates[state].append((id_, seed))
        chosen, used = [], set()
        for state in STATES:
            candidate = next(((id_, seed) for id_, seed in candidates[state] if id_ not in used), None)
            if candidate is not None:
                chosen.append((state, candidate))
                used.add(candidate[0])
    else:
        chosen = [(state, (id_, 17)) for id_, state in zip(IDS, STATES)]
    cases = []
    for number, (state, (id_, chosen_seed)) in enumerate(chosen, 1):
        query = lookup[id_]
        truth = query['label']
        full = scores['full', chosen_seed][id_]
        off = scores['no_m3', chosen_seed][id_]
        is_full = int(full.argmax()) == truth
        is_off = int(off.argmax()) == truth
        assert (is_full, is_off) == {
            'full_only': (True, False), 'no_m3_only': (False, True),
            'both_correct': (True, True), 'both_wrong': (False, False)}[state]
        rival = int(off.argmax()) if is_full else int(full.argmax())
        if rival == truth:
            rival = int(np.argsort(full)[-2])
        allowed = [r for r in refs if analysis.base.eligible(query, r)]
        same = [r for r in allowed if r['label'] == truth][:2]
        other = [r for r in allowed if r['label'] == rival][:2]
        selected = [query, *same, *other]
        roles = ['QUERY'] + ['TRUE REF'] * len(same) + ['RIVAL REF'] * len(other)
        panels = {}
        for pol in ('vv', 'vh'):
            path = root / f'case_{number:02d}_{pol}.png'
            analysis.base.panel(selected, roles, pol, path, native)
            panels[pol] = str(path)
        cases.append(dict(id=id_, state=state, selected_seed=chosen_seed, label=truth, rival=rival,
                          selected_refs=[r['id'] for r in selected[1:]], panels=panels,
                          predictions={str(seed): {arm: int(scores[arm, seed][id_].argmax())
                                                   for arm in ('full', 'no_m3')}
                                       for seed in analysis.SEEDS}))
    (root / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    print('NATIVE_CASES', len(cases), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('fit', 'adapt'), default='fit')
    main(parser.parse_args().stage)
