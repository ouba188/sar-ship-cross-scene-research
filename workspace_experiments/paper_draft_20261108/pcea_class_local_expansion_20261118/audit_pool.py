"""Metadata-only census for frozen PCEA evaluation expansion."""
from collections import Counter
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
STORE = Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def passage(row):
    if 'pass_group' in row:
        return '_'.join(row['pass_group'].split('_')[:2])
    parts = row['product'].split('_')
    return '_'.join((parts[0], parts[6]))


def vessel(row):
    return row['mmsi'].removesuffix('.0')


def describe(rows):
    counts = Counter(int(r.get('class_id', r.get('label'))) for r in rows)
    return dict(n=len(rows), vessels=len({vessel(r) for r in rows}),
                passages=len({passage(r) for r in rows}),
                ports=dict(Counter(r['port'] for r in rows)),
                classes=[counts[i] for i in range(8)])


def main():
    rows, groups = read(STORE/'manifest.json'), read(STORE/'groups.json')
    with (WORK/'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv').open(
            encoding='utf-8-sig', newline='') as handle:
        corpus = list(csv.DictReader(handle))
    used = {r['corpus_global_row'] for r in rows}
    source = [rows[i] for i in groups['source_fit']+groups['source_adapt']]
    validation = [rows[i] for i in groups['validation']]
    test = [rows[i] for i in groups['evaluation']]
    source_vessels = {vessel(r) for r in source}
    source_passages = {passage(r) for r in source}
    held_vessels = {vessel(r) for r in validation+test}
    test_passages = {passage(r) for r in test}
    validation_passages = {passage(r) for r in validation}
    stages = {}
    candidates = [(i, r) for i, r in enumerate(corpus) if i not in used]
    stages['outside_current_manifest'] = describe([r for _, r in candidates])
    for name, predicate in (
        ('valid_identity', lambda r: vessel(r).isdigit()),
        ('unseen_source_vessel', lambda r: vessel(r) not in source_vessels),
        ('unseen_source_passage', lambda r: passage(r) not in source_passages),
        ('no_validation_or_test_vessel', lambda r: vessel(r) not in held_vessels),
        ('no_test_passage', lambda r: passage(r) not in test_passages),
    ):
        candidates = [(i, r) for i, r in candidates if predicate(r)]
        stages[name] = describe([r for _, r in candidates])
    eligible_ids = {int(r['id'][5:]) for r in read(STORE/'exclusions.json')
                    if r['id'].startswith('full_') and r['reason'] in
                    ('heldout_ship', 'heldout_orbit', 'source_adapt_ship_overlap')}
    eligible = [(i, r) for i, r in candidates if i in eligible_ids]
    stages['existing_usable_target_view'] = describe([r for _, r in eligible])
    new_passages = [(i, r) for i, r in eligible if passage(r) not in validation_passages]
    stages['also_new_validation_passage'] = describe([r for _, r in new_passages])
    summary = dict(corpus=len(corpus), current_manifest=len(rows), source=describe(source),
                   validation=describe(validation), reserved_test=describe(test), stages=stages,
                   pixels_loaded=False, model_loaded=False, predictions_computed=False,
                   quality_rule='Reuse existing accepted target view; no threshold relaxation',
                   prior_development_exposure='Must separately audit legacy cohorts before inference')
    (ROOT/'pool_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT/'candidates_local.json').write_text(json.dumps(
        [dict(corpus_global_row=i, **r) for i, r in eligible], ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
