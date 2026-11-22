"""Join verified alignment, reviewed Q and frozen split; never enable training."""
from pathlib import Path
import csv
import copy
import json

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'oss2_development_dataset'


def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def main():
    OUT.mkdir(exist_ok=True)
    original = json.loads(Path('E:/临时会话/m1_coverage_20261101/manifest.json').read_text(encoding='utf8'))
    restored = {r['id']: r for r in json.loads((ROOT/'oss2_alignment/manifest_alignment_restored_local.json').read_text(encoding='utf8'))}
    q_records = {r['sample_id']: r for r in json.loads((ROOT/'oss2_q_pilot/q_qualification_local.json').read_text(encoding='utf8'))}
    with (ROOT/'oss2_development_split/split_assignment_local.csv').open(encoding='utf-8-sig', newline='') as f:
        split = {r['sample_id']: r for r in csv.DictReader(f)}
    groups = {}
    for old in original:
        sid = old['id']; role = split[sid]['development_split']
        r = copy.deepcopy(restored.get(sid, old))
        r.update(development_split=role, training_qualified=False, leakage_controlled=False,
                 background_validated=False, qualification_scope='development_data_draft_not_training_ready')
        r['alignment_audit_scope'] = 'verified_source_shared_grid' if sid in restored else 'pending_original_pixel_chain'
        if sid not in restored:
            r['polarizations_coregistered'] = False
        if role == 'test':
            r['test_exposure'] = 'legacy_exposed_regression_not_unseen'
        for p in ('vv', 'vh'):
            if not r.get(p):
                continue
            r[p]['background_validated'] = False
            r[p]['training_qualified'] = False
            r[p]['leakage_controlled'] = False
            if sid not in restored:
                r[p]['alignment_valid'] = False
        if sid in q_records:
            q = q_records[sid]
            assert role == 'train'
            r['q_review'] = q
            r['background_validated'] = q['background_validated']
            for p in ('vv', 'vh'):
                r[p]['background_mask'] = q['background_mask']
                r[p]['background_validated'] = q['background_validated']
                r[p]['q_review_scope'] = q['review_scope']
        groups.setdefault(role, []).append(r)
    assert {k:len(v) for k,v in groups.items()} == {'train':719,'val':228,'test':378,'quarantine_historical_selection':194}
    for role, rows in groups.items():
        assert all(r['training_qualified'] is False and r['leakage_controlled'] is False for r in rows)
        save(f'{role}_qualification_draft_local.json', rows)
    save('summary.json', dict(counts={k:len(v) for k,v in groups.items()},
                             train_dual_polarization=sum(bool(r['vv'] and r['vh']) for r in groups['train']),
                             reviewed_background_records=sum(r['background_validated'] for r in groups['train']),
                             validation_test_alignment_pending=606, training_qualified=0, leakage_controlled=0,
                             note='No training entry point is invoked. Current loader does not enforce these draft status fields.'))
    print((OUT/'summary.json').read_text(encoding='utf8'))


if __name__ == '__main__':
    main()
