"""Metadata-only feasibility audit for source reference replacement."""
from pathlib import Path
from collections import Counter
import json

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
STORE = Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def passage(row):
    return '_'.join(row['pass_group'].split('_')[:2])


def main():
    rows = read(STORE/'manifest.json')
    groups = read(STORE/'groups.json')
    compact = read(PAPER/'m3_coarse_supplement_20261118/evaluation_inputs/manifest_local.json')
    fit = read(PAPER/'pcea_routed_correction_20261118/split_local.json')['fit']
    query, refs = compact[:358], compact[358:]
    qships = {r['mmsi'] for r in query}
    qpasses = {passage(r) for r in query}
    rships = {r['mmsi'] for r in refs}
    qports = {r['port'] for r in query}
    rpasses = {passage(r) for r in refs}
    pool = [i for i in fit if rows[i]['mmsi'] not in qships | rships
            and passage(rows[i]) not in qpasses | rpasses]
    by_class = []
    for c in range(8):
        candidates = [rows[i] for i in pool if rows[i]['label'] == c]
        by_class.append(dict(label=c, n=len(candidates), vessels=len({r['mmsi'] for r in candidates}),
            ports=dict(Counter(r['port'] for r in candidates)),
            outside_query_ports=sum(r['port'] not in qports for r in candidates)))
    report = dict(source_fit=len(fit), eligible=len(pool),
        query_ports=dict(Counter(r['port'] for r in query)),
        query_classes=dict(Counter(r['label'] for r in query)),
        reference_classes=dict(Counter(r['label'] for r in refs)),
        references=[dict(label=r['label'], port=r['port'], both=bool(r.get('vv') and r.get('vh'))) for r in refs],
        by_class=by_class, original_ref_ids_match=[r['id'] for r in refs] == [rows[i]['id'] for i in groups['references']])
    save(ROOT/'feasibility.json', report)
    save(ROOT/'eligible_local.json', pool)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
