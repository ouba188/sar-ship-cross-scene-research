"""Aggregate locked OSS2 metadata and existing safe prepared caches; no models."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
CODE = ROOT.parent/'port_prototype_model_20261121'
sys.path.insert(0, str(CODE))
from sar_port.two_stage.cache import cache_key
from sar_port.two_stage.data import InputConfig
from sar_port.two_stage.geometry import GeometryConfig


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def csvread(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def fraction(n, total):
    return dict(count=n, denominator=total, percent=round(100*n/total, 3) if total else None)


def known(port):
    return bool(port) and port not in ('unknown', 'outside_150km')


def main():
    torch.set_num_threads(1)
    manifest = ROOT/'experiment_manifest_local.json'
    rows = read(manifest)
    preflight = {r['id']: r for r in read(ROOT/'geometry_preflight_local.json')}
    assignment = {r['sample_id']: r for r in csvread(WORK/'data_qualification_audit_20261122/oss2_development_split/split_assignment_local.csv')}
    source = {r['sample_id']: r for r in csvread(Path('E:/临时会话/oss2_port_prototypes_20261121/source_observations_local.csv'))}
    config = read(ROOT/'configs/disentangle_seed17.json')
    inp, geom = InputConfig(**config['input']), GeometryConfig(**config['geometry'])
    classes = sorted({r['class_name'] for r in rows})
    splits = {s: [r for r in rows if r['development_split'] == s] for s in ('train', 'val', 'test')}
    atom = {}
    atom_metadata = []
    missing_cache = 0
    for r in rows:
        assert r['development_split'] == assignment[r['id']]['development_split']
        key = cache_key(r, ROOT, inp, geom)
        path = ROOT/'prepared_cache_local'/f'{key}.pt'
        if not path.exists():
            missing_cache += 1
            continue
        payload = torch.load(path, map_location='cpu', weights_only=True)
        assert payload['key'] == key
        prepared = payload['prepared']['fields']
        geometry = prepared['geometry']['fields']
        assert int(geometry['canvas_size']) == preflight[r['id']]['canvas']
        atom[r['id']] = int(geometry['atom_count'])
        atom_metadata.append(dict(id=r['id'], class_name=r['class_name'], label=r['label'],
                                  development_split=r['development_split'],
                                  atom_count=int(geometry['atom_count']),
                                  region_count=len(geometry['area']),
                                  canvas_size=int(geometry['canvas_size']),
                                  axis_valid=bool(geometry['axis_valid'])))
        del payload, prepared, geometry
    assert not missing_cache, f'{missing_cache} exact caches missing; no geometry recomputed'
    (ROOT/'atom_metadata_local.json').write_text(json.dumps(atom_metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

    def counts(selected):
        values = [atom[r['id']] for r in selected]
        return dict(observations=len(selected), vessels=len({r['vessel_id'] for r in selected}),
                    absolute_orbits=len({r['group_id'] for r in selected}),
                    scenes=len({r['scene_id'] for r in selected}),
                    joint_vessel_orbit_components=len({assignment[r['id']]['joint_component_full'] for r in selected}),
                    atoms={k: fraction(sum(test(v) for v in values), len(values))
                           for k, test in [('1', lambda v:v==1), ('2', lambda v:v==2), ('more_than_2', lambda v:v>2)]},
                    exact_atom_counts=dict(sorted(Counter(values).items())),
                    port_proxies=dict(sorted(Counter(r['port'] for r in selected).items())))

    report = dict(date='2026-11-22', scope='Locked OSS2 development split; no model/prediction access',
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  geometry_source='Existing content-matched cache; torch.load(weights_only=True); no GEOS recomputation',
                  cached_geometry_records=len(atom), cache_misses=missing_cache,
                  splits={s:dict(**counts(rs), by_class={c:counts([r for r in rs if r['class_name']==c]) for c in classes}) for s,rs in splits.items()})
    report['pairwise_split_overlap'] = []
    for a,b in [('train','val'),('train','test'),('val','test')]:
        report['pairwise_split_overlap'].append(dict(splits=[a,b],
            shared_vessels=len({r['vessel_id'] for r in splits[a]} & {r['vessel_id'] for r in splits[b]}),
            shared_absolute_orbits=len({r['group_id'] for r in splits[a]} & {r['group_id'] for r in splits[b]})))
    train = splits['train']
    ports = sorted({r['port'] for r in train})
    report['train_by_port'] = {p:dict(**counts([r for r in train if r['port']==p]),
        class_counts={c:sum(r['port']==p and r['class_name']==c for r in train) for c in classes}) for p in ports}
    report['port_proxy_definition'] = dict(semantics='Nearest-anchor water-area proxies; not official port labels',
        nominal_radius_km=150, conservative_max_distance_km=75, conservative_min_second_nearest_gap_km=10,
        conservative_source_observations=sum(source[r['id']]['conservative_assignment']=='True' for r in train),
        validation_and_test_port_metadata='unknown in the locked manifest; not inferred or modified')
    support = {}
    q = {}
    for c in classes:
        category = [r for r in train if r['class_name']==c]
        eligible = []
        strict_counts = []
        conservative_queries = []
        conservative_supported = []
        for query in category:
            candidates = [r for r in category if known(query['port']) and known(r['port'])
                          and query['port'] != r['port'] and query['vessel_id'] != r['vessel_id']
                          and query['group_id'] != r['group_id']]
            if candidates:
                eligible.append(query)
            if known(query['port']):
                strict_counts.append(len({r['vessel_id'] for r in candidates}))
            if known(query['port']) and source[query['id']]['conservative_assignment']=='True':
                conservative_queries.append(query)
                if any(source[r['id']]['conservative_assignment']=='True' for r in candidates):
                    conservative_supported.append(query)
        named = sum(known(r['port']) for r in category)
        support[c] = dict(all_class_queries=fraction(len(eligible),len(category)),
            named_proxy_queries=fraction(len(eligible),named),
            conservative_proxy_queries=fraction(len(conservative_supported),len(conservative_queries)),
            named_source_ports=len({r['port'] for r in category if known(r['port'])}),
            cross_port_independent_support_vessels_min=min(strict_counts),
            cross_port_independent_support_vessels_median=float(np.median(strict_counts)),
            cross_port_independent_support_vessels_max=max(strict_counts))
        reviewed = [r for r in category if all(r[p].get('background_validated') and r[p].get('background_mask') for p in ('vv','vh'))]
        queries = [r for r in category if r.get('leakage_regularization_eligible')]
        q[c] = dict(reviewed_q=fraction(len(reviewed),len(category)),empirical_queries=fraction(len(queries),len(category)),
                    donor_count=sum(r['id'] in config['donor_ids'] for r in category),
                    reviewed_q_port_counts=dict(Counter(r['port'] for r in reviewed)),
                    empirical_query_port_counts=dict(Counter(r['port'] for r in queries)),
                    reviewed_q_atom_counts=dict(Counter(atom[r['id']] for r in reviewed)),
                    empirical_query_atom_counts=dict(Counter(atom[r['id']] for r in queries)))
    report['potential_same_class_cross_source_proxy_support'] = dict(
        definition='Full training metadata pool, known different port proxy, different vessel and absolute orbit. Potential support before learned-bank selection.',
        by_class=support)
    report['q_supervision_by_class'] = q
    report['q_supervision_total'] = dict(reviewed_q=fraction(sum(v['reviewed_q']['count'] for v in q.values()), len(train)),
        empirical_queries=fraction(sum(v['empirical_queries']['count'] for v in q.values()), len(train)),
        donor_count=8, controlled_qualified_count=sum(bool(r.get('leakage_controlled')) for r in train),
        pixels_per_polarization_per_reviewed_q=128)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)+'\n'
    assert 'pub2_opensarship2_' not in encoded
    (ROOT/'aggregate_data_composition.json').write_text(encoded, encoding='utf8')
    print(json.dumps({k:report[k] for k in ('cached_geometry_records','pairwise_split_overlap','potential_same_class_cross_source_proxy_support','q_supervision_by_class')},ensure_ascii=False,indent=2))
    for s in splits:
        print(s,report['splits'][s]['atoms'])


if __name__ == '__main__':main()
