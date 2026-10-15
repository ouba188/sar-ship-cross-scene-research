"""Bounded, label-free audit of existing chip-local OSM assets. No training."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

ROOT = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline')
OUT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
DATA = ART / 'eight_class_adaptive_20260916'
CURRENT = DATA / 'knowledge_real_recovered_20260917_pm5'
CACHE = Path('E:/Hermes/paper_tracking/sensitivity/pred_cache')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from local_relation_gate_features import LAYERS, _number, _osm_columns, derive_relations


def read_columns(path, columns):
    return pd.read_csv(path, usecols=lambda c: c in columns, low_memory=False,
                       dtype=str).fillna('')


def flag(series):
    return series.astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy()


def unique_table(table, keys, scope, collisions):
    """Exclude every side of a collision, including identical duplicate records."""
    duplicate = table.duplicated(keys, keep=False)
    for key, group in table.loc[duplicate].groupby(keys, dropna=False, sort=False):
        collisions.append(dict(scope=scope, key=repr(key), rows=len(group),
                               ids='|'.join(group.get('instance_id', group.get('pair_id', pd.Series([], dtype=str))).astype(str))))
    populated = table[keys].ne('').all(axis=1)
    return table.loc[~duplicate & populated].copy()


def aligned(table, key, ids, scope, collisions):
    table = unique_table(table, [key], scope, collisions)
    return table.set_index(key).reindex(pd.Index(ids)).reset_index(drop=True)


def resolve_source(text):
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path / 'osm_topology_features_model_visible.csv' if path.is_dir() else path


def role_mapping():
    tree = ast.parse((ROOT / 'run_crca_development.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ROLE_LAYERS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Existing CRCA role definitions missing')


def image_keys(frame):
    result = frame[['instance_id', 'product']].copy()
    for pol in ('vv', 'vh'):
        result[pol] = frame['image_relpath_' + pol].map(lambda s: PureWindowsPath(s).name)
    return result


def old_roles(rows, collisions):
    folder = ART / 'crca_3class_10port_development_v1'
    columns = ['instance_id', 'product', 'image_relpath_vv', 'image_relpath_vh']
    old = read_columns(folder / 'manifest.csv', columns)
    old['_old_row'] = np.arange(len(old))
    source = image_keys(old).assign(_old_row=old._old_row).rename(columns={'instance_id': 'old_instance_id'})
    target = image_keys(rows).assign(row_index=np.arange(len(rows)))
    keys = ['product', 'vv', 'vh']
    source = unique_table(source, keys, 'old_manifest_exact_filenames', collisions)
    target = unique_table(target, keys, 'current_manifest_exact_filenames', collisions)
    joined = target.merge(source, on=keys, how='left', validate='one_to_one').set_index('row_index').reindex(range(len(rows)))
    matched = joined._old_row.notna().to_numpy()
    supported = np.zeros(len(rows), bool)
    with np.load(folder / 'role_inputs.npz', allow_pickle=False) as pack:
        quality = pack['quality']
        supported[matched] = quality[joined.loc[matched, '_old_row'].astype(int)].sum(axis=1) > 0
    audit = rows[['instance_id', 'port']].copy()
    audit['old_instance_id'] = joined.old_instance_id.fillna('').to_numpy()
    audit['exact_product_and_two_filenames'] = matched
    audit['old_role_supported'] = supported
    # IDs are regenerated across datasets. Report the danger without using them as a join.
    same_id = old.set_index('instance_id').reindex(rows.instance_id)
    same_id_present = same_id['product'].notna().to_numpy()
    old_key = image_keys(same_id.reset_index().fillna(''))
    new_key = image_keys(rows)
    compatible = np.ones(len(rows), bool)
    for c in ('product', 'vv', 'vh'):
        compatible &= old_key[c].to_numpy() == new_key[c].to_numpy()
    audit['same_id_different_chip'] = same_id_present & ~compatible
    return audit


def metadata_for(path, selected, rows, collisions):
    """Follow only sidecars belonging to the recorded selected source."""
    folder = path.parent
    columns = ['instance_id', 'product', 'coordinate_kind', 'coordinate_source',
               'map_date', 'source_kind', 'gpkg', 'map_path', 'provenance',
               'source_id', 'map_missing_reason', 'full_10p1km_footprint_supported',
               'map_neighborhood_covered', 'map_reuse_eligible']
    candidates = [folder / n for n in ('map_recovery_sources_private.csv',
                  'map_support_provenance.csv', 'support_provenance.csv')]
    if folder.name == 'class_neutral_osm_v1':
        candidates = [ART / 'class_neutral_coordinates_v1/coordinates_private.csv']
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        return pd.DataFrame(index=range(len(selected))), '', np.zeros(len(selected), bool)
    table = read_columns(found, columns)
    table = aligned(table, 'instance_id', selected.instance_id, str(found), collisions)
    product_ok = (table['product'].fillna('').to_numpy() == selected['product'].to_numpy())
    footprint = np.zeros(len(table), bool)
    for col in ('full_10p1km_footprint_supported', 'map_neighborhood_covered', 'map_reuse_eligible'):
        if col in table:
            footprint |= flag(table[col])
    if folder.name == 'real_k_asset_audit_20260916':
        # This exact protocol requires the 10.1 km footprint inside both boundaries.
        footprint |= table.map_missing_reason.eq('recovered').to_numpy()
    if folder.name == 'class_neutral_osm_v1':
        table['coordinate_kind'] = 'inherited_verified_sample_center'
        table['map_date'] = '2025-01-01'
        table['source_kind'] = 'historical_pbf_current_provider_boundary'
    return table, str(found), footprint & product_ok


def reconstruct(rows, provenance, roles, collisions):
    columns = _osm_columns() + [v for role in roles for v in ('role_' + role, 'role_' + role + '_availability')]
    raw = pd.DataFrame(np.nan, index=range(len(rows)), columns=columns)
    detail = rows[['instance_id', 'port', 'product', 'data_origin']].copy()
    for col in ('raw_source', 'raw_pair_id', 'join_method', 'source_metadata', 'coordinate_kind',
                'coordinate_source', 'map_date', 'source_kind', 'gpkg', 'provenance'):
        detail[col] = ''
    detail['raw_join_verified'] = False
    detail['footprint_verified'] = False
    inventory = []
    for label in sorted(provenance.osm_source.dropna().unique()):
        if not label:
            continue
        index = np.flatnonzero(provenance.osm_source.eq(label))
        selected = rows.iloc[index]
        path = resolve_source(label)
        if not path.is_file():
            inventory.append(dict(path=str(path), selected_rows=len(index), exists=False))
            continue
        table = read_columns(path, ['pair_id', *columns])
        native_ids = selected.instance_id.to_numpy()
        method = 'current_instance_id_to_pair_id'
        join_ok = np.ones(len(index), bool)
        meta, meta_path, footprint = metadata_for(path, selected, rows, collisions)
        if path.parent.name == 'local_knowledge_v4':
            # A source-object identity join avoids even the opaque class-containing old key.
            keys = ['product', 'source_tiff_vv', 'source_tiff_vh', 'obj_uid_vv', 'obj_uid_vh']
            private_path = path.parent / 'local_knowledge_by_chip.csv'
            private = read_columns(private_path, ['pair_id', 'coordinate_method', 'coordinate_available', *keys])
            private = unique_table(private, keys, str(private_path) + ':source_object', collisions)
            current = selected[['instance_id', *keys]].copy()
            current = unique_table(current, keys, 'current_returned_source_objects', collisions)
            bridge = current.merge(private, on=keys, how='left', validate='one_to_one')
            bridge = bridge.set_index('instance_id').reindex(selected.instance_id).reset_index()
            native_ids = bridge.pair_id.fillna('').to_numpy()
            join_ok = native_ids != ''
            meta = bridge.rename(columns={'coordinate_method': 'coordinate_kind'})
            meta['coordinate_source'] = str(private_path)
            meta_path = str(private_path)
            method = 'exact_product_two_source_tiffs_two_object_uids'
        elif path.parent.name == 'shared_assets_local_knowledge_v6':
            source_rows = read_columns(path.parent / 'manifest_order.csv',
                                      ['instance_id', 'product', 'image_relpath_vv', 'image_relpath_vh'])
            source_rows = aligned(source_rows, 'instance_id', selected.instance_id,
                                  str(path.parent / 'manifest_order.csv'), collisions).fillna('')
            join_ok = source_rows['product'].to_numpy() == selected['product'].to_numpy()
            for pol in ('vv', 'vh'):
                join_ok &= source_rows['image_relpath_' + pol].map(lambda s: PureWindowsPath(s).name).to_numpy() == selected['image_relpath_' + pol].map(lambda s: PureWindowsPath(s).name).to_numpy()
            meta_path = str(path.parent / 'manifest_order.csv')
            method = 'instance_id_product_two_filenames'
        elif 'product' in meta:
            join_ok &= meta['product'].fillna('').to_numpy() == selected['product'].to_numpy()
        else:
            join_ok[:] = False
        good_table = unique_table(table, ['pair_id'], str(path), collisions)
        join_ok &= pd.Series(native_ids).isin(good_table.pair_id).to_numpy()
        block = good_table.set_index('pair_id').reindex(native_ids).reindex(columns=columns)
        block = block.apply(pd.to_numeric, errors='coerce').to_numpy()
        raw.loc[index[join_ok], columns] = block[join_ok]
        detail.loc[index, 'raw_source'] = str(path)
        detail.loc[index, 'raw_pair_id'] = native_ids
        detail.loc[index, 'join_method'] = method
        detail.loc[index, 'source_metadata'] = meta_path
        detail.loc[index, 'raw_join_verified'] = join_ok
        detail.loc[index, 'footprint_verified'] = footprint & join_ok
        for col in ('coordinate_kind', 'coordinate_source', 'map_date', 'source_kind', 'gpkg', 'provenance'):
            source_col = 'map_path' if col == 'gpkg' and 'map_path' in meta else col
            if source_col in meta:
                detail.loc[index, col] = meta[source_col].fillna('').to_numpy()
        inventory.append(dict(path=str(path), selected_rows=len(index), exists=True,
                              source_rows=len(table), unique_source_keys=len(good_table),
                              verified_join_rows=int(join_ok.sum()), footprint_rows=int((footprint & join_ok).sum()),
                              metadata=meta_path, join_method=method))
    return raw, detail, pd.DataFrame(inventory)


def role_features(raw, roles, layer_known):
    values = np.zeros((len(raw), len(roles), 4), np.float32)
    known = np.zeros(values.shape, bool)
    quality = np.zeros((len(raw), len(roles)), np.float32)
    for r, (role, layers) in enumerate(roles.items()):
        present = layer_known[:, [LAYERS.index(layer) for layer in layers]]
        denominator = present.sum(1).clip(1)
        stored_quality = _number(raw, 'role_' + role + '_availability')
        evidence = _number(raw, 'role_' + role)
        quality[:, r] = present.mean(1)
        known[:, r, :] = present.any(1)[:, None]
        known[:, r, 0] &= np.isfinite(evidence) & np.isclose(stored_quality, quality[:, r])
        values[:, r, 0] = np.where(known[:, r, 0], evidence, 0)
        d = np.column_stack([_number(raw, layer + '_distance_km') for layer in layers])
        values[:, r, 1] = np.where(present, np.exp(-d / 2), 0).sum(1) / denominator
        for c, radius, scale in ((2, 1, np.log1p(20)), (3, 3, np.log1p(50))):
            count = np.column_stack([_number(raw, layer + f'_count_{radius}km_log1p') for layer in layers])
            values[:, r, c] = np.where(present, np.clip(count / scale, 0, 1), 0).sum(1) / denominator
    return np.where(known, values, 0), known, quality


def query_coverage(rows, detail):
    summaries, query_rows = [], []
    for port in sorted(rows.port.unique()):
        file = CACHE / (port + '.npz')
        base = dict(port=port, manifest_rows=int(rows.port.eq(port).sum()), cache=str(file))
        if not file.is_file():
            summaries.append(dict(**base, status='missing_cache'))
            continue
        with np.load(file, allow_pickle=False) as pack:
            if 'skipped' in pack and bool(pack['skipped']):
                summaries.append(dict(**base, status='skipped_outer', query_rows=0))
                continue
            # Never access y_ids, logits, ranking, or other predictive arrays.
            q, s, ports = pack['qpos'], pack['spos'], pack['ports_all']
            if not np.array_equal(ports, rows.port.to_numpy()):
                raise ValueError(f'Cache port ordering mismatch: {port}')
            if not np.array_equal(np.sort(q), np.flatnonzero(rows.port.eq(port))):
                raise ValueError(f'Cache query positions mismatch: {port}')
            if len(np.unique(s)) != len(s) or np.intersect1d(q, s).size or (s < 0).any() or (s >= len(rows)).any():
                raise ValueError(f'Invalid cached source positions: {port}')
            item = dict(**base, status='aligned', query_rows=len(q), source_rows=len(s))
            for scope, indices in (('query', q), ('source', s)):
                for column in ('historical_supported', 'bounded_supported', 'bounded_near_infra_3km'):
                    item[scope + '_' + column] = int(detail.iloc[indices][column].sum())
            summaries.append(item)
            qframe = detail.iloc[q][['instance_id', 'bounded_supported', 'historical_supported']].copy()
            qframe.insert(0, 'row_index', q)
            qframe.insert(0, 'outer_port', port)
            query_rows.append(qframe)
    return pd.DataFrame(summaries), pd.concat(query_rows, ignore_index=True)


def run(output=OUT):
    output.mkdir(parents=True, exist_ok=True)
    columns = ['instance_id', 'port', 'product', 'data_origin', 'image_relpath_vv',
               'image_relpath_vh', 'source_tiff_vv', 'source_tiff_vh', 'obj_uid_vv', 'obj_uid_vh']
    rows = read_columns(DATA / 'manifest.csv', columns)
    if rows.instance_id.eq('').any() or not rows.instance_id.is_unique:
        raise ValueError('Manifest row IDs must be unique and populated')
    collisions = []
    provenance_path = CURRENT / 'relation_sources_private.csv'
    provenance = aligned(read_columns(provenance_path, ['instance_id', 'product', 'osm_source']),
                         'instance_id', rows.instance_id, str(provenance_path), collisions)
    good_product = provenance['product'].eq(rows['product'])
    provenance.loc[~good_product, 'osm_source'] = ''
    roles = role_mapping()
    raw, detail, inventory = reconstruct(rows, provenance, roles, collisions)
    empty = pd.DataFrame(index=range(len(rows)))
    derived, masks, names = derive_relations(empty, empty, raw, np.zeros(len(rows), bool))
    osm_idx = [i for i, name in enumerate(names) if name.startswith('osm_')]
    names = [names[i] for i in osm_idx]
    historical_values = derived[:, osm_idx].astype(np.float32)
    historical_known = masks[:, osm_idx]
    with np.load(CURRENT / 'relations.npz', allow_pickle=False) as pack:
        if not np.array_equal(pack['instance_id'], rows.instance_id.to_numpy()):
            raise ValueError('Existing facility archive does not match current manifest order')
        source_names = json.loads((CURRENT / 'relation_features.json').read_text())
        selected = [source_names.index(name) for name in names]
        cached_values, cached_known = pack['knowledge'][:, selected], pack['support'][:, selected]
    comparable = detail.raw_join_verified.to_numpy()
    mask_disagreement = np.any(historical_known != cached_known, axis=1) & comparable
    value_disagreement = np.any(~np.isclose(historical_values, cached_values, rtol=1e-5, atol=1e-6) & cached_known, axis=1) & comparable
    bad = mask_disagreement | value_disagreement
    if bad.any():
        detail.loc[bad].to_csv(output / 'cache_mismatches.csv', index=False)
        raise ValueError(f'{bad.sum()} raw/cache discrepancies; inspect cache_mismatches.csv before export')
    footprint = detail.footprint_verified.to_numpy(bool)
    known = historical_known & footprint[:, None]
    values = np.where(known, historical_values, 0)
    layer_known = historical_known[:, :len(LAYERS)]
    role_values, role_known, quality = role_features(raw, roles, layer_known)
    bounded_role_known = role_known & footprint[:, None, None]
    raw_names = [f'{layer}_{suffix}' for layer in LAYERS for suffix in
                 ('distance_km', 'count_1km_log1p', 'count_3km_log1p')]
    raw_values = raw[raw_names].to_numpy(np.float32)
    raw_known = np.repeat(layer_known & footprint[:, None], 3, axis=1)
    distances = raw[[layer + '_distance_km' for layer in LAYERS]].to_numpy(float)
    # Seamark routes are navigation context, not a nearby physical facility.
    near = ((distances[:, 1:] <= 3) & layer_known[:, 1:]).any(1)
    detail['historical_supported'] = historical_known.any(1)
    detail['bounded_supported'] = known.any(1)
    detail['bounded_near_infra_3km'] = near & footprint
    detail['historical_near_infra_3km'] = near
    detail['bounded_known_zero_cells'] = (known & (values == 0)).sum(1)
    detail['bounded_known_features'] = known.sum(1)
    detail['historical_known_features'] = historical_known.sum(1)
    detail['status'] = np.select([detail.bounded_supported, detail.historical_supported,
                                provenance.osm_source.fillna('').ne('')],
                               ['bounded_context', 'historical_footprint_unverified', 'join_unverified'],
                               default='no_existing_osm_support')
    coords_path = ART / 'real_k_asset_audit_20260917/coordinates_private.csv'
    coords = aligned(read_columns(coords_path, ['instance_id', 'product', 'coordinate_available', 'coordinate_kind']),
                     'instance_id', rows.instance_id, str(coords_path), collisions)
    detail['current_georef_center_available'] = flag(coords.coordinate_available) & coords['product'].eq(rows['product']).to_numpy()
    detail['current_coordinate_kind'] = coords.coordinate_kind.fillna('')
    # Current location availability is separate from the location actually used by an old map row.
    missing_path = ART / 'real_k_asset_audit_20260917/osm_recovery/final_remaining_reasons_private.csv'
    missing = aligned(read_columns(missing_path, ['instance_id', 'source_gap']),
                      'instance_id', rows.instance_id, str(missing_path), collisions)
    detail['remaining_map_gap'] = missing.source_gap.fillna('')
    old = old_roles(rows, collisions)
    detail['old_exact_chip_match'] = old.exact_product_and_two_filenames
    detail['old_role_supported'] = old.old_role_supported
    old.to_csv(output / 'old_role_mapping.csv', index=False)
    query, query_rows = query_coverage(rows, detail)
    query.to_csv(output / 'query_source_coverage.csv', index=False)
    query_rows.to_csv(output / 'query_row_ids.csv', index=False)
    detail.insert(0, 'row_index', np.arange(len(rows)))
    detail.to_csv(output / 'row_provenance.csv', index=False)
    inventory.to_csv(output / 'source_inventory.csv', index=False)
    pd.DataFrame(collisions, columns=['scope', 'key', 'rows', 'ids']).to_csv(output / 'key_collisions.csv', index=False)
    count_cols = ['historical_supported', 'bounded_supported', 'historical_near_infra_3km',
                  'bounded_near_infra_3km', 'current_georef_center_available', 'old_exact_chip_match',
                  'old_role_supported', 'bounded_known_zero_cells']
    coverage = detail.groupby('port')[count_cols].sum().join(rows.groupby('port').size().rename('rows')).reset_index()
    coverage.to_csv(output / 'coverage_by_port.csv', index=False)
    detail.groupby(['port', 'raw_source', 'status', 'map_date', 'coordinate_kind'], dropna=False).size().rename('rows').reset_index().to_csv(output / 'coverage_by_port_source.csv', index=False)
    np.savez_compressed(output / 'facility_features.npz',
        row_index=np.arange(len(rows)), instance_id=rows.instance_id.to_numpy(dtype=str),
        values=values, known=known, feature_names=np.asarray(names),
        historical_values=historical_values, historical_known=historical_known,
        raw_values=np.where(raw_known, raw_values, 0), raw_known=raw_known, raw_feature_names=np.asarray(raw_names),
        role_values=np.where(bounded_role_known, role_values, 0), role_known=bounded_role_known,
        role_quality=np.where(footprint[:, None], quality, 0), role_names=np.asarray(list(roles)),
        historical_role_values=role_values, historical_role_known=role_known)
    summary = dict(rows=len(rows), ports=rows.port.nunique(), features=len(names),
                   historical_supported=int(historical_known.any(1).sum()),
                   bounded_supported=int(known.any(1).sum()),
                   bounded_near_infra_3km=int(detail.bounded_near_infra_3km.sum()),
                   known_zero_cells=int((known & (values == 0)).sum()),
                   current_georef_center_rows=int(detail.current_georef_center_available.sum()),
                   old_exact_matches=int(old.exact_product_and_two_filenames.sum()),
                   old_supported_matches=int(old.old_role_supported.sum()),
                   naive_old_same_id_different_chip=int(old.same_id_different_chip.sum()),
                   source_key_collisions=len(collisions),
                   raw_cache_mask_disagreement_rows=int(mask_disagreement.sum()),
                   raw_cache_value_disagreement_rows=int(value_disagreement.sum()),
                   cached_osm_supported=int(cached_known.any(1).sum()),
                   active_outer_ports=int(query.status.eq('aligned').sum()),
                   query_members_read=['skipped', 'qpos', 'spos', 'ports_all'],
                   model_training=False, query_labels_read=False, downloads=False,
                   mapping='Existing source pointer plus ID/product; returned exact product/two source TIFFs/two object UIDs.',
                   limits=['Support is observed map context, not field-surveyed completeness or demonstrated model benefit.',
                           'Historical support is retained separately; default known additionally requires documented 10.1 km footprint.',
                           'Missing/empty semantic layers remain unsupported; known zero applies only to supported queried layers.',
                           'Distances are capped at 10 km. Co-proximity is not berth occupancy, actual cargo or navigable topology.',
                           'Inherited sample centers and AIS position proxies retain their own coordinate_kind; they are not all exact crop centers.',
                           '2025 PBF snapshots and 2026 queries are mixed; current provider boundaries are not certified historical boundaries.',
                           'No port-centroid features, identity fields, ship-type fields, AIS classes or outcomes enter feature values.'])
    (output / 'audit_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args().output)
