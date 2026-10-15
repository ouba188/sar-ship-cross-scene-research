"""Read-only inventory of exported AIS dimensions and raster header spacing."""
import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
MANIFEST = Path('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv')
AIS = Path('E:/Hermes/paper_tracking/port_ais_2024_2025_from_zip')
SYNC = Path('E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata/local_knowledge_sync_v1')


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) and v > 0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def main():
    obs = pd.read_csv(BASE/'observations.csv', dtype={'mmsi': str})
    manifest = pd.read_csv(MANIFEST, dtype={'mmsi': str}, low_memory=False).fillna('')
    aligned = manifest.iloc[obs.manifest_row].reset_index(drop=True)
    for key in ['instance_id', 'mmsi', 'product', 'port']:
        assert np.array_equal(obs[key].astype(str), aligned[key].astype(str)), key
    z = np.load(BASE/'incidence_transport_20261011/source_cohorts.npz')
    fit, held = z['fit_obs_ids'], z['held_obs_ids']
    assert len(fit) == 1296 and len(held) == 823
    assert not set(obs.iloc[fit].mmsi) & set(obs.iloc[held].mmsi)
    cohort = obs.iloc[np.r_[fit, held]].copy()
    cohort['split'] = ['fit']*len(fit)+['held']*len(held)
    cohort = cohort.set_index('obs_id', drop=False)

    cols = ['mmsi', 'length_mode', 'beam_mode', 'local_label_source', 'provenance_note']
    modes = pd.read_csv(AIS/'dedup_tables/vessels_to_lookup_dedup.csv', usecols=cols,
                        dtype=str, keep_default_na=False)
    assert not modes.mmsi.duplicated().any()
    joined = obs.merge(modes, on='mmsi', how='left', validate='many_to_one')
    joined['mode_length'] = joined.length_mode.map(number)
    joined['mode_beam'] = joined.beam_mode.map(number)
    joined['mode_pair'] = joined.mode_length.notna() & joined.mode_beam.notna()
    cohort = cohort.join(joined.set_index('obs_id')[['mode_length', 'mode_beam', 'mode_pair']])

    filemap = pd.concat([pd.read_csv(p, dtype=str).fillna('')
                         for p in (SYNC/'private').glob('server*_file_manifest.csv')])
    good = filemap[filemap.status.isin(['downloaded', 'reused_local'])]
    jsonmap = dict(zip(good[good.kind.eq('source_json')].remote_path,
                       good[good.kind.eq('source_json')].local_path))
    georef = {}
    for path in (SYNC/'private').glob('server*_georef.jsonl'):
        for line in path.open(encoding='utf-8'):
            row = json.loads(line)
            if row.get('status') == 'ok':
                georef[row['remote_source_tiff']] = row
    spacing = {}
    for key, geo in georef.items():
        epsg = int(geo['epsg'])
        if 32601 <= epsg <= 32660 or 32701 <= epsg <= 32760:
            t = geo['transform']
            spacing[key] = (math.hypot(t[1], t[4]), math.hypot(t[2], t[5]))
    for pol in ['vv', 'vh']:
        joined['gsd_'+pol] = aligned['source_tiff_'+pol].map(
            lambda key: spacing.get(key, (np.nan, np.nan))[0])
    cohort = cohort.join(joined.set_index('obs_id')[['gsd_vv', 'gsd_vh']])

    records = []
    # Open only JSON files actually referenced by the current source cohorts.
    for pol in ['vv', 'vh']:
        for remote, rows in aligned.iloc[cohort.index].groupby('source_json_'+pol):
            if remote not in jsonmap:
                continue
            path = Path(jsonmap[remote])
            if not path.is_file():
                continue
            objects = json.loads(path.read_text(encoding='utf-8')).get('objects', [])
            byuid = defaultdict(list)
            for index, obj in enumerate(objects):
                byuid[str(obj.get('obj_uid') or obj.get('source_obj_uid') or '')].append((index, obj))
            for obsid, row in rows.iterrows():
                matches = byuid.get(str(row['obj_uid_'+pol]), [])
                if len(matches) != 1:
                    continue
                index, obj = matches[0]
                m = obj.get('meta', {})
                records.append(dict(obs_id=obsid, pol=pol, mmsi=str(m.get('mmsi', '')),
                    json_length=number(m.get('length_m')), json_beam=number(m.get('beam_m')),
                    source=m.get('source', ''), local_json=str(path),
                    uid=row['obj_uid_'+pol], source_index_agrees=index == int(row['source_object_index_'+pol])))
    obj = pd.DataFrame(records)
    obj['mmsi_agrees'] = obj.mmsi.eq(obj.obs_id.map(cohort.mmsi))
    valid = obj[obj.mmsi_agrees & obj.json_length.notna() & obj.json_beam.notna()]
    obj.to_csv(OUT/'object_dimension_audit.csv', index=False)
    perobs = valid.groupby('obs_id').agg(json_length=('json_length', 'first'),
        json_beam=('json_beam', 'first'), json_length_variants=('json_length', 'nunique'),
        json_beam_variants=('json_beam', 'nunique'))
    cohort = cohort.join(perobs)

    # Primary-key MMSI lookup, no full-table scan or database writes.
    conn = sqlite3.connect((AIS/'dedup_tables/ais_points_dedup.sqlite').as_uri()+'?mode=ro', uri=True)
    plan = conn.execute('EXPLAIN QUERY PLAN SELECT length,beam FROM points WHERE mmsi=?',
                        (cohort.mmsi.iloc[0],)).fetchall()
    raw = defaultdict(set)
    exact = defaultdict(set)
    keys = set(zip(cohort.mmsi, cohort['product']))
    db_rows = 0
    for mmsi in cohort.mmsi.unique():
        for length, beam, product in conn.execute('SELECT length,beam,product FROM points WHERE mmsi=?', (mmsi,)):
            db_rows += 1
            pair = (number(length), number(beam))
            if np.isfinite(pair).all():
                raw[mmsi].add(pair)
                if (mmsi, product) in keys:
                    exact[(mmsi, product)].add(pair)
    conn.close()
    # Synced product TXT is an additional provenance source, not a web recovery.
    raw_sources = []
    products = set(cohort['product'])
    for row in good[good.kind.eq('ais') & good['product'].isin(products)].itertuples():
        path = Path(row.local_path)
        if not path.is_file():
            continue
        raw_sources.append(str(path))
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for fields in csv.reader(stream):
                if len(fields) != 18:
                    continue
                mmsi = fields[1].strip()
                key = (mmsi, row.product)
                if key not in keys:
                    continue
                pair = (number(fields[14]), number(fields[15]))
                if np.isfinite(pair).all():
                    raw[mmsi].add(pair)
                    exact[key].add(pair)
    for obsid, row in cohort.iterrows():
        candidates = exact.get((row.mmsi, row['product']), set())
        allpairs = raw.get(row.mmsi, set())
        cohort.loc[obsid, 'raw_pair_variants_identity'] = len(allpairs)
        cohort.loc[obsid, 'raw_pair_variants_product'] = len(candidates)
        cohort.loc[obsid, 'json_pair_in_raw_same_product'] = (row.json_length, row.json_beam) in candidates
        if len(candidates) == 1:
            length, beam = next(iter(candidates))
            cohort.loc[obsid, 'raw_exact_length'] = length
            cohort.loc[obsid, 'raw_exact_beam'] = beam
        if len(allpairs) == 1:
            length, beam = next(iter(allpairs))
            cohort.loc[obsid, 'raw_identity_length'] = length
            cohort.loc[obsid, 'raw_identity_beam'] = beam
    variants = [dict(mmsi=mmsi, length_beam_pairs=sorted(pairs))
                for mmsi, pairs in sorted(raw.items()) if len(pairs) > 1]
    (OUT/'dimension_conflicts.json').write_text(
        json.dumps(variants, indent=2), encoding='utf-8')
    (OUT/'read_source_txt_paths.json').write_text(
        json.dumps(raw_sources, ensure_ascii=False, indent=2), encoding='utf-8')
    cohort['dimension_pair_verified'] = (cohort.json_pair_in_raw_same_product &
        cohort.json_length_variants.eq(1) & cohort.json_beam_variants.eq(1))
    cohort['raw_identity_single_pair'] = cohort.raw_pair_variants_identity.eq(1)
    cohort['raw_exact_single_pair'] = cohort.raw_pair_variants_product.eq(1)
    cohort['raw_identity_pair_available'] = cohort.raw_pair_variants_identity.gt(0)
    cohort['raw_exact_pair_available'] = cohort.raw_pair_variants_product.gt(0)
    cohort['raw_identity_conflict'] = cohort.raw_pair_variants_identity.gt(1)
    cohort['json_pair_available'] = cohort.json_length.notna() & cohort.json_beam.notna()
    cohort['mode_json_disagreement'] = (cohort.mode_pair & cohort.json_pair_available &
        (~np.isclose(cohort.mode_length, cohort.json_length) | ~np.isclose(cohort.mode_beam, cohort.json_beam)))
    cohort['gsd_available'] = cohort.gsd_vv.notna() | cohort.gsd_vh.notna()
    cohort['incidence_available'] = cohort.incidence_deg.notna()
    cohort['relative_available'] = cohort.relative_signed_deg.notna()
    cohort['raw_identity_ratio'] = cohort.raw_identity_length/cohort.raw_identity_beam
    cohort['raw_exact_ratio'] = cohort.raw_exact_length/cohort.raw_exact_beam
    # These are review flags, NOT a declaration that all other values are true.
    review = cohort.raw_identity_single_pair & (
        cohort.raw_identity_length.le(cohort.raw_identity_beam) |
        cohort.raw_identity_length.ge(500) | cohort.raw_identity_ratio.gt(30))
    cohort.loc[review, ['obs_id','split','mmsi','class_name','raw_identity_length',
        'raw_identity_beam','raw_identity_ratio']].to_csv(OUT/'dimension_review_flags.csv', index=False)
    coverage = []
    fields = ['mode_pair', 'json_pair_available', 'dimension_pair_verified',
              'raw_identity_pair_available', 'raw_identity_single_pair', 'raw_exact_pair_available',
              'raw_exact_single_pair', 'raw_identity_conflict', 'mode_json_disagreement',
              'gsd_available', 'incidence_available', 'relative_available']
    for split, rows in cohort.groupby('split'):
        for field in fields:
            yes = rows[rows[field].fillna(False)]
            coverage.append(dict(split=split, field=field, observations=len(yes),
                all_observations=len(rows), identities=yes.mmsi.nunique(), all_identities=rows.mmsi.nunique()))
    for field, mask in [('mode_pair', joined.mode_pair),
                        ('gsd_available', joined.gsd_vv.notna() | joined.gsd_vh.notna())]:
        coverage.append(dict(split='all_observations', field=field, observations=int(mask.sum()),
            all_observations=len(joined), identities=joined.loc[mask, 'mmsi'].nunique(),
            all_identities=joined.mmsi.nunique()))
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv', index=False)
    selected_cols = ['obs_id','split','instance_id','mmsi','product','class_name',
        'mode_length','mode_beam','json_length','json_beam','raw_pair_variants_identity',
        'raw_pair_variants_product','raw_identity_length','raw_identity_beam','raw_identity_ratio',
        'raw_exact_length','raw_exact_beam','raw_exact_ratio','json_pair_in_raw_same_product']+fields
    cohort[list(dict.fromkeys(selected_cols))].to_csv(OUT/'source_dimension_join.csv',index=False)
    cohort.groupby(['split','class_name'])[fields].sum().to_csv(OUT/'class_coverage.csv')
    summary = dict(observations=len(obs), identities=obs.mmsi.nunique(),
        manifest_rows=len(manifest), obs_manifest_alignment=True, db_query_plan=plan,
        db_rows_read=db_rows, synced_txt_files_read=len(raw_sources),
        object_matched_views=len(obj), object_mmsi_disagreements=int((~obj.mmsi_agrees).sum()),
        object_index_disagreements=int((~obj.source_index_agrees).sum()),
        object_sources=obj.source.value_counts().to_dict(),
        spacing_pairs=sorted(set(spacing.values())),
        raw_column_semantics='AIS TXT 0-based columns 14 length, 15 beam per existing export consumers; no independent registry validation',
        all_observation_scope='MMSI mode table and returned raster headers only; exact source/raw audits limited to fit+held',
        dimension_scope='Positive numeric pairs are availability, not accuracy or independent true-measurement certification',
        geometry_scope='UTM raster-header sampling spacing, not SAR resolving power or resized-network spacing',
        source_paths=dict(manifest=str(MANIFEST),ais=str(AIS),sync=str(SYNC)))
    (OUT/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(coverage).to_string(index=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
