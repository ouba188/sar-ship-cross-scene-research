"""Bounded current-cohort metadata join and first 32 eligible paired tracks."""
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASELINE = ROOT / 'visual_reliable_baseline'
OLD = ROOT / 'same_vessel_condition_study_20261010'
sys.path[:0] = [str(BASELINE), str(OLD)]

import numpy as np
import pandas as pd
from pyproj import Geod
from build_returned_local_knowledge_v1 import load_file_maps, normal_mmsi
from recover_v6_relation_observations import parse_export
from extract_motion_knowledge import decode_motion, epoch_utc, midpoint, resolve, stored_coordinates

SYNC = Path('E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata/local_knowledge_sync_v1')
MANIFEST = BASELINE / 'artifacts/eight_class_adaptive_20260916/manifest.csv'
INVENTORY = ROOT / 'obb_consistent_repair_20261016/inventory.csv'
PAIRS = ROOT / 'oss1_polar_environment_20261017/same_vessel_pairs.csv'
MODES = Path('E:/Hermes/paper_tracking/port_ais_2024_2025_from_zip/dedup_tables/vessels_to_lookup_dedup.csv')
GEOD = Geod(ellps='WGS84')
MOTION = ['sog_knots', 'cog_deg', 'true_heading_deg']
RAW_BUDGET = 128 * 1024 * 1024


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def positive(value):
    v = number(value)
    return v if v > 0 else np.nan


def js(value):
    return json.dumps(value, ensure_ascii=False)


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace('+00:00', 'Z')


def angle_delta(a, b):
    return (b - a + 180) % 360 - 180


def stats(values):
    v = np.asarray(list(values), float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0)
    return dict(n=len(v), min=float(v.min()), median=float(np.median(v)),
                p95=float(np.quantile(v, .95)), max=float(v.max()))


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_cohort():
    inv = pd.read_csv(INVENTORY, dtype={'mmsi': str})
    full = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    rows = full.iloc[inv.global_row].copy().reset_index(drop=True)
    rows.insert(0, 'global_row', inv.global_row.to_numpy())
    inv['mmsi'] = inv.mmsi.map(normal_mmsi)
    for key in ['instance_id', 'mmsi', 'product', 'port', 'class_id']:
        assert rows[key].astype(str).tolist() == inv[key].astype(str).tolist(), key
    assert len(rows) == 2713 and rows.global_row.is_unique
    pairs = pd.read_csv(PAIRS, dtype={'mmsi': str})
    pairs = pairs[pairs.dataset.eq('24port_local')].copy()
    pairs['mmsi'] = pairs.mmsi.map(normal_mmsi)
    ids = rows.set_index('global_row')
    for p in pairs.itertuples():
        for end, product in [(p.first, p.first_product), (p.second, p.second_product)]:
            r = ids.loc[end]
            assert r.mmsi == p.mmsi and r['product'] == product
    return rows, pairs


def object_metadata(rows, json_map):
    cache, result, views = {}, [], []
    for row in rows.to_dict('records'):
        rr = {k: row[k] for k in ['global_row', 'instance_id', 'mmsi', 'product', 'port', 'class_id', 'class_name', 'data_origin']}
        rr['sar_midpoint_epoch'] = midpoint(row['product'])
        vv = []
        for pol in ['vv', 'vh']:
            remote, uid = row['source_json_' + pol], row['obj_uid_' + pol]
            path = json_map.get(remote)
            v = dict(global_row=row['global_row'], pol=pol, uid=uid, local_json=str(path or ''),
                     status='no_uid_or_synced_json', epoch=np.nan)
            if path and uid:
                if str(path) not in cache:
                    data = json.loads(path.read_text(encoding='utf-8'))
                    byuid = defaultdict(list)
                    for idx, obj in enumerate(data.get('objects', [])):
                        byuid[str(obj.get('obj_uid') or obj.get('source_obj_uid') or '')].append((idx, obj))
                    cache[str(path)] = (data.get('image', ''), byuid)
                image, index = cache[str(path)]
                matches = index.get(uid, [])
                if len(matches) != 1:
                    v['status'] = 'ambiguous_or_missing_uid'
                else:
                    idx, obj = matches[0]
                    meta = obj.get('meta', {})
                    v.update(mmsi=normal_mmsi(meta.get('mmsi', '')), ais_time_utc=meta.get('ais_time_utc', ''),
                             length_m=positive(meta.get('length_m')), beam_m=positive(meta.get('beam_m')),
                             attribute_source=meta.get('source', ''), position_method=meta.get('ais_position_method', ''),
                             source_index_agrees=idx == int(float(row['source_object_index_' + pol])),
                             coordinates=stored_coordinates(obj))
                    if v['mmsi'] != row['mmsi'] or row['product'] not in image:
                        v['status'] = 'object_identity_or_product_mismatch'
                    else:
                        try:
                            v['epoch'] = epoch_utc(v['ais_time_utc'])
                            v['status'] = 'ok'
                        except (TypeError, ValueError):
                            v['status'] = 'invalid_time'
            vv.append(v)
            views.append(v)
        good = [v for v in vv if v['status'] == 'ok']
        times = {v['epoch'] for v in good}
        rr['uid_time_status'] = ('verified_both_views' if len(good) == 2 and len(times) == 1 else
                                 'no_uid_or_synced_json' if not good else 'incomplete_or_conflicting_views')
        rr['ais_epoch'] = next(iter(times)) if rr['uid_time_status'] == 'verified_both_views' else np.nan
        rr['ais_time_utc'] = utc(rr['ais_epoch']) if np.isfinite(rr['ais_epoch']) else ''
        rr['ais_minus_sar_midpoint_s'] = rr['ais_epoch'] - rr['sar_midpoint_epoch']
        rr['abs_time_delta_seconds'] = abs(rr['ais_minus_sar_midpoint_s'])
        rr['source_json_paths'] = js(sorted({v['local_json'] for v in vv if v['local_json']}))
        rr['uids_json'] = js({v['pol']: v['uid'] for v in vv})
        rr['position_methods_json'] = js(sorted({v.get('position_method', '') for v in good}))
        rr['object_coordinates_json'] = js(sorted({tuple(x) for v in good for x in v['coordinates']}))
        for field in ['length_m', 'beam_m']:
            vals = {v[field] for v in good if np.isfinite(v[field])}
            rr['object_' + field] = next(iter(vals)) if len(vals) == 1 else np.nan
            rr['object_' + field + '_variants'] = len(vals)
        rr.update(motion_match_status='not_read_in_bounded_raw_scope' if np.isfinite(rr['ais_epoch']) else 'no_exact_object_timestamp',
                  motion_provenance='', source_txt='', source_line=np.nan, draught_raw=np.nan,
                  raw_length_m=np.nan, raw_beam_m=np.nan, raw_lon=np.nan, raw_lat=np.nan,
                  draught_units='unverified_export_units', draught_update_time='',
                  sog_knots=np.nan, cog_deg=np.nan, true_heading_deg=np.nan,
                  raw_sog='', raw_cog='', raw_true_heading='')
        result.append(rr)
    return pd.DataFrame(result), pd.DataFrame(views), len(cache)


def reuse_prior(table):
    prior = pd.read_csv(OLD / 'motion_knowledge_20261011/motion_knowledge.csv',
                        dtype={k: str for k in ['mmsi','raw_sog','raw_cog','raw_true_heading']})
    assert prior.instance_id.is_unique
    byid = prior.set_index('instance_id')
    reused = 0
    for idx, r in table.iterrows():
        if r.instance_id not in byid.index:
            continue
        p = byid.loc[r.instance_id]
        if p.match_status != 'matched' or not np.isfinite(r.ais_epoch):
            continue
        assert p.mmsi == r.mmsi and p['product'] == r['product']
        assert p.ais_epoch == r.ais_epoch
        for k in MOTION + ['raw_sog', 'raw_cog', 'raw_true_heading', 'raw_lon', 'raw_lat']:
            table.at[idx, k] = p[k]
        table.at[idx, 'motion_match_status'] = 'exact_prior_cache'
        table.at[idx, 'motion_provenance'] = str(OLD / 'motion_knowledge_20261011/motion_knowledge.csv')
        refs = json.loads(p.source_records_json)
        table.at[idx, 'source_txt'] = refs[0]['path']
        table.at[idx, 'source_line'] = refs[0]['line']
        reused += 1
    return reused


def load_records(path, wanted_mmsis):
    # Existing parser owns time/MMSI/position validation. CSV adds raw attribute provenance.
    parsed = parse_export(path)
    accepted = {(r['epoch'], r['mmsi'], r['lon'], r['lat']) for r in parsed if r['mmsi'] in wanted_mmsis}
    selected, line_count = [], 0
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for line, fields in enumerate(csv.reader(stream), 1):
            line_count = line
            if len(fields) != 18 or normal_mmsi(fields[1]) not in wanted_mmsis:
                continue
            key = (number(fields[0]), normal_mmsi(fields[1]), number(fields[6]), number(fields[7]))
            if key not in accepted:
                continue
            motion = decode_motion(*(fields[j] for j in (4, 8, 9)))
            selected.append(dict(epoch=key[0], mmsi=key[1], lon=key[2], lat=key[3],
                                 sog_knots=motion[0], cog_deg=motion[1], true_heading_deg=motion[2],
                                 raw_motion=tuple(fields[j].strip() for j in (4, 8, 9)),
                                 raw_length=fields[14], raw_beam=fields[15], draught_raw=fields[16],
                                 accuracy_raw=fields[5], rot_raw=fields[3], raw_fields=fields,
                                 source_txt=str(path), source_line=line))
    return selected, dict(path=str(path), bytes=path.stat().st_size, rows_read=line_count,
                         parser_valid_rows=len(parsed), cohort_records=len(selected))


def exact_join(row, records):
    candidates = [r for r in records if r['mmsi'] == row.mmsi and r['epoch'] == row.ais_epoch]
    chosen, status = resolve(candidates, json.loads(row.object_coordinates_json))
    if chosen is None:
        return dict(motion_match_status=status)
    out = {k: chosen[k] for k in MOTION}
    out.update(motion_match_status='exact_raw_record', motion_provenance='UID/product/MMSI/exact_UTC_epoch',
               source_txt=chosen['source_txt'], source_line=chosen['source_line'],
               raw_lon=chosen['lon'], raw_lat=chosen['lat'],
               raw_sog=chosen['raw_motion'][0], raw_cog=chosen['raw_motion'][1], raw_true_heading=chosen['raw_motion'][2],
               exact_candidate_count=len(candidates))
    # Static attributes are resolved separately from dynamic-field equality.
    for key, rawkey, conv in [('raw_length_m', 'raw_length', positive), ('raw_beam_m', 'raw_beam', positive),
                             ('draught_raw', 'draught_raw', number)]:
        vals = {conv(c[rawkey]) for c in candidates if np.isfinite(conv(c[rawkey]))}
        out[key] = next(iter(vals)) if len(vals) == 1 else np.nan
        out[key + '_variants'] = len(vals)
    return out


def product_window(product):
    t = [datetime.strptime(s, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc).timestamp()
         for s in product.split('_')[4:6]]
    return t[0] - 180, t[1] + 180


def dedup_records(records):
    unique = {}
    for r in records:
        # No packet identifier exists: count distinct exported dynamic records.
        key = tuple(r['raw_fields'][:10])
        if key not in unique:
            unique[key] = dict(r, duplicate_rows=0)
        unique[key]['duplicate_rows'] += 1
    return sorted(unique.values(), key=lambda r: (r['epoch'], r['source_line']))


def summarize_track(row, records):
    lo, hi = product_window(row['product'])
    allship = [r for r in records if r['mmsi'] == row.mmsi]
    window = [r for r in allship if lo <= r['epoch'] <= hi]
    unique = dedup_records(window)
    bytime = defaultdict(list)
    for r in unique:
        bytime[r['epoch']].append(r)
    times = sorted(bytime)
    if len(times) < 2:
        return None, [], []
    gaps = np.diff(times)
    consistent = {}
    ambiguous = 0
    for t, rr in bytime.items():
        variants = {(r['lon'], r['lat'], *r['raw_motion']) for r in rr}
        if len(variants) == 1:
            consistent[t] = rr[0]
        else:
            ambiguous += 1
    segments = []
    for ta, tb in zip(times[:-1], times[1:]):
        if ta not in consistent or tb not in consistent:
            continue
        a, b = consistent[ta], consistent[tb]
        azimuth, _, distance = GEOD.inv(a['lon'], a['lat'], b['lon'], b['lat'])
        dt = tb - ta
        speed = distance / dt / (1852 / 3600)
        sog_ok = all(np.isfinite(q['sog_knots']) and q['raw_motion'][0] != '1022' for q in (a, b))
        avg_sog = (a['sog_knots'] + b['sog_knots']) / 2 if sog_ok else np.nan
        moving_cog = sog_ok and distance > 0 and min(a['sog_knots'], b['sog_knots']) >= 1 and all(np.isfinite(q['cog_deg']) for q in (a, b))
        cog_mean = np.degrees(np.arctan2(np.sin(np.radians([a['cog_deg'], b['cog_deg']])).sum(),
                                       np.cos(np.radians([a['cog_deg'], b['cog_deg']])).sum())) % 360 if moving_cog else np.nan
        segments.append(dict(global_row=int(row.global_row), mmsi=row.mmsi, product=row['product'],
                             epoch_a=ta, epoch_b=tb, gap_s=dt, displacement_m=distance,
                             displacement_speed_knots=speed, reported_endpoint_mean_sog_knots=avg_sog,
                             speed_residual_knots=speed-avg_sog,
                             displacement_bearing_deg=azimuth % 360 if distance > 0 else np.nan,
                             moving_cog_eligible=moving_cog,
                             abs_bearing_cog_delta_deg=abs(angle_delta(cog_mean, azimuth % 360)),
                             abs_cog_change_deg=abs(angle_delta(a['cog_deg'], b['cog_deg'])),
                             abs_true_heading_change_deg=abs(angle_delta(a['true_heading_deg'], b['true_heading_deg']))))
    sog = np.asarray([r['sog_knots'] for r in unique if np.isfinite(r['sog_knots']) and r['raw_motion'][0] != '1022'])
    summary = dict(global_row=int(row.global_row), instance_id=row.instance_id, mmsi=row.mmsi,
                   product=row['product'], port=row.port, class_id=int(row.class_id),
                   raw_ship_rows_in_file=len(allship), rows_in_window=len(window),
                   distinct_dynamic_records=len(unique), duplicate_dynamic_rows=len(window)-len(unique),
                   distinct_export_rows=len({tuple(r['raw_fields'][:17]) for r in window}),
                   distinct_timestamps=len(times), ambiguous_timestamps=ambiguous,
                   first_ais_time_utc=utc(times[0]), last_ais_time_utc=utc(times[-1]),
                   observed_span_s=times[-1]-times[0], gap_min_s=float(gaps.min()),
                   gap_median_s=float(np.median(gaps)), gap_p95_s=float(np.quantile(gaps,.95)),
                   gap_max_s=float(gaps.max()), gaps_gt60s=int((gaps>60).sum()), gaps_gt120s=int((gaps>120).sum()),
                   brackets_sar_midpoint=times[0] <= row.sar_midpoint_epoch <= times[-1],
                   valid_sog_records=len(sog), sog_min_knots=float(sog.min()) if len(sog) else np.nan,
                   sog_max_knots=float(sog.max()) if len(sog) else np.nan,
                   sog_mean_knots=float(sog.mean()) if len(sog) else np.nan,
                   sog_sd_knots=float(sog.std()) if len(sog) else np.nan,
                   sog_range_knots=float(np.ptp(sog)) if len(sog) else np.nan,
                   low_speed_record_fraction_le0p5kn=float((sog<=.5).mean()) if len(sog) else np.nan,
                   adjacent_segments=len(segments),
                   draught_raw_variants=len({r['draught_raw'] for r in window}),
                   draught_positive_raw_variants=len({number(r['draught_raw']) for r in window if number(r['draught_raw']) > 0}),
                   draught_zero_records=sum(number(r['draught_raw']) == 0 for r in unique),
                   draught_positive_records=sum(number(r['draught_raw']) > 0 for r in unique),
                   dimension_pair_variants=len({(r['raw_length'], r['raw_beam']) for r in window}))
    for field in ['speed_residual_knots', 'abs_bearing_cog_delta_deg', 'abs_true_heading_change_deg']:
        st = stats(abs(s[field]) for s in segments)
        summary[field + '_median'] = st.get('median', np.nan)
        summary[field + '_n'] = st['n']
    messages = []
    for r in unique:
        messages.append(dict(global_row=int(row.global_row), product=row['product'], **{
            k: r[k] for k in ['mmsi', 'epoch', 'lon', 'lat', *MOTION, 'accuracy_raw', 'rot_raw', 'raw_length',
                              'raw_beam', 'draught_raw', 'source_txt', 'source_line', 'duplicate_rows']},
            ais_time_utc=utc(r['epoch']), raw_sog=r['raw_motion'][0], raw_cog=r['raw_motion'][1],
            raw_true_heading=r['raw_motion'][2]))
    return summary, messages, segments


def main():
    started = time.monotonic()
    rows, pairs = load_cohort()
    json_map, ais_map, _ = load_file_maps(SYNC)
    table, views, json_count = object_metadata(rows, json_map)
    reused = reuse_prior(table)
    modes = pd.read_csv(MODES, dtype=str, keep_default_na=False)
    assert modes.mmsi.is_unique
    modecols = ['mmsi', 'length_mode', 'beam_mode', 'draught_or_aux_mode', 'first_seen_utc', 'last_seen_utc', 'provenance_note']
    table = table.merge(modes[modecols].rename(columns={k:'identity_'+k for k in modecols if k!='mmsi'}), on='mmsi', how='left', validate='many_to_one')
    table['identity_length_mode'] = table.identity_length_mode.map(positive)
    table['identity_beam_mode'] = table.identity_beam_mode.map(positive)
    table['raw_file_located'] = table['product'].isin(ais_map)
    table['sog_lower_bounded_102p2'] = table.raw_sog.astype(str).isin(['1022','1022.0'])
    save_json('INITIAL_COVERAGE.json', dict(rows=len(table), vessels=int(table.mmsi.nunique()), products=int(table['product'].nunique()),
        uid_time_status=table.uid_time_status.value_counts().to_dict(), exact_prior_cache_rows=reused,
        object_size_pair=int((table.object_length_m.notna() & table.object_beam_m.notna()).sum()),
        identity_mode_size_pair=int((table.identity_length_mode.notna() & table.identity_beam_mode.notna()).sum()),
        read_json_files=json_count, selected_pair_rows=len(pairs), selected_pair_ships=int(pairs.mmsi.nunique())))
    table.to_csv(OUT/'cohort_attributes.csv', index=False)
    views.to_csv(OUT/'uid_views.csv', index=False)
    print('Metadata complete:', (OUT/'INITIAL_COVERAGE.json').read_text(encoding='utf8'), flush=True)

    byrow = table.set_index('global_row', drop=False)
    perproduct = rows.groupby('product').mmsi.agg(set).to_dict()
    cache, reads, selections, tracks, messages, segments = {}, [], [], [], [], []
    budget_used = 0
    selected_ships = []
    # Keep the preselected polarization pairs; one deterministic canonical pair selects a ship.
    canonical = pairs.sort_values(['mmsi', 'first', 'second', 'pol']).drop_duplicates('mmsi')
    for pair in canonical.itertuples():
        if len(selected_ships) >= 32:
            break
        ends = [byrow.loc[int(pair.first)], byrow.loc[int(pair.second)]]
        choice = dict(mmsi=pair.mmsi, first=int(pair.first), second=int(pair.second), class_id=int(pair.class_id))
        if any(r.uid_time_status != 'verified_both_views' for r in ends):
            selections.append(dict(choice, status='missing_exact_object_timestamp'))
            continue
        if any(r['product'] not in ais_map for r in ends):
            selections.append(dict(choice, status='no_synced_raw_file'))
            continue
        needed = {r['product'] for r in ends} - cache.keys()
        cost = sum(ais_map[p].stat().st_size for p in needed)
        if budget_used + cost > RAW_BUDGET:
            selections.append(dict(choice, status='raw_byte_budget_reached'))
            break
        for prod in sorted(needed):
            cache[prod], receipt = load_records(ais_map[prod], perproduct[prod])
            receipt['product'] = prod
            reads.append(receipt)
            budget_used += receipt['bytes']
            for idx, r in table[table['product'].eq(prod) & table.ais_epoch.notna()].iterrows():
                updates = exact_join(r, cache[prod])
                for key, value in updates.items():
                    table.at[idx, key] = value
        byrow = table.set_index('global_row', drop=False)
        ends = [byrow.loc[int(pair.first)], byrow.loc[int(pair.second)]]
        if any(r.motion_match_status != 'exact_raw_record' for r in ends):
            selections.append(dict(choice, status='exact_raw_join_failed'))
            continue
        summaries = [summarize_track(r, cache[r['product']]) for r in ends]
        if any(s[0] is None for s in summaries):
            selections.append(dict(choice, status='fewer_than_two_distinct_timestamps'))
            continue
        selected_ships.append(pair.mmsi)
        selections.append(dict(choice, status='selected', selection_rank=len(selected_ships)))
        for summary, mm, ss in summaries:
            tracks.append(summary)
            messages.extend(mm)
            segments.extend(ss)
        print('Selected', len(selected_ships), pair.mmsi, 'files',len(reads), 'MiB',round(budget_used/1024**2,2), flush=True)

    table['sog_lower_bounded_102p2'] = pd.to_numeric(table.raw_sog, errors='coerce').eq(1022)
    table['sog_exact_numeric_valid'] = table.sog_knots.notna() & ~table.sog_lower_bounded_102p2
    table['draught_positive_raw_available'] = table.draught_raw.gt(0)
    table['raw_size_pair_available'] = table.raw_length_m.notna() & table.raw_beam_m.notna()
    table['object_size_pair_available'] = table.object_length_m.notna() & table.object_beam_m.notna()
    table['identity_mode_size_pair_available'] = table.identity_length_mode.notna() & table.identity_beam_mode.notna()
    table['selected_32_ship'] = table.mmsi.isin(selected_ships)
    table.to_csv(OUT/'cohort_attributes.csv', index=False)
    pd.DataFrame(reads).to_csv(OUT/'read_sources.csv', index=False)
    pd.DataFrame(selections).to_csv(OUT/'ship_selection.csv', index=False)
    tt, mm, ss = pd.DataFrame(tracks), pd.DataFrame(messages), pd.DataFrame(segments)
    tt.to_csv(OUT/'track_summary.csv', index=False)
    mm.to_csv(OUT/'track_messages.csv', index=False)
    ss.to_csv(OUT/'track_segments.csv', index=False)

    pp = pairs.copy()
    attrs = ['global_row', 'motion_match_status', 'ais_epoch', 'ais_minus_sar_midpoint_s', *MOTION,
             'raw_length_m', 'raw_beam_m', 'draught_raw', 'object_length_m', 'object_beam_m',
             'source_txt', 'source_line', 'sog_lower_bounded_102p2']
    for end in ['first', 'second']:
        pp = pp.merge(table[attrs].rename(columns={c: end+'_'+c for c in attrs}),
                      left_on=end, right_on=end+'_global_row', how='left', validate='many_to_one')
    pp['selected_32_ship'] = pp.mmsi.isin(selected_ships)
    pp['both_motion_exact'] = pp.first_motion_match_status.isin(['exact_prior_cache','exact_raw_record']) & pp.second_motion_match_status.isin(['exact_prior_cache','exact_raw_record'])
    pp['sog_change_knots'] = pp.second_sog_knots - pp.first_sog_knots
    pp.loc[pp.first_sog_lower_bounded_102p2 | pp.second_sog_lower_bounded_102p2, 'sog_change_knots'] = np.nan
    pp['cog_change_deg'] = angle_delta(pp.first_cog_deg, pp.second_cog_deg)
    pp['true_heading_change_deg'] = angle_delta(pp.first_true_heading_deg, pp.second_true_heading_deg)
    pp['cog_change_moving_both_deg'] = pp.cog_change_deg.where(pp.first_sog_knots.ge(1) & pp.second_sog_knots.ge(1))
    pp['raw_length_change_m'] = pp.second_raw_length_m - pp.first_raw_length_m
    pp['raw_beam_change_m'] = pp.second_raw_beam_m - pp.first_raw_beam_m
    pp['draught_change_raw_unverified_units'] = pp.second_draught_raw - pp.first_draught_raw
    pp['draught_positive_both_change_raw'] = pp.draught_change_raw_unverified_units.where(pp.first_draught_raw.gt(0)&pp.second_draught_raw.gt(0))
    pp.to_csv(OUT/'same_vessel_conditions.csv', index=False)
    coverage = []
    for grouping in ['data_origin', 'class_name', 'port']:
        for name, g in table.groupby(grouping):
            coverage.append(dict(grouping=grouping, group=name, rows=len(g), vessels=g.mmsi.nunique(),
                exact_motion=int(g.motion_match_status.isin(['exact_prior_cache','exact_raw_record']).sum()),
                valid_sog=int(g.sog_exact_numeric_valid.sum()), valid_cog=int(g.cog_deg.notna().sum()),
                valid_true_heading=int(g.true_heading_deg.notna().sum()), object_size_pair=int(g.object_size_pair_available.sum()),
                raw_size_pair=int(g.raw_size_pair_available.sum()), positive_raw_draught=int(g.draught_positive_raw_available.sum())))
    pd.DataFrame(coverage).to_csv(OUT/'coverage_by_group.csv', index=False)
    distinct_pairs = pp.drop_duplicates(['mmsi','first','second'])
    selected_canonical = pd.DataFrame(selections).query("status == 'selected'")[['mmsi','first','second']]
    cp = distinct_pairs.merge(selected_canonical, on=['mmsi','first','second'], validate='one_to_one')
    cp.to_csv(OUT/'selected32_condition_changes.csv', index=False)
    summary = dict(scope='2713-row metadata/cache coverage; first 32 eligible MMSI-sorted preselected pairs; raw bounded at 128 MiB',
        rows=len(table), vessels=int(table.mmsi.nunique()), products=int(table['product'].nunique()), ports=int(table.port.nunique()),
        uid_time_status=table.uid_time_status.value_counts().to_dict(), initial_exact_cache_rows=reused,
        match_status=table.motion_match_status.value_counts().to_dict(),
        coverage={f:int(table[f].notna().sum()) for f in MOTION+['ais_epoch','raw_length_m','raw_beam_m','draught_raw']},
        raw_size_pairs=int(table.raw_size_pair_available.sum()), object_size_pairs=int(table.object_size_pair_available.sum()),
        identity_mode_size_pairs=int(table.identity_mode_size_pair_available.sum()),
        positive_raw_draught=int(table.draught_positive_raw_available.sum()), raw_draught_zero=int(table.draught_raw.eq(0).sum()),
        sog_lower_bound_rows=int(table.sog_lower_bounded_102p2.sum()),
        ais_midpoint_delta_s=stats(table.ais_minus_sar_midpoint_s),
        exact_motion_abs_time_delta_s=stats(table.loc[table.sog_knots.notna(),'abs_time_delta_seconds']),
        pairs_input_rows=len(pairs), pairs_input_ships=int(pairs.mmsi.nunique()),
        both_exact_pair_rows=int(pp.both_motion_exact.sum()), both_exact_pair_ships=int(pp.loc[pp.both_motion_exact,'mmsi'].nunique()),
        selected_ships=len(selected_ships), selected_ship_class_counts=cp.class_id.value_counts().sort_index().to_dict(),
        selected_ship_ports=sorted(tt.port.unique().tolist()), selected_tracks=len(tt),
        selection_status=pd.DataFrame(selections).status.value_counts().to_dict(),
        read_json_files=json_count, read_raw_files=len(reads), read_raw_bytes=budget_used,
        raw_rows_read=sum(r['rows_read'] for r in reads), selected_raw_rows=int(tt.rows_in_window.sum()),
        selected_distinct_dynamic_records=int(tt.distinct_dynamic_records.sum()),
        selected_distinct_timestamps=int(tt.distinct_timestamps.sum()), selected_duplicate_dynamic_rows=int(tt.duplicate_dynamic_rows.sum()),
        ambiguous_timestamps=int(tt.ambiguous_timestamps.sum()),
        track_stats={f:stats(tt[f]) for f in ['distinct_dynamic_records','distinct_timestamps','observed_span_s','gap_max_s',
            'sog_mean_knots','sog_range_knots','low_speed_record_fraction_le0p5kn','speed_residual_knots_median','abs_bearing_cog_delta_deg_median']},
        segment_stats={f:stats(ss[f].abs()) for f in ['gap_s','speed_residual_knots','abs_bearing_cog_delta_deg','abs_true_heading_change_deg']},
        low_speed_all_records_tracks=int(tt.low_speed_record_fraction_le0p5kn.eq(1).sum()),
        tracks_gaps_gt60=int(tt.gaps_gt60s.gt(0).sum()), tracks_gaps_gt120=int(tt.gaps_gt120s.gt(0).sum()),
        tracks_bracketing_midpoint=int(tt.brackets_sar_midpoint.sum()),
        paired_changes={f:stats(cp[f].abs()) for f in ['sog_change_knots','cog_change_deg','cog_change_moving_both_deg',
            'true_heading_change_deg','raw_length_change_m','raw_beam_change_m','draught_positive_both_change_raw']},
        paired_counts=dict(sog_change_ge2=int(cp.sog_change_knots.abs().ge(2).sum()),
            heading_change_ge30=int(cp.true_heading_change_deg.abs().ge(30).sum()),
            positive_draught_changed=int(cp.draught_positive_both_change_raw.abs().gt(0).sum()),
            length_changed=int(cp.raw_length_change_m.abs().gt(0).sum()), beam_changed=int(cp.raw_beam_change_m.abs().gt(0).sum())),
        elapsed_seconds=round(time.monotonic()-started,2),
        semantics=dict(index5='accuracy, NOT navigation status', index16='raw draught, units and update time unverified',
            length_beam='index14/index15 reported length/beam, never summed; source object labels metres',
            time='dynamic message time minus product midpoint, not vessel imaging time or draught update time',
            messages='distinct exported dynamic fields0:10 excluding repeated copies; no packet ID, not certified transmitted-packet count',
            low_speed='observed record fraction SOG<=0.5kn; no stop-duration estimate',
            trajectory_window='SAR product start-180s to stop+180s; no bridging across acquisitions',
            moving_cog='both segment endpoint SOG>=1kn, descriptive reliability convention'))
    save_json('summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
