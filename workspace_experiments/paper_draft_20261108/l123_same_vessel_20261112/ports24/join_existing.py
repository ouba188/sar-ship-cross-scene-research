"""Join existing 24-port evidence only; no inference or source-file writes."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
NATURAL = ROOT / 'same_vessel_condition_study_20261010'
R22 = ROOT / 'same_vessel_gallery_20261019/causal_frequency_side_20261024/round22_classification_coverage'
INPUT = ROOT / 'paper_evidence_20261027/full_model_interventions_20261027'
M3 = ROOT / 'm3_frequency_increment_20261028'
MATCH = M3 / 'matching_vs_aggregation_20261031'
IDS = {'26628', '12457', '37321', '10488', '34278', '30167', '9140', '294'}
SHORTLIST = {'26628': 1, '12457': 2, '37321': 3}


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            yield dict(row, source_path=str(path), source_line=reader.line_num)


def write(name, data):
    data = list(data)
    fields = list(dict.fromkeys(key for row in data for key in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)
    return len(data)


def main():
    cohort = {r['obs_id']: r for r in rows(INPUT / 'query_cohort.csv') if r['obs_id'] in IDS}
    ships = {r['mmsi'] for r in cohort.values()}
    natural = [r for r in rows(NATURAL / 'observations.csv') if r['mmsi'] in ships]
    predictions = {r['obs_id']: r for r in rows(NATURAL / 'observation_predictions.csv') if r['mmsi'] in ships}
    for row in natural:
        pred = predictions.get(row['obs_id'], {})
        for key in ('r50_prediction', 'r50_margin', 'r18_prediction', 'r18_margin', 'dual_prediction', 'dual_margin'):
            row[key] = pred.get(key, '')
        row['prediction_source'] = pred.get('source_path', '')
        row['prediction_line'] = pred.get('source_line', '')
        row['vv_exists'] = Path(row['image_relpath_vv']).is_file()
        row['vh_exists'] = Path(row['image_relpath_vh']).is_file()
    pairs = [r for r in rows(NATURAL / 'pairs.csv') if r['mmsi'] in ships]
    l2 = [r for r in rows(INPUT / 'paired_effects.csv') if r['mmsi'] in ships]
    r22 = [r for r in rows(R22 / 'interventions.csv') if r['mmsi'] in ships]
    r22base = [r for r in rows(R22 / 'baseline.csv') if r['mmsi'] in ships]
    matching = [r for r in rows(MATCH / 'predictions.csv') if r['id'] in IDS | {'24470'}]
    trace = list(rows(INPUT / 'm3_path_trace_20261030/path_rows.csv'))
    off = {(r['seed'], r['id']): r for r in matching if r['arm'] == 'm3_off'}
    rescue = []
    for row in matching:
        if row['arm'] == 'soft':
            before = off[row['seed'], row['id']]
            if row['prediction'] == row['label'] and before['prediction'] != row['label']:
                rescue.append(dict(obs_id=row['id'], seed=row['seed'], carrier='N_level_pattern_spatial',
                    off_pred=before['prediction'], on_pred=row['prediction'], off_margin=before['margin'],
                    on_margin=row['margin'], margin_type='true_minus_max_other',
                    source_path=row['source_path'], source_line=row['source_line'], off_source_line=before['source_line']))
    for row in trace:
        if row['variant'] == 'original' and row['full_pred'] == row['target'] and row['off_pred'] != row['target']:
            rescue.append(dict(obs_id=row['obs_id'], seed=row['seed'], carrier='selected_full_20261027',
                off_pred=row['off_pred'], on_pred=row['full_pred'], off_margin=row['off_margin'],
                on_margin=row['full_margin'], margin_type='true_minus_fixed_rival_NOT_max_other',
                source_path=row['source_path'], source_line=row['source_line'], off_source_line=row['source_line']))
    summaries = []
    notes = {
        '26628': 'Same-carrier original rescue s17; VV body fine suppression revokes rescue; mean control preserves it. Natural dual predictions all correct.',
        '12457': 'Original rescue s17/29/43; detailed regional writeback removals; simple mean also rescues. L2 is an earlier carrier; natural partner 12459 is correct.',
        '37321': 'Original rescue s17/29/43; same-obs L2 plus R22 interventions on same-vessel 37326/37327. No regional M3 removal for this obs.',
        '10488': 'Original rescue s29; s17 rescue only after structure intervention. Natural dual observations both wrong; R18 changes correct to wrong.',
        '34278': 'Original rescue s17/29/43; natural dual W-C-C-W. Same-obs L2 exists, no detailed local M3 removal.',
        '30167': 'Original rescue s17/29/43; same-obs L2. Natural dual both correct.',
        '9140': 'Original rescue s17, revoked by VH side fine input suppression. Natural dual all wrong.',
        '294': 'Original rescue s29/s43 by argmax. s43 off fixed-rival margin is positive although another class wins; do not infer correctness from that margin.'
    }
    for obs_id in sorted(IDS, key=lambda value: (SHORTLIST.get(value, 99), int(value))):
        c = cohort[obs_id]
        n = [r for r in natural if r['mmsi'] == c['mmsi']]
        rs = [r for r in rescue if r['obs_id'] == obs_id]
        summaries.append(dict(rank=SHORTLIST.get(obs_id, ''), obs_id=obs_id, mmsi=c['mmsi'],
            class_id=c['class_id'], port=c['port'], pass_group=c['pass_group'],
            natural_obs_ids=';'.join(r['obs_id'] for r in n), natural_products=len({r['product'] for r in n}),
            rescue_seeds=';'.join(r['seed'] for r in rs),
            l2_same_obs_rows=sum(r['obs_id'] == obs_id for r in l2),
            r22_same_ship_obs=';'.join(sorted({r['obs_id'] for r in r22 if r['mmsi'] == c['mmsi']})),
            notes=notes[obs_id], cohort_source=c['source_path'], cohort_line=c['source_line']))
    counts = {}
    for name, data in [('candidates.csv', summaries), ('shortlist.csv', [r for r in summaries if r['rank']]),
        ('l1_observations.csv', natural), ('l1_pairs.csv', pairs), ('l2_same_obs.csv', l2),
        ('l2_round22_same_ship.csv', r22), ('l1_round22_baseline.csv', r22base),
        ('l3_matching.csv', matching), ('l3_path_rows.csv', trace), ('original_m3_rescues.csv', rescue),
        ('query_cohort_selected.csv', cohort.values())]:
        counts[name] = write(name, data)
    receipt = dict(action='Read existing evidence and write joins only; no model execution.',
        root=str(ROOT), counts=counts, natural_views_missing=[
            {'obs_id': r['obs_id'], 'pol': pol} for r in natural for pol in ('vv', 'vh') if not r[f'{pol}_exists']],
        caveats=['Historical model stages differ.', 'Natural same_pass means orbit direction; use product/acq_time for distinct overpasses.',
                 'path_rows margins use fixed rivals; top-1 predictions determine rescue.',
                 'Matching seeds share selected_s17 upstream backbone.', 'Source rows are posthoc historical development evidence.'])
    (OUT / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
