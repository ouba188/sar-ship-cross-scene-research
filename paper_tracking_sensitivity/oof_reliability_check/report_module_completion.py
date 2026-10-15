"""Read completed module experiments; never substitute an unfinished run."""

import csv
import json
from pathlib import Path


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT/'module_completion_20261008'
ARMS = ('reference', 'complement_only', 'aligned_only', 'joint')
CORE = (
    ('self24ports', 'matched_ports_joint_reference'),
    ('F0921', 'matched_F0921_reference'),
    ('OSS1', 'original_reference'),
    ('OSS2', 'original_reference'),
    ('pacs', 'retrained_rgb_reference'),
    ('officehome', 'retrained_rgb_reference'),
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def result_row(dataset, protocol, value, path):
    return dict(dataset=dataset, protocol=protocol, arm=value['arm'],
                oa_pct=100*value['acc'], ba_pct=100*value['ba'],
                expert_oa_pct=100*value['expert_acc'], expert_ba_pct=100*value['expert_ba'],
                artifact=str(path))


def write_module_coverage(rows):
    snapshot = ROOT/'module_evidence_20261008/tables/module_coverage.csv'
    with snapshot.open(encoding='utf-8-sig', newline='') as stream:
        coverage = list(csv.DictReader(stream))
    names = {'self24ports': 'self24ports', 'correctedF0921': 'F0921',
             'publicOSS1': 'OSS1', 'publicOSS2': 'OSS2', 'PACS': 'pacs', 'OfficeHome': 'officehome'}
    protocols = dict(CORE)
    modules = {'M2': 'complement_only', 'M3': 'aligned_only', 'joint_M2_M3': 'joint'}
    index = {(r['dataset'], r['protocol'], r['arm']): r for r in rows}
    for record in coverage:
        record['matched_reference_oa_pct'] = ''
        record['matched_reference_ba_pct'] = ''
        arm = modules.get(record['module'])
        if arm is None or record['dataset'] not in names:
            continue
        dataset = names[record['dataset']]
        protocol = protocols[dataset]
        if dataset in ('pacs', 'officehome') and record['target'] != 'equal_domain_mean':
            dataset += '/'+record['target'].replace(' ', '_')
        value = index.get((dataset, protocol, arm))
        if value is None:
            record.update(status='pending_current_matched_run',
                          missing_reason='See current completion status; no substitute reference is used.')
            continue
        control = index[dataset, protocol, 'reference']
        record.update(status='completed_matched_ablation', oa_pct=value['oa_pct'],
                      ba_pct=value['ba_pct'], delta_oa_pp=value['delta_oa_pp'],
                      delta_ba_pp=value['delta_ba_pp'], artifact_path=value['artifact'],
                      comparison=arm+'-matched_reference', round_id=protocol, missing_reason='',
                      matched_reference_oa_pct=control['oa_pct'], matched_reference_ba_pct=control['ba_pct'])
    with (OUT/'module_coverage_completed.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)


def main():
    rows = []
    sources = [
        ('self24ports', 'original_reference', OUT/'ports_projection/summary.json', 'means'),
        ('F0921', 'matched_F0921_reference', ROOT/'joint_residual_pilot_20261008/summary.json', 'means'),
        ('OSS1', 'original_reference', OUT/'oss1/summary.json', 'means'),
        ('OSS2', 'original_reference', OUT/'oss2/summary.json', 'means'),
    ]
    for dataset, protocol, path, key in sources:
        result = read(path)
        if result:
            for value in result[key]:
                rows.append(result_row(dataset, protocol, value, path))
    generic_path = OUT/'generic/summary.json'
    generic = read(generic_path)
    if generic:
        for value in generic['domain_macro_means']:
            rows.append(result_row(value['dataset'], 'original_reference', value, generic_path))
    ports_joint_path = OUT/'ports_projection/joint_summary.json'
    ports_joint = read(ports_joint_path)
    if ports_joint and ports_joint.get('n_ports') == 23:
        for value in ports_joint['means']:
            rows.append(result_row('self24ports', 'matched_ports_joint_reference', value, ports_joint_path))
    retrained = OUT/'generic/rgb_retrained'
    for dataset in ('pacs', 'officehome'):
        domain_files = sorted((retrained/dataset).glob('*/summary.json'))
        for path in domain_files:
            result = read(path)
            for value in result['domain_macro_means']:
                rows.append(result_row(f'{dataset}/{path.parent.name}', 'retrained_rgb_reference', value, path))
        if len(domain_files) == 4:
            for arm in ARMS:
                selected = [r for r in rows if r['dataset'].startswith(dataset+'/') and r['arm'] == arm]
                rows.append(dict(dataset=dataset, protocol='retrained_rgb_reference', arm=arm,
                    oa_pct=sum(r['oa_pct'] for r in selected)/4,
                    ba_pct=sum(r['ba_pct'] for r in selected)/4,
                    expert_oa_pct=sum(r['expert_oa_pct'] for r in selected)/4,
                    expert_ba_pct=sum(r['expert_ba_pct'] for r in selected)/4,
                    artifact='; '.join(r['artifact'] for r in selected)))
    for row in rows:
        controls = [r for r in rows if r['dataset'] == row['dataset'] and
                    r['protocol'] == row['protocol'] and r['arm'] == 'reference']
        control = controls[0]
        row['delta_oa_pp'] = row['oa_pct']-control['oa_pct']
        row['delta_ba_pp'] = row['ba_pct']-control['ba_pct']
        row['delta_expert_oa_pp'] = row['expert_oa_pct']-control['expert_oa_pct']
        row['delta_expert_ba_pp'] = row['expert_ba_pct']-control['expert_ba_pct']
    (OUT/'completion_results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    with (OUT/'completion_results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Module completion: completed results', '', 'Updated 2026-10-08. OA/BA: percent; changes: percentage points.', '',
        'Only completed summary files enter this table. F0921 is reused from the earlier completed pilot.',
        'Original-reference and newly retrained RGB-reference comparisons remain separate.', '',
        '| Dataset | Reference protocol | Arm | OA | BA | Delta OA | Delta BA |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f'| {row["dataset"]} | {row["protocol"]} | {row["arm"]} | '
                     f'{row["oa_pct"]:.6f} | {row["ba_pct"]:.6f} | '
                     f'{row["delta_oa_pp"]:+.6f} | {row["delta_ba_pp"]:+.6f} |')
    lines += ['', 'Absent cells are not complete. The CSV/JSON preserve the exact source artifact for every row.',
        'Budget: shared 64 labels per target task, five support draws, at most 5% query handoffs.',
        'These are development evaluations, not independent blind confirmation or five neural-training seeds.',
        'Port/domain averages are unweighted across eligible target tasks, not pooled by sample count.']
    (OUT/'COMPLETED_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    core_rows = [r for dataset, protocol in CORE for r in rows
                 if r['dataset'] == dataset and r['protocol'] == protocol]
    with (OUT/'core_matched_results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(core_rows)
    core_lines = ['# Matched four-arm results', '',
        'Each cell is full-system OA / BA (%), not an isolated module accuracy.',
        'M1 and M4 remain active in every arm. Each row uses its own matched reference.', '',
        '| Dataset | Reference | M2 only | M3 only | M2+M3 | Joint delta OA / BA (pp) |',
        '| --- | --- | --- | --- | --- | --- |']
    complete = 0
    for dataset, protocol in CORE:
        selected = {r['arm']: r for r in core_rows if r['dataset'] == dataset}
        cells = []
        for arm in ARMS:
            value = selected.get(arm)
            cells.append(f'{value["oa_pct"]:.6f} / {value["ba_pct"]:.6f}' if value else 'pending')
        value = selected.get('joint')
        delta = f'{value["delta_oa_pp"]:+.6f} / {value["delta_ba_pp"]:+.6f}' if value else 'pending'
        core_lines.append('| '+dataset+' | '+' | '.join(cells)+' | '+delta+' |')
        complete += len(selected) == 4
    core_lines += ['', f'Complete dataset groups: {complete}/6.', '',
        '## Expert versus full-system changes', '',
        'Expert scores evaluate the expert alone on every query, without the 5% handoff rule.', '',
        '| Dataset | Reference expert OA / BA | Joint expert OA / BA | Expert delta OA / BA (pp) |',
        '| --- | --- | --- | --- |']
    for dataset, protocol in CORE:
        selected = {r['arm']: r for r in core_rows if r['dataset'] == dataset}
        if 'joint' not in selected:
            continue
        control, value = selected['reference'], selected['joint']
        core_lines.append(f'| {dataset} | {control["expert_oa_pct"]:.6f} / {control["expert_ba_pct"]:.6f} | '
                          f'{value["expert_oa_pct"]:.6f} / {value["expert_ba_pct"]:.6f} | '
                          f'{value["delta_expert_oa_pp"]:+.6f} / {value["delta_expert_ba_pp"]:+.6f} |')
    core_lines += ['',
        'Ports: 23 eligible targets, equal-port averaging; Melbourne has 64 rows and no remaining query.',
        'Ports use the same float16-prefix initial tail in all four arms; old-cache M2 is reported separately.',
        'PACS/OfficeHome: four-domain averages of rebuilt source RGB encoders, not historical feature replay.',
        'F0921: reuse the completed matched pilot. OSS1/OSS2: original reference replay is exact.',
        'Shared64 target labels, five support draws, at most 5% handoffs; one neural-training seed.',
        'No target-query model selection; historical development data, not blind confirmation.',
        'FUSAR, FMoW and Camelyon17 are not completed current-protocol joint experiments in this table.']
    (OUT/'CORE_MATCHED_RESULTS.md').write_text('\n'.join(core_lines)+'\n', encoding='utf-8')
    write_module_coverage(rows)
    print(f'Saved {len(rows)} completed result rows')


if __name__ == '__main__':
    main()
