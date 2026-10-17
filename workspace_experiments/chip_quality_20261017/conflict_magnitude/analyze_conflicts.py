import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('E:/临时会话/chip_quality_20261017')
SRC = BASE / 'ais_metadata'
OUT = BASE / 'conflict_magnitude'


def rows(name):
    with (SRC / name).open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def valid_pair(values):
    try:
        l, w = map(float, values)
        return (l, w) if math.isfinite(l) and math.isfinite(w) and l >= w > 0 else None
    except (ValueError, TypeError):
        return None


def imos(values):
    return {str(x).strip() for x in values if str(x).strip() not in ('', '0', 'None', 'nan')}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def write_csv(name, data):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def percentile(values, p):
    a = sorted(values)
    pos = (len(a) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vessels = defaultdict(lambda: dict(pairs=set(), classes=set(), ports=set(), imos=set(), n=0, ids=set()))
    input_counts = {}
    conflict_file_mmsi = set()
    for name in ('ais_metadata_sidecar.csv', 'dimension_conflicts.csv'):
        count = 0
        for row in rows(name):
            count += 1
            m = row['mmsi']
            v = vessels[m]
            for values in json.loads(row['vessel_static_variants_json'] or '[]'):
                pair = valid_pair(values)
                if pair:
                    v['pairs'].add(pair)
            v['imos'].update(imos(json.loads(row['vessel_imo_variants_json'] or '[]')))
            if name == 'ais_metadata_sidecar.csv':
                v['n'] += 1
                v['ids'].add(row['instance_id'])
                v['classes'].add(row['class_name'])
                v['ports'].add(row['port'])
            else:
                conflict_file_mmsi.add(m)
        input_counts[name] = count
    selected = {m: v for m, v in vessels.items() if len(v['pairs']) > 1}
    evidence = defaultdict(list)
    evidence_count = 0
    for row in rows('dimension_evidence.csv'):
        evidence_count += 1
        if row['mmsi'] in selected:
            evidence[row['mmsi']].append(row)
    input_counts['dimension_evidence.csv'] = evidence_count
    results = []
    for m, v in sorted(selected.items()):
        ls, ws = zip(*v['pairs'])
        ev = evidence[m]
        ev_imos = imos([r['imo'] for r in ev])
        all_imos = v['imos'] | ev_imos
        result = dict(mmsi=m, class_names=encoded(sorted(v['classes'])), class_conflict=len(v['classes']) > 1,
                      ports=encoded(sorted(v['ports'])), sidecar_record_count=v['n'],
                      unique_instance_count=len(v['ids']), evidence_record_count=len(ev),
                      unique_evidence_source_lines=len({(r['source_txt'], r['source_line']) for r in ev}),
                      dimension_variant_count=len(v['pairs']), valid_LW_variants_json=encoded(sorted(v['pairs'])),
                      L_min_m=min(ls), L_max_m=max(ls), L_range_m=max(ls)-min(ls), L_max_min_ratio=max(ls)/min(ls),
                      W_min_m=min(ws), W_max_m=max(ws), W_range_m=max(ws)-min(ws), W_max_min_ratio=max(ws)/min(ws),
                      imo_variants_json=encoded(sorted(all_imos)), imo_conflict=len(all_imos)>1,
                      sidecar_imo_variants_json=encoded(sorted(v['imos'])), evidence_imo_variants_json=encoded(sorted(ev_imos)))
        result['L_or_W_at_least_2x'] = result['L_max_min_ratio'] >= 2 or result['W_max_min_ratio'] >= 2
        result['mild_L_le5_W_le2'] = result['L_range_m'] <= 5 and result['W_range_m'] <= 2
        results.append(result)
    n = len(results)
    bins = {
        'L_range_le2m': sum(r['L_range_m'] <= 2 for r in results),
        'L_range_gt2_le10m': sum(2 < r['L_range_m'] <= 10 for r in results),
        'L_range_gt10_le50m': sum(10 < r['L_range_m'] <= 50 for r in results),
        'L_range_gt50m': sum(r['L_range_m'] > 50 for r in results),
        'W_range_le1m': sum(r['W_range_m'] <= 1 for r in results),
        'W_range_gt1_le5m': sum(1 < r['W_range_m'] <= 5 for r in results),
        'W_range_gt5m': sum(r['W_range_m'] > 5 for r in results),
        'L_or_W_at_least_2x': sum(r['L_or_W_at_least_2x'] for r in results),
        'mild_L_le5_W_le2': sum(r['mild_L_le5_W_le2'] for r in results),
    }
    summary = dict(conflict_mmsi_count=n, expected_290_confirmed=n == 290,
                   input_rows=input_counts, all_sidecar_mmsi_count=sum(1 for m, v in vessels.items() if m.strip() and v['n'] > 0),
                   dimension_conflicts_unique_mmsi=len(conflict_file_mmsi),
                   missing_from_conflicts_csv=sorted(set(selected)-conflict_file_mmsi),
                   conflicts_csv_without_multiple_valid_pairs=sorted(conflict_file_mmsi-set(selected)),
                   size_validity='finite L >= W > 0; union of vessel_static_variants_json by MMSI',
                   percentile_method='linear interpolation at (n-1)*p, each MMSI weighted once',
                   thresholds_are_descriptive_not_error_truth=True, bins=bins,
                   class_conflict_mmsi=sum(r['class_conflict'] for r in results),
                   imo_conflict_mmsi=sum(r['imo_conflict'] for r in results),
                   sidecar_records_for_conflicting_mmsi=sum(r['sidecar_record_count'] for r in results),
                   class_memberships=dict(Counter(c for v in selected.values() for c in v['classes'])))
    for key in ('L_range_m', 'W_range_m', 'L_max_min_ratio', 'W_max_min_ratio'):
        vals = [r[key] for r in results]
        summary[key] = dict(median=percentile(vals, .5), p90=percentile(vals, .9), maximum=max(vals))

    # Select real pairs; independent length/width extrema need not form a real pair.
    examples = []
    chosen = set()
    selectors = [
        ('轻微', lambda r: r['mild_L_le5_W_le2'], lambda r: (r['L_range_m']+r['W_range_m']), False),
        ('中等', lambda r: 5 < r['L_range_m'] <= 10 and r['W_range_m'] <= 5, lambda r: r['L_range_m'], False),
        ('大幅', lambda r: 10 < r['L_range_m'] <= 50, lambda r: r['L_range_m'], True),
        ('极端长度', lambda r: True, lambda r: r['L_range_m'], True),
        ('极端宽度', lambda r: True, lambda r: r['W_range_m'], True),
    ]
    for label, predicate, key, reverse in selectors:
        candidates = sorted([r for r in results if predicate(r) and r['mmsi'] not in chosen], key=key, reverse=reverse)
        r = candidates[0]
        m = r['mmsi']
        chosen.add(m)
        pairs = sorted(selected[m]['pairs'])
        axis = 1 if label == '极端宽度' or r['L_range_m'] == 0 else 0
        pair1, pair2 = min(pairs, key=lambda p:p[axis]), max(pairs, key=lambda p:p[axis])
        for variant, pair in enumerate((pair1, pair2), 1):
            e = next(e for e in evidence[m] if valid_pair((e['length_m'], e['width_m'])) == pair)
            examples.append(dict(group=label, mmsi=m, variant=variant, length_m=pair[0], width_m=pair[1],
                                 L_range_m=r['L_range_m'], W_range_m=r['W_range_m'],
                                 imo_conflict=r['imo_conflict'], source_txt=e['source_txt'],
                                 source_line=int(float(e['source_line'])), evidence_imo=e['imo'],
                                 raw_verified=False, raw_line=''))
    by_path = defaultdict(list)
    for e in examples:
        by_path[e['source_txt']].append(e)
    for path, targets in by_path.items():
        lookup = defaultdict(list)
        for e in targets:
            lookup[e['source_line']].append(e)
        with Path(path).open(encoding='utf-8-sig', errors='replace') as f:
            for line_no, line in enumerate(f, 1):
                if line_no in lookup:
                    fields = next(csv.reader([line]))
                    for e in lookup[line_no]:
                        e['raw_line'] = line.rstrip('\r\n')
                        e['raw_verified'] = (fields[1] == e['mmsi'] and
                                             valid_pair((fields[14], fields[15])) == (e['length_m'], e['width_m']) and
                                             fields[12] == e['evidence_imo'])
                if line_no >= max(lookup):
                    break
    summary['raw_verification'] = dict(mmsi_count=len(chosen), records=len(examples),
                                       verified_records=sum(e['raw_verified'] for e in examples),
                                       raw_columns_1_based=dict(mmsi=2, imo=13, length=15, width=16))
    write_csv('mmsi_conflict_magnitude.csv', results)
    write_csv('verified_examples.csv', examples)
    with (OUT / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    lines = [
        '# MMSI 静态尺寸冲突幅度', '',
        f'按 MMSI 去重并合并有效尺寸集合后，共 **{n} 个 MMSI** 存在多组尺寸；是否确认 290：{n == 290}。',
        f'dimension_conflicts.csv 去重为 {len(conflict_file_mmsi)} 个 MMSI；两集合差集分别为 {summary["missing_from_conflicts_csv"]}、{summary["conflicts_csv_without_multiple_valid_pairs"]}。',
        '', '## 幅度统计', '', '|指标|中位数|P90|最大值|', '|---|---:|---:|---:|',
    ]
    for key in ('L_range_m', 'W_range_m', 'L_max_min_ratio', 'W_max_min_ratio'):
        s = summary[key]
        lines.append(f'|{key}|{s["median"]:.6g}|{s["p90"]:.6g}|{s["maximum"]:.6g}|')
    lines += ['',
              f'长度最大/最小比的中位数为 **{summary["L_max_min_ratio"]["median"]:.6f}**，即相对较短尺寸记录的长度差中位数为 **{(summary["L_max_min_ratio"]["median"]-1)*100:.2f}%**。'
              f'这 {n} 个冲突 MMSI 中有 **{summary["class_memberships"].get("fishing_vessel", 0)} 个标注为渔船**；不能仅凭长度绝对差中位数为 6 m，就将这些冲突都理解为不重要。相对差与类别背景应一并考虑，但它们仍不构成错误真值。',
              '', '## 明确分档', '', '|条件|MMSI 数|占比|', '|---|---:|---:|']
    for key, count in bins.items():
        lines.append(f'|{key}|{count}|{count/n:.2%}|')
    lines += ['', '**这些阈值只描述差异大小，不是错误真值，也不确定哪组尺寸正确。** L/W 分档各自互斥完备；两倍组与轻微组为额外条件，不应与分档相加。',
              '', '## 类别、记录数与 IMO', '',
              f'冲突 MMSI 对应 sidecar 样本记录 {summary["sidecar_records_for_conflicting_mmsi"]} 条；类别多值 {summary["class_conflict_mmsi"]} 个 MMSI；非空非零 IMO 多值 {summary["imo_conflict_mmsi"]} 个 MMSI。',
              f'类别归属计数（多类别 MMSI 会计入各类别）：{encoded(summary["class_memberships"])}。',
              '逐 MMSI CSV 包含类别、港口、样本记录数、唯一 instance 数、尺寸证据记录数、唯一源文件/行号数、全部有效尺寸、L/W 最小/最大/差值/倍数和 IMO 集合。证据记录数不代表独立测量次数。',
              '', '## 原始 TXT 回读示例', '', '|幅度标签|MMSI|尺寸组 1 (L,W), m|尺寸组 2 (L,W), m|ΔL / ΔW, m|两条原始记录验证|', '|---|---|---|---|---|---|']
    for a, b in zip(examples[::2], examples[1::2]):
        lines.append(f'|{a["group"]}|{a["mmsi"]}|({a["length_m"]:g}, {a["width_m"]:g})|({b["length_m"]:g}, {b["width_m"]:g})|{a["L_range_m"]:g} / {a["W_range_m"]:g}|{a["raw_verified"] and b["raw_verified"]}|')
    lines += ['', '上述标签用于抽样展示，不构成另一个完备分档。verified_examples.csv 保留 source_txt、1-based source_line 和完整原始行，可逐条定位。仅回读这些示例所需 TXT；验证同一 MMSI 下两组尺寸确实存在，不等于证明属于同一条物理船。',
              '', '## 统计口径与限制', '',
              '- 有效尺寸为有限数且 L >= W > 0；仅合并指定两表的 vessel_static_variants_json，dimension_evidence 用于记录数、IMO 和原始行验证。',
              '- 每个 MMSI 等权；P90 使用排序后 (n-1)*0.9 位置的线性插值。min/max 按维度分别计算，两个维度的极值未必属于同一组实测尺寸。',
              '- IMO 冲突依据 sidecar 的 vessel_imo_variants_json 与 dimension_evidence.imo 的非空非零并集，不将 IMO 单值作为物理身份已确认。',
              '- 只描述源记录冲突；不据此认定录入错误、身份复用、尺度真值或变化原因。不涉及 4558 长度差异成因交叉、检测或训练。', '']
    (OUT / 'FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('EXAMPLES', encoded(examples))
    if not all(e['raw_verified'] for e in examples):
        raise RuntimeError('Some raw examples did not verify; inspect verified_examples.csv')


if __name__ == '__main__':
    main()
