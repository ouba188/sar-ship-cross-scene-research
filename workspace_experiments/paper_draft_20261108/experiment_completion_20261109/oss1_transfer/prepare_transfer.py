"""Prepare the fixed OSS1 cohort and source exposure audit, CPU input work only."""
import csv
import importlib.util
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
CLOSE = WORK / 'scene_aware_classification_20261018/model_closeout_20261023'
OLD = CLOSE / 'public_m123_cross_dataset_20261026/oss2-to-oss1'
OSS1 = CLOSE / 'oss1_m123_replication_20261026'
PUBLIC = CLOSE / 'final_oss2_replication_20261029'
CURRENT = WORK / 'm1_guided_m2_20261106'
COVERAGE = WORK / 'm1_coverage_20261101'
M3 = WORK / 'm3_frequency_increment_20261028/source_hard_task_20261101/oss2'
INPUT = WORK / 'sar_m123_standalone/sar_m123/input.py'
CLASS_IDS = (0, 2, 3, 4)
CLASSES = ('bulk_carrier', 'general_cargo', 'product_chemical_tanker', 'container_ship')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def csv_write(name, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flag(value):
    return str(value).lower() in ('true', '1')


def orbit(row):
    # Product token 6 is absolute orbit; data-take suffix is deliberately excluded.
    scene = row['scene_id'].split('_')
    key = scene[0] + '_' + scene[6]
    assert len(scene[6]) == 6 and scene[6].isdigit(), row['scene_id']
    assert key == '_'.join(row['pass_group'].split('_')[:2]), row
    return key


def summary(rows):
    return dict(n=len(rows), ships=len({r['mmsi'] for r in rows}),
                platform_absolute_orbits=len({orbit(r) for r in rows}),
                legacy_pass_groups=len({r['pass_group'] for r in rows}),
                classes={str(c): sum(r['label'] == c for r in rows) for c in range(4)})


def corners(raw):
    cx, cy, width, height, angle = raw
    c, s = math.cos(angle), math.sin(angle)
    return [[cx+x*c-y*s, cy+x*s+y*c] for x, y in
            ((-width/2, -height/2), (width/2, -height/2),
             (width/2, height/2), (-width/2, height/2))]


def make_view(v, geometry):
    if not flag(v['exists']):
        return None
    detections, raw, axis = [], None, None
    if flag(v['detected']) and float(v['confidence']) >= .25:
        raw = json.loads(v['raw_xywhr'])
        assert len(raw) == 5 and all(math.isfinite(x) for x in raw)
        assert min(raw[2:4]) > 0
        detections = [dict(corners=corners(raw), confidence=float(v['confidence']))]
        if flag(v['axis_valid']):
            axis = dict(center=[float(v['cx']), float(v['cy'])],
                        length_px=float(v['length']), width_px=float(v['width']),
                        angle=float(v['ship_axis']))
    return dict(image=v['path'], detections=detections,
                target_index=0 if detections else None, ship_axis=axis,
                radar_axis_rad=float(geometry['radar_axis_rad'])
                if flag(geometry['radar_axis_valid']) else None, raw_xywhr=raw)


def inspect_view(record, pol, prepare_view):
    view = record[pol]
    result = dict(id=record['id'], obs_id=record['indices']['obs_id'], pol=pol,
                  image=view['image'] if view else '', exists_now=False,
                  configured=view is not None, prepare_ok=False, valid_target=False,
                  error='')
    if view is None:
        result['error'] = 'view_not_configured'
        return result
    result['exists_now'] = Path(view['image']).is_file()
    try:
        prepared = prepare_view(view)
        result.update(prepare_ok=True, valid_target=bool(prepared['valid_target']),
                      native_h=int(prepared['raw'].shape[0]),
                      native_w=int(prepared['raw'].shape[1]),
                      dtype=str(prepared['raw'].dtype),
                      canvas_h=int(prepared['pixels'].shape[0]),
                      canvas_w=int(prepared['pixels'].shape[1]),
                      placement=json.dumps(prepared['placement']),
                      prepared_geometry=json.dumps(prepared['geometry'].tolist())
                      if prepared['geometry'] is not None else '',
                      nonzero_pixels=int((prepared['pixels'] != 0).sum()))
    except (OSError, ValueError) as exc:
        result['error'] = str(exc)
    return result


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    print('Loading CPU input reader', flush=True)
    # Import only the established NumPy/Pillow input reader, never model modules.
    spec = importlib.util.spec_from_file_location('oss1_native_input', INPUT)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    print('Loading source manifests and training exposure', flush=True)
    groups = read(CURRENT / 'groups.json')
    rows = read(COVERAGE / 'manifest.json')
    parts = {name: [rows[i] for i in ids] for name, ids in groups.items()}
    for name in ('source_fit', 'source_adapt', 'validation', 'selection', 'evaluation', 'references'):
        assert parts[name] == read(PUBLIC / 'manifests' / (name + '.json')), name
    assert len(rows) == 1519 and len(parts['small']) == 719
    assert groups['small'] == groups['source_fit'] + groups['source_adapt']
    assert len(parts['references']) == 32
    assert set(groups['references']) <= set(groups['source_fit'])
    assert Counter(r['label'] for r in parts['references']) == Counter({c: 8 for c in range(4)})
    old_bank = csv_read(CLOSE / 'oss2_m123_replication_20261026/m3_results/reference_bank.csv')
    assert [r['indices']['obs_id'] for r in parts['references']] == [int(r['obs_id']) for r in old_bank]

    occurrences, evidence = {}, []
    order_paths = [(f'current_class_s{s}', CURRENT / f'class_s{s}/orders.json') for s in (17, 29, 43)]
    order_paths += [('upstream_m1_s17', COVERAGE / 'small_class_s17/orders.json')]
    for stage, path in order_paths:
        batches = read(path)
        counts = Counter(i for batch in batches for i in batch)
        assert set(counts) == set(groups['small']), stage
        occurrences[stage] = counts
        evidence.append(dict(stage=stage, evidence=str(path), batches=len(batches),
                             observations=len(counts), exposures=sum(counts.values())))
    episodes = read(M3 / 'aggregate_17/episodes.json')
    for key, stage in (('ids', 'm3_training_queries'), ('references', 'm3_episode_references')):
        counts = Counter(i for episode in episodes for i in episode[key])
        assert set(counts) <= set(groups['small'])
        if key == 'ids':
            assert set(counts) == set(groups['small'])
        occurrences[stage] = counts
        evidence.append(dict(stage=stage, evidence=str(M3 / 'aggregate_17/episodes.json'),
                             batches=len(episodes), observations=len(counts), exposures=sum(counts.values())))

    print('Loading target inventory and geometry', flush=True)
    target_table = csv_read(OLD / 'target_table.csv')
    eligibility = {int(r['obs_id']): r for r in csv_read(OLD / 'target_eligibility.csv')}
    inventory = csv_read(OSS1 / 'view_inventory.csv')
    views = {(int(v['obs_id']), v['pol']): v for v in inventory}
    assert len(views) == len(inventory)
    geometry_path = WORK / 'oss1_joint_geometry_20261017/geometry.csv'
    geometry = {r['sample_id']: r for r in csv_read(geometry_path)}
    print('Preparing target chips sequentially', flush=True)
    targets, view_results, joined = [], [], []
    for i, row in enumerate(target_table):
        obs = int(row['obs_id'])
        assert int(row['query_row']) == i and flag(eligibility[obs]['included'])
        g = geometry[row['sample_id']]
        label = int(row['class_id'])
        assert int(g['class_id']) == CLASS_IDS[label]
        assert g['mmsi'] == row['mmsi'] and g['scene_id'] == row['scene_id']
        record = dict(id=row['sample_id'], mmsi=row['mmsi'], pass_group=row['pass_group'],
                      scene_id=row['scene_id'], label=label, original_class_id=CLASS_IDS[label],
                      class_name=CLASSES[label], split='evaluation', role='evaluation',
                      dataset='OpenSARShip1', input_scale=1.0, canvas_size=160,
                      indices=dict(obs_id=obs, target_manifest_row=int(row['target_manifest_row']),
                                   query_row=i, legacy_combined_row=int(row['combined_row'])),
                      source_paths={p: row['chip_path_'+p] for p in ('vv', 'vh')},
                      raw_patch_path=row['raw_patch_path'],
                      calibrated_patch_path=row['calibrated_patch_path'],
                      ship_xml_path=row['ship_xml_path'], metadata_xml_path=row['metadata_xml_path'])
        record['platform_absolute_orbit'] = orbit(record)
        trace = dict(row, id=record['id'], platform_absolute_orbit=orbit(record),
                     original_class_id=CLASS_IDS[label], class_name=CLASSES[label])
        checks = []
        for pol in ('vv', 'vh'):
            v = views[(obs, pol)]
            assert v['sample_id'] == record['id']
            assert Path(v['path']) == Path(row['chip_path_'+pol])
            record[pol] = make_view(v, g)
            check = inspect_view(record, pol, native.prepare_view)
            check.update({f'inventory_{k}': value for k, value in v.items()})
            check['historical_valid'] = flag(row['valid_'+pol])
            check['availability_matches_history'] = check['valid_target'] == check['historical_valid']
            view_results.append(check)
            checks.append(check)
            trace[pol+'_exists_now'] = check['exists_now']
            trace[pol+'_prepare_ok'] = check['prepare_ok']
            trace[pol+'_usable_now'] = check['valid_target']
        trace['input_ready'] = all(c['prepare_ok'] for c in checks if c['configured']) and any(c['valid_target'] for c in checks)
        record['native_size'] = [int(views[(obs, 'vv')]['native_w']), int(views[(obs, 'vv')]['native_h'])]
        for key in ('raw_patch_path', 'calibrated_patch_path', 'ship_xml_path', 'metadata_xml_path'):
            trace[key+'_exists_now'] = Path(row[key]).is_file()
        targets.append(record)
        joined.append(trace)
        if (i+1) % 200 == 0:
            print(f'CPU input preparation: {i+1}/{len(target_table)}', flush=True)
    assert len(targets) == 921 and len({r['id'] for r in targets}) == 921
    assert Counter(r['label'] for r in targets) == Counter({0: 675, 1: 71, 2: 72, 3: 103})

    reference_checks = [inspect_view(r, p, native.prepare_view) for r in parts['references'] for p in ('vv', 'vh')]
    ref_status = {r['id']: all(c['prepare_ok'] for c in reference_checks if c['id'] == r['id'] and c['configured'])
                  and any(c['valid_target'] for c in reference_checks if c['id'] == r['id'])
                  for r in parts['references']}
    cohorts = {name: parts[name] for name in ('small', 'references', 'validation', 'selection', 'evaluation')}
    cohorts['train_and_development_947'] = parts['small'] + parts['validation']
    cohorts['historical_four_roles_1141'] = parts['small'] + parts['validation'] + parts['selection']
    overlaps, overlap_summary = [], {}
    for name, source in cohorts.items():
        ship_index, orbit_index = {}, {}
        for s in source:
            ship_index.setdefault(s['mmsi'], []).append(s['id'])
            orbit_index.setdefault(orbit(s), []).append(s['id'])
        ship_hits, orbit_hits = [], []
        for r, trace in zip(targets, joined):
            sm, so = ship_index.get(r['mmsi'], []), orbit_index.get(orbit(r), [])
            trace[name+'_mmsi_overlap'] = bool(sm)
            trace[name+'_orbit_overlap'] = bool(so)
            if sm: ship_hits.append(r)
            if so: orbit_hits.append(r)
            if sm or so:
                overlaps.append(dict(cohort=name, target_id=r['id'], target_obs_id=r['indices']['obs_id'],
                                     mmsi=r['mmsi'], platform_absolute_orbit=orbit(r),
                                     matching_source_mmsi_ids=json.dumps(sm), matching_source_orbit_ids=json.dumps(so)))
        overlap_summary[name] = dict(source=summary(source), target_same_mmsi=summary(ship_hits),
                                     target_same_orbit=summary(orbit_hits))
    for row in joined:
        row['strict_eligible'] = not (row['historical_four_roles_1141_mmsi_overlap'] or row['historical_four_roles_1141_orbit_overlap'])
        row['eligible_and_input_ready'] = row['strict_eligible'] and row['input_ready']
        row['independent_of_oss2_evaluation'] = not (row['evaluation_mmsi_overlap'] or row['evaluation_orbit_overlap'])

    source_audit = []
    for i, r in enumerate(rows):
        item = dict(current_row=i, id=r['id'], obs_id=r['indices']['obs_id'], mmsi=r['mmsi'],
                    split=r['split'], class_id=r['label'], platform_absolute_orbit=orbit(r),
                    pass_group=r['pass_group'], scene_id=r['scene_id'],
                    fixed_reference=i in groups['references'])
        for stage, counts in occurrences.items():
            item[stage+'_exposures'] = counts[i]
        for p in ('vv', 'vh'):
            item[p+'_image'] = r[p]['image'] if r[p] else ''
            item[p+'_exists_now'] = Path(r[p]['image']).is_file() if r[p] else False
            item[p+'_raw_path'] = r['source_paths'][p]
        source_audit.append(item)

    write('target_manifest.json', targets)
    write('references.json', parts['references'])
    write('source719_manifest.json', parts['small'])
    independent = [r for r, t in zip(targets, joined) if t['eligible_and_input_ready'] and t['independent_of_oss2_evaluation']]
    write('target_manifest_no_oss2_eval_overlap.json', independent)
    csv_write('target_manifest.csv', joined)
    csv_write('target_views.csv', view_results)
    csv_write('reference_views.csv', reference_checks)
    csv_write('source_exposure.csv', source_audit)
    csv_write('source719.csv', [source_audit[i] for i in groups['small']])
    csv_write('overlap_details.csv', overlaps, ['cohort', 'target_id', 'target_obs_id', 'mmsi',
                                             'platform_absolute_orbit', 'matching_source_mmsi_ids', 'matching_source_orbit_ids'])
    class_rows = []
    for c in range(4):
        tt = [t for t in joined if int(t['class_id']) == c]
        class_rows.append(dict(class_id=c, original_class_id=CLASS_IDS[c], class_name=CLASSES[c],
            n=len(tt), ships=len({t['mmsi'] for t in tt}),
            both_usable=sum(t['vv_usable_now'] and t['vh_usable_now'] for t in tt),
            vv_only=sum(t['vv_usable_now'] and not t['vh_usable_now'] for t in tt),
            vh_only=sum(t['vh_usable_now'] and not t['vv_usable_now'] for t in tt),
            missing_vv=sum(not t['vv_exists_now'] for t in tt),
            missing_vh=sum(not t['vh_exists_now'] for t in tt),
            input_not_ready=sum(not t['input_ready'] for t in tt),
            strict_excluded=sum(not t['strict_eligible'] for t in tt),
            oss2_eval_mmsi_overlap=sum(t['evaluation_mmsi_overlap'] for t in tt)))
    csv_write('class_counts.csv', class_rows)
    missing = dict(target_not_ready=sum(not t['input_ready'] for t in joined),
        strict_excluded=sum(not t['strict_eligible'] for t in joined),
        references_not_ready=sum(not ok for ok in ref_status.values()),
        target_vv_missing=sum(not t['vv_exists_now'] for t in joined),
        target_vh_missing=sum(not t['vh_exists_now'] for t in joined),
        target_views_prepare_error=sum(c['configured'] and not c['prepare_ok'] for c in view_results),
        availability_changed=sum(not c['availability_matches_history'] for c in view_results),
        raw_patch_missing=sum(not t['raw_patch_path_exists_now'] for t in joined),
        calibrated_patch_missing=sum(not t['calibrated_patch_path_exists_now'] for t in joined),
        ship_xml_missing=sum(not t['ship_xml_path_exists_now'] for t in joined),
        metadata_xml_missing=sum(not t['metadata_xml_path_exists_now'] for t in joined))
    ready = not any(missing[k] for k in ('target_not_ready', 'strict_excluded', 'references_not_ready', 'availability_changed'))
    payload = dict(status='ready_for_frozen_raw_inference' if ready else 'partial_preparation',
        generated_at=datetime.now().astimezone().isoformat(), target=summary(targets),
        target_manifest=str(HERE / 'target_manifest.json'), references=str(HERE / 'references.json'),
        no_oss2_eval_overlap=summary(independent), class_counts=class_rows, missing=missing,
        overlap=overlap_summary, actual_exposure_evidence=evidence,
        current_version=read(CURRENT / 'current_version.json'),
        provenance=dict(target_table=str(OLD / 'target_table.csv'), eligibility=str(OLD / 'target_eligibility.csv'),
            views=str(OSS1 / 'view_inventory.csv'), geometry=str(geometry_path),
            source_records=str(COVERAGE / 'manifest.json'), source_groups=str(CURRENT / 'groups.json'),
            public_manifests=str(PUBLIC / 'manifests'), input_reader=str(INPUT)),
        runtime=dict(python=sys.executable, torch_imported='torch' in sys.modules,
                     gpu_used=False, npz_opened=0, weights_loaded=0, training=False),
        inference_contract=dict(polarization_order=['vv', 'vh'], canvas_size=160, input_scale=1.0,
            images='existing uint8 display TIFFs; raw/calibrated SAR retained as provenance only',
            labels='evaluation labels supplied as requested; no fitting or target selection',
            observation_order='unchanged historical target_table query_row 0..920',
            no_legacy_cache_index='indices.legacy_combined_row is provenance only, never a current cache index',
            preprocessing='sar_m123.input.prepare_view; native crop/pad, no resize; pixels /255 downstream',
            model_forward='owned by main agent; raw_current.py, not run here'))
    assert 'torch' not in sys.modules
    write('preparation_summary.json', payload)
    write_report(payload)
    print(json.dumps(dict(status=payload['status'], target=payload['target'], missing=missing,
                          manifest=payload['target_manifest']), ensure_ascii=False, indent=2), flush=True)


def write_report(p):
    target, miss = p['target'], p['missing']
    lines = ['# OSS2 当前模型到 OSS1 冻结外推：数据准备', '',
        f"状态：`{p['status']}`。仅 CPU 原图预处理，无 GPU、训练、模型前向、权重读取、下载或 Git 操作。", '',
        '## 可执行入口', '',
        f"- 目标清单：`{p['target_manifest']}`。保留 G 组全部 {target['n']} 条及原查询顺序。",
        f"- 固定参考：`{p['references']}`；32 条、每类 8 条，顺序与当前模型及旧 reference_bank 一致。",
        '- 目标记录含 id、mmsi、pass_group、label、scene_id、vv/vh、image、OBB corners、target_index、ship_axis 和 radar_axis_rad。',
        '- JSON 为列表，可直接交给 task.data.read_manifest；数据准备不调用 raw_current.py。',
        '- obs_id 在 OSS1/OSS2 内各自编号；跨库按 MMSI 和平台绝对轨道比对，不能直接比较局部 obs_id。',
        '- indices.query_row 为 0..920；legacy_combined_row 仅保留历史溯源，不沿用旧缓存索引。', '',
        '## 源暴露与隔离', '',
        '- source719 = source_fit478 + source_adapt241。当前 class_s17/29/43 的实际 orders 与上游 small_class_s17 orders 均覆盖同一719条。',
        '- M3 aggregate_17/last.pt 的 episodes 已逐项核对查询与动态参考；动态参考属于719源池，不能把训练暴露缩成固定32参考。',
        '- current groups、m1_coverage manifest、公开管线六组 manifests 逐条一致；固定参考全部来自 source_fit。',
        '- 平台绝对轨道从 scene_id 解析，并与 pass_group 前两段交叉核对；数据段后缀不参与轨道隔离。',
        '- 当前源拟合基础为 final_oss2 full_s17 fit478/adapt241；aggregate M3、后续M1和末端M2沿此源池继续训练。初始化重放由主代理处理，此处不加载权重。', '',
        '| 比对组 | 源观测 | 目标同MMSI观测 | 目标同平台绝对轨道观测 |',
        '|---|---:|---:|---:|']
    for name, value in p['overlap'].items():
        lines.append(f"| {name} | {value['source']['n']} | {value['target_same_mmsi']['n']} | {value['target_same_orbit']['n']} |")
    ev = p['overlap']['evaluation']['target_same_mmsi']
    lines += ['', f"OSS2历史evaluation与本目标有 {ev['n']} 条/{ev['ships']} 船同MMSI关联。它不是上述719训练暴露，但两套评价不能称完全独立船只证据。",
        f"辅助清单 target_manifest_no_oss2_eval_overlap.json 保留 {p['no_oss2_eval_overlap']['n']} 条，供需要时做独立船只敏感性分析；不替换正式921队列。",
        '921本身已有历史OSS1迁移结果，不称新盲测。validation为开发暴露；selection为历史候选/诊断暴露，不能等同当前选定分支的梯度训练。', '',
        '## 类别与原图', '',
        f"共 {target['n']} 观测、{target['ships']} 船、{target['platform_absolute_orbits']} 平台绝对轨道（旧pass_group为{target['legacy_pass_groups']}）。",
        '类别映射保持旧八类 [0,2,3,4] 到新四类 [0,1,2,3]。', '',
        '| 新类ID | 类别 | 数量 | 双可用极化 | 仅VV | 仅VH | 输入未就绪 |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for c in p['class_counts']:
        lines.append(f"| {c['class_id']} | {c['class_name']} | {c['n']} | {c['both_usable']} | {c['vv_only']} | {c['vh_only']} | {c['input_not_ready']} |")
    lines += ['', '- 逐张使用已有 sar_m123.input.prepare_view：原始显示TIFF、OBB角点及native160裁剪/补零，不缩放、不重做检测、不转换原始SAR幅度。',
        '- target_views.csv 保存全部VV/VH库存字段、OBB原始xywhr、置信度、轴、真实图像尺寸、prepare后几何与placement及错误。',
        '- 未配置/无有效OBB的极化不等于原始文件缺失；文件存在性与模型可用性分别报告。',
        '- 原始Patch、Patch_Cal、Ship.xml和Metedata.xml路径及当前存在性保留在target_manifest.csv；不假设多波段原始SAR的极化通道顺序。', '',
        '缺失/不一致统计：', '', '```json', json.dumps(miss, ensure_ascii=False, indent=2), '```', '',
        '## 交付与复跑', '',
        '- prepare_transfer.py：重建全部清单、实际暴露统计和此报告；逐张读取小chip后释放，不打开任何NPZ。',
        '- target_manifest.json / references.json / source719_manifest.json：兼容当前公共数据reader。',
        '- target_manifest.csv / target_views.csv / reference_views.csv：路径、OBB、极化及CPU预处理溯源。',
        '- source_exposure.csv / source719.csv：1519角色全表及源719，记录实际每阶段暴露次数。',
        '- overlap_details.csv / class_counts.csv / preparation_summary.json：重叠明细、类别统计和机器可读结果。', '',
        '```powershell', f"& '{sys.executable}' -B '{HERE / 'prepare_transfer.py'}'", '```', '',
        '本轮环境记录：默认Anaconda Python在首张TIFF像素解码时无traceback退出，原因未确认；改用项目已有ClearSAR环境后全部完成。无需安装或下载，复跑采用上面的ClearSAR解释器。', '',
        '## 最小剩余工作', '']
    if p['status'] == 'ready_for_frozen_raw_inference':
        lines += ['921条输入与32参考的数据入口已就绪，没有阻断原图准备的缺失。后续由主代理将target_manifest.json交给已验证的raw_current.py执行冻结外推。此处不声称模型输出已生成，也不替代主代理的原图前向与共同初始化核对。']
    else:
        lines += ['尚未满足完整921输入准备；按上述缺失计数及target_views/reference_views错误定位最小缺口，不应静默丢行后仍称921条完成。']
    (HERE / 'PREPARATION_REPORT_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
