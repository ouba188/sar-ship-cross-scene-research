"""Convert existing OSS2 inventories without torch, numpy, detection or image reads."""
import csv
import math
from pathlib import Path

from protocol import HERE, SOURCE, CLASS_INDICES, CLASSES, ROLES, read, write, assert_rows


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def flag(value):
    return str(value).lower() in ('true', '1')


def corners(raw):
    cx, cy, width, height, angle = raw
    c, s = math.cos(angle), math.sin(angle)
    return [[cx+x*c-y*s, cy+x*s+y*c] for x, y in
            ((-width/2, -height/2), (width/2, -height/2),
             (width/2, height/2), (-width/2, height/2))]


def view_spec(view):
    if not (flag(view['valid_target']) and flag(view['prepare_ok']) and
            flag(view['detected']) and float(view['confidence']) >= .25):
        return None
    import json
    raw = json.loads(view['raw_xywhr'])
    if len(raw) != 5 or not all(math.isfinite(v) for v in raw) or min(raw[2:4]) <= 0:
        raise ValueError(f"Invalid raw OBB: {view['sample_id']} {view['pol']}")
    axis = None
    if flag(view['axis_valid']):
        axis = dict(center=[float(view['cx']), float(view['cy'])],
                    length_px=float(view['length']), width_px=float(view['width']),
                    angle=float(view['ship_axis']))
    return dict(image=view['path'], detections=[dict(corners=corners(raw),
                confidence=float(view['confidence']))], target_index=0,
                ship_axis=axis, radar_axis_rad=None, raw_xywhr=raw)


def convert():
    table = csv_rows(SOURCE/'stack_table.csv')
    inventory = csv_rows(SOURCE/'view_inventory.csv')
    views = {(v['sample_id'], v['pol']): v for v in inventory}
    assert len(views) == len(inventory)
    indices = read(SOURCE/'stack_indices.json')
    groups = {name: [] for name in ROLES.values()}
    records, by_obs = [], {}
    for i, row in enumerate(table):
        assert int(row['table_index']) == i
        label = int(row['class_id'])
        assert CLASS_INDICES[label] == int(row['original_class_id'])
        assert row['class_name'] == CLASSES[label]
        record = dict(id=row['sample_id'], label=label, class_name=CLASSES[label],
            original_class_id=int(row['original_class_id']), mmsi=row['mmsi'],
            pass_group=row['pass_group'], scene_id=row['scene_id'],
            role=row['role'], split=ROLES[row['role']], input_scale=1.0, canvas_size=160,
            indices={key: int(row[key]) for key in ('legacy_row_index', 'target_manifest_row',
                'manifest_position', 'obs_id', 'table_index', 'row_index', 'up_index')},
            source_paths={p: row['source_path_'+p] for p in ('vv', 'vh')},
            native_size=[int(row['chip_width_px']), int(row['chip_height_px'])])
        for pol in ('vv', 'vh'):
            view = views[(row['sample_id'], pol)]
            assert int(view['obs_id']) == int(row['obs_id'])
            assert Path(view['path']) == Path(row['chip_path_'+pol])
            record[pol] = view_spec(view)
            assert (record[pol] is not None) == flag(row['valid_'+pol])
        assert record['vv'] or record['vh']
        assert int(row['obs_id']) not in by_obs
        by_obs[int(row['obs_id'])] = record
        records.append(record)
        groups[record['split']].append(record)
    assert len({r['id'] for r in records}) == len(records)
    for role, split in ROLES.items():
        assert indices[role] == [r['indices']['table_index'] for r in groups[split]]
        assert_rows(groups[split], split)
    refs = []
    for ref in csv_rows(SOURCE/'m3_results/reference_bank.csv'):
        record = by_obs[int(ref['obs_id'])]
        assert record['role'] == ref['role'] == 'fit_train'
        assert record['mmsi'] == ref['mmsi'] and record['label'] == int(ref['class_id'])
        refs.append(record)
    for name, group in dict(groups, references=refs, all=records).items():
        write(HERE/'manifests'/f'{name}.json', group)
    write(HERE/'conversion.json', dict(status='converted_cpu_only', source=str(SOURCE),
        classes=list(CLASSES), class_indices=list(CLASS_INDICES),
        counts={k: len(v) for k, v in groups.items()}, references=len(refs),
        torch_imported=False, images_opened=0, gpu_checks='NOT_RUN'))
    return groups, refs


if __name__ == '__main__':
    groups, refs = convert()
    print({k: len(v) for k, v in groups.items()}, 'references', len(refs))
