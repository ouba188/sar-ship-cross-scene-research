"""Freeze an existing leave-port task pool and legal PCA gallery, before outcomes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / 'prepool_local_probe_20261015'
EXP = OLD / 'body_context_expansion'
PORT = 'Singapore'


def main():
    frozen = json.loads((EXP/'frozen.json').read_text(encoding='utf-8'))
    pool = next(x for x in frozen['pools'] if x['port'] == PORT)
    folder = Path(pool['source_folder'])
    train = pd.read_csv(folder/'source_train.csv', dtype={'mmsi':str})
    val = pd.read_csv(folder/'source_validation.csv', dtype={'mmsi':str})
    rng = np.random.default_rng(20261015)
    gallery = np.array([int(i) for _, group in train.groupby(['port','class_id'], sort=True)
                       for i in rng.choice(group.row_index.to_numpy(int), min(8,len(group)), replace=False)])
    allowed = train.loc[~train.acquisition.isin(val.acquisition)]
    gallery = gallery[np.isin(gallery, allowed.row_index)]
    assert not allowed.port.eq(PORT).any()
    source = json.loads((EXP/'ring_relaxed/query_coverage/frozen.json').read_text(encoding='utf-8'))
    source_pool = next(x for x in source['pools'] if x['pool'] == PORT)
    roles = []
    # Include the existing loader's full meta rows. Evaluation remains B32 only.
    needed = set(gallery.tolist())
    for kind, path, tasks in (
        ('source',OLD/'body_context'/PORT,source_pool['splits']),
        ('target',EXP/PORT,{'target':json.loads((EXP/PORT/'tasks.json').read_text())})):
        meta = pd.read_csv(path/'metadata.csv',dtype={'mmsi':str})
        for split, ts in tasks.items():
            retained = [t for t in ts if kind=='target' or t['budget']==32]
            for t in retained:
                for role in ('s','q'):
                    rows = meta.iloc[t[role]].global_row.to_numpy(int)
                    needed.update(rows.tolist())
                    if t['budget']==32:
                        roles.extend(dict(global_row=int(g),split=split,role=role) for g in rows)
    roles.extend(dict(global_row=int(g),split='gallery',role='fit') for g in gallery)
    roles = pd.DataFrame(roles).drop_duplicates().sort_values(['global_row','split','role'])
    manifest = pd.read_csv(ROOT/'geometry_source_recovery_20261016/geometry_input_manifest.csv',low_memory=False)
    records = manifest.iloc[sorted(needed)].copy()
    records.insert(0,'global_row', sorted(needed))
    regions = pd.read_csv(ROOT/'region_cue_replication_20261013/ports/frame.csv').set_index('sample_id')
    selected_regions = regions.loc[records.instance_id].reset_index()
    for pol in ('vv','vh'):
        records['native_exists_'+pol] = [Path(p).is_file() for p in records['image_relpath_'+pol].astype(str)]
        for field in ('native_h','native_w','bbox_x0','bbox_y0','bbox_x1','bbox_y1'):
            records[field+'_'+pol] = selected_regions[field+'_'+pol].to_numpy()
    columns = ['global_row','instance_id','port','product','mmsi','class_id'] + [
        f'{field}_{pol}' for pol in ('vv','vh') for field in
        ('image_relpath','native_exists','native_h','native_w','bbox_x0','bbox_y0','bbox_x1','bbox_y1')]
    records[columns].to_csv(HERE/'inventory.csv',index=False)
    roles.to_csv(HERE/'roles.csv',index=False)
    payload = dict(port=PORT,rows=sorted(needed),gallery_rows=gallery.tolist(),
        source_folder=str(folder),source_checkpoint=str(folder.parent/PORT/'source_only_r18_best.pt'),
        budgets=[32],seeds=[0,1,2],steps=200,variants=['wrong','fixed'],
        target_parameter_selection=False,detector_model=str(ROOT/'v5_server_toolkit_20260912/models/550k.pt'),
        region_rule='same fresh OBB; raw w/h vs sorted w/h, fixed angle; pad1.25; floor/ceil; VV/VH union',
        missing_detection_policy='retain historical box for both arms, no sample removal',
        geometry_angles='unchanged recovered sidecar; B retains its old geometry inputs',
        native_available={p:int(records['native_exists_'+p].sum()) for p in ('vv','vh')})
    (HERE/'inventory.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('rows',len(needed),'gallery',len(gallery),'native_available',payload['native_available'])
    print(roles.groupby(['split','role']).global_row.nunique().to_string())


if __name__=='__main__':
    main()
