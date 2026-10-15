"""Keep sklearn 1.7.2 splits fixed when numerical replay uses another runtime."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from run_expansion import OUT, TASK_ROOT, ROOT, old, cv, ship_groups


def main(task):
    if sklearn.__version__ != '1.7.2':
        raise ValueError('Generate the original recipe only with sklearn 1.7.2')
    folder = TASK_ROOT/task
    info = json.loads((folder/'PREPARED.json').read_text())
    with np.load(folder/'target.npz') as z:
        data = {k:z[k] for k in ('rows','ids','support','support_labels')}
    column = 'mmsi' if task=='f0921' else 'mmsi_or_ship_id'
    frame = pd.read_csv(ROOT/old.DATASETS[task][0]/'target_manifest.csv',
        usecols=['sample_id', column],dtype=str,keep_default_na=False).iloc[data['rows']]
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str),data['ids'])
    groups,_ = ship_groups(frame,'sample_id',column)
    ids = np.full((5,3,64),-1,int)
    for draw,support in enumerate(data['support']):
        for repeat,seed in enumerate(cv.SEEDS):
            for fold,(train,val) in enumerate(cv.folds(data['support_labels'][draw],groups[support],seed)):
                if set(groups[support][train])&set(groups[support][val]):
                    raise ValueError('Vessel overlap')
                ids[draw,repeat,val]=fold
    dst = OUT/'tasks'/info['dataset']/info['task']
    dst.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dst/'CV_FOLDS.npz',fold_ids=ids,support=data['support'],
        labels=data['support_labels'],groups=groups[data['support']],seeds=np.array(cv.SEEDS))
    print('FROZEN CV FOLDS',task,sklearn.__version__,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=['f0921','oss2'])
    main(parser.parse_args().task)
