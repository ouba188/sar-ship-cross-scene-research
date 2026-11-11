"""Build figure tables from archived predictions, without model inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
WORK = PAPER.parent
SEEDS = (17, 29, 43)
LABELS = {
    'OSS2': ['散货船', '普通货船', '化学品/成品油轮', '集装箱船'],
    '24ports': ['散货船', '渔船', '普通货船', '化学品/成品油轮', '集装箱船', '原油轮', '拖轮/拖带', '海工补给船'],
}


def save_csv(name, data):
    pd.DataFrame(data).to_csv(ROOT / 'result_data' / name, index=False, encoding='utf-8-sig')


def main():
    (ROOT / 'result_data').mkdir(parents=True, exist_ok=True)
    sources = {
        'OSS2': (WORK / 'm1_coverage_20261101', PAPER / 'staged_ablation_20261109'),
        '24ports': (Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110'),
                    Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')),
    }
    output = {}
    all_cells, all_scene, metrics = [], [], []
    for dataset, (meta, store) in sources.items():
        rows = json.loads((meta / 'manifest.json').read_text(encoding='utf-8'))
        groups = json.loads((meta / 'groups.json').read_text(encoding='utf-8'))
        ids = np.asarray(groups['evaluation'])
        query = [rows[i] for i in ids]
        y = np.asarray([r['label'] for r in query])
        count = len(LABELS[dataset])
        n = np.bincount(y, minlength=count)
        assert (n > 0).all()
        predictions, matrices = {}, {}
        for arm in ('plain', 'full'):
            cm_seeds = []
            for seed in SEEDS:
                path = store / f'{arm}_s{seed}_lr1' / 'evaluation.npz'
                with np.load(path) as data:
                    assert np.array_equal(data['ids'], ids), path
                    pred = data['score'].argmax(1)
                predictions[arm, seed] = pred
                cm = np.zeros((count, count), int)
                np.add.at(cm, (y, pred), 1)
                cm_seeds.append(cm)
                metrics.append(dict(dataset=dataset, arm=arm, seed=seed, n=len(y),
                                    oa=100*np.mean(pred == y), ba=100*np.mean(cm.diagonal()/n)))
                for i in range(count):
                    for j in range(count):
                        all_cells.append(dict(dataset=dataset, arm=arm, seed=seed, true_label=i,
                            predicted_label=j, true_name=LABELS[dataset][i], predicted_name=LABELS[dataset][j],
                            count=int(cm[i,j]), class_n=int(n[i]), percentage=100*cm[i,j]/n[i]))
            matrices[arm] = (100*np.mean(cm_seeds, axis=0)/n[:, None]).tolist()
            assert np.allclose(np.sum(matrices[arm],axis=1),100)
            save_csv(f'confusion_{dataset}_{arm}_percent.csv',
                     [dict(true_class=LABELS[dataset][i], n=int(n[i]), **dict(zip(LABELS[dataset], row)))
                      for i,row in enumerate(matrices[arm])])
        levels = ('port', 'pass_group') if dataset == '24ports' else ('pass_group',)
        scenes = {}
        for level in levels:
            group_values = np.asarray([str(r[level]) for r in query])
            paired = []
            for group in sorted(set(group_values)):
                mask = group_values == group
                present = np.unique(y[mask])
                per_seed = []
                for seed in SEEDS:
                    right_f = predictions['full',seed][mask] == y[mask]
                    right_p = predictions['plain',seed][mask] == y[mask]
                    diff = right_f.astype(float)-right_p
                    oa = 100*diff.mean()
                    ba = 100*np.mean([diff[y[mask] == c].mean() for c in present])
                    per_seed.append((oa, ba))
                    all_scene.append(dict(dataset=dataset, level=level, group=group, seed=seed,
                        n=int(mask.sum()), present_classes=len(present), oa_delta=oa, ba_delta=ba))
                values = np.asarray(per_seed)
                paired.append(dict(group=group, n=int(mask.sum()), present_classes=len(present),
                    oa_delta=float(values[:,0].mean()), oa_std=float(values[:,0].std(ddof=1)),
                    ba_delta=float(values[:,1].mean()), ba_std=float(values[:,1].std(ddof=1)),
                    oa_positive_seeds=int((values[:,0]>0).sum()),
                    ba_positive_seeds=int((values[:,1]>0).sum())))
            scenes[level] = paired
            save_csv(f'scenes_{dataset}_{level}.csv', paired)
        output[dataset] = dict(labels=LABELS[dataset], n=n.tolist(), observations=len(y),
            ships=len({r['mmsi'] for r in query}), seeds=list(SEEDS), confusion=matrices, scenes=scenes,
            manifest=str(meta/'manifest.json'), prediction_store=str(store))
    save_csv('confusion_by_seed.csv', all_cells)
    save_csv('scene_by_seed.csv', all_scene)
    save_csv('metrics.csv', metrics)
    # Match the known archived summary, catching the wrong training scale/version.
    frame = pd.DataFrame(metrics)
    expected = {('OSS2','full'):(61.0229276896,55.6299705825),
                ('24ports','full'):(48.4119241192,43.9019382978)}
    for (dataset,arm), values in expected.items():
        actual = frame[(frame.dataset==dataset)&(frame.arm==arm)][['oa','ba']].mean().to_numpy()
        assert np.allclose(actual,values,atol=1e-6),(dataset,actual,values)
    (ROOT/'result_data'/'figures.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(frame.groupby(['dataset','arm'])[['oa','ba']].mean().to_string())


if __name__ == '__main__':
    main()
