"""Replay missing-class decisions; separate own score from strongest rival score."""

import numpy as np
import pandas as pd
import torch
from target_loss_repair import BASE, CACHE, OLD, OUT, evaluate, ResponseAdapter, build_blocks, save_json


def run():
    torch.set_num_threads(1)
    obs = pd.read_csv(BASE/'observations.csv')
    maps = np.load(CACHE/'boundary_maps.npy', mmap_mode='r')
    block = build_blocks()[1]
    records, replays = [], []
    for path in sorted(OUT.glob('*_models.npz')):
        tag = path.stem.removesuffix('_models')
        with np.load(OLD/f'{tag}_split.npz') as z:
            support, query, labels, ids = [z[k] for k in
                ('support_map_rows', 'query_map_rows', 'support_labels', 'query_obs_ids')]
        meta = obs.iloc[ids]
        truth = meta.class_id.to_numpy()
        absent = np.bincount(labels, minlength=8) == 0
        select = np.flatnonzero(absent[truth])
        if not len(select):
            continue
        with np.load(OLD/f'{tag}_models.npz') as z:
            state = {k: z[k] for k in z.files}
        head, _, _ = evaluate(block, maps, support, query[select], labels,
                              ResponseAdapter(state['channel_scale']).eval(), state)
        if not np.array_equal(head.argmax(1), state['head_only_pred'][select]):
            raise ValueError('Missing-class head predictions failed replay')
        y = truth[select]
        index = np.arange(len(y))
        head_rivals = head.copy()
        head_rivals[index, y] = -np.inf
        head_own, head_rival = head[index, y], head_rivals.max(1)
        with np.load(path) as saved:
            original_correct = saved['ordinary_adapter_pred'][select] == y
            for arm in ('ordinary_adapter', 'mask_only', 'source_supervision', 'mask_and_source'):
                score = saved[arm+'_scores'][select]
                rival = score.copy()
                rival[index, y] = -np.inf
                own, competitor = score[index, y], rival.max(1)
                for j, qi in enumerate(select):
                    records.append(dict(task=tag, port=meta.iloc[qi].port, draw=int(tag.rsplit('draw', 1)[1]),
                        obs_id=int(ids[qi]), mmsi=str(meta.iloc[qi].mmsi), class_id=int(y[j]), arm=arm,
                        head_correct=bool(head.argmax(1)[j] == y[j]),
                        original_correct=bool(original_correct[j]), correct=bool(score.argmax(1)[j] == y[j]),
                        head_margin=float(head_own[j]-head_rival[j]), margin=float(own[j]-competitor[j]),
                        delta_own=float(own[j]-head_own[j]), delta_rival=float(competitor[j]-head_rival[j]),
                        delta_margin=float(own[j]-competitor[j]-head_own[j]+head_rival[j])))
        replays.append(dict(task=tag, samples=len(select), prediction_disagreements=0))
    rows = pd.DataFrame(records)
    rows.to_csv(OUT/'missing_class_score_cases.csv', index=False)
    class_cells = rows.groupby(['task', 'arm', 'class_id']).agg(
        head_recall=('head_correct', 'mean'), old_recall=('original_correct', 'mean'), recall=('correct', 'mean'),
        head_margin=('head_margin', 'mean'), margin=('margin', 'mean'), delta_own=('delta_own', 'mean'),
        delta_rival=('delta_rival', 'mean'), delta_margin=('delta_margin', 'mean')).reset_index()
    task_cells = class_cells.groupby(['task', 'arm']).mean(numeric_only=True).drop(columns='class_id').reset_index()
    task_cells.to_csv(OUT/'missing_class_score_by_task.csv', index=False)
    summary = task_cells.groupby('arm').mean(numeric_only=True).reset_index()
    summary.to_csv(OUT/'missing_class_score_summary.csv', index=False)
    save_json(OUT/'MISSING_SCORE_REPLAY.json', dict(replays=replays,
        boundary='Development-query diagnosis only; strongest rival may change between arms; not physical causal attribution',
        sample_exposures=len(rows)//4, unique_observations=int(rows.obs_id.nunique()),
        unique_vessels=int(rows.mmsi.nunique())))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    run()
