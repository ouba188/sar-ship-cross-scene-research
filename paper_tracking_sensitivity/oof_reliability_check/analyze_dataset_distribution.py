"""Describe the exact six-dataset budget-sweep cohorts without fitting models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
SWEEP = ROOT / 'handoff_budget_sweep_20261008'
OUT = ROOT / 'dataset_distribution_20261008'
ORDER = ('self24ports', 'F0921', 'OSS1', 'OSS2', 'pacs', 'officehome')


def counts(y, k):
    return np.bincount(np.asarray(y, dtype=int), minlength=k)


def describe(n):
    positive = n[n > 0]
    return dict(total=int(n.sum()), present_classes=int(len(positive)),
                min_present_class=int(positive.min()), max_class=int(n.max()),
                largest_class_share=float(n.max() / n.sum()),
                max_min_ratio=float(n.max() / positive.min()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((SWEEP / 'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    source_path = Path(inventory[0]['manifest_path'])
    source = pd.read_csv(source_path, usecols=['class_id', 'class_name', 'port', 'mmsi', 'product'],
                         dtype={'mmsi': str, 'product': str}, keep_default_na=False)
    names = {'sar': source[['class_id', 'class_name']].drop_duplicates().sort_values('class_id').class_name.tolist()}
    generic = {}
    for dataset, folder in [('pacs', 'pacs'), ('officehome', 'office_home')]:
        files = sorted((Path('E:/Hermes/xdomain_data') / folder).glob('*.parquet'))
        metadata = pq.read_schema(files[0]).metadata
        names[dataset] = json.loads(metadata[b'huggingface'])['info']['features']['label']['names']
        generic[dataset] = pd.concat([pd.read_parquet(p, columns=['label', 'domain']) for p in files], ignore_index=True)

    task_records, draw_records, class_records, support_records = [], [], [], []
    dataset_labels = {d: [] for d in ORDER}
    metadata_records, groups = [], []
    source_labels = source.class_id.to_numpy(int)
    for item in inventory:
        dataset, task, k = item['dataset'], item['task'], item['n_classes']
        class_names = names[dataset if dataset in generic else 'sar']
        with np.load(item['frozen_path'], allow_pickle=False) as policy:
            row_ids = policy['row_ids']
            support = policy['support_local_rows']
            evaluation = policy['eval_mask']
        if dataset == 'self24ports':
            frame = source.iloc[row_ids].reset_index(drop=True)
            with np.load(item['policy_path'], allow_pickle=False) as original:
                src = original['source_rows']
            sy = source_labels[src]
        else:
            frame = pd.read_csv(item['manifest_path'], low_memory=False, keep_default_na=False)
            if dataset in generic:
                full = generic[dataset]
                np.testing.assert_array_equal(full.iloc[row_ids].label, frame.class_id)
                sy = full.loc[full.domain.ne(task), 'label'].to_numpy(int)
            else:
                previous = {'F0921': 'f0921', 'OSS1': 'opensarship1', 'OSS2': 'opensarship2'}[dataset]
                with np.load(ROOT / f'{previous}_transfer_20261007/source_split.npz', allow_pickle=False) as split:
                    sy = source_labels[split['source_rows']]
        y = frame.class_id.to_numpy(int)
        assert len(y) == item['target_rows']
        assert len(np.unique(row_ids)) == len(y)
        target_counts, source_counts = counts(y, k), counts(sy, k)
        dataset_labels[dataset].append(y)
        stats = describe(target_counts)
        draw_stats = []
        for draw in range(5):
            s = support[draw]
            assert len(s) == len(np.unique(s)) == 64
            expected = np.ones(len(y), dtype=bool)
            expected[s] = False
            np.testing.assert_array_equal(evaluation[draw], expected)
            sc, qc = counts(y[s], k), counts(y[expected], k)
            np.testing.assert_array_equal(sc + qc, target_counts)
            absent = (sc == 0) & (target_counts > 0)
            record = dict(dataset=dataset, task=task, draw=draw, target_rows=len(y),
                          support_rows=64, query_rows=int(qc.sum()),
                          target_present_classes=int((target_counts > 0).sum()),
                          support_present_classes=int((sc > 0).sum()),
                          query_present_classes=int((qc > 0).sum()),
                          target_classes_missing_in_support=int(absent.sum()),
                          missing_class_names=';'.join(np.asarray(class_names)[absent]),
                          query_rows_in_support_unseen_classes=int(qc[sc == 0].sum()),
                          query_share_in_support_unseen_classes=float(qc[sc == 0].sum()/qc.sum()),
                          majority_query_share=float(qc.max()/qc.sum()),
                          unique_support_rows_across_draws=int(np.unique(support).size))
            draw_records.append(record)
            draw_stats.append(record)
            for c, name in enumerate(class_names):
                support_records.append(dict(dataset=dataset, task=task, draw=draw, class_id=c,
                    class_name=name, target_count=int(target_counts[c]), support_count=int(sc[c]),
                    query_count=int(qc[c]), support_observed=bool(sc[c] > 0)))
        ds = pd.DataFrame(draw_stats)
        task_records.append(dict(dataset=dataset, task=task, model_classes=k,
            **stats, source_rows=len(sy), source_present_classes=int((source_counts > 0).sum()),
            support_fraction=64/len(y), query_rows=len(y)-64,
            support_classes_mean=float(ds.support_present_classes.mean()),
            support_classes_min=int(ds.support_present_classes.min()),
            support_classes_max=int(ds.support_present_classes.max()),
            query_classes_min=int(ds.query_present_classes.min()),
            query_classes_max=int(ds.query_present_classes.max()),
            unseen_query_share_mean=float(ds.query_share_in_support_unseen_classes.mean()),
            majority_query_share_mean=float(ds.majority_query_share.mean()),
            unique_support_rows_across_draws=int(np.unique(support).size)))
        for c, name in enumerate(class_names):
            class_records.append(dict(dataset=dataset, task=task, class_id=c, class_name=name,
                target_count=int(target_counts[c]), target_share=float(target_counts[c]/len(y)),
                source_count=int(source_counts[c]), source_share=float(source_counts[c]/len(sy))))
        identity_key = {'self24ports': 'mmsi', 'F0921': 'mmsi', 'OSS1': 'mmsi_or_ship_id', 'OSS2': 'mmsi_or_ship_id'}.get(dataset)
        group_key = {'self24ports': 'product', 'F0921': 'scene', 'OSS1': 'scene_id', 'OSS2': 'scene_id'}.get(dataset)
        if group_key:
            for value, part in frame.groupby(group_key, sort=True):
                groups.append(dict(dataset=dataset, task=task, group_field=group_key, group=str(value), rows=len(part)))
        if identity_key:
            raw = frame[identity_key].astype(str)
            valid = ~raw.isin(['', '0', 'nan', 'None', 'unknown', '-1'])
            metadata_records.append(dict(dataset=dataset, task=task, id_field=identity_key,
                nonempty_id_rows=int(valid.sum()), distinct_recorded_ids=int(raw[valid].nunique()),
                group_field=group_key, distinct_groups=int(frame[group_key].nunique())))

    tables = dict(task_summary=pd.DataFrame(task_records), support_draws=pd.DataFrame(draw_records),
                  task_class_counts=pd.DataFrame(class_records), support_class_counts=pd.DataFrame(support_records),
                  metadata_summary=pd.DataFrame(metadata_records), acquisition_group_counts=pd.DataFrame(groups))
    dataset_records, pooled_class_records = [], []
    for dataset in ORDER:
        part = tables['task_summary'].loc[lambda f: f.dataset.eq(dataset)]
        draws = tables['support_draws'].loc[lambda f: f.dataset.eq(dataset)]
        k = int(part.model_classes.iloc[0])
        y = np.concatenate(dataset_labels[dataset])
        n = counts(y, k)
        dataset_records.append(dict(dataset=dataset, model_classes=k, tasks=len(part), **describe(n),
            query_rows_per_all_tasks_draw=int(part.query_rows.sum()),
            support_rows_per_all_tasks_draw=64*len(part),
            source_rows_min=int(part.source_rows.min()), source_rows_max=int(part.source_rows.max()),
            target_rows_min=int(part.total.min()), target_rows_max=int(part.total.max()),
            target_classes_per_task_min=int(part.present_classes.min()), target_classes_per_task_max=int(part.present_classes.max()),
            query_classes_per_task_min=int(part.query_classes_min.min()), query_classes_per_task_max=int(part.query_classes_max.max()),
            support_classes_mean=float(draws.support_present_classes.mean()),
            support_classes_min=int(draws.support_present_classes.min()), support_classes_max=int(draws.support_present_classes.max()),
            support_complete_draws=int(draws.target_classes_missing_in_support.eq(0).sum()),
            task_draw_contexts=len(draws),
            unseen_query_share_taskmean=float(part.unseen_query_share_mean.mean()),
            majority_query_share_taskmean=float(part.majority_query_share_mean.mean())))
        for c, name in enumerate(names[dataset if dataset in generic else 'sar']):
            pooled_class_records.append(dict(dataset=dataset, class_id=c, class_name=name, count=int(n[c]), share=float(n[c]/n.sum())))
    tables['dataset_summary'] = pd.DataFrame(dataset_records)
    tables['dataset_class_counts'] = pd.DataFrame(pooled_class_records)
    corpus = source.groupby(['port', 'class_id', 'class_name']).size().rename('count').reset_index()
    tables['all24ports_class_counts'] = corpus
    for name, table in tables.items():
        table.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    assert len(tables['task_summary']) == 34 and len(tables['support_draws']) == 170
    summary = dict(date='2026-10-08', scope='Exact fixed-joint budget sweep; no fitting or rerouting',
        inventory=str(SWEEP/'POLICIES_SAVED.json'), source_manifest=str(source_path),
        corpus_rows=len(source), corpus_ports=int(source.port.nunique()),
        corpus_counts=counts(source_labels, 8).tolist(), class_names=names,
        datasets=dataset_records, checks=dict(tasks=34, task_draw_contexts=170,
            support_query_partition=True, class_counts_reconcile=True, no_feature_arrays_loaded=True),
        interpretation='Counts are stored samples, not unique vessels. Dataset pooled counts describe data; accuracy remains equal-task mean.')
    (OUT/'distribution_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(tables['dataset_summary'].to_string(index=False))
    print('\nSAR CLASS COUNTS\n' + tables['dataset_class_counts'].query("dataset in ['self24ports','F0921','OSS1','OSS2']").pivot(index='class_name', columns='dataset', values='count').to_string())


if __name__ == '__main__':
    main()
