"""Fixed source-only condition-response experiment before classifier changes."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from conditioned_vessel_drift import (split_observations, condition_differences,
                                      fit_response, response_metrics)
from same_vessel_temporal import collect_observations, make_pairs
from run_same_vessel_temporal_probe import latent
from task_stable_sensitivity import SOURCE, PARENT, ROOT
from sar_joint_completion import write_json


OUT = ROOT/'conditioned_vessel_drift_20261009'
SIDECAR = Path('E:/Hermes/paper_tracking/pixel_correction_review/chip_incidence_v1/incidence_by_instance.csv')
MODELS = {'incidence': [0, 1], 'position': [2, 3], 'incidence_position': [0, 1, 2, 3]}


def load_frame():
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id','port','class_id','mmsi','product'],
                        dtype={'mmsi':str}, keep_default_na=False)
    side = pd.read_csv(SIDECAR, usecols=['instance_id','product','port','status','incidence_deg',
                                        'latitude','longitude','coordinate_kind'])
    frame = frame.merge(side, on=['instance_id','product','port'], how='left',
                        validate='one_to_one', sort=False)
    frame.loc[frame.status.ne('ok'), ['incidence_deg','latitude','longitude']] = np.nan
    frame.groupby(['port','status']).size().rename('rows').to_csv(OUT/'metadata_coverage.csv')
    write_json(OUT/'metadata_provenance.json', dict(rows=len(frame),
        usable_rows=int(frame.incidence_deg.notna().sum()), products=frame['product'].nunique(),
        sidecar=str(SIDECAR),
        grid_producer='E:/Hermes/scripts/fetch_s1_grid.py',
        grid_source='Product-specific AWS sentinel-s1-l1c annotation/iw-vv.xml geolocationGridPoint',
        coordinate_source='Sidecar status=ok only; pair-reference positions retained and labelled',
        representative_safe_templates_used=False, missing_conditions_imputed=False,
        causal_identification=False))
    return frame


def run_port(folder, frame):
    out = OUT/'ports'/folder.name
    out.mkdir(parents=True, exist_ok=True)
    rows = np.load(folder/'source_rows.npy')
    target = np.load(folder/'target_rows.npy')
    with (folder/'geometry.pkl').open('rb') as f:
        geometry = pickle.load(f)
    bank = np.load(folder/'source_bank.npy', mmap_mode='r')
    reduced = latent(bank, geometry)
    width = 128 + reduced.shape[1]
    conditions = frame.loc[rows, ['incidence_deg','latitude','longitude']].to_numpy(float)
    values = np.column_stack((bank[:, :128], reduced, conditions))
    obs, aggregate = collect_observations(frame, rows, values)
    # All contributing rows need accepted conditions; do not average away missingness.
    good = np.isfinite(aggregate[:, width:]).all(axis=1)
    obs = obs.loc[good].reset_index(drop=True)
    aggregate = aggregate[good]
    del values, bank
    if np.intersect1d(obs.mmsi, frame.loc[target,'mmsi']).size:
        raise ValueError('Target identity in source observation set')
    split = split_observations(obs)
    spaces = {'retained128': slice(0,128), 'stable_drift': slice(128,width)}
    fitted = {}
    records, coverage, by_class, by_source_port = [], [], [], []
    train_products = set(obs.loc[split['fit'],'product'])
    train_ships = set(obs.loc[split['fit'],'mmsi'])
    for name, mask in split.items():
        o = obs.loc[mask].reset_index(drop=True)
        z = aggregate[mask]
        pairs, delta, weights = make_pairs(o, z[:, :width])
        if pairs.empty:
            raise ValueError(f'No pairs for {folder.name}/{name}')
        if name in ('new_month','new_ship_month') and train_products & set(o['product']):
            raise ValueError('Product overlap in held-month evaluation')
        if name in ('new_ship','new_ship_month') and train_ships & set(o.mmsi):
            raise ValueError('Identity overlap in held-ship evaluation')
        x = condition_differences(o, z[:,width:], pairs)
        pairs['product_a'] = o.iloc[pairs.obs_a]['product'].to_numpy()
        pairs['product_b'] = o.iloc[pairs.obs_b]['product'].to_numpy()
        pairs['weight'] = weights
        pairs['delta_incidence_deg'] = x[:,0]*5
        pairs['east_km'] = x[:,2]*10
        pairs['north_km'] = x[:,3]*10
        pairs.to_csv(out/(name+'_pairs.csv'), index=False)
        coverage.append(dict(target_port=folder.name, split=name, observations=len(o),
            pairs=len(pairs), ships=pairs.mmsi.nunique(), classes=pairs.class_id.nunique(),
            source_ports=pairs.port.nunique(), products=o['product'].nunique(),
            incidence_delta_abs_median=float(np.median(np.abs(x[:,0]*5))),
            incidence_delta_abs_p90=float(np.quantile(np.abs(x[:,0]*5),.9)),
            condition_rank=int(np.linalg.matrix_rank(x)),
            position_incidence_r2=float(1-np.sum(weights*(x[:,0]-x[:,2:] @ fit_response(x[:,2:],x[:,:1],weights).ravel())**2)
                                           / np.sum(weights*x[:,0]**2))))
        for space, cols in spaces.items():
            y = delta[:,cols]
            for model, features in MODELS.items():
                xx = x[:,features]
                if name == 'fit':
                    fitted[space+'_'+model] = fit_response(xx, y, weights)
                pred = xx @ fitted[space+'_'+model]
                row = dict(target_port=folder.name, split=name, space=space, model=model,
                           **response_metrics(y,pred,weights))
                records.append(row)
                for c in sorted(pairs.class_id.unique()):
                    use = pairs.class_id.eq(c).to_numpy()
                    ww = weights[use]/weights[use].sum()
                    by_class.append(dict(target_port=folder.name,split=name,space=space,model=model,
                        class_id=int(c),pairs=int(use.sum()), **response_metrics(y[use],pred[use],ww)))
                for port in sorted(pairs.port.unique()):
                    use = pairs.port.eq(port).to_numpy()
                    ww = weights[use]/weights[use].sum()
                    by_source_port.append(dict(target_port=folder.name,split=name,space=space,model=model,
                        source_port=port,pairs=int(use.sum()), **response_metrics(y[use],pred[use],ww)))
    np.savez_compressed(out/'response_models.npz', **fitted)
    write_json(out/'COMPLETE.json',dict(target_port=folder.name, target_queries_read=0,
        source_observations=len(obs), regression_models=len(fitted)))
    print(json.dumps(dict(port=folder.name,coverage=coverage,
        held=[r for r in records if r['split']=='new_ship_month'])),flush=True)
    return records, coverage, by_class, by_source_port


def report():
    scores = pd.read_csv(OUT/'response_metrics.csv')
    coverage = pd.read_csv(OUT/'pair_coverage.csv')
    mean = scores.groupby(['space','split','model']).mean(numeric_only=True).reset_index()
    mean.to_csv(OUT/'response_means.csv',index=False)
    pivot = scores.pivot(index=['target_port','space','split'],columns='model',values='skill').reset_index()
    pivot['incidence_increment_over_position'] = pivot.incidence_position-pivot.position
    pivot.to_csv(OUT/'incremental_skill.csv',index=False)
    lines = ['# 同船观测条件与隐空间变化', '',
        '日期：2026-10-09。仅源数据条件归因诊断，主模型未改动，未读取目标查询标签。', '',
        '## 输入与问题', '',
        '逐目标入射角来自按完整产品编号下载的公开annotation网格及已有位置插值；仅用sidecar status=ok的28628/38091条记录。缺失不填补，代表性SAFE模板不参与归因。',
        '同港同MMSI同采集时间重复观测先平均；任一参与行缺少条件的聚合观测不进入本轮。标签冲突身份按各源划分剔除。MMSI仍为身份代理。',
        '同船差分同时包含位置、姿态、船舶状态及观测条件变化，不预设它等于噪声。本轮以位置差分为对照，检验入射角是否提供额外的可迁移解释。', '',
        '## 固定方法', '',
        '沿用23个排除目标身份/产品的源划分。按MMSI固定随机留出约1/3船；另固定留出3/6/9/12月采集。拟合只用其余船和其余月份。先按子集筛选观测，再在同港同船内配对相邻时相；四组的间隔分布可以不同。',
        '报告拟合、新船、新月份、新船且新月份四组。新月份组不含拟合产品；新船组不含拟合船身份。原有PCA/Qs/Qd利用过外层全部源数据，因此这里检验给定表示下条件响应的外推，不是整个表示学习流程的独立验证。',
        '入射角输入为t=(angle-38)/5及t平方的两次观测差；位置输入为局部东西、南北位移/10km。零截距加权岭回归，固定ridge=0.01，不调参。三种模型为角度、位置、角度+位置。权重按类别、类内港、港内船、船内配对逐级等权。',
        '输出分别为原128维标准化R50保留特征和39至43维既有[Qs,Qd]坐标。skill=1-预测残差能量/原同船差分能量，以预测零变化为基准；它不是分类准确率，也不是因果解释率。', '',
        '## 条件响应结果', '',
        '| 空间 | 留出方式 | 输入 | 平均skill (%) | skill为正的源划分数 |',
        '|---|---|---|---:|---:|']
    for _,r in mean.iterrows():
        group=scores[(scores.space==r.space)&(scores['split']==r['split'])&(scores.model==r.model)]
        lines.append(f"| {r.space} | {r['split']} | {r.model} | {100*r.skill:.4f} | {int((group.skill>0).sum())}/23 |")
    lines += ['', '23个源划分大量重叠，不能当作23次独立发现。正skill只表示预测比零差分好，不能自动证明条件的因果效应。', '',
        '## 覆盖及定位', '',
        '| 留出方式 | 每折配对数范围 | 每折源港数范围 | 每折类别数范围 |',
        '|---|---:|---:|---:|']
    for name,c in coverage.groupby('split'):
        lines.append(f'| {name} | {c.pairs.min()}-{c.pairs.max()} | {c.source_ports.min()}-{c.source_ports.max()} | {c.classes.min()}-{c.classes.max()} |')
    lines += ['', 'incremental_skill.csv保存角度+位置相对位置的额外skill；class_metrics.csv和source_port_metrics.csv保留负结果，防止总体平均掩盖特定船型/港口退化。best_scalar仅是留出结果的事后方向/幅度诊断，不用于重新调节模型或报告校正性能。', '',
        '## 本轮边界', '',
        '对位置的控制只包括一阶局部位移，不能排除泊位背景、船舶姿态、状态及其他未测因素。缺失覆盖不均；插值角度也不等于地形校正局部入射角。',
        '尚未进行条件校正或新的完整模型训练，不提供虚构的BA/OA提升。本轮结论与下一步见DECISION_ZH.md。']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(mean.to_string(index=False),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed condition diagnostic')
    write_json(OUT/'PROTOCOL.json',dict(date='2026-10-09', parent=str(PARENT),
        source_only=True,conditions=['incidence','local_position_displacement'],
        models=MODELS,held_months=[3,6,9,12],vessel_seed=20261009,ridge=.01,
        target_query_labels_used=0,main_model_unchanged=True,
        scope='Conditional association replication before any feature correction',
        representation='Existing outer-source fitted geometry, not nested held representation',
        gate='Do not promote unreplicated associations as identified causal drift'))
    frame=load_frame()
    all_records,all_coverage,all_classes,all_ports=[],[],[],[]
    folders=[f for f in sorted((PARENT/'ports').iterdir()) if (f/'policies.npz').exists()]
    with threadpool_limits(limits=2):
        for folder in folders:
            r,c,k,p=run_port(folder,frame)
            all_records.extend(r); all_coverage.extend(c); all_classes.extend(k); all_ports.extend(p)
    for name,data in [('response_metrics',all_records),('pair_coverage',all_coverage),
                      ('class_metrics',all_classes),('source_port_metrics',all_ports)]:
        pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
    report()
    write_json(OUT/'COMPLETE.json',dict(ports=len(folders),records=len(all_records),
        source_condition_models=len(folders)*6,new_classifier_fits=0,target_queries_read=0))


if __name__=='__main__':
    main()
