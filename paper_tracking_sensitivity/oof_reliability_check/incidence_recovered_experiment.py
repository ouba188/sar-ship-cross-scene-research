"""Fixed F0921 recovered-incidence experiment, independent of the old runner.

Default --stage verify only replays the original low-coverage experiment.
Run --stage run after the coordinating agent explicitly releases the final
recovery sidecar. COVERAGE.json is required but does not trigger execution.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from external_knowledge_run import ROOT, OUT as OLD, TASKS, TARGETS, load, unpickle
from external_knowledge_metadata import write
from compact_fusion_source import _expert_inputs
from cross_dataset_c4 import base_coordinates
from direct_incidence_conditioning import condition_features
from unified_head_expert import ExpertContext
from unified_head_run import base_kernel
from unified_head_core import score, combined, late

OUT = ROOT/'incidence_recovered_experiment_20261010'
RECOVERY = ROOT/'f0921_incidence_recovery_20261010'
ARMS = ('matched_refit', 'angle_only_old36', 'angle_only_recovered')
READOUTS = ('base', 'expert', 'fusion', 'unified')


def aligned_angles(side, ids):
    """Join by stable sample ID, rejecting absent or duplicate metadata rows."""
    if side.sample_id.duplicated().any() or len(side) != len(ids):
        raise ValueError('Incidence sidecar has duplicate IDs or a different cohort size')
    keyed = side.set_index('sample_id')
    if set(keyed.index) != set(ids):
        raise ValueError('Incidence sidecar IDs do not match the frozen F0921 cohort')
    keyed = keyed.loc[ids]
    angles = pd.to_numeric(keyed.incidence_deg, errors='coerce').to_numpy(copy=True)
    ok = keyed.status.to_numpy() == 'ok'
    if np.any(ok & (~np.isfinite(angles) | (angles <= 0) | (angles >= 90))):
        raise ValueError('status=ok needs a finite incidence angle in degrees, strictly between0 and90')
    angles[~ok] = np.nan
    return angles


def validate_recovery(side, ids, old_angles, coverage):
    angles = aligned_angles(side, ids)
    old_valid, new_valid = np.isfinite(old_angles), np.isfinite(angles)
    if int(coverage['rows']) != len(ids) or int(coverage['usable']) != int(new_valid.sum()):
        raise ValueError('COVERAGE.json does not describe this completed sidecar')
    if int(coverage['before_usable']) != int(old_valid.sum()):
        raise ValueError('Recovery baseline differs from the original low-coverage cohort')
    if not new_valid[old_valid].all():
        raise ValueError('Recovery removed original supported angles')
    np.testing.assert_allclose(angles[old_valid], old_angles[old_valid], atol=1e-10, rtol=0)
    return angles


def source_prior(x, heads, arm, head, u=None):
    values = np.asarray(x, float)
    if arm == 'angle_only':
        values = np.column_stack((values, u))
    h = heads[arm][head]
    return values@h['coef']+h['intercept']


def readouts(kb, ke, pb, pe, support, labels):
    base = score(kb, pb, support, labels, strength=8.)[0]
    expert = score(ke, pe, support, labels, strength=8.)[0]
    unified = combined(kb, ke, pb, pe, support, labels, ratio=.5, strength=8.)[0]
    return base, expert, late(base, expert, ratio=.5), unified


def angle_readouts(kb, ke, pb, pe, u, valid, support, labels, fallback):
    ku = u@u[support].T
    result = list(readouts(kb+.5*ku, ke+.125*ku, pb, pe, support, labels))
    for i in (0, 1, 3):
        result[i][~valid] = fallback[i][~valid]
    result[2] = late(result[0], result[1], ratio=.5)
    return tuple(result)


def targeted_checks():
    ids = np.array(['sample_b', 'sample_a', 'sample_c'])
    old = np.array([37., np.nan, 42.])
    side = pd.DataFrame(dict(sample_id=['sample_c', 'sample_a', 'sample_b'],
                             status=['ok', 'ok', 'ok'], incidence_deg=[42., 39., 37.]))
    marker = dict(rows=3, usable=3, before_usable=2)
    np.testing.assert_array_equal(validate_recovery(side, ids, old, marker), [37., 39., 42.])
    rejected = []
    for name, broken, receipt in [
        ('duplicate_id', side.assign(sample_id=['sample_c', 'sample_a', 'sample_a']), marker),
        ('wrong_id', side.assign(sample_id=['sample_c', 'sample_a', 'alien']), marker),
        ('invalid_degree', side.assign(incidence_deg=[42., 100., 37.]), marker),
        ('stale_coverage', side, dict(rows=3, usable=2, before_usable=2)),
    ]:
        try:
            validate_recovery(broken, ids, old, receipt)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Validation missed '+name)
    rng = np.random.default_rng(20261010)
    x, z = rng.normal(size=(12, 5)), rng.normal(size=(12, 4))
    support, labels = np.arange(8), np.arange(8)
    kb, ke = x@x[support].T+1, z@z[support].T+1
    pb, pe = rng.normal(size=(12, 8)), rng.normal(size=(12, 8))
    fallback = readouts(kb, ke, pb, pe, support, labels)
    valid = np.arange(12) % 3 != 0
    angles = np.where(valid, np.linspace(32, 44, 12), np.nan)
    u = condition_features(angles)
    np.testing.assert_array_equal(u[~valid], np.zeros((int((~valid).sum()), 2)))
    actual = angle_readouts(kb, ke, pb, pe, u, valid, support, labels, fallback)
    for a, b in zip(actual, fallback):
        np.testing.assert_array_equal(a[~valid], b[~valid])
    # Appended explicit features independently verify the two incidence weights.
    xb, xe = np.c_[x, np.ones(12), u/np.sqrt(2)], np.c_[z, np.ones(12), u/np.sqrt(8)]
    explicit = readouts(xb@xb[support].T, xe@xe[support].T, pb, pe, support, labels)
    for a, b in zip(actual, explicit):
        np.testing.assert_allclose(a[valid], b[valid], atol=1e-12)
    return dict(metadata_failures_rejected=rejected, id_based_reordering=True,
                missing_angle_fallback_exact=True, incidence_feature_kernel_weights_verified=True)


class FrozenInputs:
    def __init__(self):
        folder = TASKS/'f0921'
        self.info = json.loads((folder/'PREPARED.json').read_text())
        self.fit = unpickle(OLD/'f0921/source_fit.pkl')
        self.data = load(folder/'target.npz', ['rows', 'ids', 'support', 'support_labels', 'eval_masks',
                                             'phi', 'mean', 'conditions', 'prior', 'source_rows'])
        data = self.data
        np.testing.assert_array_equal(self.fit['source_rows'], data['source_rows'])
        self.old = load(OLD/'f0921/policies.npz')
        for key in ('rows', 'ids', 'support', 'support_labels', 'eval_masks'):
            np.testing.assert_array_equal(self.old[key], data[key])
        np.testing.assert_array_equal(self.old['readouts'], READOUTS)
        embedding = load(folder/'selective_embedding.npz', ['rows', 'embedding'])
        np.testing.assert_array_equal(embedding['rows'], data['rows'])
        self.embedding = embedding['embedding'].astype(float)
        self.context = ExpertContext(folder)
        base_model = unpickle(Path(self.info['model_path']))
        c4 = np.load(ROOT/'cross_dataset_c4_20261009/f0921/pooled_c4.npy', mmap_mode='r')
        self.xb = base_coordinates(c4, base_model).astype(float)
        raw = np.load(ROOT/TARGETS['f0921']/'paired_r50.npy', mmap_mode='r')
        extra = np.load(self.context.joint/'target_extra.npy', mmap_mode='r')
        self.xe = np.concatenate([_expert_inputs(raw[lo:lo+2048], extra[lo:lo+2048], self.context.model)
                                  for lo in range(0, len(raw), 2048)])
        self.old_angles = aligned_angles(pd.read_csv(OLD/'f0921_incidence.csv',
            usecols=['sample_id', 'status', 'incidence_deg']), data['ids'])
        np.testing.assert_array_equal(np.isfinite(self.old_angles), self.old['angle_valid'])
        self.matched_prior = self.priors('matched_refit')

    def priors(self, arm, u=None):
        heads = self.fit['heads']
        return (source_prior(self.xb, heads, arm, 'base', u),
                source_prior(self.xe, heads, arm, 'expert', u))

    def kernels(self, support):
        kb = base_kernel(self.data, self.embedding, support, self.info['eta'])
        _, ke = self.context.target_components(self.data, support)
        return kb, ke


def experiment(stage):
    recovering = stage == 'run'
    if recovering and (OUT/'FROZEN.json').exists():
        print('REUSE frozen recovered experiment; no inputs reloaded', flush=True)
        return
    if recovering:
        marker = RECOVERY/'COVERAGE.json'
        if not marker.exists():
            raise FileNotFoundError('Recovery is not complete: '+str(marker))
        coverage = json.loads(marker.read_text(encoding='utf-8'))
    fixtures = targeted_checks()
    inputs_ = FrozenInputs()
    data = inputs_.data
    old_valid = np.isfinite(inputs_.old_angles)
    old_u = condition_features(inputs_.old_angles)
    old_prior = inputs_.priors('angle_only', old_u)
    if recovering:
        side = pd.read_csv(RECOVERY/'f0921_incidence.csv', usecols=['sample_id', 'status', 'incidence_deg'])
        new_angles = validate_recovery(side, data['ids'], inputs_.old_angles, coverage)
        new_valid, new_u = np.isfinite(new_angles), condition_features(new_angles)
        new_prior = inputs_.priors('angle_only', new_u)
        predictions = np.empty((5, 3, 4, len(data['rows'])), np.int8)
    started, replay = time.perf_counter(), []
    for draw, support in enumerate(data['support']):
        labels, ev = data['support_labels'][draw], data['eval_masks'][draw]
        if len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('Expected the original64 distinct supports, excluded from queries')
        kb, ke = inputs_.kernels(support)
        matched = readouts(kb, ke, *inputs_.matched_prior, support, labels)
        old = angle_readouts(kb, ke, *old_prior, old_u, old_valid, support, labels, matched)
        for arm, outputs in [('matched_refit', matched), ('angle_only', old)]:
            actual = np.stack([values.argmax(1) for values in outputs])
            slot = list(inputs_.old['arms']).index(arm)
            np.testing.assert_array_equal(actual, inputs_.old['predictions'][draw, slot])
        replay.append(dict(draw=draw, matched_four_readouts_exact=True, old36_four_readouts_exact=True))
        if recovering:
            new = angle_readouts(kb, ke, *new_prior, new_u, new_valid, support, labels, matched)
            for ai, outputs in enumerate((matched, old, new)):
                for ri, values in enumerate(outputs):
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite recovered experiment scores')
                    predictions[draw, ai, ri] = values.argmax(1)
            np.testing.assert_array_equal(predictions[draw, 2, :, ~new_valid], predictions[draw, 0, :, ~new_valid])
        print(f'{stage.upper()} draw{draw}: matched and old36 four-readout replay exact', flush=True)
    checks = dict(fixtures=fixtures, replay=replay, source_cache_reused=True, source_refit=False,
                  target_query_labels_used=False, cpu_threads=2, seconds=time.perf_counter()-started)
    if not recovering:
        write(OUT/'VERIFIED.json', checks)
        return
    np.savez_compressed(OUT/'policies.npz', predictions=predictions, arms=np.array(ARMS),
        readouts=np.array(READOUTS), ids=data['ids'], rows=data['rows'], support=data['support'],
        support_labels=data['support_labels'], eval_masks=data['eval_masks'],
        old_incidence_deg=inputs_.old_angles, recovered_incidence_deg=new_angles,
        old_valid=old_valid, recovered_valid=new_valid)
    write(OUT/'RECOVERY_COVERAGE_SNAPSHOT.json', coverage)
    write(OUT/'FROZEN.json', dict(**checks, rows=len(data['rows']),
        before_usable=int(old_valid.sum()), usable=int(new_valid.sum()),
        source_fit=str(OLD/'f0921/source_fit.pkl'), incidence_input=str(RECOVERY/'f0921_incidence.csv'),
        source_angle_coverage=json.loads((OLD/'f0921/SOURCE_READY.json').read_text())['source_angle_coverage'],
        hyperparameters='Inherited ridge8, base incidence0.5, expert0.125, equal fusion/unified0.5; no selection'))


def markdown_table(frame, columns):
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for values in frame[columns].itertuples(index=False, name=None):
        text = [f'{v:.5f}' if isinstance(v, (float, np.floating)) else str(v) for v in values]
        lines.append('| '+' | '.join(text)+' |')
    return '\n'.join(lines)


def write_findings(summary, policies):
    p = policies
    old, new = p['old_valid'], p['recovered_valid']
    full = summary[summary.subset == 'all']
    subset = summary[summary.arm == 'angle_only_recovered']
    supports = pd.DataFrame([dict(draw=draw, old_supported=int(old[s].sum()),
                                 recovered_supported=int(new[s].sum()),
                                 newly_supported=int((new[s] & ~old[s]).sum()))
                             for draw, s in enumerate(p['support'])])
    columns = ['arm', 'readout', 'oa', 'ba', 'delta_ba_vs_matched', 'delta_ba_vs_old36']
    metrics = pd.read_csv(OUT/'metrics.csv')
    classes = pd.read_csv(OUT/'class_metrics.csv')
    coverage = pd.read_csv(OUT/'coverage_subsets.csv')
    manifest = pd.read_csv(ROOT/TARGETS['f0921']/'target_manifest.csv',
                           usecols=['sample_id', 'class_id', 'class_name'])
    np.testing.assert_array_equal(manifest.sample_id.to_numpy(str), p['ids'])
    names = manifest[['class_id', 'class_name']].drop_duplicates().set_index('class_id').class_name.sort_index()
    receipt = json.loads((OUT/'RECOVERY_COVERAGE_SNAPSHOT.json').read_text(encoding='utf-8'))
    frozen = json.loads((OUT/'FROZEN.json').read_text(encoding='utf-8'))
    status_table = pd.DataFrame([dict(status=k, rows=v) for k, v in receipt['statuses'].items()])
    class_coverage = coverage[coverage.class_id.notna()].groupby(['class_id', 'subset']).total.first().unstack()
    class_coverage.insert(0, 'class_name', names)
    class_coverage['coverage_percent'] = 100*class_coverage.recovered_supported/class_coverage['all']
    class_coverage = class_coverage.reset_index()
    class_coverage['class_id'] = class_coverage.class_id.astype(int)
    class_tables = []
    all_classes = classes[classes.subset == 'all']
    for readout in READOUTS:
        selected = all_classes[all_classes.readout == readout]
        values = selected.groupby(['class_id', 'arm']).recall.mean().unstack()
        values.insert(0, 'class_name', names)
        values['query_n_mean'] = selected.groupby('class_id').n.mean()
        values['delta_vs_matched'] = values.angle_only_recovered-values.matched_refit
        values['delta_vs_old36'] = values.angle_only_recovered-values.angle_only_old36
        paired = selected.pivot(index=['draw', 'class_id'], columns='arm', values='recall')
        counts = selected[selected.arm == 'matched_refit'].set_index(['draw', 'class_id']).n
        total_queries = metrics[(metrics.subset == 'all') & (metrics.arm == 'matched_refit') &
                                (metrics.readout == readout)].set_index('draw').n
        denominators = paired.index.get_level_values('draw').map(total_queries)
        contributions = ((paired.angle_only_recovered-paired.matched_refit)*counts/denominators)
        values['oa_contribution_vs_matched'] = contributions.groupby('class_id').mean()
        observed = full[(full.arm == 'angle_only_recovered') & (full.readout == readout)].delta_oa_vs_matched.iloc[0]
        np.testing.assert_allclose(values.oa_contribution_vs_matched.sum(), observed, atol=1e-10)
        class_tables.append('### '+readout+'\n\n'+markdown_table(values.reset_index(),
            ['class_id', 'class_name', 'query_n_mean', 'matched_refit', 'angle_only_old36',
             'angle_only_recovered', 'delta_vs_matched', 'delta_vs_old36', 'oa_contribution_vs_matched']))
    draw_metrics = metrics[(metrics.subset == 'all') & (metrics.arm == 'angle_only_recovered')]
    draw_effects = draw_metrics.pivot(index='draw', columns='readout', values='delta_ba_vs_matched')
    draw_effects = supports.merge(draw_effects.reset_index(), on='draw')
    per_class_supports = []
    for draw, support in enumerate(p['support']):
        row = dict(draw=draw)
        sy = p['support_labels'][draw]
        for c in range(8):
            added = int(np.sum(new[support] & ~old[support] & (sy == c)))
            row[names[c]] = f'{added}/{int(np.sum(sy == c))}'
        per_class_supports.append(row)
    diagnostic = OUT/'diagnostic_prior_kernel'
    diagnostic_text = '固定先验/核2x2诊断尚未完成，主结果不据此改写。'
    if (diagnostic/'COMPLETE.json').exists():
        ds = pd.read_csv(diagnostic/'summary.csv')
        factors = ds.pivot(index='readout', columns='arm', values='ba').reindex(READOUTS)
        factors['prior_delta'] = factors.angle_prior_only_10-factors.matched_00
        factors['kernel_delta'] = factors.angle_kernel_only_01-factors.matched_00
        factors['both_delta'] = factors.angle_both_11-factors.matched_00
        factors['interaction'] = factors.angle_both_11-factors.angle_prior_only_10-factors.angle_kernel_only_01+factors.matched_00
        diagnostic_text = '''固定恢复后的角度、五组支持和全部权重；00为matched，10仅换角度源先验，01仅加角度核，11同时更换。00/11在全部五draw、四读出逐预测精确复现主实验。四臂缺角查询均回退matched。此处“角度源先验”是原有输入匹配重拟合得到的整个角度头（包含非角度系数的变化），不是单独一个角度线性系数；10仍使用原matched核完成支持残差适配。01保留matched源先验，只改变支持核与查询到支持关系。

''' + markdown_table(ds, ['readout', 'arm', 'oa', 'ba']) + '\n\nBA作用分解（百分点；interaction=11-10-01+00）：\n\n' + markdown_table(factors.reset_index(), ['readout', 'prior_delta', 'kernel_delta', 'both_delta', 'interaction']) + '''

融合中仅换源角度先验使BA下降0.19497个百分点，仅加角度核反而提高0.01922；正交互0.06914部分抵消先验损失，联合仍下降0.10662。因此本次融合BA损失主要定位到源先验替换后的分类/残差适配关系，而不是角度支持相似度单独造成。unified仅换先验-0.13451、仅加核-0.03104，二者单独均负；expert也以先验负作用更大。base则先验-0.29554、核-0.25722，不能把融合结论一概推广为所有读出的核都有效。

诊断只定位本次固定接入中先验与核及其交互的作用，不替代主实验、不用于选优。它不直接分离“新增查询角度”和“恢复支持角度”，也不能单凭源先验项退化就断定真实入射角与类别无关。源域分类关系迁移失配仍需与源拟合覆盖和场景分布差异区分；本轮到此收口，不增加权重搜索。
'''
    text = f'''# F0921恢复入射角固定对照

新旧两个角度臂使用相同源拟合缓存、角度源头参数、五组64支持标签与所有超参，仅改变目标入射角元数据覆盖；无角度matched仍使用原有独立匹配源先验。原36%结果不覆盖，全部新预测冻结后才读查询真值。源端角度覆盖仍沿用旧版，未因目标在线恢复而增加。

## 结论

高覆盖融合OA为34.74904%、BA为27.54332%；相对matched为OA +0.21292、BA -0.10662个百分点，相对旧36%角度臂为OA +0.08685、BA -0.07232个百分点。元数据接入与覆盖恢复已完成，但本协议下不能声称全量BA提升。

融合逐类相对matched的主要损失为offshore_supply -1.02269、tug_towing -0.79997、general_cargo -0.62710个百分点召回；product_chemical_tanker +1.39899、crude_oil_tanker +0.39005、container_ship +0.35458。较大类收益使按样本加权的OA提高，但不足以抵消八类等权BA损失。融合BA在3/5个draw下降，draw1降幅最大（-0.34149）；不是单次异常即可解释，也不能凭支持数量相关性断定原因。

相对旧36%角度臂，old_supported融合BA +0.18415、unified +0.14566，来自支持端适配改变；newly_supported融合BA -0.15579、unified -0.13781，同时包含查询端和支持端变化。不能把全量负结果解释为所有查询都受损，也不能把新增子集变化全部归于查询角度。

原覆盖{int(old.sum())}/{len(old)}（{100*old.mean():.5f}%），恢复后{int(new.sum())}/{len(new)}（{100*new.mean():.5f}%）；新增{int((new & ~old).sum())}，仍缺失{int((~new).sum())}。剩余缺角查询逐预测回退到matched，不删除查询。

覆盖侧表是全部样本口径；每draw排除原64支持后有123425条查询。源角度覆盖仍为{100*frozen['source_angle_coverage']:.5f}%。角度来自精确产品匹配的Sentinel-1 annotation地理网格插值，单位为度，不是船舶航向；不对网格外查询外推，不以传感器典型值填充。原44765条角度逐值保留。`differs_from_tile_first`只是来源比较标记，分类输入未读取该列；后来单独修订此标记不改变本次冻结角度、ID或预测。

{markdown_table(status_table, ['status', 'rows'])}

{markdown_table(class_coverage, ['class_id', 'class_name', 'all', 'old_supported', 'newly_supported', 'recovered_missing', 'coverage_percent'])}

## 全量结果

OA/BA为百分数，五次支持抽样均值；差值为百分点。

{markdown_table(full, columns)}

## 逐类召回与OA/BA分歧

以下召回均为五draw均值，差值为百分点；`oa_contribution_vs_matched`是按每draw该类查询数计算后取均值的OA贡献，可加回全量OA差值。BA则对八类等权，不能用总救回数代替BA判断。

{chr(10).join(class_tables)}

## 覆盖子集的解释

`old_supported`：这些查询自身的角度逐值不变，新旧角度源头参数也完全相同，故查询源先验不变。相对旧36%角度臂的预测变化只能来自支持样本角度恢复后改变的64标签残差适配：支持源先验、支持核矩阵及查询到支持的角度关系相应变化。不能描述成又给这些查询加了角度。这里的归因针对新旧角度臂之间的差异，不适用于新角度臂与无角度matched的比较。

`newly_supported`：查询由缺角回退切换到可用角度路径，同时支持角度也发生变化，预测差异兼有查询端和支持端作用，当前对照不能把全部变化归因于其中一个因素。

`recovered_supported`是上述两组的并集；`recovered_missing`继续采用matched回退。各子集使用原始五组支持、同一全量预测，不对子集重新拟合或抽样。子集类别比例不同，BA不能按样本覆盖率直接加权回全量BA。

{markdown_table(subset, ['subset', 'readout', 'oa', 'ba', 'delta_oa_vs_old36', 'delta_ba_vs_old36'])}

## 支持角度变化

{markdown_table(supports, ['draw', 'old_supported', 'recovered_supported', 'newly_supported'])}

逐draw高覆盖相对matched的BA差值（百分点）：

{markdown_table(draw_effects, ['draw', 'newly_supported', 'base', 'expert', 'fusion', 'unified'])}

各draw各类“新增可用角度支持数/原支持总数”（原标签和样本选择未变）：

{markdown_table(pd.DataFrame(per_class_supports), ['draw']+names.tolist())}

五draw不是独立数据集，支持角度数量与效果的对应只作描述，不能据五个点推断因果。旧有角度子集提供了支持端变化的直接固定对照，但新增角度子集没有隔离查询端与支持端效应。

draw1原支持集中fishing_vessel与offshore_supply均为0，draw4的offshore_supply也为0，这些类没有对应标签残差可直接适配；这是既有抽样条件，不是在线恢复新引入的缺陷。matched使用相同支持，因此仅凭“缺该类支持”也不能解释角度臂相对matched的增量退化。保留原draw、不重抽支持。

## 最小先验/核诊断

{diagnostic_text}

## 文件与限制

`policies.npz`保存三个臂四读出的完整预测、原支持/查询掩码、两版实际角度及覆盖掩码；`metrics.csv`含逐draw配对OA/BA、救回/误改，`class_metrics.csv`含逐类召回，`coverage_subsets.csv`含子集与逐类覆盖。`RECOVERY_COVERAGE_SNAPSHOT.json`保存本次输入覆盖记录。

`summary.csv`为五draw汇总；`PROTOCOL.json`记录固定协议，`VERIFIED.json`、`FROZEN.json`和`COMPLETE.json`记录验证与冻结状态。`diagnostic_prior_kernel/`独立保存诊断的`policies.npz`、`metrics.csv`、`class_metrics.csv`、`summary.csv`、`FROZEN.json`和`COMPLETE.json`，不覆盖主结果。脚本为`oof_reliability_check/incidence_recovered_experiment.py`及`oof_reliability_check/external_knowledge_recovered_diagnose.py`。

139条采集归属歧义不采用旧row_in_shp候选替代，实验只读取最终恢复侧表。本次不调参、不重拟合源先验或稳定分支；五次支持抽样并非独立训练，结果仍属既有开发数据。
'''
    (OUT/'FINDINGS_ZH.md').write_text(text, encoding='utf-8')


def report():
    if not (OUT/'FROZEN.json').exists():
        raise RuntimeError('Freeze recovered predictions before reading query truth')
    p = load(OUT/'policies.npz')
    frame = pd.read_csv(ROOT/TARGETS['f0921']/'target_manifest.csv', usecols=['sample_id', 'class_id'])
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str), p['ids'])
    truth = frame.class_id.to_numpy()
    np.testing.assert_array_equal(truth[p['support']], p['support_labels'])
    old, new = p['old_valid'], p['recovered_valid']
    subsets = dict(all=np.ones(len(truth), bool), old_supported=old,
                   newly_supported=new & ~old, recovered_supported=new, recovered_missing=~new)
    metrics, classes, coverage = [], [], []
    for draw, ev in enumerate(p['eval_masks']):
        for subset, eligible in subsets.items():
            mask = ev & eligible
            sy = p['support_labels'][draw]
            coverage.append(dict(draw=draw, subset=subset, total=int(eligible.sum()),
                                 support_n=int(eligible[p['support'][draw]].sum()), query_n=int(mask.sum())))
            y = truth[mask]
            present = np.unique(y)
            for c in range(8):
                coverage.append(dict(draw=draw, subset=subset, class_id=c,
                    total=int(np.sum(eligible & (truth == c))),
                    support_n=int(np.sum(eligible[p['support'][draw]] & (sy == c))),
                    query_n=int(np.sum(y == c))))
            if not len(y):
                continue
            for ai, arm in enumerate(ARMS):
                for ri, readout in enumerate(READOUTS):
                    pred = p['predictions'][draw, ai, ri, mask]
                    oa = 100*np.mean(pred == y)
                    ba = np.mean([100*np.mean(pred[y == c] == c) for c in present])
                    record = dict(draw=draw, subset=subset, arm=arm, readout=readout,
                                  n=len(y), classes=len(present), oa=oa, ba=ba)
                    for label, index in [('matched', 0), ('old36', 1)]:
                        ref = p['predictions'][draw, index, ri, mask]
                        record['delta_oa_vs_'+label] = oa-100*np.mean(ref == y)
                        record['delta_ba_vs_'+label] = ba-np.mean([100*np.mean(ref[y == c] == c) for c in present])
                        record['rescues_vs_'+label] = int(np.sum((pred == y) & (ref != y)))
                        record['harms_vs_'+label] = int(np.sum((pred != y) & (ref == y)))
                    metrics.append(record)
                    for c in present:
                        classes.append(dict(draw=draw, subset=subset, arm=arm, readout=readout,
                            class_id=int(c), n=int(np.sum(y == c)), support_n=int(np.sum(sy == c)),
                            recall=100*np.mean(pred[y == c] == c)))
    m = pd.DataFrame(metrics)
    m.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage_subsets.csv', index=False)
    fields = [c for c in m if c not in ('draw', 'subset', 'arm', 'readout')]
    summary = m.groupby(['subset', 'arm', 'readout'])[fields].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    write_findings(summary, p)
    write(OUT/'COMPLETE.json', dict(rows=len(truth), draws=5, arms=ARMS, readouts=READOUTS,
        all_predictions_frozen_before_query_readout=True, parameter_search=False,
        original_low_coverage_results_untouched=True, source_refit=False))
    print(summary[summary.subset == 'all'][['arm', 'readout', 'oa', 'ba', 'delta_ba_vs_matched', 'delta_ba_vs_old36']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('verify', 'run', 'report'), default='verify')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT/'PROTOCOL.json').exists():
        write(OUT/'PROTOCOL.json', dict(dataset='f0921', arms=ARMS, readouts=READOUTS,
            source_cache=str(OLD/'f0921/source_fit.pkl'), source_refit=False,
            recovery=str(RECOVERY), receipt='COVERAGE.json plus explicit coordinating-agent release before run',
            fixed=dict(source_ridge=1., support_ridge=8., base_angle_kernel=.5,
                       expert_angle_kernel=.125, frequency=.25, fusion=.5, unified=.5),
            features='t=(incidence_deg-38)/5, t^2; missing basis0; missing query uses exact matched scores',
            cohort='Original123489 rows; originalfive64-support draws; no query filtering',
            subsets=['all', 'old_supported', 'newly_supported', 'recovered_supported', 'recovered_missing'],
            source_coverage='Original source incidence remains unchanged; only target metadata coverage changes',
            query_labels='Read only in report after policy freeze; no tuning',
            no_gpu=True, cpu_threads=2, default_stage='verify; never auto-run on receipt appearance'))
    with threadpool_limits(limits=2):
        if args.stage in ('verify', 'run'):
            experiment(args.stage)
        if args.stage in ('run', 'report'):
            report()


if __name__ == '__main__':
    main()
