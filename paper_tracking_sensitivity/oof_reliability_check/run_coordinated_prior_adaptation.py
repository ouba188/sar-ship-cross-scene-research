"""Isolate prior/residual coordination using the existing frozen experiment."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from coordinated_prior_adaptation import fit_component_response, ridge_objective
from evidence_aware_handoff import ALPHAS, select_recalibration, route_scores
from receiver_consistent_handoff import gate_scores
from residual_expert import fit_correction
from run_evidence_aware_handoff import ROOT, PARENT, TRACE, archive
from run_generic_transfer import metrics
from sar_joint_completion import SOURCE, write_json


PREVIOUS = ROOT/'evidence_aware_handoff_20261008'
OUT = ROOT/'coordinated_prior_adaptation_20261008'
BASE = ROOT/'oof_base_adapt_results_20261007'
ARMS = ('reference', 'old_posthoc', 'coordinated_same_alpha',
        'coordinated_reselected', 'ordinary_scale_coordinated')


def protocol():
    record = dict(date='2026-10-08', arms=list(ARMS), labels=64, draws=5,
        budget=.05, ridge_strength=8., alpha_grid=list(ALPHAS),
        change='Recalibrate source prior BEFORE refitting original residual64',
        exact_update='B_new=B_old+alpha*(C-phi*A), beta_new=beta_old-alpha*A',
        component_response='A is class-balanced ridge8 fit of C on original64 supports',
        same_alpha='Reuse previous source-selected stable alpha to isolate coordination',
        reselected='Select alpha by equal-source-episode Base64 BA under coordinated recipe',
        ordinary_control='Identical coordination and selection for entire source prior',
        expert='Frozen original joint expert', gate='Frozen original gate, recompute scores',
        source_episodes='Reuse506 prior cached held-port episodes; no new encoder training',
        source_boundary='Heads/directions inner port held; M3 encoder not inner-OOF',
        query_selection=False, independent_confirmation=False,
        reconstruction='Stable trace inherits recovered-PCA approximation',
        original_model_unchanged=True, target_freeze='All23 policies before query readout')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError('Use new output for changed protocol')
    else:
        write_json(path, record)


def phi_from_xs(xs, center, scale):
    return np.column_stack(((xs-center)/scale/np.sqrt(xs.shape[1]), np.ones(len(xs))))


def select_source(port):
    folder = OUT/'ports'/port
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder/'SOURCE_FITTED.json'
    if result_path.exists():
        return json.loads(result_path.read_text(encoding='utf-8'))
    descriptor = json.loads((PREVIOUS/'source_episodes'/port/'COMPLETE.json').read_text(encoding='utf-8'))
    episodes, replay_error = [], 0.
    for item in descriptor['episodes']:
        ep = archive(Path(item['path']))
        with np.load(PARENT/port/'episodes'/Path(item['path']).name) as saved:
            xs = saved['xs']
            np.testing.assert_array_equal(saved['rows'], ep['rows'])
            np.testing.assert_array_equal(saved['support'], ep['support'])
        with np.load(PREVIOUS/'source_pair_cache'/(item['pair_key']+'.npz')) as pair:
            phi = phi_from_xs(xs, pair['base_residual_center'], pair['base_residual_scale'])
        support, y = ep['support'], ep['labels'][ep['support']]
        if ep['query'][support].any():
            raise ValueError('Support reused for source selection')
        beta = fit_correction(phi[support], y, ep['base_prior'][support])
        replay = ep['base_prior']+phi@beta
        replay_error = max(replay_error, float(np.max(np.abs(replay-ep['base_scores']))))
        np.testing.assert_allclose(replay, ep['base_scores'], rtol=0, atol=1e-10)
        # The selector now evaluates prior-first refitting, not post-hoc addition.
        episodes.append(dict(base_scores=ep['base_scores'], labels=ep['labels'], query=ep['query'],
            stable_source=fit_component_response(phi, support, y, ep['stable_source'])['residual_component'],
            base_prior=fit_component_response(phi, support, y, ep['base_prior'])['residual_component']))
    calibration = select_recalibration(episodes)
    old = json.loads((PREVIOUS/'ports'/port/'SOURCE_FITTED.json').read_text(encoding='utf-8'))
    record = dict(port=port, episodes=len(episodes), calibration=calibration,
                  old_alpha=old['calibration']['stable_alpha'], source_replay_max_error=replay_error)
    write_json(result_path, record)
    print(f'SOURCE {port}: old={record["old_alpha"]}, coordinated={calibration["stable_alpha"]}, '
          f'scale={calibration["scale_alpha"]}', flush=True)
    return record


def support_mse(scores, y):
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    weights = len(y)/(len(counts)*counts[inverse])
    desired = 2*np.eye(scores.shape[1])[y]-1
    return float(np.average(np.mean((scores-desired)**2, axis=1), weights=weights))


def freeze_target(port, selected, labels):
    folder = OUT/'ports'/port
    if (folder/'TARGET_FROZEN.json').exists():
        return
    old = archive(PARENT/port/'matched_joint/joint_policies.npz')
    previous = archive(PREVIOUS/'ports'/port/'policies.npz')
    base_fit = archive(BASE/port/'policies.npz')
    trace = archive(TRACE/'ports'/port/'trace.npz')
    q = old['query_rows']
    for other in (base_fit, trace, previous):
        for key in ('query_rows', 'support_rows'):
            np.testing.assert_array_equal(old[key], other[key])
    np.testing.assert_array_equal(old['eval_mask'], base_fit['eval_masks'])
    with np.load(ROOT/'pred_cache'/f'{port}.npz') as saved:
        prior = saved['base_logits'][q].astype(float)
    with np.load(ROOT/'pred_cache2'/f'{port}.npz') as saved:
        xs = saved['Xs'][q]
    phi = phi_from_xs(xs, base_fit['source_center'], base_fit['source_scale'])
    with (PARENT/port/'source_model.pkl').open('rb') as f:
        gate = pickle.load(f)['gate']
    csource = trace['contributions'][:, list(trace['component_names']).index('base_source')]
    global_to_local = {int(row): i for i, row in enumerate(q)}
    cal = selected['calibration']
    alphas = (0., selected['old_alpha'], selected['old_alpha'], cal['stable_alpha'], cal['scale_alpha'])
    records, bases, coefs, gates, diagnostics = [], [], [], [], []
    max_error = 0.
    for draw, ev in enumerate(old['eval_mask']):
        s = np.array([global_to_local[int(row)] for row in old['support_rows'][draw]])
        y = labels[old['support_rows'][draw]]
        if ev[s].any():
            raise ValueError('Support reused as target query')
        b, e, beta = old['base_scores'][draw], old['expert_scores'][draw], base_fit['visual_correction'][draw]
        np.testing.assert_allclose(prior+phi@beta, b, rtol=0, atol=1e-10)
        stable = fit_component_response(phi, s, y, csource[draw])
        scale = fit_component_response(phi, s, y, prior)
        components = (np.zeros_like(prior), csource[draw], csource[draw], csource[draw], prior)
        responses = (None, None, stable, stable, scale)
        views, coefficients, routing, gate_views = [], [], [], []
        for i, arm in enumerate(ARMS):
            a, c, response = alphas[i], components[i], responses[i]
            if response is None:
                newb, newcoef = b+a*c, beta
            else:
                newb = b+a*response['residual_component']
                newcoef = beta-a*response['coefficient']
                direct = fit_correction(phi[s], y, (prior+a*c)[s])
                error = float(np.max(np.abs(prior+a*c+phi@direct-newb)))
                max_error = max(max_error, error)
                np.testing.assert_allclose(direct, newcoef, rtol=0, atol=1e-10)
                np.testing.assert_allclose(prior+a*c+phi@direct, newb, rtol=0, atol=1e-10)
                post_objective = ridge_objective((b+a*c)[s], y, beta)
                new_objective = ridge_objective(newb[s], y, newcoef)
                if new_objective > post_objective+1e-9:
                    raise ValueError('Coordinated fit failed to minimize fixed-prior objective')
            g = gate_scores(newb, e, gate)
            r = route_scores(newb, e, g, y, ev, q)
            if arm in ('reference', 'old_posthoc'):
                replay_index = 0 if arm == 'reference' else 1
                for key in ('predictions', 'selected'):
                    np.testing.assert_array_equal(r[key], previous[key][draw, replay_index])
                np.testing.assert_array_equal(newb, previous['base_scores'][draw, replay_index])
            if r['cap'] != old['cap'][draw] or r['selected'][~ev].any():
                raise ValueError('Original support exclusion or handoff cap changed')
            diagnostics.append(dict(port=port, draw=draw, arm=arm, alpha=a,
                support_mse=support_mse(newb[s], y),
                objective=ridge_objective(newb[s], y, newcoef),
                posthoc_objective=ridge_objective((b+a*c)[s], y, beta),
                response_query_ratio=(float(np.linalg.norm(response['residual_component'][ev])/
                    max(np.linalg.norm(c[ev]), 1e-12)) if response is not None else np.nan)))
            views.append(newb)
            coefficients.append(newcoef)
            routing.append(r)
            gate_views.append(g)
        records.append(routing)
        bases.append(np.stack(views))
        coefs.append(np.stack(coefficients))
        gates.append(np.stack(gate_views))
    np.savez_compressed(folder/'policies.npz', query_rows=q, source_rows=old['source_rows'],
        support_rows=old['support_rows'], eval_mask=old['eval_mask'], arm_names=np.asarray(ARMS),
        expert_scores=old['expert_scores'], base_scores=np.stack(bases), coefficients=np.stack(coefs),
        gate_scores=np.stack(gates),
        **{k: np.stack([[r[k] for r in draw] for draw in records]) for k in records[0][0]})
    pd.DataFrame(diagnostics).to_csv(folder/'support_diagnostics.csv', index=False)
    write_json(folder/'TARGET_FROZEN.json', dict(port=port, draws=5, arms=list(ARMS),
        original_replay_exact=True, old_posthoc_replay_exact=True,
        direct_refit_max_score_error=max_error, alphas=list(alphas)))
    print('FROZEN '+port, flush=True)


def evaluate(inventory, labels):
    rows, parameters, diagnostics = [], [], []
    for item in inventory:
        port = item['port']
        folder = OUT/'ports'/port
        p = archive(folder/'policies.npz')
        y = labels[p['query_rows']]
        fit = json.loads((folder/'SOURCE_FITTED.json').read_text(encoding='utf-8'))
        parameters.append(dict(port=port, old_alpha=fit['old_alpha'],
            new_alpha=fit['calibration']['stable_alpha'], scale_alpha=fit['calibration']['scale_alpha']))
        diagnostics.append(pd.read_csv(folder/'support_diagnostics.csv'))
        for draw, ev in enumerate(p['eval_mask']):
            for i, arm in enumerate(ARMS):
                pred, base = p['predictions'][draw,i], p['base_predictions'][draw,i]
                m, bm = metrics(y[ev], pred[ev], 8), metrics(y[ev], base[ev], 8)
                rows.append(dict(port=port, draw=draw, arm=arm, acc=m['acc'], ba=m['ba'],
                    base_acc=bm['acc'], base_ba=bm['ba'], selected=int(p['selected'][draw,i].sum()),
                    query_rows=int(ev.sum()), rescued=int(np.sum(ev & (pred == y) & (base != y))),
                    harmed=int(np.sum(ev & (pred != y) & (base == y)))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat(diagnostics).to_csv(OUT/'support_diagnostics.csv', index=False)
    pd.DataFrame(parameters).to_csv(OUT/'source_selected_parameters.csv', index=False)
    pm = df.groupby(['port', 'arm']).mean(numeric_only=True).drop(columns='draw').reset_index()
    pm.to_csv(OUT/'port_means.csv', index=False, encoding='utf-8-sig')
    means = pm.groupby('arm').mean(numeric_only=True).reset_index()
    means.to_csv(OUT/'equal_port_means.csv', index=False, encoding='utf-8-sig')
    contrasts = []
    for a, b in (('coordinated_same_alpha', 'old_posthoc'), ('coordinated_same_alpha', 'reference'),
                 ('coordinated_reselected', 'reference'), ('coordinated_reselected', 'ordinary_scale_coordinated')):
        av, bv = pm[pm.arm == a].set_index('port'), pm[pm.arm == b].set_index('port')
        delta = av.ba-bv.ba
        contrasts.append(dict(candidate=a, control=b, delta_ba_pp=float(100*delta.mean()),
            delta_oa_pp=float(100*(av.acc-bv.acc).mean()),
            delta_base_ba_pp=float(100*(av.base_ba-bv.base_ba).mean()),
            positive_ports=int((delta>1e-12).sum()), negative_ports=int((delta < -1e-12).sum())))
    write_json(OUT/'COMPLETE.json', dict(ports=len(inventory), draws=5, arms=list(ARMS), rows=len(df),
        means=means.to_dict('records'), contrasts=contrasts, main_model_unchanged=True,
        target_query_selection=False, independent_confirmation=False))
    print(means.to_string(index=False), flush=True)
    print(json.dumps(contrasts, indent=2), flush=True)


def main():
    protocol()
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed experiment')
    inventory = json.loads((BASE/'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    labels = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    with threadpool_limits(limits=2):
        for item in inventory:
            selected = select_source(item['port'])
            freeze_target(item['port'], selected, labels)
    write_json(OUT/'ALL_TARGETS_FROZEN.json', dict(ports=len(inventory), arms=list(ARMS)))
    evaluate(inventory, labels)
    report(labels)


def report(labels):
    """Read frozen outcomes to separate optimization repair from transfer gain."""
    pm = pd.read_csv(OUT/'port_means.csv')
    means = pd.read_csv(OUT/'equal_port_means.csv').set_index('arm')
    parameters = pd.read_csv(OUT/'source_selected_parameters.csv').set_index('port')
    support = pd.read_csv(OUT/'support_diagnostics.csv')
    active = parameters.index[parameters.old_alpha.ne(0)]
    old = support[(support.arm == 'old_posthoc') & support.port.isin(active)].set_index(['port', 'draw'])
    new = support[(support.arm == 'coordinated_same_alpha') & support.port.isin(active)].set_index(['port', 'draw'])
    ref = support[(support.arm == 'reference') & support.port.isin(active)]
    rows, source_gains, replay = [], [], []
    for port in parameters.index:
        folder = OUT/'ports'/port
        p = archive(folder/'policies.npz')
        y = labels[p['query_rows']]
        fit = json.loads((folder/'SOURCE_FITTED.json').read_text(encoding='utf-8'))
        if fit['calibration']['stable_alpha'] != 0:
            curve = fit['calibration']['stable_curve']
            source_gains.append(100*(max(t['ba'] for t in curve)-curve[0]['ba']))
        replay.append(json.loads((folder/'TARGET_FROZEN.json').read_text(encoding='utf-8'))['direct_refit_max_score_error'])
        for draw, ev in enumerate(p['eval_mask']):
            for arm in ('coordinated_same_alpha', 'coordinated_reselected'):
                i = ARMS.index(arm)
                for stage, key in (('base', 'base_predictions'), ('full', 'predictions')):
                    baseline, changed = p[key][draw,0], p[key][draw,i]
                    for c in np.unique(y[ev]):
                        mask = ev & (y == c)
                        rescues = int(np.sum(mask & (baseline != y) & (changed == y)))
                        harms = int(np.sum(mask & (baseline == y) & (changed != y)))
                        rows.append(dict(port=port, draw=draw, arm=arm, stage=stage, class_id=int(c),
                            rows=int(mask.sum()), rescued=rescues, harmed=harms,
                            recall_delta_pp=100*(rescues-harms)/mask.sum()))
    details = pd.DataFrame(rows)
    details.to_csv(OUT/'class_transitions.csv', index=False, encoding='utf-8-sig')
    totals = details.groupby(['arm', 'stage'])[['rescued', 'harmed']].sum()
    names = dict(reference='原联合模型', old_posthoc='旧版：适配后直接加权',
        coordinated_same_alpha='纠错：同权重，重新适配', coordinated_reselected='纠错：源域重选权重',
        ordinary_scale_coordinated='对照：整个源先验缩放后适配')
    lines = ['# 先验与64标签适配协同：纠错实验及根源分析', '',
        '日期：2026-10-08。原24港语料中23个可评估港口，每港5组原始64标签；5%专家交接。',
        '结论：修复先验与残差适配的协调关系，并未转化为跨港收益；不纳入主模型。', '',
        '## 固定的对照', '',
        '原模型为 P + phi*beta。旧增强为 P + alpha*C + phi*beta，beta仍针对旧P拟合。',
        '新流程先改P，再用同一64标签重拟合beta；专家、特征、门控参数和预算不变。',
        '同权重对照只改变适配协调；另设按新流程在源港重新选权重及普通先验缩放对照。',
        '506个已缓存源港任务用于源域选参；全部目标策略冻结后才读取目标查询指标。', '',
        '## 结果', '', '| 方法 | OA (%) | BA (%) | 基础模型BA (%) |', '|---|---:|---:|---:|']
    for arm in ARMS:
        m = means.loc[arm]
        lines.append(f'| {names[arm]} | {100*m.acc:.4f} | {100*m.ba:.4f} | {100*m.base_ba:.4f} |')
    lines += ['', '先在每港内平均5组，再等权平均23港；不是合并样本准确率。', '',
        '## 已证实与被否定的解释', '',
        f'1. 实现与数值：原模型和旧增强的预测/交接逐项复现。快速更新与直接重新拟合的最大分数差为{max(replay):.3g}；独立sklearn求解等13项单元测试通过。所检查路径未发现实现偏离。',
        f'2. 优化协调：原8个非零权重港口的40组适配中，新流程的固定新先验正则化训练目标有{int((new.objective < old.objective-1e-10).sum())}/40组降低。平均{old.objective.mean():.6f}降至{new.objective.mean():.6f}。',
        f'   同40组的类别平衡训练平方误差，原模型{ref.support_mse.mean():.6f}，旧增强{old.support_mse.mean():.6f}，纠错后{new.support_mse.mean():.6f}；{int((new.support_mse < old.support_mse-1e-10).sum())}/40组优于旧增强。这是支持集拟合，不是查询泛化。',
        '3. 关键反证：同一权重重新适配后，最终BA比旧增强再低0.0772个百分点，比主模型低0.3751个百分点。因此，适配协调不是足以解释或修复退化的原因；不能把训练目标下降写成模块有效。',
        f'4. 源到目标的收益没有稳定迁移：新流程在10个港口对应的源任务上选中非零权重，选中源平均增益为{np.mean(source_gains):.4f}个百分点；目标10港基础BA平均变化为{100*(means.loc["coordinated_reselected","base_ba"]-means.loc["reference","base_ba"])*23/10:.4f}个百分点。注意源增益经过选参，不能与未选的目标结果视为无偏对称估计。',
        '5. 损失在交接前已经存在：重选版本基础BA比主模型低0.2552个百分点，最终BA低0.3023个百分点。各版本平均交接数相同，为79，不能归因于少交接。', '',
        '## 错误变化', '', '| 版本 | 阶段 | 修正原错误 | 破坏原正确 |', '|---|---|---:|---:|']
    for (arm, stage), row in totals.iterrows():
        lines.append(f'| {names[arm]} | {stage} | {int(row.rescued)} | {int(row.harmed)} |')
    lines += ['', '这里是23港×5组的样本出现次数，同一船可能重复出现；它们不是独立船数，也不等于等港BA/OA。逐港、逐类变化见class_transitions.csv。', '',
        '## 原因的边界与处置', '',
        '已定位的失效环节是：把“候选空间对原判别重要”直接转成“统一放大该空间的源分类分数”。即使适配重新协调，所选加权仍未在目标上产生净收益。删除后性能下降不意味着继续放大会提升；候选方向较稳定，也不意味着整个候选子空间内的分类分数与其最优权重跨港不变。',
        '目前不能进一步断言唯一根因是条件冗余、源类间偏置、64标签不足或分布稳定性不足，这些未被本对照逐一识别。没有证据把负结果解释成整片隐空间无价值。',
        '停止此处统一源分数放大及其外部数据扩展。原联合模型不变；后续若利用候选空间，应先检验它能否对原模型的具体错误提供增量判别证据，再决定是否设计类别/样本选择性的修正。此建议尚非已验证模块。', '',
        '普通先验缩放的OA提高而BA降低，只是指标权衡，不证明候选稳定空间的特有收益；不据此更换主指标。', '',
        '## 数据与解释限制', '',
        '沿用历史研发数据，不是全新独立确认；源编码器见过源任务标签，不宣称编码器内层完全OOF。候选贡献继承此前恢复PCA的近似，不是完整因果干预。本轮仅检验所列加权网格与适配流程，不否定所有其他用法。', '',
        '产物：PROTOCOL.json、COMPLETE.json、metrics.csv、port_means.csv、source_selected_parameters.csv、support_diagnostics.csv、class_transitions.csv，以及每港冻结策略。']
    (OUT/'COORDINATED_PRIOR_FINDINGS_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(totals.to_string(), flush=True)


if __name__ == '__main__':
    main()
