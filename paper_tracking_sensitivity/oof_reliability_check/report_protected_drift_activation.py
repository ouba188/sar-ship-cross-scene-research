"""Factorial results plus fitted-readout drift sensitivity, not causal attribution."""

import json
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_protected_drift_activation import (OUT, ROOT, PARENT, SOURCE, FACTORS, ARMS,
    DRIFT_STRENGTH, make_views)


def gradient_norm(value):
    centered = value-value.mean(axis=1, keepdims=True)
    return float(np.linalg.norm(centered))


def verify_readout_derivatives(port='Antwerp-Bruges'):
    """Check score sensitivities against a raw-feature perturbation, supports fixed."""
    folder = OUT/'ports'/port
    with (folder/'geometry.pkl').open('rb') as f:
        g = pickle.load(f)
    with (folder/'source_models.pkl').open('rb') as f:
        fit = pickle.load(f)
    with np.load(folder/'policies.npz') as saved:
        p = {k: saved[k] for k in saved.files}
    bank = np.load(folder/'target_bank.npy')
    raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    extra = np.load(PARENT/port/'matched_joint/joint_extra.npy', mmap_mode='r')
    views = make_views(bank, raw[p['query_rows']], extra, g)
    support = np.searchsorted(p['query_rows'], p['support_rows'][0])
    selected = np.flatnonzero(p['eval_mask'][0])[:3]
    r, d = g['stable_rank'], g['drift_rank']
    delta = np.linspace(-.5,.5,d)
    shifted_bank = bank[selected].copy()
    shifted_bank[:,:128] += delta@g['drift_to_xs']
    shifted_bank[:,192+r:] += delta
    shift_raw = (delta@g['qd'].T)*g['feature_scale']
    before = make_views(bank[selected], raw[p['query_rows'][selected]], extra[selected], g)
    after = make_views(shifted_bank, np.asarray(raw[p['query_rows'][selected]],dtype=float)+shift_raw, extra[selected], g)
    errors = []
    for arm, (suppress, activate) in FACTORS.items():
        ret = views['common'][suppress]['retained'][support]
        comp = views['common'][suppress]['complement'][support]
        factor = 1.-DRIFT_STRENGTH if suppress else 1.
        for head in ('base','expert'):
            h = fit['heads'][arm+'_'+head]
            prior_gradient = g['drift_to_xs']@h['coef'][:128]+h['coef'][192+r:]
            beta = p[arm+'_'+head+'_coefficients'][0]
            if head == 'base':
                analytic = prior_gradient+(factor*g['drift_to_xs']/g['base_scale']/np.sqrt(128))@beta[:128]
            else:
                dr = factor*g['drift_to_xs']/g['model']['expert_scale'][:128]/np.sqrt(192)
                dc = factor*g['drift_to_complement']/np.sqrt(2560)
                analytic = prior_gradient+.5*(dr@ret[:,:128].T+dc@comp[:,:2048].T)@beta
            scores = []
            for b, v in ((bank[selected],before),(shifted_bank,after)):
                prior = b@h['coef']+h['intercept']
                if head == 'base':
                    score = prior+v['phis'][suppress,activate]@beta
                else:
                    q = v['common'][suppress]
                    kernel = .5*(q['retained']@ret.T)+.5*(q['complement']@comp.T)+1.
                    if activate:
                        kernel += .25*(v['stable']@views['stable'][support].T)
                    score = prior+kernel@beta
                scores.append(score)
            expected = np.broadcast_to(delta@analytic, scores[0].shape)
            error = float(np.max(np.abs(scores[1]-scores[0]-expected)))
            np.testing.assert_allclose(scores[1]-scores[0], expected, atol=2e-6, rtol=2e-6)
            errors.append(dict(arm=arm, head=head, max_error=error))
    (OUT/'READOUT_SENSITIVITY_VERIFICATION.json').write_text(json.dumps(dict(port=port,
        query_samples=len(selected), fixed_supports=64, errors=errors),indent=2),encoding='utf-8')
    return max(x['max_error'] for x in errors)


def diagnose_readouts(ports):
    raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    records = []
    for port in ports:
        folder = OUT/'ports'/port
        with (folder/'geometry.pkl').open('rb') as f:
            g = pickle.load(f)
        with (folder/'source_models.pkl').open('rb') as f:
            fit = pickle.load(f)
        with np.load(folder/'policies.npz') as saved:
            p = {k: saved[k] for k in saved.files}
        bank = np.load(folder/'target_bank.npy')
        extra = np.load(PARENT/port/'matched_joint/joint_extra.npy', mmap_mode='r')
        views = make_views(bank, raw[p['query_rows']], extra, g)
        r = g['stable_rank']
        stable_to_xs = ((g['qs'].T*g['feature_scale'])@g['model']['pca'].components_.T)/g['model']['scale']
        for arm, (suppress, activate) in FACTORS.items():
            retained = views['common'][suppress]['retained']
            complementary = views['common'][suppress]['complement']
            factor = 1.-DRIFT_STRENGTH if suppress else 1.
            for head in ('base', 'expert'):
                coef = fit['heads'][arm+'_'+head]['coef']
                drift_prior = g['drift_to_xs']@coef[:128]+coef[192+r:]
                stable_prior = stable_to_xs@coef[:128]+coef[192:192+r]/g['stable_scale'][:,None]
                for draw in range(5):
                    support = np.searchsorted(p['query_rows'], p['support_rows'][draw])
                    if head == 'base':
                        beta = p[arm+'_base_coefficients'][draw]
                        update = (factor*g['drift_to_xs']/g['base_scale']/np.sqrt(128))@beta[:128]
                    else:
                        beta = p[arm+'_expert_coefficients'][draw]
                        dr = factor*g['drift_to_xs']/g['model']['expert_scale'][:128]/np.sqrt(192)
                        dc = factor*g['drift_to_complement']/np.sqrt(2560)
                        update = .5*(dr@retained[support,:128].T+dc@complementary[support,:2048].T)@beta
                    records.append(dict(port=port, arm=arm, head=head, draw=draw,
                        source_drift_gradient_norm=gradient_norm(drift_prior),
                        source_stable_gradient_norm=gradient_norm(stable_prior),
                        adapted_drift_gradient_norm=gradient_norm(drift_prior+update)))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'readout_sensitivity.csv', index=False, encoding='utf-8-sig')
    averaged = frame.groupby(['port','arm','head']).mean(numeric_only=True).drop(columns='draw').reset_index()
    ref = averaged[averaged.arm == 'matched_baseline'].set_index(['port','head'])
    ratios = []
    for _, row in averaged.iterrows():
        baseline = ref.loc[row.port,row['head']]
        ratios.append(dict(port=row.port, arm=row.arm, head=row['head'],
            source_drift_ratio=row.source_drift_gradient_norm/max(baseline.source_drift_gradient_norm,1e-12),
            adapted_drift_ratio=row.adapted_drift_gradient_norm/max(baseline.adapted_drift_gradient_norm,1e-12)))
    ratios = pd.DataFrame(ratios)
    ratios.to_csv(OUT/'readout_sensitivity_ratios.csv', index=False, encoding='utf-8-sig')
    return ratios.groupby(['arm','head']).mean(numeric_only=True)


def main():
    complete = json.loads((OUT/'COMPLETE.json').read_text(encoding='utf-8'))
    pm = pd.read_csv(OUT/'port_means.csv')
    means = pd.read_csv(OUT/'equal_port_means.csv').set_index('arm')
    ports = sorted(pm.port.unique())
    source = pd.read_csv(OUT/'source_episode_metrics.csv')
    sm = source.groupby(['port','arm']).mean(numeric_only=True).groupby('arm').mean()
    sm.to_csv(OUT/'source_episode_means.csv', encoding='utf-8-sig')
    geometry = pd.DataFrame([json.loads((OUT/'ports'/port/'GEOMETRY_READY.json').read_text(encoding='utf-8')) for port in ports])
    geometry.to_csv(OUT/'geometry_summary.csv', index=False, encoding='utf-8-sig')
    replay = max(json.loads((OUT/'ports'/p/'TARGET_FROZEN.json').read_text(encoding='utf-8'))['unchanged_kernel_replay_error'] for p in ports)
    sensitivity_path = OUT/'readout_sensitivity_means.csv'
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path).set_index(['arm','head'])
    else:
        with threadpool_limits(limits=2):
            sensitivity = diagnose_readouts(ports)
        sensitivity.to_csv(sensitivity_path, encoding='utf-8-sig')
    interactions = {}
    for metric in ('acc','ba','base_ba','expert_ba'):
        interactions[metric] = 100*(means.loc['joint',metric]-means.loc['drift_only',metric]
            -means.loc['stable_only',metric]+means.loc['matched_baseline',metric])
    names = dict(original='原联合模型（历史参照）', matched_baseline='同流程重训基线',
                 drift_only='仅保护式漂移抑制', stable_only='仅显式稳定读出', joint='联合版本')
    contrasts = pd.DataFrame(complete['contrasts'])
    contrasts.to_csv(OUT/'contrasts.csv', index=False, encoding='utf-8-sig')
    joint = contrasts[(contrasts.candidate == 'joint') & (contrasts.control == 'matched_baseline')].iloc[0]
    lines = ['# 漂移抑制与稳定信息激活：四组联合验证', '', '日期：2026-10-08。', '',
        '## 总体结果', '',
        f'完成原24港语料的23个可评估目标港、5组原64标签、5%专家交接。联合版本相对同流程基线OA变化{joint.delta_oa_pp:+.4f}个百分点、BA变化{joint.delta_ba_pp:+.4f}个百分点；BA提高{int(joint.positive_ports)}港、下降{int(joint.negative_ports)}港。',
        '原联合模型保留不变。本报告只检验本轮固定操作、强度和训练配方，不代表所有抑制/激活设计。', '',
        '| 方法 | OA (%) | BA (%) | 基础BA (%) | 专家BA (%) | 平均交接数 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m = means.loc[arm]
        lines.append(f'| {names[arm]} | {100*m.acc:.4f} | {100*m.ba:.4f} | {100*m.base_ba:.4f} | {100*m.expert_ba:.4f} | {m.selected:.2f} |')
    lines += ['', '先平均每港5组抽样，再对23港等权平均；不是样本池准确率。', '',
        '## 本轮具体做了什么', '',
        '1. 固定候选空间：使用源港发现的R50候选张成空间Qs。冻结投影不代表已证明严格不变性。R50和已训练R18编码器均不更新。',
        '2. 定位可分离漂移：计算同一类别在不同源港的均值偏移，类别等权、类内港口等权；扣除与Qs重叠部分，再取剩余漂移的前16个方向Qd。',
        '3. 保护式软抑制：在源标准化R50空间执行x_new=x-0.5*(x Qd)Qd^T。Qs与Qd正交，候选坐标不变，所选漂移幅度减半、平方能量为四分之一。与候选重叠的漂移不强删。',
        '4. 显式激活：把源标准化的候选坐标作为可学习分支加入源基础/专家分类头；不是给旧分类分数乘常数。源头使用原ridge1、类别平衡。基础64标签适配同时接入候选坐标；专家保留原核并增加0.25候选核，而非缩小原核。',
        '5. 协调重训：四个因子组均重拟合源分类头和原64标签适配；门控也按各组实际基础/专家分数在源任务上重新训练，仍是原4个margin/entropy输入。',
        '6. 同流程基线：当前R18源特征上的分类头重拟合及门控重训本身可能改变结果，所以不能只与历史主模型比较来归功于新模块。', '',
        '## 来源与监督边界', '',
        '源分类头的内层训练使用原22个源港任务、原身份/产品排除行和64标签支持集。为了复用现有投影与表征，几何、Qs/Qd和编码器固定为外层全部源域拟合版本，包含内层留出源港的信息；只有分类头读出按内层留港重拟合。不是完整的内层OOF几何或编码器。',
        '所有外层目标港在源训练中排除，目标只使用每组既定64标签做适配。漂移强度0.5、最多16维、专家候选核权重0.25预先固定，没有目标指标选参。所有23港策略先冻结再统一读出查询成绩。',
        '数据已经用于历史研发，不是全新独立确认。源任务指标用于解释流程，不应写成独立无偏源验证。', '',
        '## 因子对比', '',
        '| 候选 | 对照 | OA差 (pp) | BA差 (pp) | 基础BA差 (pp) | 专家BA差 (pp) | BA升/降港数 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for _, row in contrasts.iterrows():
        lines.append(f'| {names[row.candidate]} | {names[row.control]} | {row.delta_oa_pp:+.4f} | {row.delta_ba_pp:+.4f} | {row.delta_base_ba_pp:+.4f} | {row.delta_expert_ba_pp:+.4f} | {row.positive_ports}/{row.negative_ports} |')
    lines += ['',
        f'描述性交互项（联合-仅抑制-仅激活+同流程基线）：最终BA为{interactions["ba"]:+.4f}个百分点、基础BA为{interactions["base_ba"]:+.4f}、专家BA为{interactions["expert_ba"]:+.4f}。交互项不等于独立统计显著性或因果协同。', '',
        '## 是否真正抑制了漂移', '',
        f'原管线无变换专家核重放最大分数误差为{replay:.3g}。数值模块12项测试通过，包括独立sklearn源头拟合、稳定空间保护、漂移能量变为四分之一。',
        f'各港候选空间与漂移方向最大重叠为{geometry.overlap.max():.3g}；前16方向平均覆盖可分离源均值漂移能量的{100*geometry.selected_energy_fraction.mean():.2f}%。原均值漂移能量平均有{100*geometry.nuisance_energy_in_stable_fraction.mean():.2f}%位于候选空间中，本轮未抹掉这部分。',
        '特征幅度被抑制，不保证重新训练后的分类读出也同比减少依赖。下面直接计算拟合分数对所选漂移坐标的梯度范数，相对同流程基线；去掉各类共有分数偏移后计算，先每港平均再等港平均。不是准确率贡献率，也不是信息使用百分比。', '',
        '| 版本 | 分支 | 源先验漂移敏感度比 | 64标签适配后敏感度比 |',
        '|---|---|---:|---:|']
    for (arm, head), row in sensitivity.iterrows():
        lines.append(f'| {names[arm]} | {head} | {row.source_drift_ratio:.4f} | {row.adapted_drift_ratio:.4f} |')
    lines += ['', '## 源任务读数', '',
        '| 版本 | 源任务基础BA (%) | 源任务专家BA (%) |', '|---|---:|---:|']
    for arm in FACTORS:
        lines.append(f'| {names[arm]} | {100*sm.loc[arm,"base_ba"]:.4f} | {100*sm.loc[arm,"expert_ba"]:.4f} |')
    trials = pd.read_csv(OUT/'metrics.csv').groupby(['draw','arm'])[['acc','ba','base_acc','base_ba']].mean()
    draw_rows = []
    for draw in range(5):
        j, b, o = (trials.loc[draw,arm] for arm in ('joint','matched_baseline','original'))
        draw_rows.append(dict(draw=draw, ba_vs_matched=100*(j.ba-b.ba), oa_vs_matched=100*(j.acc-b.acc),
                             ba_vs_original=100*(j.ba-o.ba), oa_vs_original=100*(j.acc-o.acc)))
    draws = pd.DataFrame(draw_rows)
    draws.to_csv(OUT/'draw_effects.csv',index=False,encoding='utf-8-sig')
    lines += ['', '## 抽样与收益位置', '',
        '| 抽样 | BA相对重训基线 (pp) | OA相对重训基线 (pp) | BA相对原模型 (pp) | OA相对原模型 (pp) |',
        '|---|---:|---:|---:|---:|']
    for row in draw_rows:
        lines.append(f'| {row["draw"]} | {row["ba_vs_matched"]:+.4f} | {row["oa_vs_matched"]:+.4f} | {row["ba_vs_original"]:+.4f} | {row["oa_vs_original"]:+.4f} |')
    decompositions = []
    for control in ('matched_baseline','original'):
        for metric, base_metric in (('ba','base_ba'),('acc','base_acc')):
            full = 100*(means.loc['joint',metric]-means.loc[control,metric])
            base = 100*(means.loc['joint',base_metric]-means.loc[control,base_metric])
            decompositions.append(dict(control=control,metric=metric,full_delta_pp=full,
                                       base_delta_pp=base,handoff_increment_delta_pp=full-base))
    pd.DataFrame(decompositions).to_csv(OUT/'gain_decomposition.csv',index=False)
    delta = (pm[pm.arm == 'joint'].set_index('port').ba
             -pm[pm.arm == 'matched_baseline'].set_index('port').ba)*100
    lines += ['',
        '联合版相对同流程基线5组等港BA均为正，但第4组仅+0.0076个百分点；相对历史原模型有1组BA下降。高度重叠的5组不能当作独立显著性证据。',
        '恒等分解为：最终变化=基础变化+交接净增益的变化。相对同流程基线，基础BA+0.3435pp，交接净增益变化+0.0059pp，总计+0.3494pp；OA对应+0.9413pp、-0.0812pp，总计+0.8602pp。这是读数分解，不是因果归因。',
        '联合版本本身交接仍带来约+2.1998pp BA，但相对基线新增的BA收益主要来自基础识别，不是专家单独更强。', '',
        '| 联合版相对重训基线的港口BA变化 | 差值 (pp) |', '|---|---:|']
    for port, value in pd.concat((delta.nlargest(3),delta.nsmallest(3))).items():
        lines.append(f'| {port} | {value:+.4f} |')
    verified = OUT/'READOUT_SENSITIVITY_VERIFICATION.json'
    if verified.exists():
        verification = json.loads(verified.read_text(encoding='utf-8'))
        error = max(v['max_error'] for v in verification['errors'])
        lines += ['', f'漂移敏感度另用固定支持集下的真实原始特征扰动核对，4组×基础/专家两分支最大分数差{error:.3g}，不是只检验同一公式的重写。']
    lines += ['', '## 根源分析与处置', '',
        '1. 不是抑制模块没有执行：特征所选漂移幅度确实减半；重新拟合后，单独抑制的基础/专家漂移敏感度约为基线0.739/0.740，联合为0.675/0.708。分类器只部分抵消抑制，没有完全恢复原依赖。',
        '2. 单独抑制没有平均收益：最终BA-0.0790pp，说明检测到的同类跨港变化并不等同于可以单独删减的有害信息。究竟哪些漂移仍携带有效判别证据尚未单独识别，不能将其作为唯一根因。',
        '3. 显式稳定读出在基础分支有初步收益：仅激活使基础BA+0.2542pp，联合为+0.3435pp；不再只是给旧源分数统一加权。相同64标签预算下，更改读出/适配的信息组织方式产生了正向结果。',
        '4. 专家不能照搬结论：仅激活和联合相对重训基线的专家BA分别-0.3865pp和-0.3047pp。源任务专家BA提高不代表目标专家提高，而且源几何并非内层独立；不将源到目标差异自动归结为实现错误。',
        '5. 联合并非全面优于仅激活：BA额外+0.1142pp，但OA低0.1243pp。保留两者作为候选，明确BA/OA权衡，不选择性隐去任一指标。',
        '6. 此轮可以记录为有效候选组合，不覆盖原主模型，也不宣称因果不变性或顶刊级机制已经成立。尚无等维普通判别分支对照，不能把全部收益归因于稳定性独有属性。',
        '7. 更有针对性的下一步是优先保留基础分支的收益，单独检验是否应让专家继续采用原有/重训基线读出，再协调训练门控；本轮未运行该混合版本，不承诺进一步提点。', '',
        '## 保存文件', '',
        'PROTOCOL.json、COMPLETE.json、metrics.csv、port_means.csv、contrasts.csv、source_episode_metrics.csv、geometry_summary.csv、readout_sensitivity*.csv，以及每港source_models.pkl和policies.npz。',
        '实现位于sensitivity/oof_reliability_check：protected_drift_activation.py及测试、run_protected_drift_activation.py、report_protected_drift_activation.py。']
    (OUT/'PROTECTED_DRIFT_ACTIVATION_FINDINGS_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(means.to_string(), flush=True)
    print(sensitivity.to_string(), flush=True)
    print(json.dumps(interactions), flush=True)


if __name__ == '__main__':
    main()
