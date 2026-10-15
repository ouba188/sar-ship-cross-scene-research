"""Read-only capacity/routing diagnosis of two previously evaluated policies."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT / 'handoff_capacity_diagnosis_followup'
MANIFEST = Path('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv')
PORTS = ('Antwerp-Bruges', 'Jebel Ali', 'Port Klang', 'Santos', 'Singapore', 'Tanger Med')
RUNS = dict(preserved='base_only_preserved_expert_pilot_20261009',
            modified='protected_drift_activation_20261008')


def load(path):
    # NPZ members decompress individually; no feature banks are loaded.
    wanted = ('arms', 'arm_names', 'query_rows', 'support_rows', 'eval_mask',
              'base_scores', 'expert_scores', 'selected', 'predictions', 'gate_scores', 'cap')
    with np.load(path, mmap_mode='r', allow_pickle=False) as f:
        return {k: f[k] for k in wanted if k in f.files}


def class_weights(y):
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[inverse])


def oracle(utility, weights, eligible, cap):
    positive = (utility > 0) & eligible
    return (min(int(positive.sum()), cap) / len(utility),
            float(np.sort(weights[positive])[-cap:].sum()) if cap else 0.)


def tiny_check():
    # Exhaustive subsets catch wrong BA weighting, negative picks and quota handling.
    y = np.array([0, 0, 0, 1])
    u = np.array([1, 0, -1, 1])
    w = class_weights(y)
    for allowed in (np.ones(4, bool), np.array([True, True, True, False])):
        for cap in (0, 1, 2):
            subsets = [c for size in range(cap + 1)
                       for c in combinations(np.flatnonzero(allowed), size)]
            expected = (max(sum(u[i] for i in c) / 4 for c in subsets),
                        max(sum(u[i] * w[i] for i in c) for c in subsets))
            np.testing.assert_allclose(oracle(u, w, allowed, cap), expected)
    np.testing.assert_allclose(oracle(u, w, np.ones(4, bool), 1), (.25, .5))


def gain(u, w, selected):
    return float(u[selected].sum() / len(u)), float((u[selected] * w[selected]).sum())


def summarize(y, base, expert, selected, score, observed, cap):
    w = class_weights(y)
    u = (expert == y).astype(int) - (base == y).astype(int)
    disagree = base != expert
    eligible = disagree & np.isin(expert, observed)
    positive_gate = eligible & (score > 0)
    assert selected.sum() <= cap and not np.any(selected & ~positive_gate)
    result = dict(query_n=len(y), quota=cap, selected_n=int(selected.sum()),
                  base_errors=int((base != y).sum()),
                  base_oa=float((base == y).mean()), base_ba=float(w[base == y].sum()),
                  expert_oa=float((expert == y).mean()), expert_ba=float(w[expert == y].sum()))
    for name, mask in (('disagree', disagree), ('eligible', eligible),
                       ('gate_positive', positive_gate), ('selected', selected)):
        n = int(mask.sum())
        result[name + '_n'] = n
        for kind, value in (('rescue', 1), ('harm', -1), ('neutral', 0)):
            count = int((mask & (u == value)).sum())
            result[name + '_' + kind + '_n'] = count
            result[name + '_' + kind + '_rate'] = count / n if n else np.nan
    actual = gain(u, w, selected)
    for eligibility, mask in (('structural', eligible), ('gate_positive', positive_gate)):
        upper = oracle(u, w, mask, cap)
        for metric, maximum, achieved in zip(('oa', 'ba'), upper, actual):
            result[eligibility + '_oracle_gain_' + metric] = maximum
            result[eligibility + '_oracle_' + metric] = result['base_' + metric] + maximum
            result[eligibility + '_oracle_gap_' + metric] = maximum - achieved
            result[eligibility + '_captured_' + metric] = achieved / maximum if maximum else np.nan
            assert achieved <= maximum + 1e-12
    for metric, value in zip(('oa', 'ba'), actual):
        result['actual_gain_' + metric] = value
        result['actual_' + metric] = result['base_' + metric] + value
    return result, dict(u=u, w=w, eligible=eligible, selected=selected,
                        rescue=eligible & (u == 1))


def main():
    tiny_check()
    labels = pd.read_csv(MANIFEST, usecols=['class_id']).class_id.to_numpy()
    records, overlaps, crosses, decompositions = [], [], [], []
    for port in PORTS:
        saved = {k: load(ROOT / v / 'ports' / port / 'policies.npz') for k, v in RUNS.items()}
        a, b = saved['preserved'], saved['modified']
        for key in ('query_rows', 'support_rows', 'eval_mask'):
            np.testing.assert_array_equal(a[key], b[key])
        indices = {k: list(v['arms'] if 'arms' in v else v['arm_names']).index('stable_only')
                   for k, v in saved.items()}
        np.testing.assert_array_equal(a['base_scores'][:, indices['preserved']],
                                      b['base_scores'][:, indices['modified']])
        original_index = list(a['arms']).index('original')
        for draw in range(5):
            ev = a['eval_mask'][draw]
            y = labels[a['query_rows'][ev]]
            observed = labels[a['support_rows'][draw]]
            cap = int(np.floor(.05 * len(y)))
            info = {}
            for arm, data in saved.items():
                index = indices[arm]
                base = data['base_scores'][draw, index, ev].argmax(1)
                expert_scores = (data['expert_scores'][draw, ev] if arm == 'preserved'
                                 else data['expert_scores'][draw, index, ev])
                expert = expert_scores.argmax(1)
                selected = data['selected'][draw, index, ev]
                score = data['gate_scores'][draw, index, ev]
                saved_cap = data['cap'][draw] if arm == 'preserved' else data['cap'][draw, index]
                assert saved_cap == cap
                np.testing.assert_array_equal(np.where(selected, expert, base),
                                              data['predictions'][draw, index, ev])
                rec, info[arm] = summarize(y, base, expert, selected, score, observed, cap)
                records.append(dict(port=port, draw=draw, arm=arm, **rec))
            masks = {k: v['selected'] for k, v in info.items()}
            masks['historical_original'] = a['selected'][draw, original_index, ev]
            values = {}
            for expert_arm, data in info.items():
                for mask_arm, mask in masks.items():
                    # No refill: keep the other policy's rows, dropping structurally invalid ones.
                    valid = mask & data['eligible']
                    raw = gain(data['u'], data['w'], mask)
                    clipped = gain(data['u'], data['w'], valid)
                    values[expert_arm, mask_arm] = clipped
                    crosses.append(dict(port=port, draw=draw, expert=expert_arm, selection=mask_arm,
                        proposed_n=int(mask.sum()), structurally_valid_n=int(valid.sum()),
                        ineligible_n=int((mask & ~data['eligible']).sum()),
                        raw_gain_oa=raw[0], raw_gain_ba=raw[1],
                        clipped_gain_oa=clipped[0], clipped_gain_ba=clipped[1]))
            ar, br = info['preserved']['rescue'], info['modified']['rescue']
            overlaps.append(dict(port=port, draw=draw, shared_rescues=int((ar & br).sum()),
                preserved_only_rescues=int((ar & ~br).sum()), modified_only_rescues=int((br & ~ar).sum()),
                union_rescues=int((ar | br).sum()),
                selected_intersection=int((masks['preserved'] & masks['modified']).sum())))
            aa, ab = values['preserved', 'preserved'], values['preserved', 'modified']
            ba, bb = values['modified', 'preserved'], values['modified', 'modified']
            record = dict(port=port, draw=draw)
            for i, metric in enumerate(('oa', 'ba')):
                expert_effect = .5 * ((ba[i] - aa[i]) + (bb[i] - ab[i]))
                selection_effect = .5 * ((ab[i] - aa[i]) + (bb[i] - ba[i]))
                delta = bb[i] - aa[i]
                np.testing.assert_allclose(expert_effect + selection_effect, delta, atol=1e-12)
                record.update({f'modified_minus_preserved_{metric}': delta,
                               f'expert_and_eligibility_effect_{metric}': expert_effect,
                               f'selection_effect_{metric}': selection_effect})
            decompositions.append(record)
        print('COMPLETE ' + port, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frames = dict(capacity_per_draw=pd.DataFrame(records), rescue_overlap=pd.DataFrame(overlaps),
                  cross_selection=pd.DataFrame(crosses), decomposition=pd.DataFrame(decompositions))
    for name, frame in frames.items():
        frame.to_csv(OUT / (name + '.csv'), index=False)
    pm = frames['capacity_per_draw'].groupby(['port', 'arm']).mean(numeric_only=True).drop(columns='draw')
    pm.to_csv(OUT / 'port_means.csv')
    means = pm.groupby('arm').mean()
    means.to_csv(OUT / 'equal_port_means.csv')
    write_report(means, frames)
    print(means[['actual_oa', 'actual_ba', 'eligible_rescue_n', 'structural_oracle_oa',
                 'structural_oracle_ba']].to_string())
    print(frames['decomposition'].mean(numeric_only=True).to_string())
    print('Report: ' + str(OUT / 'FINDINGS_ZH.md'))


def write_report(means, frames):
    p, m = means.loc['preserved'], means.loc['modified']
    d = frames['decomposition'].mean(numeric_only=True)
    overlap = frames['rescue_overlap'].mean(numeric_only=True)
    cross = frames['cross_selection'].groupby(['expert', 'selection']).mean(numeric_only=True)
    lines = ['# 专家容量与交接路由诊断', '',
        '## 结论与下一步', '',
        '此次保留专家方案的 OA 退步，主要体现为既有路由选样效果变化，不是同配额可纠错容量耗尽。两套专家 OA 上界相同；保留专家的 BA 上界反而更高。不能根据专家单独 BA 更高，就推断交接组合一定更好。', '',
        f'改造专家方案实际 OA 高 {100*d.modified_minus_preserved_oa:.4f} 个百分点，其中固定集合交换分解的选择项为 {100*d.selection_effect_oa:.4f}，专家及资格项为 {100*d.expert_and_eligibility_effect_oa:.4f}。BA 的微小反向差异也主要对应选择集合项。', '',
        '行动建议：下一项改进优先研究如何识别“能救回且不会改错”的样本，保持分类器固定便于判断增益来源；不因本轮保留失败而优先扩充专家容量。已有净效用及响应门控负结果仍成立，本诊断不证明再拟合同一个门控会有效，也不授权参数搜索。两专家各自独有救回样本约 123/126 个，说明它们互补而非简单强弱替代；是否能部署地识别这些样本仍未知。', '',
        f'保留现有 gate_score > 0 限制后的 OA 上界仍分别为 {100*p.gate_positive_oracle_oa:.4f}% / {100*m.gate_positive_oracle_oa:.4f}%；BA 上界为 {100*p.gate_positive_oracle_ba:.4f}% / {100*m.gate_positive_oracle_ba:.4f}%。因此 OA 未达上界不能只归咎于正分阈值，正分候选池内的选样仍有诊断空间。', '',
        '仅复用两轮 stable_only 的六港、各五次抽样缓存。基础分数逐元素相同；64 个支持标签、查询集合和 5% 配额相同。没有训练、调参、方法选择或修改主模型。', '',
        '“保留专家”来自 base_only_preserved_expert_pilot_20261009；“改造专家”来自 protected_drift_activation_20261008。历史日期只是既有目录标签。', '',
        '## 结果', '',
        '| 指标 | 保留专家 | 改造专家 |', '|---|---:|---:|']
    for label, key, scale in (
        ('基础 OA (%)', 'base_oa', 100), ('基础 BA (%)', 'base_ba', 100),
        ('专家 OA (%)', 'expert_oa', 100), ('专家 BA (%)', 'expert_ba', 100),
        ('实际最终 OA (%)', 'actual_oa', 100), ('实际最终 BA (%)', 'actual_ba', 100),
        ('平均查询数', 'query_n', 1), ('平均交接配额', 'quota', 1),
        ('符合结构资格的可纠错数量', 'eligible_rescue_n', 1),
        ('实际交接救回数', 'selected_rescue_n', 1), ('实际交接伤害数', 'selected_harm_n', 1),
        ('同配额 OA 上界 (%)', 'structural_oracle_oa', 100),
        ('同配额 BA 上界 (%)', 'structural_oracle_ba', 100),
        ('实际 OA 增益 / 可用上界增益 (%)', 'structural_captured_oa', 100),
        ('实际 BA 增益 / 可用上界增益 (%)', 'structural_captured_ba', 100)):
        lines.append(f'| {label} | {p[key]*scale:.4f} | {m[key]*scale:.4f} |')
    lines += ['', '## 固定样本交换诊断', '',
        '以下值为相对同一个基础模型的增益（百分点），只交换已保存的选择集合，不重新排名、不补满名额。跨专家后不符合结构资格的行回退基础预测。', '',
        '| 使用专家 | 使用选择集合 | OA 增益 | BA 增益 | 平均资格不符行数 |', '|---|---|---:|---:|---:|']
    names = dict(preserved='保留专家', modified='改造专家', historical_original='历史原方案')
    for (expert, selection), row in cross.iterrows():
        lines.append(f'| {names[expert]} | {names[selection]} | {100*row.clipped_gain_oa:.4f} | {100*row.clipped_gain_ba:.4f} | {row.ineligible_n:.2f} |')
    lines += ['', '## 差异分解', '',
        '对四种“专家 × 固定选择集合”组合做双顺序平均分解，方向为改造专家方案减保留专家方案。专家项含其结构资格变化；选择项包含既有路由、分数变化共同导致的选样变化，不等同于单独门控算法的因果效果。', '',
        '| 指标 | 总差异 pp | 专家及资格项 pp | 选择集合项 pp |', '|---|---:|---:|---:|']
    for metric in ('oa', 'ba'):
        lines.append(f'| {metric.upper()} | {100*d["modified_minus_preserved_"+metric]:.4f} | {100*d["expert_and_eligibility_effect_"+metric]:.4f} | {100*d["selection_effect_"+metric]:.4f} |')
    lines += ['', f'可纠错样本平均交集 {overlap.shared_rescues:.2f}；仅保留专家能救回 {overlap.preserved_only_rescues:.2f}；仅改造专家能救回 {overlap.modified_only_rescues:.2f}。', '',
        '## 解释边界', '',
        '- 结构资格：基础与专家预测不同，且专家预测类别出现在该次 64 个支持标签中。另单列保留既有 gate_score > 0 条件后的上界。',
        '- OA 上界只选择正效用救回行，最多 floor(0.05 × 查询数) 行；BA 上界对查询真实类别赋权 1/(出现类别数 × 该类查询数)，选择权重最大的正效用行。两个上界可选不同样本。',
        '- 上界使用历史已读真标签，纯诊断、不可部署；不得作为实际精度或训练/参数选择结果。',
        '- 中性行指两模型均错且预测不同。比率分母在 CSV 前缀中指明（分歧、结构资格、门控正分或实际选中）。',
        '- 先五次抽样平均再六港等权平均；重复抽样并非独立新数据。原始交换列忽略资格仅供定位，资格截断后的交换值也不是重新执行另一套门控。',
        '- 已核对基础分数、支持与查询行完全一致，保存预测可由已选行重建；小型穷举独立核验 OA/BA 上界及零配额，无新增测试框架。', '',
        '逐港见 port_means.csv；逐抽样见 capacity_per_draw.csv；交换及交集见 cross_selection.csv、rescue_overlap.csv；精确加法分解见 decomposition.csv。']
    (OUT / 'FINDINGS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
