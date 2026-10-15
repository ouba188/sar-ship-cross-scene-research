"""Report the prespecified five-arm development experiment without cherry-picking."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path('E:/Hermes/paper_tracking/sensitivity/evidence_aware_handoff_20261008')
NAMES = dict(reference='原联合模型', stable_recalibration='稳定证据重标定',
             source_scale_control='普通源域分数缩放', matched_score_gate='匹配分数门控',
             evidence_gate='证据感知门控')


def main():
    result = json.loads((ROOT/'COMPLETE.json').read_text(encoding='utf-8'))
    means = pd.read_csv(ROOT/'equal_port_means.csv').set_index('arm')
    params = pd.read_csv(ROOT/'source_selected_parameters.csv')
    contrasts = result['contrasts']
    stable = next(c for c in contrasts if c['candidate'] == 'stable_recalibration' and c['control'] == 'reference')
    aware = next(c for c in contrasts if c['candidate'] == 'evidence_gate' and c['control'] == 'matched_score_gate')
    lines = ['# 稳定证据重标定与证据感知交接：第一阶段实验', '',
        '日期：2026-10-08。使用既有开发语料的外层留港验证；不是全新独立盲测。', '',
        '## 结论', '',
        f'稳定证据重标定相对原联合模型：OA {stable["delta_oa_pp"]:+.4f}、BA {stable["delta_ba_pp"]:+.4f} 个百分点。'
        f'证据感知门控相对同源训练的匹配分数门控：OA {aware["delta_oa_pp"]:+.4f}、BA {aware["delta_ba_pp"]:+.4f} 个百分点。', '',
        '本轮未显示稳定证据输入带来明确额外收益；稳定重标定没有通过独立改动的收益检查，'
        '因此不追加联合版本、不扩展外部数据、不替换原主方案。两个门控都只是OA/BA小幅权衡，'
        '不能把门控重新训练带来的变化归为稳定证据模块的成功。', '',
        '这不推翻前轮“原模型已经使用候选空间”的归因结果。信息被使用与重加权后能进一步提点是两个不同问题。', '',
        '## 1. 本轮问题与设计', '',
        '前轮已经定位到候选稳定空间主要通过源域分类头影响原模型。本轮不新增特征、不重训骨干，'
        '分别检验：重标定既有稳定证据是否有益，以及门控识别证据来源后是否更会选择专家。', '',
        '- 稳定证据重标定：B_new = Base64 + alpha * c_base_source；专家固定，使用原门控。',
        '- 普通源域缩放对照：B_new = Base64 + alpha_control * source_base_logits；专家固定，使用原门控。'
        '这不是对整个Base64做温度缩放，而是重标定源域先验与既有64标签适配项的相对权重。',
        '- 两种alpha独立在源港留一episode上选取，网格[0,-0.25,0.25,-0.5,0.5]，按等港平均Base64 BA，平局优先不改动。',
        '- 匹配分数门控：专家/基础margin、专家/基础entropy、两模型在各自预测类别对上的总分数优势，共5维。',
        '- 证据感知门控：在相同5维基础上增加候选稳定分量的类别对优势，共6维。'
        '其余分量优势等于总优势减稳定优势，不重复添加线性冗余特征。',
        '- 两个新门控使用完全相同的源episode、样本、权重及Logistic C=1，只差最后一项证据特征。'
        '因此“证据感知门控减匹配分数门控”才是稳定证据输入的直接对照。',
        '- 保持原64标签、原5组目标支持集、原专家、原5%交接上限以及资格规则。'
        '重标定可改变基础预测，5%仍仅限制专家交接，不限制基础模型改变的样本数。', '',
        '## 2. 数据访问与验证边界', '',
        '每个外层目标港均沿用原MMSI/产品排除后的源数据。源港内层验证时，稳定方向、标准化统计、'
        'PCA与分类头排除被验证源港；64个支持样本只用于适配，其他源港样本用于选参和门控监督。'
        '所有外层目标策略冻结后才读取目标查询指标。', '',
        '源episode的R18特征由当前外层M3冻结骨干生成。该骨干此前训练见过源episode标签，'
        '所以这是固定源训练骨干下的头部/方向留港验证，不是整个编码器的内层OOF。外层目标港未参与骨干、'
        '方向、alpha或门控训练。目标侧证据归因沿用已记录的近似恢复PCA映射，不能宣称原历史映射逐位精确。', '',
        '## 3. 23港完整结果', '',
        '先平均每港5组支持抽样，再对23港等权平均。指标为八分类OA/BA，单位%。', '',
        '| 方法 | 完整OA | 完整BA | 基础OA | 基础BA | 平均交接数/港/抽样 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in NAMES:
        row = means.loc[arm]
        lines.append(f'| {NAMES[arm]} | {100*row.acc:.4f} | {100*row.ba:.4f} | '
                     f'{100*row.base_acc:.4f} | {100*row.base_ba:.4f} | {row.selected:.3f} |')
    lines += ['', '## 4. 配对比较', '', '| 候选减对照 | OA变化/百分点 | BA变化/百分点 | BA改善港 | BA下降港 |',
              '|---|---:|---:|---:|---:|']
    for c in contrasts:
        lines.append(f'| {NAMES[c["candidate"]]} - {NAMES[c["control"]]} | {c["delta_oa_pp"]:+.4f} | '
                     f'{c["delta_ba_pp"]:+.4f} | {c["positive_ports"]} | {c["negative_ports"]} |')
    lines += ['', '五个方法均为每港每抽样平均交接79条，门控变化不是通过减少交接数量获得。'
        '原模型平均纠正43.522条、误改16.017条；匹配分数门控40.800/12.339条，'
        '证据感知门控40.722/12.478条。新门控减少误改，也放弃了部分纠正。'
        '这些是等港平均计数，不能用平均计数之差或比值替代等港平均准确率。']
    lines += ['', '## 5. 源港选参', '', '| alpha | 稳定证据重标定使用港数 | 普通源域缩放使用港数 |',
              '|---:|---:|---:|']
    for a in [0., -.25, .25, -.5, .5]:
        lines.append(f'| {a:+.2f} | {int((params.stable_alpha == a).sum())} | '
                     f'{int((params.source_scale_alpha == a).sum())} |')
    lines += ['', '## 6. 如何解释', '',
        '两种改动必须分别判断：稳定重标定需要与原模型及普通源域缩放比较；证据门控需要与原模型及'
        '匹配分数门控比较。门控重训本身带来的变化不能全部归功于稳定空间。', '',
        '此阶段没有联合两个改动，也没有按目标结果修改alpha网格、门控参数或筛选港口。'
        '若两项各自有效，再另存联合实验；若没有收益，则保留失败，不把独立信息归因直接包装成有效新模块。', '',
        '候选空间此前只是平均类条件漂移较小，未证明严格或因果不变性。这里的改动也不构成因果识别。'
        '门控阈值>0仍然存在，不是纯排序机制；两分支均计算，5%交接不是实测推理节省。', '',
        '## 7. 产物与复现', '',
        '代码目录：`E:/Hermes/paper_tracking/sensitivity/oof_reliability_check`。', '',
        '1. GPU环境运行 `extract_evidence_source_features.py`，冻结提取各外层源港R18特征。',
        '2. ClearSAR运行 `prepare_evidence_source_episodes.py`，拟合并缓存内层源方向与头部读出。',
        '3. ClearSAR运行 `run_evidence_aware_handoff.py`，源选参/拟合门控、冻结目标策略、全港完成后统一评估。',
        '4. ClearSAR运行 `report_evidence_aware_handoff.py`，生成本报告。', '',
        '本目录保存PROTOCOL.json、source_pair_cache、source_episodes、各港source_models.pkl与policies.npz，'
        '以及metrics.csv、port_means.csv和source_selected_parameters.csv。原联合模型文件未修改。', '']
    (ROOT/'EVIDENCE_AWARE_FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print('REPORT '+str(ROOT/'EVIDENCE_AWARE_FINDINGS_ZH.md'), flush=True)


if __name__ == '__main__':
    main()
