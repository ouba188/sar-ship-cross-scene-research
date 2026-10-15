"""Matched32-support narrow/broad source curricula; all target supports remain64."""

from condition_kernel_residual import (
    BASE, CACHE, FACTORS, OUT as PREVIOUS, ARMS, STEPS, SEED,
    FactorNet, source_readout, design, fit, evaluate_targets, source_holdout, save_json,
)
import time
import numpy as np
import pandas as pd
import torch
from channel_operator_report import table

OUT = BASE/'condition_coverage_20261010'
NARROW = ('Jebel Ali', 'Singapore')


def make_episodes(meta, raw):
    episodes, records = [], []
    for port, group in meta.groupby('port', sort=True):
        ships = np.sort(group.mmsi.unique())
        if len(ships) < 48:
            continue
        reference = np.flatnonzero(~meta.port.eq(port)&~meta.mmsi.isin(ships))
        if meta.iloc[reference].class_id.nunique() != 8:
            continue
        state = source_readout(raw[reference], meta.iloc[reference].reset_index(drop=True))
        with torch.no_grad():
            ph, prior = design(torch.from_numpy(raw), state)
        for draw in range(3):
            rng = np.random.default_rng(SEED+draw)
            sampled = rng.choice(ships, 48, replace=False)
            rows = np.array([rng.choice(group.index[group.mmsi.eq(ship)].to_numpy()) for ship in sampled])
            support, query = rows[:32], rows[32:]
            if set(meta.iloc[support].mmsi)&set(meta.iloc[query].mmsi):
                raise ValueError('Support/query identity overlap')
            if set(meta.iloc[reference].mmsi)&set(sampled):
                raise ValueError('Episode source prior has query/support identities')
            missing = set(meta.iloc[query].class_id)-set(meta.iloc[support].class_id)
            tag = port.replace(' ', '_')+f'_draw{draw}'
            episodes.append(dict(tag=tag, port=port, draw=draw, support=support, query=query,
                                 reference=reference, phi=ph, prior=prior))
            records.append(dict(port=port, draw=draw, reference_chips=len(reference), support_ships=32,
                query_ships=16, support_classes=meta.iloc[support].class_id.nunique(),
                query_classes=meta.iloc[query].class_id.nunique(), missing_support_query_classes=len(missing)))
    # Interleave ports so the incomplete last cycle does not favor a whole block of one port.
    episodes.sort(key=lambda e: (e['draw'], e['port']))
    return episodes, pd.DataFrame(records)


def report():
    rows = pd.read_csv(OUT/'target_results.csv')
    summary = rows.groupby('arm')[['OA', 'BA']].mean().reset_index()
    ports = rows.groupby(['port', 'arm'])[['OA', 'BA']].mean().reset_index()
    cl = pd.read_csv(OUT/'target_by_class.csv')
    coverage = cl.groupby(['port', 'draw', 'arm', 'coverage']).recall.mean().reset_index().groupby(
        ['arm', 'coverage']).agg(recall=('recall', 'mean'), task_groups=('draw', 'size')).reset_index()
    held = pd.read_csv(OUT/'source_holdout.csv').groupby('arm')[['OA', 'BA']].mean().reset_index()
    diag = pd.read_csv(OUT/'target_diagnosis.csv').groupby('arm').mean(numeric_only=True).reset_index()
    contrasts = []
    for context in ARMS:
        narrow = rows[rows.arm.eq('narrow32_'+context)].set_index(['port', 'draw'])
        broad = rows[rows.arm.eq('broad32_'+context)].set_index(['port', 'draw'])
        diff = broad[['OA', 'BA']]-narrow[['OA', 'BA']]
        for (port, draw), r in diff.iterrows():
            contrasts.append(dict(context=context, port=port, draw=draw, delta_OA=r.OA, delta_BA=r.BA))
    contrasts = pd.DataFrame(contrasts)
    contrast_mean = contrasts.groupby('context').agg(delta_OA=('delta_OA', 'mean'), delta_BA=('delta_BA', 'mean'),
        positive_tasks=('delta_BA', lambda x: int((x > 0).sum())), negative_tasks=('delta_BA', lambda x: int((x < 0).sum()))).reset_index()
    for name, frame in [('summary', summary), ('port_summary', ports), ('coverage_summary', coverage),
                        ('source_holdout_summary', held), ('diagnosis_summary', diag),
                        ('coverage_contrasts', contrasts), ('coverage_contrast_summary', contrast_mean)]:
        frame.to_csv(OUT/(name+'.csv'), index=False)
    lines = ['# 条件引导修正：源训练港口覆盖对照', '',
        '## 结论', '',
        '在相同32支持/16查询、120步预算下，源任务从2港扩到9港没有提高目标成绩。条件版BA38.274% -> 38.095%（-0.179个百分点），普通上下文38.414% -> 38.153%（-0.261个百分点）。本轮不能将先前未提点简单解释为覆盖港口太少。',
        '原基线38.304%。2港普通上下文有+0.110个百分点BA的小正结果，完整保留；但它不是关系监督条件分支带来的增益，不能用来证明因子分离有效。单次神经初始化及重叠查询抽样也不足以确认稳定提升。', '',
        '## 目标结果', '',
        table(summary, list(summary.columns)), '', table(contrast_mean, list(contrast_mean.columns)), '',
        '主比较为相同32支持/16查询、相同120步训练下的broad32减narrow32。真实目标评估仍为原3港9次64标签支持，不改变目标标注预算。', '',
        '## 分港与缺类', '', table(ports, list(ports.columns)), '', table(coverage, list(coverage.columns)), '',
        '缺类召回仅4组任务；各支持抽样的查询人群有重叠。', '',
        '## 源域身份留出', '', table(held, list(held.columns)), '',
        '仍为原302艘留出源船，64支持来自混合源港，与逐目标港64支持设定不同。没有用这组诊断选轮次或超参数。', '',
        '## 更新是否生效', '', table(diag, list(diag.columns)), '',
        '所有模型保留原源分类分数，仅调整原512维案例关系；支持缺类分数严格不变，但其他类别竞争仍可改变缺类召回。', '',
        '## 控制范围', '',
        '- narrow32仅Jebel Ali和Singapore；broad32加入所有至少48艘且参考源集仍有8类的源港。窄覆盖任务直接取宽覆盖池中的同一配对，无独立重新抽样。',
        '- 每任务32支持船、16独立查询船，各船只取一张。两组均120步、Adam0.01、16896个参数、相同±10%限制；宽覆盖不增加梯度步数。',
        '- 普通上下文/条件上下文编码器、归一化、原源分类器与302艘诊断身份全部冻结。全部使用原1744源图/605船，没有新增人工标签或真实目标训练数据。',
        '- 各港按轮次交错，宽覆盖港训练次数尽可能均衡；重复源查询参与训练，不是独立确认。参考分类器排除整个模拟目标港及其船舶，冻结上下文网络的旧源训练曝光未撤销。',
        '- 比上轮64支持训练减少源支持数以容纳更多港口。因此上轮结果只作历史参考，不能将其与broad32的差值全部归因于港口覆盖。',
        '- 不按目标成绩挑港口、权重或版本；CPU单线程，无主线程或骨干改动。冻结R18开发小试，不是完整主模型或因果鉴定。', '']
    diagnostic = OUT/'source_diagnosis'
    if (diagnostic/'COMPLETE.json').exists():
        source = pd.read_csv(diagnostic/'summary.csv')
        gradients = pd.read_csv(diagnostic/'gradient_summary.csv')
        lines += ['## 为什么扩覆盖仍然没有迁移收益', '',
            table(source, list(source.columns)), '',
            '这是对同一批固定源训练任务的回放，不是独立验证。条件版在新增7港的MSE由未修正0.317587降至0.312667，2港训练模型在这些任务上为0.317471。说明扩覆盖确实学到了新增源任务，不是完全没有更新或训练没有作用。',
            '但原2港的MSE由窄覆盖0.392529升至宽覆盖0.396246，仍好于未修正0.400227。扩覆盖产生了源任务间的取舍；源身份留出BA也从窄覆盖40.870%降至宽覆盖40.225%，并非只在目标3港出现下降。', '',
            table(gradients, list(gradients.columns)), '',
            '在零修正初始化处，先求各港3个任务的平均梯度，再比较9港间36对方向：条件版15对余弦为负，普通上下文8对为负，均值分别0.0097和0.0181。这是局部优化方向冲突/差异的线索，不是港口物理机制鉴定，也不能唯一证明最终退化的原因。Adam、非线性与后续梯度变化没有被这个静态诊断完整描述。',
            '固定120步时，窄覆盖每港60步，宽覆盖每港13或14步。因此已检验的是固定总计算预算下扩覆盖，不是每港保持同等训练次数的比较；训练不足与任务冲突尚未完全分离，本轮不追加步数搜索。', '',
            '## 实现与机制边界', '',
            '- 4项检查通过，包括32支持训练切片、身份排除、零修正恒等及固定先验；9组原基线预测回放一致。',
            '- 宽覆盖条件版实际特征更新RMS约7.29%，6.76%的查询更换了点积最高的支持案例，说明模块确实介入。每组平均纠正3.89张、误改5.44张，仍未产生净收益。',
            '- 所有目标支持缺类分数的最大变化为0。新增27个源任务只有1个存在查询类别未被支持覆盖，不能宣称已充分训练缺类恢复。',
            '- 本轮没有证明条件因子无价值，也没有发现足以解释结果的接入错误。更值得检验的是：编码出的条件差异，是否对应可复用的判别修正方向，而非只会区分场景。',
            '- 不再把增加源港数量当成现成修复，不将本轮条件模块并入主模型。后续若继续，应针对条件信息与正确修正方向的对应关系设计实验，而不是继续无差别扩港或加大改动幅度。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)
    print(contrast_mean.to_string(index=False), flush=True)
    print(held.to_string(index=False), flush=True)


def run():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed results')
    start = time.perf_counter()
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(FACTORS/'source_split.npz') as z:
        train_ids, val_ids, center, scale = [z[k] for k in ('train_obs_ids', 'diagnostic_obs_ids', 'center', 'scale')]
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[obs.manifest_row], dtype=np.float32)
    x = (raw-center)/scale
    tm = obs.iloc[train_ids].reset_index(drop=True)
    base_context = {}
    for arm, checkpoint in zip(ARMS, ('task_only', 'relations')):
        encoder = FactorNet()
        encoder.load_state_dict(torch.load(FACTORS/(checkpoint+'.pt'), map_location='cpu', weights_only=True))
        encoder.eval().requires_grad_(False)
        with np.load(PREVIOUS/(arm+'_context_scale.npz')) as z:
            dc, ds = torch.from_numpy(z['center']), torch.from_numpy(z['scale'])
        with torch.no_grad():
            d = torch.cat([encoder.condition(torch.from_numpy(x[lo:lo+256])) for lo in range(0, len(x), 256)])
            base_context[arm] = (d-dc)/ds
    episodes, designs = make_episodes(tm, raw[train_ids])
    curricula = {'narrow32': [e for e in episodes if e['port'] in NARROW], 'broad32': episodes}
    if set(e['port'] for e in curricula['narrow32']) != set(NARROW):
        raise ValueError('The narrow control lost a required source port')
    designs.to_csv(OUT/'source_episode_design.csv', index=False)
    schedule = []
    for name, es in curricula.items():
        for step in range(STEPS):
            e = es[step % len(es)]
            schedule.append(dict(curriculum=name, step=step+1, episode=e['tag'], port=e['port']))
    pd.DataFrame(schedule).to_csv(OUT/'source_schedule.csv', index=False)
    for ep in episodes:
        np.savez_compressed(OUT/(ep['tag']+'_episode.npz'),
            **{k+'_obs_ids': train_ids[ep[k]] for k in ('support', 'query', 'reference')})
    save_json(OUT/'PROTOCOL.json', dict(source_steps=STEPS, source_support=32, source_query=16, target_support=64,
        narrow_ports=list(NARROW), broad_ports=sorted(designs.port.unique()), broad_episodes=len(episodes),
        arms=[c+'_'+a for c in curricula for a in ARMS],
        frozen='Previous condition encoders and scales; original prior; unchanged target9x64 splits',
        selection='All ports selected by source metadata only; no grid or target-selected checkpoint', resources='CPU1, no parent changes'))
    models, contexts, history = {}, {}, []
    for curriculum, eps in curricula.items():
        for arm in ARMS:
            name = curriculum+'_'+arm
            contexts[name] = base_context[arm]
            models[name], records = fit(contexts[name][train_ids], tm, eps, name)
            history.extend(records)
            torch.save(models[name].state_dict(), OUT/(name+'.pt'))
    results, classes, diagnoses, replays = evaluate_targets(obs, raw, contexts, models, output_dir=OUT)
    held = source_holdout(obs, raw, train_ids, val_ids, contexts, models)
    for name, data in [('target_results', results), ('target_by_class', classes), ('target_diagnosis', diagnoses),
                       ('source_training', history), ('source_holdout', held)]:
        pd.DataFrame(data).to_csv(OUT/(name+'.csv'), index=False)
    save_json(OUT/'REPLAYS.json', replays)
    save_json(OUT/'COMPLETE.json', dict(source_fits=4, target_tasks=9, seconds=time.perf_counter()-start,
        neural_target_fits=0, query_labels_used_for_training=False))
    report()


if __name__ == '__main__':
    run()
