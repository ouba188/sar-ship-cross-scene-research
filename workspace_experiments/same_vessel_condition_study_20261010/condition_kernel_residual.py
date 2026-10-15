"""Identity-initialized context correction of geometry, with raw source priors fixed."""

from relational_factors import (
    BASE, CACHE, OLD, OUT as FACTORS, SEED, FactorNet, source_readout, design,
    residual_head, sample_weights, metrics, save_json,
)
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from target_response_adapter import support_design
from channel_operator_report import table

OUT = BASE/'condition_kernel_residual_20261010'
ARMS = ('ordinary_context', 'condition_context')
STEPS = 120


class KernelResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = nn.Linear(32, 512)
        nn.init.zeros_(self.map.weight)
        nn.init.zeros_(self.map.bias)

    def gains(self, context):
        return .1*torch.tanh(self.map(context))

    def forward(self, phi, context):
        updated = phi[:, :512]*(1+self.gains(context))
        return torch.cat([updated, phi[:, 512:]], 1)


def make_episodes(meta, raw):
    episodes, records = [], []
    for port, group in meta.groupby('port', sort=True):
        if group.mmsi.nunique() < 72:
            continue
        reference = np.flatnonzero(~meta.port.eq(port)&~meta.mmsi.isin(group.mmsi))
        if meta.iloc[reference].class_id.nunique() != 8:
            continue
        state = source_readout(raw[reference], meta.iloc[reference].reset_index(drop=True))
        with torch.no_grad():
            ph, prior = design(torch.from_numpy(raw), state)
        local, global_rows = group.reset_index(drop=True), group.index.to_numpy()
        for draw in range(3):
            support, candidates = support_design(local, draw)
            rng = np.random.default_rng(SEED+1000+draw)
            qm = local.iloc[candidates]
            query = np.array([rng.choice(q.index.to_numpy()) for _, q in qm.groupby('mmsi')])
            support, query = global_rows[support], global_rows[query]
            if set(meta.iloc[support].mmsi)&set(meta.iloc[query].mmsi):
                raise ValueError('Episode support/query identity overlap')
            if set(meta.iloc[reference].mmsi)&set(meta.iloc[np.r_[support, query]].mmsi):
                raise ValueError('Episode source-head identity overlap')
            missing = set(meta.iloc[query].class_id)-set(meta.iloc[support].class_id)
            tag = port.replace(' ', '_')+f'_draw{draw}'
            episodes.append(dict(tag=tag, support=support, query=query, reference=reference, phi=ph, prior=prior))
            records.append(dict(port=port, draw=draw, reference_chips=len(reference), support_ships=len(support),
                query_ships=len(query), query_classes=meta.iloc[query].class_id.nunique(), missing_support_query_classes=len(missing)))
    return episodes, pd.DataFrame(records)


def fit(context, meta, episodes, arm):
    model = KernelResidual()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    y = torch.from_numpy(meta.class_id.to_numpy(copy=True)).long()
    traces = []
    for step in range(STEPS):
        episode = episodes[step % len(episodes)]
        support, query = episode['support'], episode['query']
        n_support = len(support)
        ids = np.r_[support, query]
        ph = model(episode['phi'][ids], context[ids])
        prior = episode['prior'][ids]
        beta, _, _ = residual_head(ph[:n_support], prior[:n_support], y[support], balanced=True)
        scores = prior[n_support:]+ph[n_support:]@beta
        target = 2*torch.eye(8)[y[query]]-1
        w = torch.from_numpy(sample_weights(meta.iloc[query].reset_index(drop=True)).astype(np.float32))
        query_loss = ((scores-target).square().mean(1)*w).mean()
        gain = model.gains(context[ids])
        regularizer = gain.square().mean()
        loss = query_loss+.1*regularizer
        optimizer.zero_grad()
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
        optimizer.step()
        if step in (0, 29, 59, 89, STEPS-1):
            traces.append(dict(arm=arm, step=step+1, episode=episode['tag'], query_loss=float(query_loss.detach()),
                gain_rms=float(regularizer.detach().sqrt()), gradient_norm=norm))
            print('EPISODE_TRAIN', traces[-1], flush=True)
    return model.eval(), traces


def score(ph, prior, labels):
    beta, _, _ = residual_head(ph[:64], prior[:64], torch.from_numpy(labels).long(), balanced=True)
    return prior[64:]+ph[64:]@beta


def diagnose_examples(base_score, score_new, base_phi, new_phi, support_labels, truth):
    delta = (score_new-base_score).numpy()
    n = len(truth)
    r = np.arange(n)
    # Margins use each score matrix's strongest incorrect class, not a fixed rival.
    b, a = base_score.numpy().copy(), score_new.numpy().copy()
    bt, at = b[r, truth].copy(), a[r, truth].copy()
    b[r, truth], a[r, truth] = -np.inf, -np.inf
    margin_change = (at-a.max(1))-(bt-b.max(1))
    old_ok, new_ok = base_score.argmax(1).numpy() == truth, score_new.argmax(1).numpy() == truth
    old_sim = base_phi[64:, :-1]@base_phi[:64, :-1].T
    new_sim = new_phi[64:, :-1]@new_phi[:64, :-1].T
    old_top, new_top = old_sim.argmax(1).numpy(), new_sim.argmax(1).numpy()
    absent = np.setdiff1d(np.arange(8), np.unique(support_labels))
    absent_change = float(abs(delta[:, absent]).max()) if len(absent) else 0.
    if absent_change != 0:
        raise ValueError('Absent-class scores must stay equal to the raw prior')
    return dict(rescue=int((~old_ok&new_ok).sum()), harm=int((old_ok&~new_ok).sum()),
        mean_true_margin_change=float(margin_change.mean()), max_abs_score_change=float(abs(delta).max()),
        top_support_switch_pct=float(100*(old_top != new_top).mean()),
        top_support_class_match_before=float(100*(support_labels[old_top] == truth).mean()),
        top_support_class_match_after=float(100*(support_labels[new_top] == truth).mean()),
        absent_class_score_max_change=absent_change)


def evaluate_targets(obs, raw, contexts, models, output_dir=OUT):
    records, classes, diagnoses, replays = [], [], [], []
    for path in sorted(OLD.glob('*_split.npz')):
        tag = path.stem.removesuffix('_split')
        with np.load(path) as z:
            support, query, labels = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
        with np.load(OLD/f'{tag}_models.npz') as z:
            state = [torch.from_numpy(z[k]) for k in ('source_center', 'source_scale', 'source_coef', 'source_bias')]
        ids = np.r_[support, query]
        with torch.no_grad():
            ph, prior = design(torch.from_numpy(raw[ids]), state)
            baseline = score(ph, prior, labels)
        with np.load(BASE/'spectral_residual_20261010'/f'{tag}_models.npz') as z:
            old = z['head_only_scores']
        disagreement = int((old.argmax(1) != baseline.argmax(1).numpy()).sum())
        replays.append(dict(task=tag, prediction_disagreement=disagreement, max_score_difference=float(abs(old-baseline.numpy()).max())))
        if disagreement:
            raise ValueError('Original classification baseline failed replay')
        saved = dict(query_obs_ids=query, support_obs_ids=support, baseline_scores=baseline.numpy(), raw_prior=prior[64:].numpy())
        all_scores = {'baseline': baseline}
        with torch.no_grad():
            for arm, model in models.items():
                changed = model(ph, contexts[arm][ids])
                all_scores[arm] = score(changed, prior, labels)
                saved[arm+'_scores'] = all_scores[arm].numpy()
                gain = model.gains(contexts[arm][ids])
                diagnoses.append(dict(task=tag, arm=arm, gain_rms=float(gain.square().mean().sqrt()),
                    relative_update_rms=float(torch.linalg.norm(changed[:, :-1]-ph[:, :-1])/torch.linalg.norm(ph[:, :-1])),
                    **diagnose_examples(baseline, all_scores[arm], ph, changed, labels, obs.iloc[query].class_id.to_numpy())))
        truth = obs.iloc[query].class_id.to_numpy()
        port, draw = obs.iloc[query].port.iloc[0], int(tag.rsplit('draw', 1)[1])
        for arm, scores in all_scores.items():
            pred = scores.argmax(1).numpy()
            records.append(dict(arm=arm, port=port, draw=draw, **metrics(pred, truth)))
            for c in np.unique(truth):
                count = int((labels == c).sum())
                classes.append(dict(arm=arm, port=port, draw=draw, class_id=int(c),
                    coverage='absent' if count == 0 else 'singleton' if count == 1 else 'multiple',
                    recall=float(100*(pred[truth == c] == c).mean())))
        np.savez_compressed(output_dir/(tag+'.npz'), **saved)
        print('TARGET_DONE', tag, records[-len(all_scores):], flush=True)
    return records, classes, diagnoses, replays


def source_holdout(obs, raw, train_ids, val_ids, contexts, models):
    tm, vm = obs.iloc[train_ids].reset_index(drop=True), obs.iloc[val_ids].reset_index(drop=True)
    state = source_readout(raw[train_ids], tm)
    records = []
    for draw in range(3):
        support, query = support_design(vm, draw)
        ids = val_ids[np.r_[support, query]]
        labels = vm.iloc[support].class_id.to_numpy(copy=True)
        truth = vm.iloc[query].class_id.to_numpy()
        with torch.no_grad():
            ph, prior = design(torch.from_numpy(raw[ids]), state)
            for arm, model in [('baseline', None), *models.items()]:
                changed = ph if model is None else model(ph, contexts[arm][ids])
                pred = score(changed, prior, labels).argmax(1).numpy()
                records.append(dict(arm=arm, draw=draw, query_chips=len(query), query_ships=vm.iloc[query].mmsi.nunique(), **metrics(pred, truth)))
    return records


def report():
    rows = pd.read_csv(OUT/'target_results.csv')
    summary = rows.groupby('arm')[['OA', 'BA']].mean().reset_index()
    ports = rows.groupby(['port', 'arm'])[['OA', 'BA']].mean().reset_index()
    cl = pd.read_csv(OUT/'target_by_class.csv')
    coverage = cl.groupby(['port', 'draw', 'arm', 'coverage']).recall.mean().reset_index().groupby(
        ['arm', 'coverage']).agg(recall=('recall', 'mean'), task_groups=('draw', 'size')).reset_index()
    diagnostic = pd.read_csv(OUT/'target_diagnosis.csv').groupby('arm').mean(numeric_only=True).reset_index()
    held = pd.read_csv(OUT/'source_holdout.csv').groupby('arm')[['OA', 'BA']].mean().reset_index()
    for name, data in [('summary', summary), ('port_summary', ports), ('coverage_summary', coverage), ('diagnosis_summary', diagnostic), ('source_holdout_summary', held)]:
        data.to_csv(OUT/(name+'.csv'), index=False)
    lines = ['# 保留原主路的条件引导案例关系修正', '',
        '## 结论', '',
        '条件版目标BA38.261%，相对基线38.304%下降0.043个百分点，相对普通上下文38.019%高0.242个百分点。原来替换整个判别表示的大幅退化没有再出现，但没有取得跨港净收益，不加入主模型。',
        '源域留出船舶任务BA由40.425%升至40.960%（+0.535个百分点）。该任务64支持来自多个源港混合人群，而目标评价每港64支持，任务组成不同；不能将两者差别仅归因于新港条件。', '',
        '## 目标港结果', '',
        table(summary, list(summary.columns)), '', table(ports, list(ports.columns)), '',
        table(coverage, list(coverage.columns)), '', '## 源域身份留出诊断', '', table(held, list(held.columns)), '',
        '留出的302艘源船不参与条件编码器或修正网络训练。每次64艘作支持，其余身份查询；可共享港口及大景。这里是独立身份诊断，不是源训练episode查询表现。', '',
        '## 本轮管线', '',
        '- 原512维特征及其源分类分数完整保留。条件/普通上下文向量均32维、冻结；输入同样的零初始化线性层，经0.1*tanh生成512维增益，调节原标准化特征后形成支持核。',
        '- ordinary_context来自上轮同结构但无关系损失网络；condition_context来自同船/同景关系监督网络。两者残差参数均16896、相同源episodes、120步Adam、学习率0.01。',
        '- 源域模拟64支持任务，在支持集之外的船舶上监督最终残差分类结果，不直接奖励同船向量相似。训练查询参与梯度，因此不把它们称为独立验证。',
        '- 源任务分类器排除整个模拟目标港及其所有船舶身份；已有冻结条件编码器训练曾使用这些源港身份，因此这不是对整个系统严格留港的验证。',
        '- 目标港只用原64标签拟合相同的残差分类头，不额外更新神经网络或按目标港挑模型。原先验固定，支持未覆盖类别的分数严格不变，但仍可能被其他类别竞争压过。',
        '- 只检验条件分支如何使用；没有让128维稳定分支替代主路，没有恢复频率实验。条件向量仍可能含船型信息，不能称纯环境因子。', '',
        '## 实际影响', '', table(diagnostic, list(diagnostic.columns)), '',
        'rescue/harm是每组查询计数的平均值，查询组重复，不能跨组相加当作独立船数。最近支持案例仅为诊断，真实预测使用全部64支持的带符号残差贡献。', '',
        '## 边界', '',
        '- 本轮两个源港可构成64支持任务，共6个固定训练episodes；修正网络源任务覆盖有限。没有用目标查询选超参数、训练轮次或版本。',
        '- Jebel Ali每个训练任务查询22艘，Singapore为14艘；6个任务的查询类别均被对应支持集覆盖。本轮不是通过缺类正例训练学会保护缺类，而是通过固定先验保证缺类原分数不变。',
        '- 冻结R18小试、单次神经初始化、原3港9次支持抽样；不是完整双路模型或独立外部确认。继承原预训练曝光与目标支持/查询共景边界。', '']
    frozen = OUT/'dynamic_diagnosis'
    if (frozen/'COMPLETE.json').exists():
        dynamic = pd.read_csv(frozen/'summary.csv')
        source_dynamic = pd.read_csv(frozen/'source_holdout_summary.csv')
        gain = pd.read_csv(frozen/'source_gain_structure.csv')
        lines += ['## 条件向量是否真的需要逐船变化', '',
            table(dynamic, list(dynamic.columns)), '', table(source_dynamic, list(source_dynamic.columns)), '',
            '固定已学权重，只将逐船增益替换为源训练1744张图上的平均增益，不重新训练。条件版在目标港由动态38.261%变为固定平均38.303%，后者基本等于原基线；源留出任务则由动态40.960%变为固定平均40.725%。动态条件使用在源内有信号，但本轮未显示跨港额外收益。', '',
            table(gain, list(gain.columns)), '',
            '平均增益RMS约1.12%，动态约7.49%。此干预同时减小更新幅度，不是等幅度静态模型对照，因此不能单独证明样本自适应一定无效，也不能据此选择目标成绩最好的变体部署。', '',
            '## 失效定位', '',
            '- 原先验严格不变；目标支持缺类分数实际最大变化为0。缺类召回仍从33.503%变为33.832%，来自其他类别竞争变化，而非给缺类新增标签或修改其源分数。',
            '- 条件版实际特征更新RMS约7.70%，约6.54%的查询更换了点积最高的支持案例。不是模块未接入或没有产生变化。最近案例诊断不等价于完整带符号残差分类。',
            '- 每组平均纠正3.89张、误改4.67张；目标平均真实类间隔变化-0.000299。更新产生了正负两方面作用，尚未形成净收益。重复抽样查询不独立，不把这些计数相加当独立样本。',
            '- 两项针对性测试通过；9组原基线、18组动态版本回放预测均一致。未发现可解释退化的接入错误，但这些检查不能证明没有任何实现问题。',
            '- 剩余假设是条件校正规则的跨港可迁移性、源任务覆盖与更新幅度之间的关系。当前6个源任务只覆盖2港是实际限制，但尚未证明它是唯一根因。', '',
            '## 后续取舍', '',
            '保留本轮的原判别主路保护和条件关系学习作为候选，不宣称域不变因子已经被纯净分离或跨港提点成功。若继续，优先扩大源域模拟任务的港口/条件覆盖，固定其他设计后观察收益能否迁移；本轮没有追加新训练或超参数搜索。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)
    print(held.to_string(index=False), flush=True)
    print(diagnostic.to_string(index=False), flush=True)


def run():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed experiment')
    started = time.perf_counter()
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(FACTORS/'source_split.npz') as z:
        train_ids, val_ids = z['train_obs_ids'], z['diagnostic_obs_ids']
        center, scale = z['center'], z['scale']
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[obs.manifest_row], dtype=np.float32)
    normalized = (raw-center)/scale
    tm = obs.iloc[train_ids].reset_index(drop=True)
    contexts = {}
    for arm, checkpoint in zip(ARMS, ('task_only', 'relations')):
        model = FactorNet()
        model.load_state_dict(torch.load(FACTORS/(checkpoint+'.pt'), map_location='cpu', weights_only=True))
        model.eval().requires_grad_(False)
        with torch.no_grad():
            d = torch.cat([model.condition(torch.from_numpy(normalized[lo:lo+256])) for lo in range(0, len(raw), 256)])
            dc = d[train_ids].mean(0)
            ds = d[train_ids].std(0, correction=0).clamp_min(1e-3)
            contexts[arm] = (d-dc)/ds
        np.savez_compressed(OUT/(arm+'_context_scale.npz'), center=dc.numpy(), scale=ds.numpy())
    episodes, designs = make_episodes(tm, raw[train_ids])
    if not episodes:
        raise ValueError('No eligible source episodes')
    designs.to_csv(OUT/'source_episode_design.csv', index=False)
    for ep in episodes:
        np.savez_compressed(OUT/(ep['tag']+'_episode.npz'),
            **{k+'_obs_ids': train_ids[ep[k]] for k in ('support', 'query', 'reference')})
    save_json(OUT/'PROTOCOL.json', dict(seed=SEED, source_steps=STEPS, source_episodes=len(episodes),
        source_ports=designs.port.unique().tolist(), correction_parameters=16896, max_coordinate_gain_change=.1,
        prior='Exact original prior scores unchanged for every arm', source_training='Final query weighted MSE through differentiable64-support ridge8 +.1 gain square; identity initialization,120 Adam steps lr.01',
        source_split='Same1744 observations/605 identities as previous representation training;302 identities remain diagnostic only',
        context_control='Previous task_only32d vs relations32d; frozen matched factor network architecture and source budget',
        target='Original9x64 splits; residual-head fit only, no target neural adaptation; no query selection',
        resources='CPU1; no backbone or parent changes'))
    models, histories = {}, []
    for arm in ARMS:
        models[arm], history = fit(contexts[arm][train_ids], tm, episodes, arm)
        histories.extend(history)
        torch.save(models[arm].state_dict(), OUT/(arm+'.pt'))
    rows, classes, diagnosis, replays = evaluate_targets(obs, raw, contexts, models)
    holdout = source_holdout(obs, raw, train_ids, val_ids, contexts, models)
    for name, data in [('target_results', rows), ('target_by_class', classes), ('target_diagnosis', diagnosis),
                       ('source_training', histories), ('source_holdout', holdout)]:
        pd.DataFrame(data).to_csv(OUT/(name+'.csv'), index=False)
    save_json(OUT/'REPLAYS.json', replays)
    save_json(OUT/'COMPLETE.json', dict(source_fits=2, source_steps_each=STEPS, target_tasks=9,
        neural_target_fits=0, seconds=time.perf_counter()-started, query_labels_used_for_training=False))
    report()


if __name__ == '__main__':
    run()
