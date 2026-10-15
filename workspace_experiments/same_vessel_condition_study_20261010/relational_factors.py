"""Source ship/scene relation supervision with matched factor-network controls."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import time
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from channel_operator import BASE, CACHE, OUT as SOURCE, SEED, metrics, save_json
from scene_training import sufficient, fit_head, sample_weights
from target_response_adapter import residual_head
from target_loss_repair import OLD, support_loss, supervised_source_loss
from channel_operator_report import table

OUT = BASE/'relational_factors_20261010'
ARMS = ('task_only', 'relations')
SOURCE_STEPS = 400


class FactorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stable = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 128), nn.LayerNorm(128))
        self.condition = nn.Sequential(nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 32), nn.LayerNorm(32))
        self.correction = nn.Linear(32, 128)
        self.classifier = nn.Linear(128, 8)
        self.decoder = nn.Sequential(nn.Linear(160, 256), nn.GELU(), nn.Linear(256, 512))

    def forward(self, x):
        stable, condition = self.stable(x), self.condition(x)
        correction = .1*torch.tanh(self.correction(condition))
        corrected = stable+correction
        reconstruction = self.decoder(torch.cat([stable, condition], dim=1))
        return stable, condition, correction, corrected, reconstruction


def relation_pools(meta):
    meta = meta.reset_index(drop=True)
    y, ships, scenes = [meta[c].to_numpy() for c in ('class_id', 'mmsi', 'product')]
    pairs = []
    for _, group in meta.groupby('mmsi'):
        for a, b in itertools.combinations(group.index, 2):
            if scenes[a] != scenes[b]:
                pairs.append((a, b))
    options = dict(semantic={}, scene={})
    for a in range(len(meta)):
        same_type = y == y[a]
        other_ship, other_scene = ships != ships[a], scenes != scenes[a]
        pos = np.flatnonzero(same_type & other_ship & other_scene)
        neg = np.flatnonzero(~same_type & other_scene)
        if len(pos) and len(neg):
            options['semantic'][a] = (pos, neg)
        pos = np.flatnonzero(~same_type & other_ship & ~other_scene)
        neg = np.flatnonzero(same_type & other_ship & other_scene)
        if len(pos) and len(neg):
            options['scene'][a] = (pos, neg)
    pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    return pairs, options


def sample_triplets(options, rng, n=32):
    anchors = rng.choice(np.array(sorted(options)), n, replace=True)
    return np.array([(a, rng.choice(options[a][0]), rng.choice(options[a][1])) for a in anchors])


def angular_gap(v, triples):
    v = F.normalize(v, dim=1)
    a, p, n = triples.T
    return (v[a]*v[p]).sum(1)-(v[a]*v[n]).sum(1)


def train_model(x, meta, arm):
    torch.manual_seed(SEED)
    model = FactorNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x = torch.from_numpy(x)
    y = torch.from_numpy(meta.class_id.to_numpy(copy=True)).long()
    weights = sample_weights(meta)
    probability = weights/weights.sum()
    pairs, pools = relation_pools(meta)
    counts = pd.Series(meta.iloc[pairs[:, 0]].mmsi.to_numpy()).value_counts()
    pair_weights = np.array([1/counts[v] for v in meta.iloc[pairs[:, 0]].mmsi])
    pair_weights /= pair_weights.sum()
    history = []
    for step in range(SOURCE_STEPS):
        rng = np.random.default_rng(SEED+step)
        ix = rng.choice(len(x), 96, replace=True, p=probability)
        pair = pairs[rng.choice(len(pairs), 32, replace=True, p=pair_weights)]
        sem, scene = [sample_triplets(pools[k], rng) for k in ('semantic', 'scene')]
        s, d, correction, r, reconstruction = model(x[ix])
        classification = .5*(F.cross_entropy(model.classifier(s), y[ix])+F.cross_entropy(model.classifier(r), y[ix]))
        recon = F.mse_loss(reconstruction, x[ix])
        ids = np.r_[pair.ravel(), sem.ravel(), scene.ravel()]
        a_s, a_d, _, a_r, _ = model(x[ids])
        ps, pr = [F.normalize(a[:64], dim=1).reshape(32, 2, -1) for a in (a_s, a_r)]
        invariant = .5*((1-(ps[:, 0]*ps[:, 1]).sum(1)).mean()+(1-(pr[:, 0]*pr[:, 1]).sum(1)).mean())
        triples = np.arange(96).reshape(32, 3)
        semantic = F.relu(.2-angular_gap(a_s[64:160], triples)).mean()
        condition = F.relu(.2-angular_gap(a_d[160:256], triples)).mean()
        loss = classification+.1*recon+.01*correction.square().mean()
        if arm == 'relations':
            loss = loss+.2*invariant+.2*semantic+.2*condition
        optimizer.zero_grad()
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
        optimizer.step()
        if step == 0 or (step+1) % 100 == 0:
            history.append(dict(arm=arm, step=step+1, classification=float(classification.detach()),
                reconstruction=float(recon.detach()), same_ship=float(invariant.detach()),
                semantic_triplet=float(semantic.detach()), scene_triplet=float(condition.detach()), gradient_norm=grad))
            print('SOURCE_TRAIN', history[-1], flush=True)
    return model.eval(), history


def encode(model, x):
    with torch.no_grad():
        parts = [model(torch.from_numpy(x[lo:lo+256])) for lo in range(0, len(x), 256)]
    return [torch.cat([p[k] for p in parts]).numpy() for k in range(5)]


def source_readout(values, meta):
    center = values.mean(0)
    scale = np.maximum(values.std(0), 1e-3)
    d = values.shape[1]
    x = (values-center)/scale/np.sqrt(d)
    w, b = fit_head(sufficient(x, meta.class_id.to_numpy(), sample_weights(meta)), np.eye(d), np.eye(d))
    return [torch.from_numpy(a.astype(np.float32)) for a in (center, scale, w, b)]


def design(values, state):
    center, scale, weight, bias = state
    x = (values-center)/scale/np.sqrt(values.shape[1])
    return torch.cat([x, torch.ones((len(x), 1))], 1), x@weight+bias


def source_diagnose(arm, train_parts, val_parts, train_meta, val_meta):
    pairs, pools = relation_pools(val_meta)
    sem, scene = [sample_triplets(pools[k], np.random.default_rng(SEED+100), 512) for k in ('semantic', 'scene')]
    records = []
    for name, tv, vv in zip(('stable', 'condition', 'corrected'),
                            [train_parts[k] for k in (0, 1, 3)], [val_parts[k] for k in (0, 1, 3)]):
        state = source_readout(tv, train_meta)
        with torch.no_grad():
            _, scores = design(torch.from_numpy(vv), state)
            v = F.normalize(torch.from_numpy(vv), dim=1)
            distance = (1-(v[pairs[:, 0]]*v[pairs[:, 1]]).sum(1)).numpy()
            semantic_gap, scene_gap = [angular_gap(v, t).numpy() for t in (sem, scene)]
        drift = pd.DataFrame(dict(ship=val_meta.iloc[pairs[:, 0]].mmsi.to_numpy(), distance=distance)).groupby('ship').distance.mean().mean()
        records.append(dict(arm=arm, factor=name, **metrics(scores.argmax(1).numpy(), val_meta.class_id.to_numpy()),
            same_ship_cosine_distance=float(drift), semantic_gap=float(semantic_gap.mean()),
            semantic_triplet_accuracy=float(100*(semantic_gap > 0).mean()), scene_gap=float(scene_gap.mean()),
            scene_triplet_accuracy=float(100*(scene_gap > 0).mean()), coordinate_variance=float(vv.var(0).mean()),
            same_ship_pairs=len(pairs), scene_triplets=len(scene)))
    return records


def adapt(stable, correction, source_s, source_c, source_labels, labels, state):
    gate = torch.zeros(128, requires_grad=True)
    optimizer = torch.optim.Adam([gate], lr=.01)
    y = torch.from_numpy(labels).long()
    counts = torch.bincount(y, minlength=8)
    present = counts > 0
    row_weight = len(y)/(present.sum()*counts[y].float())
    with torch.no_grad():
        _, reference = design(source_s+source_c, state)
    history = []
    for step in range(24):
        rng = np.random.default_rng(SEED+step)
        replay = np.concatenate([rng.choice(np.flatnonzero(source_labels == c), 4, replace=False) for c in range(8)])
        ph, prior = design(stable+correction*(1+gate), state)
        _, loo, target = residual_head(ph, prior, y, balanced=True)
        target_loss = support_loss(loo, target, row_weight, present, True)
        _, source_score = design(source_s[replay]+source_c[replay]*(1+gate), state)
        keep = (source_score-reference[replay]).square().mean()
        source_loss = supervised_source_loss(source_score, torch.from_numpy(source_labels[replay]).long())
        loss = target_loss+.25*keep+.25*source_loss+.1*gate.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            gate.clamp_(-.25, .25)
        if step in (0, 23):
            history.append(dict(step=step+1, target_loss=float(target_loss.detach()), source_keep=float(keep.detach()),
                source_loss=float(source_loss.detach()), gate_rms=float(gate.detach().square().mean().sqrt())))
    return gate.detach(), history


def target_evaluate(arm, s, correction, train, test, obs):
    source_meta = obs.iloc[train].reset_index(drop=True)
    states = {name: source_readout(v[train], source_meta) for name, v in [('stable', s), ('corrected', s+correction)]}
    records, classes, traces = [], [], []
    for path in sorted(OLD.glob('*_split.npz')):
        tag = path.stem.removesuffix('_split')
        with np.load(path) as z:
            support, query, labels = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
        if set(obs.iloc[support].mmsi) & set(obs.iloc[query].mmsi):
            raise ValueError('Support/query ship overlap')
        ss, cs, sr, cr = [torch.from_numpy(v[ix]) for v, ix in [(s, support), (correction, support), (s, train), (correction, train)]]
        gate, history = adapt(ss, cs, sr, cr, source_meta.class_id.to_numpy(), labels, states['corrected'])
        traces.extend([dict(arm=arm, task=tag, **h) for h in history])
        predictions = {}
        saved = dict(query_obs_ids=query, support_obs_ids=support, gate=gate.numpy())
        for mode in ('stable', 'corrected', 'adapted'):
            ids = np.r_[support, query]
            values = torch.from_numpy(s[ids])
            if mode != 'stable':
                values = values+torch.from_numpy(correction[ids])*(1+gate if mode == 'adapted' else 1)
            state = states['stable' if mode == 'stable' else 'corrected']
            with torch.no_grad():
                ph, prior = design(values, state)
                beta, _, _ = residual_head(ph[:64], prior[:64], torch.from_numpy(labels), balanced=True)
                score = prior[64:]+ph[64:]@beta
            predictions[mode] = score.argmax(1).numpy()
            saved[mode+'_scores'], saved[mode+'_prior'] = score.numpy(), prior[64:].numpy()
        truth = obs.iloc[query].class_id.to_numpy()
        port, draw = obs.iloc[query].port.iloc[0], int(tag.rsplit('draw', 1)[1])
        for mode, pred in predictions.items():
            records.append(dict(arm=arm, mode=mode, port=port, draw=draw, **metrics(pred, truth)))
            for c in np.unique(truth):
                count = int((labels == c).sum())
                classes.append(dict(arm=arm, mode=mode, port=port, draw=draw, class_id=int(c),
                    coverage='absent' if count == 0 else 'singleton' if count == 1 else 'multiple',
                    recall=float(100*(pred[truth == c] == c).mean())))
        np.savez_compressed(OUT/f'{arm}_{tag}.npz', **saved)
        print('TARGET_DONE', arm, tag, records[-3:], flush=True)
    np.savez_compressed(OUT/f'{arm}_source_heads.npz', **{name+'_'+key: value.numpy()
        for name, state in states.items() for key, value in zip(('center', 'scale', 'coef', 'bias'), state)})
    return records, classes, traces


def report():
    rows = pd.read_csv(OUT/'target_results.csv')
    summary = rows.groupby(['arm', 'mode'])[['OA', 'BA']].mean().reset_index()
    ports = rows.groupby(['port', 'arm', 'mode'])[['OA', 'BA']].mean().reset_index()
    cl = pd.read_csv(OUT/'target_by_class.csv')
    coverage = cl.groupby(['port', 'draw', 'arm', 'mode', 'coverage']).recall.mean().reset_index().groupby(
        ['arm', 'mode', 'coverage']).agg(recall=('recall', 'mean'), task_groups=('draw', 'size')).reset_index()
    for name, data in [('summary', summary), ('port_summary', ports), ('coverage_summary', coverage)]:
        data.to_csv(OUT/(name+'.csv'), index=False)
    diag = pd.read_csv(OUT/'source_diagnosis.csv')
    lines = ['# 同船与同景关系监督的双因子小试', '',
        '## 本轮结论', '',
        '关系监督产生了候选分工信号，但没有提高跨港分类：稳定分支目标BA由同结构对照33.763%降至33.503%，加条件校正33.310%，64标签微调校正后33.371%。原始特征加分类头基线为38.304%，本轮新网络不加入主模型。',
        '在未参与表示训练的302艘源船上，条件分支场景排序正确率50.391%升至68.164%，但仍可用来判船型（BA30.072%），不是纯场景或纯噪声。稳定分支同船余弦距离0.635降至0.364，但后续去整体均值诊断发现，这一降幅部分来自公共偏移，不可全部解释为跨条件不变性增强。', '',
        '## 本轮实际管线', '',
        '冻结R18的512维池化特征 -> 源训练数据标准化 -> 稳定候选MLP128维 / 条件候选MLP32维；条件向量经线性层和0.1*tanh生成128维修正；重建头从稳定+条件重建输入，分类头同时监督稳定及校正后的表示。',
        '表示训练：1744张、605艘、446景；诊断：823张、302艘、336景。身份无重叠，但有290个大景共用，因此诊断是新船检验，不是新大景检验。源训练采用4670对同船异景观测，含缺少细粒度条件元数据的配对；它们不是4670次已知物理干预。',
        '源训练固定400步后冻结两分支。目标阶段以原939标签拟合源先验，再用原64标签拟合残差分类头；额外适配版只训练128个条件修正增益（相对源修正幅度上下25%），沿用缺类屏蔽及源回放。', '',
        '## 源域未参与表示训练的船舶诊断', '',
        table(diag, list(diag.columns)), '',
        'same_ship_cosine_distance越小表示同船更一致；semantic_gap/scene_gap及对应排序正确率越大越好。条件排序正例为同景异类异船，负例为同类异景异船。所有诊断在留出身份上进行，但部分场景在表示训练中出现过；不能称为独立新场景确认。', '',
        '## 原3港9组64标签目标评估', '', table(summary, list(summary.columns)), '',
        table(ports, list(ports.columns)), '', table(coverage, list(coverage.columns)), '',
        'stable为稳定候选分支；corrected增加源域学到的条件残差；adapted再用64标签调整该残差的128个通道系数。三个版本都重拟合相同形式的源分类头和目标残差头。', '',
        '## 对照与边界', '',
        '- task_only与relations网络、初始化、源样本、训练步数相同，区别仅为是否加入同船一致性、同类跨场景排序和同景异类排序约束。',
        '- 两版均有船型监督、重建任务和条件残差。不能将两版差异归因为凭空增加分支或额外标注。稳定/条件为设计名称，是否成功分工要看诊断，不能依名称认定因果因素。',
        '- 源表示训练只用fold1/2的源船；fold0用于诊断，不用于选超参数或轮次。目标源头重拟合使用原939个允许源标签，故诊断和目标源头的训练范围不同。',
        '- 重复观测使用已知源MMSI船型标签继承，没有新增人工船型标注，但增加了观测及身份关系信息。与历史原始特征基线比较并非等训练量。',
        '- 冻结R18的512维池化特征上学习轻量网络，不更新骨干空间算子；无法据此判定全部SAR信息可分离或物理根因。',
        '- 源域共景不表示船只局部观测条件完全一致。场景排序只提供共享条件代理；条件分支仍可能携带船型信息。',
        '- 固定400步源训练与24步目标适配；CPU1，无目标查询拟合，不按港选版本。支持与查询可共享大景，历史预训练曝光保留。', '']
    diagnosis = OUT/'diagnosis'
    if (diagnosis/'COMPLETE.json').exists():
        paths = pd.read_csv(diagnosis/'path_summary.csv')
        centered = pd.read_csv(diagnosis/'centered_relations.csv')
        lines += ['## 为什么更相似却没有分得更好', '',
            table(centered, list(centered.columns)), '',
            '去除源训练均值后，稳定分支同船距离为0.759 -> 0.672，同类异船距离也从0.818降至0.749。同船/同类异船距离比0.928 -> 0.897，说明还存在相对同船一致性信号，但远小于未经居中时的表面降幅。不能只用整体余弦距离下降宣称找到域不变因素。',
            '条件排序限制为同一港口内比较，并去除源均值后，从44.531%升至57.031%。因此场景关系信号不只来自港口差别，但仍是弱监督代理，可能包含船只状态及局部观测差异。512个抽样三元组会重复锚点，不是512个独立确认样本。', '',
            table(paths, list(paths.columns)), '',
            'raw表示原冻结特征；new表示对应的新特征。prior替换源分类分数，kernel替换支持案例关系，所有组合仍重拟合同一64标签残差头。原始/原始及两个已训练模型的对角预测共27次回放完全一致。交叉组合用于定位路径，不作为按目标成绩挑选的新部署模型。',
            '原始/原始BA38.304%；关系模型只替换源先验降至32.813%，只替换案例关系37.466%，两者都换33.503%。因此大部分直接损害沿新源先验路径出现，存在路径交互，不能把降幅简单相加。只换回旧先验仍未超过基线，不是现成修复。',
            'PCA128对照在同一表示训练源集无监督拟合：全替换BA36.404%，优于两个学习型128维表示但仍低于原始512维。128维压缩可能参与损失，却不足以解释学习型表示的全部退化；同结构无关系约束版也只有33.763%，大幅下降并非全由关系损失引起。', '',
            '## 已定位与未定位', '',
            '- 已定位：关系损失确实改变两分支的关系偏好；部分表面稳定性来自公共偏移；主要直接分类损害与新表示的源分类分数有关。',
            '- 尚未定位：哪些SAR散射变化被条件分支编码；为何新源分类分数跨港变差的唯一根因；可分离因子能否在未见大景上保持同样分工。',
            '- 实现检查：三项针对性测试通过；两分支均有梯度；预测路径回放一致。Pandas只读标签转Tensor曾提示警告，但没有标签原地修改；已改为显式复制，未据此认定训练错误或重跑结果。',
            '- 下一步设计重点：保留原512维识别主路，用条件因子约束有限修正，而不直接让128维候选因子替换整个判别通路；同时用源域未见船舶任务的分类效果约束分离，不能继续把“更相似”当作唯一成功条件。这是建议，尚未运行。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)
    print(diag.to_string(index=False), flush=True)


def run():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed results')
    started = time.perf_counter()
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(SOURCE/'split.npz') as z:
        train, test = z['train_obs_ids'], z['test_obs_ids']
    labeled = obs.iloc[train]
    target_ports = set(obs.iloc[test].port)
    target_ships = set(obs.iloc[test].mmsi)
    extra = obs[obs.mmsi.isin(labeled.mmsi)&~obs.port.isin(target_ports)].copy()
    if set(extra.mmsi)&target_ships or (extra.groupby('mmsi').fold.nunique()>1).any():
        raise ValueError('Invalid identity split')
    if (extra.groupby('mmsi').class_id.nunique()>1).any():
        raise ValueError('Conflicting inherited ship-type labels')
    label_map = labeled.drop_duplicates('mmsi').set_index('mmsi').class_id
    extra['class_id'] = extra.mmsi.map(label_map)
    tm, vm = extra[extra.fold.ne(0)].copy(), extra[extra.fold.eq(0)].copy()
    raw = np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')
    x = np.array(raw[obs.manifest_row], dtype=np.float32)
    center, scale = x[tm.index].mean(0), np.maximum(x[tm.index].std(0), 1e-3)
    x = ((x-center)/scale).astype(np.float32)
    train_ids, val_ids = tm.index.to_numpy(), vm.index.to_numpy()
    tm, vm = tm.reset_index(drop=True), vm.reset_index(drop=True)
    pools = {}
    for name, meta in [('train', tm), ('diagnostic', vm)]:
        pair, options = relation_pools(meta)
        pools[name] = dict(chips=len(meta), ships=int(meta.mmsi.nunique()), scenes=int(meta['product'].nunique()),
            same_ship_pairs=len(pair), semantic_anchors=len(options['semantic']), scene_anchors=len(options['scene']))
        if not len(pair) or not all(options.values()):
            raise ValueError('Missing required relations')
    save_json(OUT/'PROTOCOL.json', dict(seed=SEED, source_steps=SOURCE_STEPS, arms=list(ARMS),
        architecture='Frozen R18 GAP512 -> stable128 / condition32; bounded .1*tanh condition-to-stable correction; source classifier plus reconstruction',
        losses='CE stable/corrected mean +.1 reconstruction +.01 correction L2; relations adds .2 same-ship cosine +.2 semantic triplet +.2 scene triplet; cosine margin .2',
        sampling='Class/vessel-balanced supervised batches; vessel-balanced same-ship pair draws; uniform eligible triplet anchors',
        source_split=pools, heldout_scene_overlap=len(set(tm['product'])&set(vm['product'])),
        relation_metadata='Same MMSI different acquisitions; semantic positive same type/different ship/different scene; scene positive same scene/different type/different ship, negative same type/different scene/different ship',
        target='Original9x64 supports. Source ridge refit on original939; fixed balanced residual ridge8; conditioned target adaptation only128 gains bounded +/-.25 with source replay and absent-class masking',
        no_target_selection=True, neural_seeds=1, resources='CPU1, no backbone updates, no parent changes',
        limitations='Identity-heldout source diagnostic can share scenes; inherited pretraining exposure; candidate factors not established physical/causal factors'))
    np.savez_compressed(OUT/'source_split.npz', train_obs_ids=train_ids, diagnostic_obs_ids=val_ids,
                        source_head_obs_ids=train, target_obs_ids=test, center=center, scale=scale)
    results, classes, diagnoses, traces, target_traces = [], [], [], [], []
    for arm in ARMS:
        model, history = train_model(x[train_ids], tm, arm)
        traces.extend(history)
        parts = encode(model, x)
        diagnoses.extend(source_diagnose(arm, [p[train_ids] for p in parts], [p[val_ids] for p in parts], tm, vm))
        records, byclass, adaptation = target_evaluate(arm, parts[0], parts[2], train, test, obs)
        results.extend(records)
        classes.extend(byclass)
        target_traces.extend(adaptation)
        torch.save(model.state_dict(), OUT/(arm+'.pt'))
        np.savez_compressed(OUT/(arm+'_source_factors.npz'), obs_ids=extra.index.to_numpy(),
            stable=parts[0][extra.index], condition=parts[1][extra.index], correction=parts[2][extra.index])
        for name, data in [('target_results', results), ('target_by_class', classes), ('source_diagnosis', diagnoses),
                           ('source_training', traces), ('target_training', target_traces)]:
            pd.DataFrame(data).to_csv(OUT/(name+'.csv'), index=False)
    save_json(OUT/'COMPLETE.json', dict(source_fits=2, target_adaptations=18, target_metric_rows=len(results),
        seconds=time.perf_counter()-started, query_labels_used_for_training=False))
    report()


if __name__ == '__main__':
    run()
