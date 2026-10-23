"""Read-only source-utility pairing diagnostic; no fit, optimizer, or new model."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
from itertools import permutations
import math
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location('pairing_existing_edge', ROOT / 'edge_difference/run.py')
edge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edge)
ROWS = []


def row(section, group, metric, value, n, variant='real'):
    ROWS.append(dict(section=section, group=group, variant=variant, metric=metric,
                     value=float(value), n=int(n)))


def summary(group, metric, values, variant='real', section='image'):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values):
        for suffix, value in [('mean', values.mean()), ('p10', np.quantile(values, .1)),
                              ('p50', np.median(values)), ('p90', np.quantile(values, .9))]:
            row(section, group, metric + '_' + suffix, value, len(values), variant)


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.corrcoef(a, b)[0, 1] if min(a.std(), b.std()) > 1e-15 else np.nan


def exact_distribution(bc, nc, perm):
    """All m! centered cross-covariance norms and traces using m x m Grams."""
    m = len(bc)
    gb, gn, cross = bc @ bc.T, nc @ nc.T, bc @ nc.T
    norm2, trace = np.zeros(len(perm)), np.zeros(len(perm))
    for i in range(m):
        norm2 += gb[i, i] * gn[perm[:, i], perm[:, i]]
        trace += cross[i, perm[:, i]]
        for j in range(i):
            norm2 += 2 * gb[i, j] * gn[perm[:, i], perm[:, j]]
    return np.maximum(norm2 / m**2, 0), trace / m


def class_kernel(group, variant, features, valid, table):
    """Descriptive cross-vessel/cross-pass class alignment, no classifier or fit."""
    ids = np.flatnonzero(valid)
    f, t = features[ids], table.iloc[ids]
    y = t.class_id.to_numpy()
    kernel = f @ f.T
    vessel, acquisition = t.mmsi.to_numpy(), t.pass_group.to_numpy()
    allowed = (vessel[:, None] != vessel[None]) & (acquisition[:, None] != acquisition[None])
    gaps, query_class = [], []
    for i in range(len(ids)):
        class_means = np.array([kernel[i, allowed[i] & (y == c)].mean()
                               if np.any(allowed[i] & (y == c)) else np.nan for c in range(8)])
        if np.isfinite(class_means).all():
            gaps.append(class_means[y[i]] - np.delete(class_means, y[i]).mean())
            query_class.append(y[i])
    gaps, query_class = np.asarray(gaps), np.asarray(query_class)
    balanced_gap = np.mean([gaps[query_class == c].mean() for c in range(8)])
    rms = np.sqrt(np.mean(kernel[allowed]**2))
    row('class_kernel', group, 'balanced_same_minus_other', balanced_gap, len(gaps), variant)
    row('class_kernel', group, 'eligible_pair_kernel_rms', rms, int(allowed.sum()), variant)
    row('class_kernel', group, 'balanced_gap_over_kernel_rms',
        balanced_gap / rms if rms > 1e-15 else 0., len(gaps), variant)
    for c in range(8):
        row('class_kernel_by_class', group, 'gap_class_' + str(c),
            gaps[query_class == c].mean(), (query_class == c).sum(), variant)
    return kernel, allowed


def main():
    tick = time.perf_counter()
    table, x, mass, bank = edge.support.load()
    utility = np.flatnonzero(table.role.to_numpy() == 'utility_train')
    assert len(utility) == 644
    t = table.iloc[utility].reset_index(drop=True)
    b, near, legal, mapping = edge.edges(x, mass)
    shuffles = {seed: edge.edges(x, mass, seed) for seed in (101, 211, 307)}
    cache = np.load(ROOT / 'edge_difference/localcache.npz')
    np.testing.assert_array_equal(cache['instance_ids'].astype(str), table.instance_id.to_numpy())
    np.testing.assert_array_equal(cache['bank'], bank)
    for seed, (sb, sn, sl, sm) in shuffles.items():
        # Full-table reconstruction is needed for the existing sequential RNG stream.
        np.testing.assert_array_equal(sl, legal)
        np.testing.assert_array_equal(sb, b)
        np.testing.assert_array_equal(sm, cache[f'mixed_shuffle{seed}_neighbor_map'])
        for g in range(4):
            for i in np.r_[utility, bank.ravel()]:
                keep = legal[i, g]
                np.testing.assert_array_equal(np.sort(sm[i, g, keep]), np.sort(mapping[i, g, keep]))
        row('check', 'all', 'cached_mapping_exact', 1, len(table), str(seed))
    for key in ('mmsi', 'pass_group', 'port'):
        row('coverage', 'all', 'unique_' + key, t[key].nunique(), len(t))
    for c, count in t.class_id.value_counts().sort_index().items():
        row('coverage', 'all', 'class_' + str(c), count, len(t))

    perms = {m: np.array(list(permutations(range(m))), dtype=np.int16) for m in range(1, 9)}
    largest_error = 0.
    for g, name in enumerate(edge.GROUPS):
        valid = legal[utility, g].any(-1)
        feats = {key: np.zeros((644, dim)) for key, dim in
                 [('body_mean', 32), ('neighbor_mean', 32), ('additive_mean', 64),
                  ('difference_mean', 32), ('outer_joint', 1024), ('outer_independent', 1024),
                  ('centered_crosscov', 1024), ('centered_exact_permutation_mean', 1024)]}
        for seed in shuffles:
            feats[f'centered_shuffle{seed}'] = np.zeros((644, 1024))
        stats = {}
        paired, independent, total_perm = [], [], 0
        def add(key, value):
            stats.setdefault(key, []).append(value)
        for j, i in enumerate(utility):
            keep = legal[i, g]
            m = int(keep.sum())
            if not m:
                continue
            bb, nn = b[i, g, keep].astype(float), near[i, g, keep].astype(float)
            bm, nm = bb.mean(0), nn.mean(0)
            bc, nc = bb - bm, nn - nm
            cv = bc.T @ nc / m
            joint, product = bb.T @ nn / m, np.outer(bm, nm)
            error = np.max(np.abs(joint - product - cv))
            largest_error = max(largest_error, error)
            feats['body_mean'][j], feats['neighbor_mean'][j] = bm, nm
            feats['additive_mean'][j] = np.r_[bm, nm] / np.sqrt(2)
            feats['difference_mean'][j] = (bm - nm) / np.sqrt(2)
            feats['outer_joint'][j], feats['outer_independent'][j] = joint.ravel(), product.ravel()
            feats['centered_crosscov'][j] = cv.ravel()
            norms, traces = exact_distribution(bc, nc, perms[m])
            total_perm += len(norms)
            true_norm = np.sum(cv**2)
            null_mean = np.sum(bc**2) * np.sum(nc**2) / (m*m*(m-1)) if m > 1 else 0.
            largest_error = max(largest_error, abs(norms.mean() - null_mean), abs(traces.mean()),
                                abs(norms[0] - true_norm))
            # Every neighbor occupies each slot (m-1)! times, so E_pi C_pi = 0.
            mean_nc = np.array([nc[perms[m][:, k]].mean(0) for k in range(m)])
            largest_error = max(largest_error, float(np.max(np.abs(bc.T @ mean_nc / m))))
            energy = np.mean(np.sum((bb - nn)**2, axis=1))
            mean_energy = np.sum((bm - nm)**2)
            residual_energy = np.mean(np.sum((bc - nc)**2, axis=1))
            largest_error = max(largest_error, abs(energy - mean_energy - residual_energy))
            cosine = np.sum(bb*nn, axis=1)
            paired.append(cosine.mean())
            independent.append(bm @ nm)
            add('legal_edges', m)
            add('paired_cosine', cosine.mean())
            add('paired_cosine_gt_0p8_fraction', np.mean(cosine > .8))
            add('independent_pair_cosine', bm @ nm)
            add('paired_minus_independent_cosine', np.trace(cv))
            add('body_mean_energy_fraction', np.sum(bm**2) / np.mean(np.sum(bb**2, axis=1)))
            add('neighbor_mean_energy_fraction', np.sum(nm**2) / np.mean(np.sum(nn**2, axis=1)))
            add('difference_image_mean_energy_fraction', mean_energy / energy)
            add('crosscov_norm2', true_norm)
            add('exact_permutation_norm2_mean', null_mean)
            add('crosscov_norm2_over_exact_null_mean', true_norm/null_mean if null_mean > 1e-15 else np.nan)
            add('crosscov_norm2_exact_midrank', np.mean(norms < true_norm-1e-12) + .5*np.mean(np.abs(norms-true_norm) <= 1e-12))
            add('crosscov_norm2_above_exact_q95', true_norm > np.quantile(norms, .95)+1e-12)
            add('trace_exact_null_sd', traces.std())
            add('trace_over_exact_null_sd', np.trace(cv)/traces.std() if traces.std() > 1e-15 else np.nan)
            add('joint_outer_centered_component_energy_ratio', true_norm/np.sum(joint**2))
            # This is a component ratio, not an orthogonal variance fraction.
            for seed, (_, sn, _, sm) in shuffles.items():
                ns = sn[i, g, keep].astype(float)
                cs = bc.T @ (ns - ns.mean(0)) / m
                feats[f'centered_shuffle{seed}'][j] = cs.ravel()
                denom = np.linalg.norm(cv) * np.linalg.norm(cs)
                add(f'shuffle{seed}_crosscov_cosine_real', np.sum(cv*cs)/denom if denom > 1e-15 else np.nan)
                add(f'shuffle{seed}_crosscov_relative_change', np.linalg.norm(cs-cv)/np.linalg.norm(cv) if true_norm > 1e-15 else np.nan)
                add(f'shuffle{seed}_fixed_edge_fraction', np.mean(sm[i, g, keep] == mapping[i, g, keep]))
                add(f'shuffle{seed}_whole_map_identity', np.array_equal(sm[i, g, keep], mapping[i, g, keep]))
                largest_error = max(largest_error, float(np.max(np.abs(ns.mean(0)-nm))))
        for metric, values in stats.items():
            summary(name, metric, values)
        row('exact_permutation', name, 'enumerated_permutations', total_perm, valid.sum())
        row('image', name, 'paired_cosine_correlation_with_independent_cosine', corr(paired, independent), valid.sum())
        row('coverage', name, 'missing_images', (~valid).sum(), 644)
        body_slots, near_slots = edge.BODY[g % 2], edge.NEIGHBOR[g % 2]
        p = g // 2
        body_ok, near_ok = mass[utility, p][:, body_slots] > 0, mass[utility, p][:, near_slots] > 0
        row('coverage', name, 'valid_body_excluded_by_paired_neighbor', (body_ok & ~near_ok).sum(), 644)
        row('coverage', name, 'valid_neighbor_excluded_by_paired_body', (~body_ok & near_ok).sum(), 644)
        kernels = {}
        for variant, feature in feats.items():
            kernels[variant], allowed = class_kernel(name, variant, feature, valid, t)
        row('kernel', name, 'body_neighbor_mean_kernel_correlation',
            corr(kernels['body_mean'][allowed], kernels['neighbor_mean'][allowed]), allowed.sum())
        for seed in shuffles:
            row('kernel', name, 'centered_kernel_correlation_real',
                corr(kernels['centered_crosscov'][allowed], kernels[f'centered_shuffle{seed}'][allowed]),
                allowed.sum(), f'shuffle{seed}')
        # Existing nonlinear score outputs: utility rows only; no recomputation or fitting.
        for arm in ('body', 'additive', 'mixed', 'mixed_shuffle101', 'mixed_shuffle211', 'mixed_shuffle307'):
            z = cache[f'{arm}_raw'][utility, :, g]
            mask = cache[f'{arm}_valid'][utility, :, g]
            common = mask.all(1)
            centered = z - z.mean(1, keepdims=True)
            if arm == 'mixed':
                real_score = centered.copy()
            if 'shuffle' in arm:
                row('existing_utility_readout', name, 'class_centered_correlation_real',
                    corr(real_score[common], centered[common]), common.sum(), arm)
                row('existing_utility_readout', name, 'class_centered_relative_l2_change',
                    np.linalg.norm(centered[common]-real_score[common])/np.linalg.norm(real_score[common]), common.sum(), arm)
            y = t.class_id.to_numpy()
            gap = z[np.arange(644), y] - (z.sum(1)-z[np.arange(644), y])/7
            value = np.mean([gap[common & (y == c)].mean() for c in range(8)])
            row('existing_utility_readout', name, 'balanced_true_minus_other_raw', value, common.sum(), arm)
        print('completed', name, 'images', int(valid.sum()), 'exact permutations', total_perm, flush=True)
    assert largest_error < 1e-10, largest_error
    row('check', 'all', 'maximum_algebra_or_permutation_error', largest_error, 644)
    for stage in ('edge_difference', 'learned_edge_metric', 'reference_relative_metric'):
        existing = pd.read_csv(ROOT / stage / 'summary.csv')
        for _, result in existing.iterrows():
            for metric in ('OA', 'BA'):
                row('existing_evaluation_NOT_design', stage, metric, result[metric], 485, result['arm'])
    row('run', 'all', 'seconds', time.perf_counter()-tick, 644)
    df = pd.DataFrame(ROWS)
    df.to_csv(HERE / 'diagnostic.csv', index=False, float_format='%.12g')
    write_report(df)
    print('DONE', len(df), 'rows', 'max error', largest_error, 'seconds', time.perf_counter()-tick, flush=True)


def write_report(df):
    def get(group, metric, variant='real', section='image'):
        return float(df[(df.group == group) & (df.metric == metric) & (df.variant == variant) & (df.section == section)].value.iloc[0])
    def table(metrics, section='image', variant='real'):
        lines = ['| group | ' + ' | '.join(metrics) + ' |', '|---|' + '---:|'*len(metrics)]
        for group in edge.GROUPS:
            lines.append('| ' + group + ' | ' + ' | '.join(f'{get(group, metric, variant, section):.6g}' for metric in metrics) + ' |')
        return '\n'.join(lines)
    parts = [
        '# 配对可辨识性：有界源端诊断\n',
        '## 当前结论\n',
        '**utility644 中存在可测、带类别相关性的真实配对二阶信息；现有 shuffle 确实大幅破坏这部分信息，'
        '而原 mixed 读出仍高度相似。该诊断不等于完整模型有效：主代理最新 pairing_excess 已完成但没有超过 anchor。**\n\n'
        '四组顺序统一为 VV_side / VV_end / VH_side / VH_end。排除同船、同pass图对后，真实中心化核的宏平均同类减异类差 / kernel RMS '
        '为0.1788/0.1232/0.1427/0.0663；三个shuffle的对应值范围为 '
        '0.0020~0.0368 / -0.0272~-0.0065 / -0.0159~0.0133 / -0.0267~-0.0007。'
        '真实中心化核与shuffle核的图对相关接近0（-0.0133~0.0100），'
        '但既有mixed类别中心化读出相关仍为0.9773~0.9963。这支持区分配对信息与既有读出保留的共同成分，不能据此认定其失效根因。\n\n'
        'body-neighbor 原始配对余弦均值仅0.108/0.047/0.028/-0.088，不是普遍近似重复token。'
        '每图配对余弦与边际均值点积的跨图相关却为0.976/0.987/0.965/0.983；'
        '差分向量能量中图内均值差占71.0%/75.6%/72.9%/78.0%。'
        '另一方面，body均值核与neighbor均值核的图对相关只有0.422/0.204/0.354/0.213，不能把上述两种相关混为一谈。\n\n'
        '精确置换下，真实crosscov范数平方相对null均值的图级中位比为1.363/1.155/1.345/1.138，'
        '超过各自精确置换q95的图比例为58.5%/20.8%/61.9%/23.7%。'
        '侧边二阶结构较明确；端边较弱且逐类有负值。单图强依赖不能替代跨图类别方向与分类收益。\n',
        '仅以 utility_train644 量化下一步设计。没有训练、模型选择、新分类器、Git 或目录外写入。'
        '全表仅用于重放既有顺序随机流；固定 bank64 仅检查置换集合。existing_evaluation_NOT_design 单独转录既有结果，不参与计算或选择。'
        '这是 token/算子诊断，不是一次新分类实验，未新增原图物理归因。\n',
        '## 实际 shuffle\n',
        '`edge_difference/run.py:edges` 每次用一个 default_rng(seed)，按 group、table row 顺序为每图合法边抽取独立的邻居置换。'
        '同一完整输入顺序和种子重放一致；不是随 query-reference 对重抽。learned Reader 在构造时缓存整个表，reference_relative 继承同一实现。'
        '每张 reference 的映射被所有 query 共享，每张 query 在所有类别/参考中固定；不同图没有强制共享位置置换，但也不是按 instance_id 派生的顺序无关 RNG。'
        '若先切子集或重排表再调用 edges，同一图的随机映射一般会改变。这不影响当前完整表用法。\n',
        'shuffle 保留原合法边交集中的 body 集合、neighbor 多重集、正值 mask 和等权边质量；它不恢复因为另一端无效而原已排除的单端区域，'
        '也不保留真实连接关系。side/end 分开置换，不跨组。随机置换允许固定点和整图恒等，不能称每条边都被破坏。'
        '下面列出可达的排除计数；不是 shuffle 新增的缺失。\n',
        table(['valid_body_excluded_by_paired_neighbor', 'valid_neighbor_excluded_by_paired_body', 'missing_images'], 'coverage'),
        '\n## 精确数学与统计口径\n',
        '对一图一组的 m 条合法边，b_i,n_i 是原 PCA32 单位 token；原 mass 仅用于合法性，所有统计等权。'
        'mu_b=mean_i b_i，mu_n=mean_i n_i，B_c[i]=b_i-mu_b，N_c[i]=n_i-mu_n。\n\n'
        '`C = B_c^T N_c / m = mean_i b_i n_i^T - mu_b mu_n^T`。'
        '精确枚举全部 m! 邻居双射（含恒等，不是只用3次随机采样）；令 C_pi=B_c^T P_pi N_c/m，则 '
        '`E_pi C_pi=0`，`E_pi ||C_pi||_F^2=||B_c||_F^2 ||N_c||_F^2/[m^2(m-1)]`（m=1 为0）。'
        '范数分布由小 Gram 矩阵精确枚举；mean、q95、midrank 不是独立图/边显著性检验。'
        'midrank=(小于真实值的个数+0.5*相等个数)/m!，浮点平局容差1e-12。\n\n'
        '`||b-n||` 的平均平方恰分解为 `||mu_b-mu_n||^2 + mean_i ||B_c[i]-N_c[i]||^2`。'
        '因此 difference_image_mean_energy_fraction 是真正的能量分解；joint_outer_centered_component_energy_ratio 的两项不正交，不能叫解释方差比例。'
        'pooled additive `[mu_b,mu_n]/sqrt(2)` 与 pooled difference `(mu_b-mu_n)/sqrt(2)` 在任何邻居置换下恒定。'
        '原 Reader 却先逐边 logsumexp 再平均，所以不能由均值不变推出原 mixed 分数不变。\n',
        table(['paired_cosine_mean', 'paired_cosine_gt_0p8_fraction_mean', 'independent_pair_cosine_mean',
               'paired_cosine_correlation_with_independent_cosine', 'difference_image_mean_energy_fraction_mean']),
        '\n',
        table(['crosscov_norm2_mean', 'exact_permutation_norm2_mean_mean', 'crosscov_norm2_over_exact_null_mean_p50',
               'crosscov_norm2_exact_midrank_p50', 'crosscov_norm2_above_exact_q95_mean']),
        '\n## 是否存在类别相关的配对方向\n',
        '仅在644内部，构造描述性图间核 `K(q,s)=<F_q,F_s>`；排除同 MMSI 和同 pass_group 的图对，缺组排除。'
        '每张图分别求各类参考核均值，gap=同类均值-7个异类均值的平均，再按查询类宏平均。'
        '除以全部可用有向图对 kernel RMS 只用于尺度比较，不是拟合。未做原型分类、调参或新的模型/预测。'
        '它是开发集描述，不是独立泛化估计；正的 gap 不是涨点承诺。\n',
    ]
    variants = ('additive_mean', 'difference_mean', 'outer_independent', 'centered_crosscov',
                'centered_shuffle101', 'centered_shuffle211', 'centered_shuffle307')
    parts.append('| group | ' + ' | '.join(variants) + ' |\n|---|' + '---:|'*len(variants))
    for group in edge.GROUPS:
        parts.append('| ' + group + ' | ' + ' | '.join(f'{get(group, "balanced_gap_over_kernel_rms", v, "class_kernel"):.6g}' for v in variants) + ' |')
    parts.extend([
        '\ncentered_exact_permutation_mean 是每张图先对全部置换求平均，再计算图间核。对不同图独立均匀置换，双线性核的精确期望也为0；'
        '这不等于单次随机置换 C_pi 为0，也不等于对所有置换训练后模型的平均性能。outer_independent=mu_b mu_n^T 保留边际均值乘积。\n',
        table(['body_neighbor_mean_kernel_correlation'], 'kernel'),
        '\n## 既有读出的配对敏感性（utility，未拟合）\n',
    ])
    for arm in ('mixed_shuffle101', 'mixed_shuffle211', 'mixed_shuffle307'):
        parts.extend(['\n### ' + arm + '\n', table(['class_centered_correlation_real', 'class_centered_relative_l2_change'],
                     'existing_utility_readout', arm)])
    parts.extend([
        '\n## 纯配对交互与主代理新算子的数学评估\n',
        '中心化 outer-product 核是 `K_C(q,s)=<C_q,C_s>_F = mean_ij[(b_qi-mu_bq)^T(b_sj-mu_bs)]'
        '`[(n_qi-mu_nq)^T(n_sj-mu_ns)]`。它隔离线性特征下的 joint 相对 independent-product 的交叉二阶矩，'
        '不是检验所有形式统计独立性的完备核；C=0 不能推出真实图像无依赖，C非零也不推出类别有效或物理因果。\n\n'
        '主代理提议：固定一个 query/reference 和组，合法边数为m/r，'
        '`B_ij=b_qi^T W b_sj/(2*tau)`、`N_kj=n_qk^T W n_sj/(2*tau)`，'
        '`L_ik=log[(1/r) sum_j exp(B_ij+N_kj)]`，`S=tr(L)/m-sum_ik L_ik/m^2`。'
        '因每个(i,k)恰在(m-1)!个双射中出现，`E_pi mean_i L_i,pi(i)=mean_ik L_ik` 严格成立。'
        'S 消去所有 `u_i+v_k` 项，m=1、query neighborconstant、reference r=1 时均为0；'
        'reference neighborconstant 也使 L 可分离从而为0。m=0/r=0 必须按已有合法性规则跳过，不能对空集合取logmeanexp。\n\n'
        '高温：令 B=t*B0，N=t*N0，t->0，则 `L_ik=t*(mean B0_i+mean N0_k)'
        '+t^2/2*Var_j(B0_ij+N0_kj)+O(t^3)`。对角减全均值后行列项消失，'
        '`S=t^2*mean_ij[(H_m B0 H_r)_ij*(H_m N0 H_r)_ij]+O(t^3)`。'
        'H_d=I-11^T/d；若直接用已经含1/(2*tau)的 B,N，首项就是两者双中心化乘积的均值，无额外1/2。'
        '等价首项为 `<C_q, W C_s W^T>_F/(4*tau^2)`。若“centered(B)”只减参考轴均值，'
        '必须再减 `mean_j mean_i(BH_r)_ij * mean_i(NH_r)_ij`；否则仍含查询边际项。\n\n'
        '该S严格隔离的是给定真实参考边对时的query配对偏离，不是自动给出同时对query和reference独立置换后的非线性joint-product差。'
        'query端全置换平均S必为0；reference端的log池化之后再平均置换一般不能与log交换，不能直接声称也为0。'
        '均匀参考图平均保留前述query-null及退化性质。相对I的S_W-S_I也保留这些性质。'
        'side与end的r=8/4不使高温近似自动成立；本协议tau=1/sqrt32且W可改变logit幅度，二阶式应作为明确简单对照而不是有限温度的精确替代。'
        '当前脚本不实现或训练该新算子。\n',
        '## 实现边界与检查\n',
        f'映射重放、合法集合、多重集及精确代数最大误差={get("all", "maximum_algebra_or_permutation_error", section="check"):.3g}。'
        '穷举检查用于识别中心化、归一化和索引错误；当前通过。未发现当前完整表路径的shuffle实现错误。'
        '可达解释风险包括：随机置换留固定点；m=1不可辨识；当前边合法交集不等于全部单端有效区域；'
        'pooled difference不敏感不能误写成log池化也不敏感；仅参考轴中心化不等于新算子的高温极限。\n',
        '现有三个阶段evaluation均值已原样记录于CSV的existing_evaluation_NOT_design，仅作历史背景；无evaluation拟合、选组、选阈值。'
        '后续是否接入和拟合由主代理决定；本报告不作因果或novelty结论。\n',
        '复现：`D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe -X utf8 -B run.py`。'
        '仅生成diagnostic.csv与本报告，不保存大型缓存。\n'
    ])
    current = ROOT / 'pairing_excess/summary.csv'
    if current.exists():
        current_results = pd.read_csv(current)
        parts.extend(['## 主代理最新已完成结果（只读转录）\n',
                      '来源：pairing_excess/summary.csv，evaluation485均值，OA/BA为百分数；未用于本诊断选择或拟合。',
                      '| arm | OA (%) | BA (%) |\n|---|---:|---:|'])
        for _, result in current_results.iterrows():
            parts.append(f'| {result.arm} | {100*result.OA:.3f} | {100*result.BA:.3f} |')
        parts.append('\n真实配对信号存在，但excess、covariance及joint本次均未超过anchor；不能以两shuffle回退anchor来宣称真实算子有效。'
                     '本子任务到此结束，新算子拟合与原图复核属于主代理。')
    (HERE / 'FINDINGS_ZH.md').write_text('\n'.join(parts), encoding='utf-8')


if __name__ == '__main__':
    if '--report-only' in sys.argv:
        write_report(pd.read_csv(HERE / 'diagnostic.csv'))
        print('Report regenerated from completed diagnostic.csv; no diagnostic rerun.', flush=True)
    else:
        main()
