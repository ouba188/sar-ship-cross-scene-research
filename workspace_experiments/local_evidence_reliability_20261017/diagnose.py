"""Frozen mechanism replacement, scoped uncertainty, and actual SAR cases."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile
import train_reliability as run
from local_evidence import EvidenceReader

HERE = Path(__file__).resolve().parent


def from_h(h, task, valid):
    ns = task['ns']
    ss = task['ss']+run.ALPHA*(h[:ns]@h[:ns].T)
    qs = task['qs']+run.ALPHA*(h[ns:]@h[:ns].T)
    root = task['root']
    coefficient = root[:, None]*torch.linalg.solve(root[:, None]*ss*root[None]+8*torch.eye(ns, dtype=ss.dtype),
                                                  root[:, None]*(task['desired']-task['prior_s']))
    p = .5*(task['base']+(task['prior_q']+qs@coefficient).softmax(-1))
    active = valid[ns:] & (valid[:ns].sum() >= 2)
    return torch.where(active[:, None], p, task['old']), coefficient, qs


def summarize():
    m = pd.read_csv(HERE/'metrics.csv')
    p = pd.read_csv(HERE/'predictions.csv')
    if (HERE/'trained_no_quality_metrics.csv').exists():
        m = pd.concat([m, pd.read_csv(HERE/'trained_no_quality_metrics.csv')], ignore_index=True)
        p = pd.concat([p, pd.read_csv(HERE/'trained_no_quality_predictions.csv')], ignore_index=True)
    p['net'] = p.prediction.eq(p.truth).astype(int)-p.baseline.eq(p.truth).astype(int)
    p['probability_gain'] = p.true_probability-p.baseline_true_probability
    p['retrieval_gain'] = p.ap-p.baseline_ap
    p.to_csv(HERE/'prediction_diagnosis.csv', index=False)
    m.groupby(['origin', 'arm'])[['oa', 'ba', 'delta_oa', 'delta_ba', 'kernel_risk_gain', 'support_map_r',
                                  'baseline_support_map_r', 'null_weight', 'descriptor_energy']].mean().to_csv(HERE/'summary.csv')
    seeds = m.groupby(['origin', 'arm', 'seed'])[['delta_oa', 'delta_ba', 'kernel_risk_gain']].mean()
    seeds.to_csv(HERE/'seed_summary.csv')
    seeds.groupby(['origin', 'arm']).agg(['mean', 'std']).to_csv(HERE/'seed_uncertainty.csv')
    m.groupby(['origin', 'arm', 'fold'])[['delta_oa', 'delta_ba', 'kernel_risk_gain']].mean().to_csv(HERE/'fold_summary.csv')
    p.groupby(['origin', 'arm', 'truth'])[['net', 'probability_gain', 'retrieval_gain']].mean().to_csv(HERE/'class_summary.csv')
    m.groupby(['origin', 'arm', 'draw'])[['delta_oa', 'delta_ba', 'kernel_risk_gain']].mean().to_csv(HERE/'support_draw_summary.csv')
    intervals = []
    for (origin, arm), group in m.groupby(['origin', 'arm']):
        folds = group.groupby('fold')[['delta_oa', 'delta_ba', 'kernel_risk_gain']].mean()
        rng = np.random.default_rng(20261017)
        boot = folds.to_numpy()[rng.integers(0, len(folds), (5000, len(folds)))].mean(1)
        for col, name in enumerate(folds):
            intervals.append(dict(origin=origin, arm=arm, metric=name, mean=folds[name].mean(),
                fold_bootstrap_low=np.quantile(boot[:, col], .025), fold_bootstrap_high=np.quantile(boot[:, col], .975),
                units=len(folds), interpretation='four acquisition clusters; seeds/draws averaged, exploratory not independent confirmation'))
    pd.DataFrame(intervals).to_csv(HERE/'fold_bootstrap.csv', index=False)
    history = []
    for path in HERE.glob('*/*/training.csv'):
        arm, seed = path.parent.name.rsplit('_', 1)
        h = pd.read_csv(path)
        h['fold'], h['arm'], h['seed'] = path.parent.parent.name, arm, int(seed)
        history.append(h)
    pd.concat(history).to_csv(HERE/'source_learning_curves.csv', index=False)
    return m, p


@torch.no_grad()
def mechanism(p):
    quality = dict(np.load(HERE/'quality.npz'))
    manifest = pd.read_csv(run.old.MANIFEST, dtype={'mmsi': str})
    records, energies, edges, receipts, weight_rows = [], [], [], [], []
    for args in run.old.folds():
        data, features, _, _, tasks, _, _ = run.prepare_fold(args, manifest, quality)
        for seed in range(3):
            folder = HERE/data['fold']/f'dynamic_{seed}'
            model = EvidenceReader('dynamic')
            model.load_state_dict(torch.load(folder/'model.pt', weights_only=True))
            saved = dict(np.load(folder/'readouts.npz'))
            raw = torch.tensor(saved['raw'])
            weights = torch.tensor(saved['weights'])
            constant = torch.tensor(saved['source_mean_correction'])
            fixed_raw, fixed_weights, _ = run.all_features(model, features, gate_override=constant)
            q = data['query']
            complete = features['valid'][q].numpy() & (features['positions'][q, :, 3].numpy() > 0).all(1)
            correction = saved['correction'][q[complete], :12]
            total_variance = correction.var(0).sum()
            common_fraction = 12*correction.mean(1).var()/max(total_variance, 1e-12)
            records.append(dict(fold=data['fold'], seed=seed, complete_queries=int(complete.sum()),
                regional_logit_common_variance_fraction=float(common_fraction),
                residual_relative_logit_rms=float(np.sqrt((correction-correction.mean(1, keepdims=True)).var(0).mean())),
                query_weight_rms_vs_constant=float((weights[q]-fixed_weights[q]).square().mean().sqrt()),
                query_null_mean=float(weights[q, -1].mean()), gate_parameter_norm=float(model.gate.weight.norm()),
                null_gate_parameter_norm=float(model.null_gate.weight.norm())))
            for index in q:
                for slot in range(13):
                    weight_rows.append(dict(fold=data['fold'], seed=seed, global_row=int(data['rows'][index]), slot=slot,
                        learned=float(weights[index, slot]), frozen=float(fixed_weights[index, slot])))
            for task in tasks:
                ids, query, ns = task['ids'], task['q'], task['ns']
                valid = features['valid'][ids]
                dynamic_p, dynamic_h = run.old.predict(raw[ids], valid, task, run.ALPHA)
                reconstructed, coefficient, qs = from_h(dynamic_h, task, valid)
                error = float(abs(dynamic_p-reconstructed).max())
                if error > 1e-10:
                    raise AssertionError('Frozen solve does not replay')
                _, fixed_h = run.old.predict(fixed_raw[ids], valid, task, run.ALPHA)
                # Diagnostic preserves the learned per-row centered feature energy.
                matched_h = fixed_h/fixed_h.norm(dim=1, keepdim=True).clamp_min(1e-12)*dynamic_h.norm(dim=1, keepdim=True)
                matched_p, _, _ = from_h(matched_h, task, valid)
                y = data['labels'][query]
                metric, before = run.old.scores(y, matched_p.numpy()), run.old.scores(y, task['old'].numpy())
                energies.append(dict(fold=data['fold'], seed=seed, origin=task['origin'], draw=task['draw'],
                    arm='constant_gate_matched_energy', oa=metric['oa'], ba=metric['ba'],
                    delta_oa=metric['oa']-before['oa'], delta_ba=metric['ba']-before['ba'],
                    kernel_risk_gain=float(run.old.risk(task['old'], torch.tensor(y))-run.old.risk(matched_p, torch.tensor(y))),
                    max_prob_change_vs_dynamic=float(abs(matched_p-dynamic_p).max()),
                    max_energy_error=float(abs(matched_h.norm(dim=1)-dynamic_h.norm(dim=1)).max())))
                mask = p.arm.eq('dynamic') & p.fold.eq(data['fold']) & p.seed.eq(seed) & p.origin.eq(task['origin']) & p.draw.eq(task['draw'])
                prior = p[mask].set_index('global_row').loc[data['rows'][query]]
                prior_p = prior[[f'p{k}' for k in range(8)]].to_numpy(float)
                saved_error = float(np.max(abs(prior_p-dynamic_p.numpy())))
                if saved_error > 1e-7:
                    raise AssertionError('Saved training results do not replay')
                receipts.append(dict(fold=data['fold'], seed=seed, origin=task['origin'], draw=task['draw'],
                                     solve_error=error, saved_probability_error=saved_error))
                old_coeff = run.public.coefficients(task['ss'].numpy(), task['prior_s'].numpy(), data['labels'][task['s']], 8.)
                for j, index in enumerate(query):
                    if not prior.iloc[j]['net']:
                        continue
                    opponent = int(prior.iloc[j].prediction if prior.iloc[j].baseline == y[j] else prior.iloc[j].baseline)
                    c0 = old_coeff[:, y[j]]-old_coeff[:, opponent]
                    c1 = coefficient[:, y[j]].numpy()-coefficient[:, opponent].numpy()
                    k0, k1 = task['qs'][j].numpy(), qs[j].numpy()
                    dk, da = (k1-k0)*c0, k1*(c1-c0)
                    np.testing.assert_allclose((dk+da).sum(), k1@c1-k0@c0, atol=1e-10)
                    for pos, support in enumerate(task['s']):
                        edges.append(dict(fold=data['fold'], seed=seed, origin=task['origin'], draw=task['draw'],
                            query_global_row=int(data['rows'][index]), support_global_row=int(data['rows'][support]),
                            query_class=int(y[j]), opponent_class=opponent, net=int(prior.iloc[j]['net']),
                            kernel_margin_change=dk[pos], support_refit_margin_change=da[pos], total_margin_change=dk[pos]+da[pos]))
    pd.DataFrame(records).to_csv(HERE/'gate_mechanism.csv', index=False)
    pd.DataFrame(energies).to_csv(HERE/'matched_energy_control.csv', index=False)
    pd.DataFrame(edges).to_csv(HERE/'changed_support_edges.csv', index=False)
    pd.DataFrame(receipts).to_csv(HERE/'diagnostic_replay.csv', index=False)
    pd.DataFrame(weight_rows).to_csv(HERE/'query_weights.csv', index=False)


def plots(m, p):
    figures = HERE/'figures'
    figures.mkdir(exist_ok=True)
    rows = m[m.origin.eq('cross')].groupby(['arm', 'seed'])[['delta_oa', 'delta_ba']].mean()
    names = ['uniform', 'static', 'dynamic', 'dynamic_constant_gate', 'dynamic_no_quality']
    display = ['Uniform', 'Static', 'Dynamic', 'Frozen gate', 'Stats zeroed']
    if 'static_no_quality' in m.arm.unique():
        names.append('static_no_quality')
        display.append('No stats retrained')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ['delta_oa', 'delta_ba']):
        for i, name in enumerate(names):
            values = rows.loc[name, metric]
            ax.scatter(i+np.linspace(-.12, .12, len(values)), values, s=25, color='#087e8b')
            ax.plot([i-.2, i+.2], [values.mean()]*2, color='#b2403b', lw=2)
        ax.axhline(0, color='.5', lw=1)
        ax.set_xticks(range(len(names)), display, rotation=30)
        ax.set_ylabel(metric+' (percentage points)')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Frozen 4-fold cross-acquisition task | dots: training seeds; line: mean')
    fig.tight_layout()
    fig.savefig(figures/'paired_seed_results.png', dpi=160)
    plt.close(fig)
    dynamic = p[p.origin.eq('cross') & p.arm.eq('dynamic')]
    totals = dynamic.groupby('global_row').net.sum().sort_values()
    chosen = [int(totals.index[-1]), int(totals.index[0])]
    geometry = pd.read_csv(run.OLD/'geometry.csv').set_index('global_row')
    meta = pd.read_csv(HERE.parent/'all_factor_probe_20261017/metadata/oss1_factor_metadata.csv').set_index('global_row')
    weights = pd.read_csv(HERE/'query_weights.csv')
    selections = []
    for global_row in chosen:
        event = dynamic[dynamic.global_row.eq(global_row) & dynamic.net.ne(0)].sort_values(['seed', 'draw']).iloc[0]
        w = weights[weights.global_row.eq(global_row) & weights.seed.eq(event.seed) & weights.fold.eq(event.fold)].sort_values('slot')
        image = tifffile.imread(geometry.loc[global_row, 'calibrated_patch_path'])
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1, 1, 1.6]})
        for ax, pol, channel in zip(axes[:2], ['VV', 'VH'], [1, 0]):
            ax.imshow(10*np.log10(image[:, :, channel].clip(1e-8)), cmap='gray', vmin=-35, vmax=5)
            ax.set_title(pol)
            ax.tick_params(labelsize=7)
        axes[2].plot(w.slot, w.learned, 'o-', label='Learned')
        axes[2].plot(w.slot, w.frozen, 'x--', label='Source-mean gate')
        axes[2].set_xticks([0, 3, 6, 9, 12], ['0', '3', '6', '9', 'Null'])
        axes[2].set_xlabel('Existing local region slot')
        axes[2].set_ylabel('Weight')
        axes[2].legend(fontsize=8)
        item = meta.loc[global_row]
        fig.suptitle(f'Outcome-selected row {global_row}, {item.class_name}, SOG {item.ais_sog_knots:.1f}kn\n'
                     f'seed {int(event.seed)}, draw {int(event.draw)}, net change {int(event.net):+d}; weights are not causal labels', fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, .89])
        fig.savefig(figures/f'case_{global_row}.png', dpi=160)
        plt.close(fig)
        selections.append(event.to_dict())
    pd.DataFrame(selections).to_csv(figures/'case_selection.csv', index=False)


def main():
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        m, p = summarize()
        mechanism(p)
        plots(m, p)
    print(pd.read_csv(HERE/'gate_mechanism.csv').to_string(index=False), flush=True)
    print(pd.read_csv(HERE/'matched_energy_control.csv').groupby('origin')[['delta_oa', 'delta_ba', 'kernel_risk_gain', 'max_prob_change_vs_dynamic']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
