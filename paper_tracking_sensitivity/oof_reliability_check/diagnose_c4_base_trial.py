"""Check affine integration and retain per-image, class, port and angle effects."""

import json
import pickle

import numpy as np
import pandas as pd

from run_c4_base_trial import OUT, SOURCE, GEOMETRY, PARENT, REFERENCE, base_phi, replace_r50


def integration():
    from direct_incidence_conditioning import condition_features
    from residual_expert import fit_correction
    from run_conditioned_vessel_drift import SIDECAR
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'port', 'class_id'])
    side = pd.read_csv(SIDECAR, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    rows = np.load(OUT/'measurable_rows.npy')
    orbit = np.load(OUT/'measurable_orbit.npy', mmap_mode='r')
    average = np.load(OUT/'pooled_c4.npy', mmap_mode='r')
    checks = []
    for port in sorted(frame.loc[rows, 'port'].unique()):
        chosen = rows[frame.loc[rows, 'port'].eq(port).to_numpy()]
        oi = np.searchsorted(rows, chosen)
        with (PARENT/port/'geometry.pkl').open('rb') as f:
            g = pickle.load(f)
        with (REFERENCE/port/'direct_heads.pkl').open('rb') as f:
            head = pickle.load(f)['joint_base']
        with np.load(OUT/'ports'/port/'policies.npz') as f:
            p = {k: f[k] for k in f.files}
        tr = p['query_rows']
        indices = np.searchsorted(tr, chosen)
        bank = np.load(PARENT/port/'target_bank.npy', mmap_mode='r')
        u = condition_features(frame.loc[tr, 'incidence_deg'].to_numpy())
        phi = base_phi(bank, g, u)
        prior = np.c_[bank, u] @ head['coef']+head['intercept']
        per_angle = []
        for k in range(4):
            b = replace_r50(orbit[oi, k].astype(float), bank[indices], g)
            per_angle.append((np.c_[b, u[indices]] @ head['coef']+head['intercept'], base_phi(b, g, u[indices])))
        max_error, flip_count, total = 0., 0, 0
        for draw, ev in enumerate(p['eval_mask']):
            support = np.searchsorted(tr, p['support_rows'][draw])
            labels = frame.loc[p['support_rows'][draw], 'class_id'].to_numpy()
            coef = fit_correction(phi[support], labels, prior[support])
            scores = np.stack([pr+ph @ coef for pr, ph in per_angle], axis=1)
            wanted = p['base_scores'][draw, list(p['base_modes']).index('query_c4'), indices]
            max_error = max(max_error, float(np.max(np.abs(scores.mean(1)-wanted))))
            eligible = ev[indices]
            flip_count += int(((scores[:, 0].argmax(1) != scores[:, 1].argmax(1)) & eligible).sum())
            total += int(eligible.sum())
        if max_error > 1e-5:
            raise ValueError(f'C4 feature/score averaging mismatch {port}: {max_error}')
        np.testing.assert_allclose(average[chosen], np.asarray(orbit[oi]).mean(1), atol=0, rtol=0)
        checks.append(dict(port=port, affine_score_averaging_max_error=max_error,
            original90_flip_events=flip_count, paired_query_events=total))
    pd.DataFrame(checks).to_csv(OUT/'affine_checks.csv', index=False)
    print(pd.DataFrame(checks).to_string(index=False), flush=True)


def angle_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    effect = pd.read_csv(OUT/'instance_effects.csv')
    metadata = pd.read_csv(GEOMETRY/'observability/observations.csv', dtype={'mmsi': str})
    metadata['manifest_row'] = np.arange(len(metadata))
    effect = effect.merge(metadata[['manifest_row', 'mmsi', 'product', 'incidence_deg']], on='manifest_row', validate='many_to_one')
    effect.loc[~effect.mmsi.str.fullmatch(r'\d{9}', na=False), 'mmsi'] = np.nan
    measured = effect[np.isfinite(effect.relative_angle)].copy()
    measured['angle_bin'] = pd.cut(measured.relative_angle, [0,15,30,45,60,75,90.0001], right=False)
    table = measured.groupby(['port', 'class_id', 'stage', 'mode', 'angle_bin'], observed=True).agg(
        images=('manifest_row', 'nunique'), vessels=('mmsi', 'nunique'), products=('product', 'nunique'),
        recall=('correct', 'mean'), rescue=('rescue', 'mean'), harm=('harm', 'mean'), incidence=('incidence_deg', 'mean')).reset_index()
    table.to_csv(OUT/'port_class_angle_coverage.csv', index=False)
    focused = measured[(measured.stage == 'base') & measured['mode'].isin(['reextracted0', 'query_c4', 'support_c4', 'matched_c4'])].copy()
    focused['band'] = np.select([(focused.relative_angle >= 30) & (focused.relative_angle < 45),
        (focused.relative_angle >= 75) & (focused.relative_angle <= 90)], ['30-45', '75-90'], default='other')
    focused = focused[(focused.band != 'other') & focused.class_id.isin([3,4])]
    cells = focused.groupby(['port', 'class_id', 'mode', 'band']).agg(n=('manifest_row', 'nunique'),
        accuracy=('correct', 'mean')).reset_index()
    cells.to_csv(OUT/'focused_angle_cells.csv', index=False)
    pair = cells.pivot(index=['port', 'class_id', 'mode'], columns='band', values=['accuracy', 'n']).dropna()
    pair.columns = ['_'.join(c) for c in pair.columns]
    pair['gap_pp'] = 100*(pair['accuracy_30-45']-pair['accuracy_75-90'])
    pair.reset_index().to_csv(OUT/'focused_angle_gaps.csv', index=False)
    records = []
    for threshold in [1,20]:
        keep = pair[(pair['n_30-45'] >= threshold) & (pair['n_75-90'] >= threshold)].reset_index()
        if not len(keep):
            continue
        avg = keep.groupby(['class_id', 'mode']).agg(ports=('port', 'nunique'),
            low=('accuracy_30-45', 'mean'), high=('accuracy_75-90', 'mean'), gap_pp=('gap_pp', 'mean')).reset_index()
        avg['minimum_images_per_bin'] = threshold
        records.append(avg)
    pd.concat(records).to_csv(OUT/'common_port_angle_gaps.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11,4), layout='constrained')
    colors = {'reextracted0':'#444444', 'query_c4':'#b24c63', 'support_c4':'#a27b22', 'matched_c4':'#15867b'}
    for axis, cl, name in zip(axes, [3,4], ['Product/chemical tanker', 'Container ship']):
        for mode, color in colors.items():
            sample = measured[(measured.class_id == cl) & (measured.stage == 'base') & (measured['mode'] == mode)]
            grouped = sample.groupby('angle_bin', observed=True).agg(accuracy=('correct','mean'), n=('manifest_row','nunique'))
            x = [b.mid for b in grouped.index]
            axis.plot(x, grouped.accuracy*100, marker='o', label=mode, color=color)
            if mode == 'reextracted0':
                for xx, nn in zip(x, grouped.n):
                    axis.text(xx, 97, f'n={nn}', ha='center', va='top', fontsize=8)
        axis.set(title=name, xlabel='Natural relative angle proxy (deg)', ylabel='Base recall (%)', ylim=(0,100), xticks=[7.5,22.5,37.5,52.5,67.5,82.5])
        axis.grid(axis='y', alpha=.2)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, .89), fontsize=8)
    fig.savefig(OUT/'class_angle_c4.png', dpi=160)
    plt.close(fig)
    print(pd.concat(records).to_string(index=False), flush=True)


def paired_transitions():
    predictions = pd.read_csv(OUT/'predictions.csv')
    base = predictions[predictions.stage.eq('base')].copy()
    wide = base.pivot(index=['manifest_row', 'port', 'draw', 'class_id'], columns='mode', values='prediction').reset_index()
    metadata = pd.read_csv(GEOMETRY/'relative_conditions.csv')
    wide['relative_angle'] = metadata.relative_acute_deg.to_numpy()[wide.manifest_row]
    wide['weight'] = 1./wide.groupby('manifest_row').manifest_row.transform('size')
    records = []
    for subset, mask in [('all', np.ones(len(wide), dtype=bool)),
            ('pchem_75_90', wide.class_id.eq(3) & wide.relative_angle.ge(75))]:
        part = wide[mask]
        for mode in ['query_c4', 'support_c4', 'matched_c4']:
            changed = part.rename(columns={'reextracted0': 'from_prediction', mode: 'to_prediction'})
            grouped = changed.groupby(['port', 'class_id', 'from_prediction', 'to_prediction']).agg(
                image_weighted_events=('weight', 'sum'), query_draw_events=('weight', 'size')).reset_index()
            grouped['mode'], grouped['subset'] = mode, subset
            records.append(grouped)
    pd.concat(records, ignore_index=True).to_csv(OUT/'base_prediction_transitions.csv', index=False)


if __name__ == '__main__':
    integration()
    angle_report()
    paired_transitions()
