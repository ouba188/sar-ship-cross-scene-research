"""Native chip review and exact diagnostic contribution bookkeeping."""
import os
os.environ['MPLBACKEND'] = 'Agg'
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'round05_margin'


def main():
    frame = pd.read_csv(OUT / 'diagnostic_input.csv', dtype={'mmsi': str})
    frame['timestamp'] = pd.to_datetime(frame.time)
    pred = pd.read_csv(OUT / 'predictions.csv', dtype={'mmsi': str})
    fits = json.loads((OUT / 'fit_manifest.json').read_text(encoding='utf8'))
    contributions = []
    replay_error = 0.
    for r in pred[pred.arm.isin(['spatial_geometry', 'plus_core'])].itertuples():
        query = frame[frame.obs_id.eq(r.obs_id) & frame.pol.eq(r.pol)].iloc[0]
        ids = [int(i) for i in r.history_ids.split()]
        history = frame[frame.obs_id.isin(ids) & frame.pol.eq(r.pol)]
        beta = fits[r.fit_id]['coefficients'][r.arm]
        terms = {k: float(b * (query[k] - history[k].mean())) for k, b in beta.items()}
        replay_error = max(replay_error, abs(r.history_margin + sum(terms.values()) - r.predicted_margin))
        contributions.extend(dict(obs_id=r.obs_id, pol=r.pol, classifier=r.classifier,
                                  arm=r.arm, feature=k, contribution=v) for k, v in terms.items())
    assert replay_error < 1e-8
    terms = pd.DataFrame(contributions)
    terms.to_csv(OUT / 'term_contributions.csv', index=False)
    selected = pd.read_csv(OUT / 'case_selection.csv')
    primary = pred[pred.pol.eq('vh') & pred.classifier.eq('interaction') & pred.arm.eq('plus_core')].copy()
    primary['state'] = (primary.latest_history_margin.gt(0).map({True: 'C', False: 'W'}) +
                        primary.actual_margin.gt(0).map({True: 'C', False: 'W'}))
    additions = []
    for state, group in primary.groupby('state'):
        change = (group.actual_margin - group.latest_history_margin).abs()
        row = group.loc[(change - change.median()).abs().idxmin()]
        additions.append(dict(pol='vh', classifier='interaction', selection='state_' + state,
                              obs_id=int(row.obs_id), gain=np.nan))
    selected = pd.concat([selected, pd.DataFrame(additions)], ignore_index=True)
    selected = selected.drop_duplicates(['pol', 'classifier', 'obs_id'])
    selected.to_csv(OUT / 'review_selection.csv', index=False)
    raw = pd.read_csv(ROOT / 'multitemporal_20261021/measurements.csv', dtype={'mmsi': str})
    spectra = pd.read_csv(HERE / 'round04_frequency/measurements.csv')
    original = pd.read_csv(ROOT / 'conditional_evidence_pilot/predictions.csv')
    facts = []
    for selected_case in selected.itertuples():
        oid, pol, clf = selected_case.obs_id, selected_case.pol, selected_case.classifier
        records = pred[pred.obs_id.eq(oid) & pred.pol.eq(pol) & pred.classifier.eq(clf)]
        row = records[records.arm.eq('plus_core')].iloc[0]
        hid = int(row.latest_history_id)
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout='constrained')
        imgs = {}
        for band in ('vv', 'vh'):
            for obs_id in (hid, oid):
                rr = raw[raw.obs_id.eq(obs_id) & raw.pol.eq(band)].iloc[0]
                imgs[(band, obs_id)] = np.asarray(Image.open(rr.image_path).convert('L'))
        limit = max(max(im.shape) for im in imgs.values())
        for i, band in enumerate(('vv', 'vh')):
            for j, obs_id in enumerate((hid, oid)):
                ax = axes[i, j]
                im = imgs[(band, obs_id)]
                h, w = im.shape
                ax.imshow(im, cmap='gray', vmin=0, vmax=255, interpolation='nearest',
                          extent=(-w / 2, w / 2, h / 2, -h / 2))
                ax.set_xlim(-limit / 2, limit / 2)
                ax.set_ylim(limit / 2, -limit / 2)
                ax.set_facecolor('#d4d4d4')
                rr = raw[raw.obs_id.eq(obs_id) & raw.pol.eq(band)].iloc[0]
                pp = original[original.obs_id.eq(obs_id) & original.arm.eq(clf)].iloc[0]
                ff = spectra[spectra.obs_id.eq(obs_id) & spectra.pol.eq(band)]
                freq = 'spectral measurement unavailable' if ff.empty else (
                    f"centroid={ff.iloc[0].core_centroid:.5f}/m; high={ff.iloc[0].core_high:.3f}")
                ax.set_title(f"{'Last history' if j == 0 else 'Query'} {obs_id} {band.upper()}\n"
                             f"class={pp.class_id}; pred={pp.prediction}; margin={pp.margin:+.3f}\n{freq}", fontsize=10)
                ax.set_xlabel('Native pixels, equal scale')
                ax.set_ylabel('Native pixels')
                facts.append(dict(case_query=oid, diagnostic_pol=pol, classifier=clf, obs_id=obs_id,
                    pol=band, role='query' if j else 'latest_history', image_path=rr.image_path,
                    class_id=int(pp.class_id), prediction=int(pp.prediction), margin=float(pp.margin),
                    native_h=h, native_w=w, incidence=rr.incidence, phi_deg=rr.phi_deg,
                    target_saturated=rr.target_saturated, other_boxes=rr.other_boxes))
        errors = records.set_index('arm').abs_error
        fig.suptitle(f"{clf}, measured {pol.upper()}, {selected_case.selection}\n"
                     f"MAE contribution: base {errors.spatial_geometry:.4f} -> core {errors.plus_core:.4f}; "
                     f"no classifier predictions changed", fontsize=12)
        fig.savefig(OUT / f'case_{oid}_{pol}_{clf}.png', dpi=140)
        plt.close(fig)
    pd.DataFrame(facts).to_csv(OUT / 'case_images.csv', index=False)
    fixed = pred[pred.obs_id.isin([28992,383,25586,26309])]
    fixed.to_csv(OUT / 'fixed_cases.csv', index=False)
    grouped = pred.groupby(['pol', 'classifier', 'arm']).agg(
        mae=('abs_error', 'mean'), rmse=('squared_error', lambda x: np.sqrt(x.mean())),
        sign_accuracy=('sign_correct', 'mean'))
    grouped.to_csv(OUT / 'absolute_metrics.csv')
    summary = pd.read_csv(OUT / 'summary.csv')
    for metric, color in [('plus_core', '#008577'), ('plus_outer', '#b55038')]:
        data = summary[summary.baseline.eq('spatial_geometry') & summary.arm.eq(metric)]
        xs = np.arange(len(data)) + (-.16 if metric == 'plus_core' else .16)
        plt.bar(xs, data.reduction_pct, width=.3, color=color, label=metric)
    plt.xticks(np.arange(4), ['VH image', 'VH interaction', 'VV image', 'VV interaction'])
    plt.axhline(0, color='black', linewidth=.8)
    plt.ylabel('Margin-prediction MAE reduction (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'margin_gain.png', dpi=160)
    plt.close()
    print('Prediction reconstruction max error:', replay_error)
    print('Panels:', len(selected))
    print(grouped.to_string())


if __name__ == '__main__':
    main()
