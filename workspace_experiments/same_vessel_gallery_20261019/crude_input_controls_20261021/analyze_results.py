"""Paired uncertainty, crude association changes, and native input case panels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import input_controls as experiment

ROOT = Path(__file__).resolve().parent


def paired_uncertainty(p):
    output, classes = [], []
    comparisons = [('full_global', 'body_global'), ('full_target', 'body_target'),
                   ('full_global', 'body_target')]
    for candidate, reference in comparisons:
        a = p[p.arm.eq(candidate)].set_index('obs_id')
        b = p[p.arm.eq(reference)].set_index('obs_id').loc[a.index]
        delta = a.correct.to_numpy(int) - b.correct.to_numpy(int)
        g = a[['mmsi', 'class_id']].copy()
        g['delta'] = delta
        g['n'] = 1
        ships = g.groupby(['mmsi', 'class_id'])[['delta', 'n']].sum().reset_index()
        assert ships.mmsi.nunique() == len(ships)
        # Resample ships within true classes, preserving each ship's observations.
        rng = np.random.default_rng(20261021)
        draws_delta, draws_n = [], []
        for c in range(8):
            h = ships[ships.class_id.eq(c)]
            ii = rng.integers(0, len(h), size=(1500, len(h)))
            d = h.delta.to_numpy()[ii].sum(1)
            n = h.n.to_numpy()[ii].sum(1)
            draws_delta.append(d); draws_n.append(n)
            lo, hi = np.quantile(d/n, [.025, .975])
            rows = g[g.class_id.eq(c)]
            classes.append(dict(candidate=candidate, reference=reference, class_id=c,
                n=len(rows), ships=len(h), delta_recall=float(rows.delta.mean()), ci_low=lo, ci_high=hi))
        ds, ns = np.array(draws_delta).T, np.array(draws_n).T
        boot = dict(oa=ds.sum(1)/ns.sum(1), ba=(ds/ns).mean(1))
        for metric in ('oa', 'ba'):
            value = delta.mean() if metric == 'oa' else g.groupby('class_id').delta.mean().mean()
            low, high = np.quantile(boot[metric], [.025, .975])
            output.append(dict(candidate=candidate, reference=reference, metric=metric,
                difference=value, ci_low=low, ci_high=high, rescue=int((delta > 0).sum()), harm=int((delta < 0).sum()),
                replicates=1500, unit='MMSI stratified within class; fixed fitted heads'))
    pd.DataFrame(output).to_csv(ROOT/'paired_uncertainty.csv', index=False)
    pd.DataFrame(classes).to_csv(ROOT/'class_differences.csv', index=False)


def correlation_changes():
    p = pd.read_csv(ROOT/'crude_pairs.csv')
    records = []
    for full, body in (('full_global', 'body_global'), ('full_target', 'body_target')):
        a = p[p.arm.eq(full)].set_index('pair_id')
        b = p[p.arm.eq(body)].set_index('pair_id').loc[a.index]
        for pol in ('vv', 'vh'):
            x = a[pol+'_contrast'].to_numpy()
            ya, yb = a.delta_margin.to_numpy(), b.delta_margin.to_numpy()
            rng = np.random.default_rng(20261021)
            changes = []
            for _ in range(1500):
                ii = rng.integers(0, len(a), len(a))
                changes.append(spearmanr(x[ii], yb[ii]).statistic - spearmanr(x[ii], ya[ii]).statistic)
            low, high = np.quantile(changes, [.025, .975])
            records.append(dict(full=full, body=body, pol=pol, n=len(a),
                rho_full=spearmanr(x, ya).statistic, rho_body=spearmanr(x, yb).statistic,
                change_low=low, change_high=high))
    pd.DataFrame(records).to_csv(ROOT/'association_changes.csv', index=False)


def case_panels(p):
    frame, views, disabled = experiment.load_data()
    position = dict(zip(frame.obs_id, frame.index))
    pair_table = pd.read_csv(experiment.PREVIOUS/'crude_pair_details.csv')
    a = p[p.arm.eq('full_global')].set_index('obs_id')
    b = p[p.arm.eq('body_global')].set_index('obs_id').loc[a.index]
    selected = [('historical_32042_32043', 32042, 32043), ('historical_30233_30237', 30233, 30237)]
    used = {32042, 32043, 30233, 30237}
    for label, condition in [('context_rescue', a.correct & ~b.correct), ('context_harm', ~a.correct & b.correct),
                             ('both_inputs_correct', a.correct & b.correct), ('both_inputs_wrong', ~a.correct & ~b.correct)]:
        candidates = a.loc[condition & a.class_id.eq(5)].copy()
        candidates['change'] = candidates.margin - b.loc[candidates.index].margin
        candidates = candidates.sort_values('change')
        available = []
        for oid in candidates.index:
            match = pair_table[pair_table.obs_id_a.eq(oid) | pair_table.obs_id_b.eq(oid)]
            if len(match) and oid not in used:
                available.append(match.iloc[0])
        if available:
            chosen = available[len(available)//2]
            oa, ob = int(chosen.obs_id_a), int(chosen.obs_id_b)
            selected.append((label, oa, ob)); used.update((oa, ob))
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 12)
    records = []
    for label, oa, ob in selected:
        canvas = Image.new('RGB', (1012, 485), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), label+' | Crude tanker (class 5) | Native pixels, no resize', fill='black', font=font)
        for j, text in enumerate(('VV full', 'VH full', 'VV body only', 'VH body only', 'VV outside only', 'VH outside only')):
            draw.text((10+166*j, 30), text, fill='black', font=small)
        for r, oid in enumerate((oa, ob)):
            i = position[oid]; row = frame.iloc[i]; y = 54+210*r
            for pol_index, pol in enumerate(('vv', 'vh')):
                pix, masks, _, _, _ = experiment.prepare_view(row, views[i]['images'][pol], pol)
                grey = np.rint(pix[0]*255).astype(np.uint8)
                for k, mask in enumerate((np.ones_like(grey), masks[0], 1-masks[0])):
                    image = Image.fromarray(np.rint(grey*mask).astype(np.uint8)).convert('RGB')
                    canvas.paste(image, (10+166*(2*k+pol_index), y))
            av, bv = a.loc[oid], b.loc[oid]
            text = f'obs {oid} fold {int(row.fold)} | full: {int(av.prediction)} margin {av.margin:+.3f} | body: {int(bv.prediction)} margin {bv.margin:+.3f}'
            draw.text((10, y+163), text, fill='black', font=font)
            draw.text((10, y+183), f'{row.port} | {row.time} | source-only heads, no target support', fill='black', font=small)
            records.append(dict(case=label, obs_id=oid, partner=ob if oid == oa else oa,
                full_prediction=av.prediction, body_prediction=bv.prediction, full_margin=av.margin, body_margin=bv.margin,
                full_correct=av.correct, body_correct=bv.correct, image_vv=row.image_vv, image_vh=row.image_vh,
                vh_disabled=bool(disabled[i])))
        canvas.save(ROOT/f'case_{label}.png')
    pd.DataFrame(records).to_csv(ROOT/'case_index.csv', index=False)


def plots():
    data = pd.read_csv(ROOT/'summary.csv').set_index('arm')
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(9, 4.4), layout='constrained')
    x = np.arange(len(experiment.ARMS))
    for offset, metric in enumerate(('oa', 'ba', 'macro_f1')):
        ax.bar(x+(offset-1)*.23, data.loc[list(experiment.ARMS), metric]*100, .23, label=metric.upper())
    ax.set_xticks(x, ['Full / target', 'Body / target', 'Full / global', 'Body / global', 'Outside / global'])
    ax.set_ylabel('Percent'); ax.set_ylim(0, 50); ax.legend(ncol=3)
    ax.set_title('Matched source-trained heads: 4,744 held-out observations, eight classes')
    fig.savefig(ROOT/'matched_metrics.png', dpi=160); plt.close(fig)
    pairs = pd.read_csv(ROOT/'crude_pairs.csv')
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True, layout='constrained')
    for col, arm in enumerate(('full_global', 'body_global')):
        g = pairs[pairs.arm.eq(arm)]
        for row, pol in enumerate(('vv', 'vh')):
            ax = axes[row, col]
            ax.scatter(g[pol+'_contrast'], g.delta_margin, s=12, alpha=.65)
            ax.axhline(0, color='grey', linewidth=.7); ax.axvline(0, color='grey', linewidth=.7)
            rho = spearmanr(g[pol+'_contrast'], g.delta_margin).statistic
            ax.set_title(f'{arm}, {pol.upper()}, rho={rho:.3f}')
            ax.set_xlabel('PNG target-minus-rest contrast change')
            ax.set_ylabel('True-vs-wrong margin change')
    fig.suptitle('Same 100 crude tanker pairs; separately trained diagnostic heads')
    fig.savefig(ROOT/'crude_association.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    predictions = pd.read_csv(ROOT/'predictions.csv', dtype={'mmsi': str})
    paired_uncertainty(predictions)
    correlation_changes()
    case_panels(predictions)
    plots()
    print('Paired estimates and six native case panels complete.')
