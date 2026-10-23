"""Native query/reference review of selected preaggregation metric changes."""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run as experiment

HERE = Path(__file__).resolve().parent
OUT = HERE / 'analysis'


def main():
    OUT.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(OUT / '.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from extract_tokens import geo, slots

    edge = experiment.edge
    table, x, mass, bank = edge.support.load()
    pred = np.load(HERE / 'predictions_17.npz')
    weights = np.load(HERE / 'weights.npz')
    features = np.load(HERE / 'features_17.npz')
    tokens = np.load(HERE.parent / 'tokens.npz')
    old_fixed = np.load(HERE.parent / 'edge_difference/predictions_17.npz')
    scales = pd.read_csv(HERE / 'scales.csv')
    divisor = scales.loc[scales.arm.eq('mixed')].sort_values('feature').divisor.to_numpy()
    ev = np.flatnonzero(table.role.eq('evaluation'))
    y = table.class_id.to_numpy(int)
    old = pred['anchor'][ev].argmax(-1) == y[ev]
    focus = 'mixed'
    if np.array_equal(pred['mixed'][ev], pred['anchor'][ev]):
        focus = 'body'
    new = pred[focus][ev].argmax(-1) == y[ev]
    cases = []
    for name, mask in [('rescued', ~old & new), ('harmed', old & ~new),
                       ('CC', old & new), ('WW', ~old & ~new)]:
        ids = ev[mask]
        if not len(ids):
            continue
        i = int(ids[np.argmin(table.iloc[ids].obs_id.to_numpy())])
        c = int(y[i])
        rival_scores = pred[focus][i] if name == 'harmed' else pred['anchor'][i]
        r = int(np.argmax(np.where(np.arange(8) == c, -np.inf, rival_scores)))
        coef = weights['mixed_17_a']
        parts = coef * (features['mixed'][i, c] - features['mixed'][i, r])
        group = int(np.argmax(np.abs(parts)))
        trace = experiment.trace_query(i, 'mixed', weights['mixed_17_W'], x, mass, bank)
        refs = {}
        for cl in (c, r):
            resp = trace[f'{edge.GROUPS[group]}_responsibility'][cl].sum((0, 2))
            refs[cl] = bank[cl, np.argsort(-resp, kind='stable')[:2]].tolist()
        cases.append(dict(outcome=name, focus=focus, n=len(ids), i=i,
                          obs_id=int(table.iloc[i].obs_id), true=c, rival=r,
                          group=group, refs=refs))

    wanted = {int(table.iloc[ix].obs_id) for ca in cases
              for ix in [ca['i'], *ca['refs'][ca['true']], *ca['refs'][ca['rival']]]}
    snaps = {int(s['obs_id']): s for s in json.loads(
        (geo.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))
        if int(s['obs_id']) in wanted}
    margins, evidence, records, contributions = [], [], [], []
    for ca in cases:
        i, c, r = ca['i'], ca['true'], ca['rival']
        for state, w in [('identity', np.eye(32)), ('learned', weights['mixed_17_W'])]:
            trace = experiment.trace_query(i, 'mixed', w, x, mass, bank)
            for g, name in enumerate(edge.GROUPS):
                gap = float(trace['raw_scores'][c, g] - trace['raw_scores'][r, g])
                a = float(weights['mixed_17_a'][g])
                contributions.append(dict(obs_id=ca['obs_id'], state=state, group=name,
                                          raw_margin=gap, coefficient=a, divisor=divisor[g],
                                          margin_contribution=a * gap / divisor[g]))
                for cl in (c, r):
                    resp = trace[name + '_responsibility'][cl]
                    valid = trace['legal_edges'][0, g]
                    pts = trace[name + '_four_points'][:, cl]
                    terms = (pts * resp[None]).sum((1, 2, 3)) / max(valid.sum(), 1)
                    terms = terms / edge.TAU * .5 * np.array([1, 1, -1, -1])
                    raw = float(trace['raw_scores'][cl, g])
                    evidence.append(dict(obs_id=ca['obs_id'], state=state, group=name,
                                         class_id=cl, raw=raw, bb=terms[0], nn=terms[1],
                                         bn=terms[2], nb=terms[3], entropy_mass=raw-terms.sum()))
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        fig.subplots_adjust(top=.83, bottom=.14, wspace=.15, hspace=.3)
        views = [(i, 'QUERY')] + [(v, f'TRUE REF {j+1}') for j, v in enumerate(ca['refs'][c])]
        views += [(v, f'RIVAL REF {j+1}') for j, v in enumerate(ca['refs'][r])]
        for p, pol in enumerate(('vv', 'vh')):
            for j, (ix, label) in enumerate(views):
                pix, fp, geometry, place = geo.native(table, snaps, ix, p)
                ax = axes[p, j]
                ax.imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                if tokens['valid'][ix, p]:
                    masks = slots(geometry, fp)
                    for (lo, hi), color in zip(((0, 8), (8, 16), (16, 20)),
                                               ('#24a878', '#2c84c6', '#d39222')):
                        region = masks[lo:hi].sum(0)
                        if region.min() < .5 < region.max():
                            ax.contour(region, levels=[.5], colors=[color], linewidths=.7)
                row = table.iloc[ix]
                ax.set(title=f'{pol.upper()} {label}: obs {int(row.obs_id)}\n{row.class_name}; {row.port}\nvalid={int(tokens["valid"][ix,p])}',
                       xlim=(-.5, 159.5), ylim=(159.5, -.5), xticks=[0, 80, 159], yticks=[0, 80, 159])
                records.append(dict(query_obs=ca['obs_id'], obs_id=int(row.obs_id), pol=pol,
                                    path=row['image_' + pol], placement=place))
        lines = []
        for arm in ('anchor', 'body', 'additive', 'mixed'):
            score = pred[arm][i]
            margin = float(score[c] - score[r])
            margins.append(dict(obs_id=ca['obs_id'], focus=ca['focus'], outcome=ca['outcome'],
                                arm=arm, prediction=int(score.argmax()), true=c, rival=r, margin=margin))
            lines.append(f'{arm}: pred={int(score.argmax())}; margin={margin:+.4f}')
        fixed_margin = float(old_fixed['mixed'][i, c] - old_fixed['mixed'][i, r])
        fig.suptitle(f"{HERE.name} | {ca['focus']} {ca['outcome']} vs anchor | obs {ca['obs_id']} | seed17 | group n={ca['n']}\nRefs: top two raw mixed responsibilities in {edge.GROUPS[ca['group']]}; not independent evidence counts", fontsize=12)
        fig.text(.04, .018, '\n'.join(lines) + f'\nFixed mixed margin={fixed_margin:+.4f}; native160 crop/pad, NO resize. Layer1 RF=35: regions are not pixel-pure.', fontsize=9)
        fig.savefig(OUT / f"case_{ca['focus']}_{ca['outcome']}_{ca['obs_id']}.png", dpi=130)
        plt.close(fig)
    pd.DataFrame(margins).to_csv(OUT / 'case_margins.csv', index=False)
    pd.DataFrame(evidence).to_csv(OUT / 'relation_evidence.csv', index=False)
    pd.DataFrame(contributions).to_csv(OUT / 'group_contributions.csv', index=False)
    (OUT / 'cases.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
    (OUT / 'image_records.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(pd.DataFrame(margins).to_string(index=False))


if __name__ == '__main__':
    main()
