"""Explain a repeated class-level signal without selecting a new model."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import analyze_results as display


def shorten_panel_names():
    cases = pd.read_csv(HERE / 'review_cases.csv')
    for ix, row in cases.iterrows():
        source = Path(row.panel)
        target = source.with_name(f'crude_{row.outcome}.png')
        if source != target:
            source.replace(target)
        cases.loc[ix, 'panel'] = str(target)
    cases.to_csv(HERE / 'review_cases.csv', index=False)


def main():
    all_rows = pd.read_csv(HERE.parent / 'predictions.csv', dtype={'mmsi': str})
    focus = all_rows[all_rows.arm.eq('pol_preserved') & all_rows.class_id.eq(5)].copy()
    rows = []
    for q in focus.itertuples(index=False):
        b = np.array([getattr(q, f'b{c}') for c in range(8)])
        s = np.array([getattr(q, f's{c}') for c in range(8)])
        old = max((c for c in range(8) if c != 5), key=lambda c: b[c])
        new = max((c for c in range(8) if c != 5), key=lambda c: s[c])
        r = dict(instance_id=q.instance_id, fold=q.fold, seed=q.seed, role=q.role,
                 mmsi=q.mmsi, port=q.port, pass_group=q.pass_group, outcome=q.outcome,
                 net=q.net, old_rival=old, new_rival=new, before=b[5]-b[new], after=s[5]-s[new])
        for pol in ['vv', 'vh']:
            r[pol+'_truth'] = getattr(q, f'{pol}_delta5')
            r[pol+'_rival'] = getattr(q, f'{pol}_delta{new}')
            r[pol+'_margin'] = r[pol+'_truth']-r[pol+'_rival']
        assert abs((r['after']-r['before'])-r['vv_margin']-r['vh_margin']) < 1e-10
        rows.append(r)
    detail = pd.DataFrame(rows)
    detail.to_csv(HERE / 'class_events.csv', index=False)
    summary = []
    for (role, seed, outcome), g in detail.groupby(['role', 'seed', 'outcome']):
        summary.append(dict(role=role, seed=seed, outcome=outcome, exposures=len(g),
                            observations=g.instance_id.nunique(), ships=g.mmsi.nunique(),
                            passes=g.pass_group.nunique(), net=int(g.net.sum()),
                            mean_vv_margin=g.vv_margin.mean(), mean_vh_margin=g.vh_margin.mean(),
                            mean_true_increment=(g.vv_truth+g.vh_truth).mean(),
                            mean_rival_increment=(g.vv_rival+g.vh_rival).mean()))
    pd.DataFrame(summary).to_csv(HERE / 'outcomes.csv', index=False)
    tests = detail[detail.role.eq('test')]
    tests.groupby(['seed', 'port', 'pass_group', 'old_rival', 'new_rival', 'outcome']).size().rename('n').reset_index().to_csv(HERE / 'locations.csv', index=False)
    recurrence = tests.pivot(index='instance_id', columns='seed', values='net')
    recurrence['positive_seeds'] = (recurrence > 0).sum(axis=1)
    recurrence.to_csv(HERE / 'recurrence.csv')
    panels, missing = display.render('polarization_readout/crude_case_diagnosis', focus[focus.role.eq('test')], True)
    shorten_panel_names()
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print('PANELS', panels, 'MISSING', missing, flush=True)


if __name__ == '__main__':
    main()
