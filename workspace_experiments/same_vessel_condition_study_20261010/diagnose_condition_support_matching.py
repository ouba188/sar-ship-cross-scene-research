"""Locate score improvements and decision changes; no refitting or parameter choices."""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


def main():
    meta = pd.read_csv(BASE/'observations.csv')
    for directory in ('condition_support_matching_20261011', 'condition_support_increment_20261011'):
        out = BASE/directory
        rows, changes = [], []
        for path in sorted(out.glob('*_source_*_predictions.npz')):
            seed, _, draw, _ = path.stem.split('_', 3)
            with np.load(path) as f:
                q = f['query_obs_ids']; y = meta.iloc[q].class_id.to_numpy()
                base = f['baseline']; active = f['valid'].any(1)
                reference = base.copy(); reference[np.arange(len(y)), y] = -np.inf
                rivals = reference.argmax(1)
                old_margin = base[np.arange(len(y)), y]-reference.max(1)
                for arm in ('condition', 'sham', 'gated'):
                    result = f[arm+'_forced1']
                    delta = result-base
                    dm = delta[np.arange(len(y)), y]-delta[np.arange(len(y)), rivals]
                    correct = base.argmax(1) == y
                    for group, mask in [('all', np.ones(len(y), bool)),
                                        ('active_correct', active & correct), ('active_wrong', active & ~correct)]:
                        rows.append(dict(seed=int(seed), draw=int(draw), arm=arm, group=group,
                            count=int(mask.sum()), margin_delta=float(dm[mask].mean()) if mask.any() else np.nan,
                            improved=int((dm[mask] > 1e-12).sum()), worsened=int((dm[mask] < -1e-12).sum()),
                            average_max_score_change=float(np.abs(delta[mask]).max(1).mean()) if mask.any() else np.nan))
                    altered = np.flatnonzero(base.argmax(1) != result.argmax(1))
                    for i in altered:
                        m = meta.iloc[q[i]]
                        changes.append(dict(seed=int(seed), draw=int(draw), arm=arm, obs_id=int(q[i]),
                            mmsi=int(m.mmsi), port=m.port, class_id=int(y[i]), class_name=m.class_name,
                            original=int(base[i].argmax()), changed=int(result[i].argmax()),
                            outcome='rescue' if result[i].argmax()==y[i] else 'harm' if correct[i] else 'wrong_to_wrong',
                            original_margin=float(old_margin[i]), fixed_rival_margin_delta=float(dm[i]),
                            incidence_deg=m.incidence_deg))
        pd.DataFrame(rows).to_csv(out/'score_change_location.csv', index=False)
        pd.DataFrame(changes).to_csv(out/'changed_decisions.csv', index=False)
        print(directory)
        print(pd.DataFrame(rows).groupby(['arm', 'group'])[['count', 'margin_delta', 'average_max_score_change']].mean().to_string())
        c = pd.DataFrame(changes)
        print(c[c.arm.eq('condition')].to_string(index=False))


if __name__ == '__main__':
    main()
