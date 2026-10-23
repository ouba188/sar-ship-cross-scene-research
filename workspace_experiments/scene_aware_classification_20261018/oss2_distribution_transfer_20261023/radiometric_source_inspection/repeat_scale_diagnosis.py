"""Source-only repeat-observation scale description, without classifier fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    data = pd.read_csv(HERE / 'radiometric_scale_probe/source_inputs.csv', dtype={'mmsi': str})
    data = data[data.role.eq('train')].copy()
    data['time'] = pd.to_datetime(data.source_time, utc=True)
    rows = []
    # One widest-time cross-pass pair per identity, selected without scores or labels.
    for mmsi, group in data.groupby('mmsi', sort=True):
        group = group.sort_values(['time', 'sample_id'])
        candidates = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group.iloc[i], group.iloc[j]
                if a.pass_group != b.pass_group:
                    candidates.append(((b.time-a.time).total_seconds(), a.sample_id, b.sample_id))
        if not candidates:
            continue
        seconds, aid, bid = sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))[0]
        a, b = group.set_index('sample_id').loc[[aid, bid]].iloc[0], group.set_index('sample_id').loc[[aid, bid]].iloc[1]
        assert a.class_id == b.class_id
        dv = np.log(b.vv_amplitude_anchor_255 / a.vv_amplitude_anchor_255)
        dh = np.log(b.vh_amplitude_anchor_255 / a.vh_amplitude_anchor_255)
        rows.append(dict(mmsi=mmsi, class_id=int(a.class_id), class_name=a.class_name,
                         first=aid, second=bid, days=seconds/86400,
                         first_pass=a.pass_group, second_pass=b.pass_group,
                         delta_vv=dv, delta_vh=dh,
                         common=(dv+dh)/2, differential=(dv-dh)/2,
                         delta_log_pixels=np.log(b.vv_height*b.vv_width/(a.vv_height*a.vv_width))))
    pairs = pd.DataFrame(rows)
    pairs.to_csv(HERE / 'repeat_pairs.csv', index=False)
    summary = dict(source_train_only=True, fitted_models=0, pairs=len(pairs),
                   class_counts=pairs.class_name.value_counts().to_dict(),
                   delta_vv_vh_correlation=float(pairs.delta_vv.corr(pairs.delta_vh)),
                   median_abs={key: float(pairs[key].abs().median())
                               for key in ['delta_vv','delta_vh','common','differential','delta_log_pixels']},
                   common_gt_differential=int((pairs.common.abs()>pairs.differential.abs()).sum()))
    (HERE / 'repeat_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    indexed = data.set_index('sample_id')
    picks = [('max_common', pairs.common.abs().idxmax()),
             ('max_differential', pairs.differential.abs().idxmax()),
             ('median_change', (pairs[['delta_vv','delta_vh']].abs().mean(axis=1)
                               -pairs[['delta_vv','delta_vh']].abs().mean(axis=1).median()).abs().idxmin())]
    cases = []
    for reason, ix in picks:
        pair = pairs.loc[ix]
        fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
        for col, sid in enumerate([pair['first'], pair['second']]):
            row = indexed.loc[sid]
            for polidx, pol in enumerate(['vv', 'vh']):
                with Image.open(row[f'chip_path_{pol}']) as im:
                    raw = np.array(im)
                ax = axes[polidx, col]
                ax.imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                ax.set_title(f'{sid[-6:]} {pol.upper()} native {raw.shape[1]}x{raw.shape[0]}\n'
                             f'float amplitude anchor={row[f"{pol}_amplitude_anchor_255"]:.5g}')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel(str(row.pass_group))
        fig.suptitle(f'{reason}: MMSI {pair.mmsi}, {pair.class_name}\n'
                     f'days={pair.days:.1f}; delta log anchors VV={pair.delta_vv:.3f}, VH={pair.delta_vh:.3f}\n'
                     'Linked identity, not verified physical cause; source train only')
        path = HERE / 'panels' / f'repeat_{reason}.png'
        fig.savefig(path, dpi=120); plt.close(fig)
        cases.append(dict(reason=reason, **pair.to_dict(), panel=str(path)))
    pd.DataFrame(cases).to_csv(HERE / 'repeat_cases.csv', index=False)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
