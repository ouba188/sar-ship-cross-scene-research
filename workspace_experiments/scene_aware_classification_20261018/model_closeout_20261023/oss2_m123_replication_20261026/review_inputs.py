"""Pretraining native-pixel inspection of class representatives and crop coverage."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import configure, extract


def main():
    configure()
    table = extract.read_table(HERE/'stack_table.csv')
    records = extract.prepare_inventory(table)
    rows = []
    representatives = []
    for c, ids in table.groupby('class_id').groups.items():
        ix = next(i for i in ids if table.iloc[i].role == 'fit_train')
        representatives.append(ix)
    panels = HERE/'input_review'
    panels.mkdir(exist_ok=True)
    for i, rec in enumerate(records):
        prepared = extract.prepare(rec)
        for p, view in enumerate(prepared):
            if view is None:
                continue
            rows.append(dict(obs_id=table.iloc[i].obs_id, pol=('vv','vh')[p],
                role=table.iloc[i].role, class_id=int(table.iloc[i].class_id),
                native_h=view['raw'].shape[0], native_w=view['raw'].shape[1],
                valid_target=view['valid_target'], **view['placement']))
        if i not in representatives:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        for p, v in enumerate(prepared):
            if v is not None:
                axes[p].imshow(v['pixels'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                axes[p].contour(v['masks'][0], levels=[.5], colors=['#19a580'], linewidths=.7)
            axes[p].set_title(('VV','VH')[p])
            axes[p].set(xticks=[],yticks=[])
        fig.suptitle(f"{table.iloc[i].sample_id} class={table.iloc[i].class_id}; native crop/pad")
        fig.tight_layout()
        fig.savefig(panels/f'class{table.iloc[i].class_id}.png', dpi=130)
        plt.close(fig)
    frame = pd.DataFrame(rows)
    frame.to_csv(panels/'placement.csv', index=False)
    receipt = dict(observations=len(table), polarizations=len(frame),
        native_over160=int(((frame.native_h>160)|(frame.native_w>160)).sum()),
        target_pixels_lost=float((frame.native_target_pixels-frame.retained_target_pixels).sum()),
        representatives=table.iloc[representatives][['obs_id','sample_id','class_id']].to_dict('records'),
        labels_used_for_model_selection=False)
    (panels/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
