"""Align training-anchor retention with validation/target retention after freeze."""

import json
import numpy as np
import pandas as pd
from frequency_preserve_experiment import OUT
from frequency_band_core import BANDS
from prepare_learned_frequency import load_npz


def main():
    if not (OUT/'retention_summary.csv').exists():
        raise ValueError('Complete frozen reporting before root-cause summary')
    rows, classes, gradients = [], [], []
    for path in sorted((OUT/'tasks').glob('*/*/preserved_training.json')):
        meta = dict(dataset=path.parent.parent.name, task=path.parent.name)
        log = json.loads(path.read_text())
        p = load_npz(path.parent/'training_anchors.npz')
        mask, labels = p['mask'], p['labels'][:, None]
        for bi, band in enumerate(BANDS):
            active = mask[:, bi]
            before = active & (p['standalone'][:, bi] == labels[:, 0])
            after = active & (p['preserved'][:, bi] == labels[:, 0])
            rows.append(dict(**meta, band=band, anchor_events=int(active.sum()),
                standalone_retained=int(before.sum()), preserved_retained=int(after.sum()),
                newly_preserved=int(np.sum(after & ~before)), newly_lost=int(np.sum(before & ~after)),
                distinct_rows=int(len(np.unique(p['rows'][active])))))
            for c in np.unique(p['labels'][active]):
                eligible = active & (p['labels'] == c)
                classes.append(dict(**meta, band=band, class_id=int(c), anchor_events=int(eligible.sum()),
                    standalone_retained=int(np.sum(eligible & before)),
                    preserved_retained=int(np.sum(eligible & after))))
        before = int(np.sum(mask & (p['standalone'] == labels)))
        after = int(np.sum(mask & (p['preserved'] == labels)))
        if before != log['standalone_anchor_correct'] or after != log['preserved_anchor_correct']:
            raise ValueError('Training marker disagrees with anchor archive')
        g = log['first_active_gradient'] or {}
        old = json.loads((path.parent/'standalone_training.json').read_text())
        gradients.append(dict(**meta, anchor_events=log['anchor_events'], unique_anchor_rows=log['anchor_distinct_source_rows'],
            protection_gradient_ratio=g.get('protection_norm', np.nan)/g.get('original_norm', np.nan),
            gradient_cosine=g.get('cosine', np.nan), preservation_first=log['epochs'][0]['preservation'],
            preservation_last=log['epochs'][-1]['preservation'],
            final_ce_delta=log['epochs'][-1]['ce']-old['epochs'][-1]['ce'],
            final_auxiliary_delta=log['epochs'][-1]['auxiliary']-old['epochs'][-1]['auxiliary']))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'training_retention.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'training_retention_classes.csv', index=False)
    pd.DataFrame(gradients).to_csv(OUT/'training_diagnostics.csv', index=False)
    fields = ['anchor_events', 'standalone_retained', 'preserved_retained', 'newly_preserved', 'newly_lost']
    summary = frame.groupby('dataset')[fields].sum().reset_index()
    summary['standalone_retention_percent'] = 100*summary.standalone_retained/summary.anchor_events
    summary['preserved_retention_percent'] = 100*summary.preserved_retained/summary.anchor_events
    summary.to_csv(OUT/'training_retention_summary.csv', index=False)
    readout = pd.read_csv(OUT/'retention_summary.csv')
    phases = [dict(split='source_train', dataset=r.dataset, anchors=r.anchor_events,
                   standalone_retained=r.standalone_retained, preserved_retained=r.preserved_retained)
              for r in summary.itertuples()]
    phases += [dict(split=r.split, dataset=r.dataset, anchors=r.baseline_anchor_count,
                   standalone_retained=r.standalone_anchor_retained, preserved_retained=r.preserved_anchor_retained)
               for r in readout.itertuples()]
    phase = pd.DataFrame(phases)
    for family in ('standalone', 'preserved'):
        phase[family+'_percent'] = 100*phase[family+'_retained']/phase.anchors
    phase.to_csv(OUT/'retention_phases.csv', index=False)
    marker = OUT/'REPORT_COMPLETE.json'
    status = json.loads(marker.read_text())
    status.update(root_diagnosis_pending=False, retention_diagnostics_complete=True, retained_main_unchanged=True)
    marker.write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(phase.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
