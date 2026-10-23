"""Summarize saved source gradient diagnostics; no gradient recomputation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    g = pd.read_csv(ROOT / 'gradient_comparison.csv')
    fields = ['max_share', 'top5_share', 'top1pct_share', 'effective_contributing_samples',
              'sum_individual_gradient_sq', 'full_batch_gradient_sq', 'jointCE']
    rows = []
    for (state, arm), sub in g.groupby(['state', 'arm']):
        for field in fields:
            original, robust = sub[field + '_original'], sub[field + '_robust']
            rows.append(dict(state=state, arm=arm, metric=field, heads=len(sub),
                original_median=original.median(), robust_median=robust.median(),
                original_mean=original.mean(), robust_mean=robust.mean(),
                decreased=int((robust < original).sum()), increased=int((robust > original).sum())))
    result = pd.DataFrame(rows)
    result.to_csv(HERE / 'mechanism_aggregate.csv', index=False)
    classes = pd.read_csv(ROOT / 'gradient_class_comparison.csv')
    classes.groupby(['state', 'arm', 'class_id'])[
        ['gradient_sq_share_original', 'gradient_sq_share_robust']
    ].mean().reset_index().to_csv(HERE / 'mechanism_class_shares.csv', index=False)
    cols = ['sample_id', 'fold', 'seed', 'arm', 'state', 'selected_epoch_original',
            'selected_epoch_robust', 'gradient_sq_original', 'gradient_sq_robust',
            'gradient_sq_share_original', 'gradient_sq_share_robust',
            'signed_aggregate_projection_original', 'signed_aggregate_projection_robust']
    pairs = pd.read_csv(ROOT / 'gradient_paired_original.csv', usecols=cols)
    focus = pairs[pairs.sample_id.eq('pub2_opensarship2_005904')]
    focus.to_csv(HERE / 'mechanism_005904.csv', index=False)
    focus_summary = focus.groupby(['state', 'arm'])[
        ['gradient_sq_share_original', 'gradient_sq_share_robust',
         'gradient_sq_original', 'gradient_sq_robust',
         'signed_aggregate_projection_original', 'signed_aggregate_projection_robust']
    ].agg(['count', 'median', 'min', 'max'])
    focus_summary.columns = ['_'.join(c) for c in focus_summary.columns]
    focus_summary.reset_index().to_csv(HERE / 'mechanism_005904_summary.csv', index=False)
    h = pd.read_csv(HERE / 'learning_paired.csv')
    assert len(h) == 36 and len(g) == 72
    info = dict(head_pairs=36, different_selected_epochs=int((h.epoch_robust != h.epoch_original).sum()),
        returned_to_B=int(h.returned_to_B.sum()), activated_from_B=int(h.activated_from_B.sum()),
        both_zero=int(h.both_zero.sum()),
        loss200_lower_heads=int((h.train_loss_step200_robust < h.train_loss_step200_original).sum()),
        selected_J_higher=int((h.selected_J_robust > h.selected_J_original + 1e-12).sum()),
        selected_J_lower=int((h.selected_J_robust < h.selected_J_original - 1e-12).sum()),
        focus_unique_objects=int(focus.sample_id.nunique()), focus_source_folds=sorted(focus.fold.unique().tolist()),
        initial_loss_max_difference=float(np.abs(g.loc[g.state.eq('initial'), 'jointCE_robust'] -
            g.loc[g.state.eq('initial'), 'jointCE_original']).max()),
        note='Head/state exposures are repeated observations, not independent ships; squared-gradient energy is not AdamW update share')
    (HERE / 'mechanism_summary.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info), flush=True)
    print(focus_summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
