"""Summarize the completed trajectories without optimization or target access."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def joint_ce(y, scores):
    shifted = scores - scores.max(1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(y)), y]
    return .5 * losses.mean() + .5 * np.mean([losses[y == c].mean() for c in range(8)])


def main():
    curves = pd.read_csv(HERE / 'selection_curves.csv')
    selections = pd.read_csv(HERE / 'selections.csv')
    predictions = pd.read_csv(HERE / 'predictions.csv')
    rows, complete = [], []
    for (seed, arm), g in curves.groupby(['seed', 'arm'], sort=False):
        g = g.sort_values('epoch').copy()
        assert g.epoch.tolist() == list(range(0, 201, 5))
        sel = selections[selections.seed.eq(seed) & selections.arm.eq(arm)].iloc[0]
        train = predictions[predictions.seed.eq(seed) & predictions.arm.eq(arm) & predictions.role.eq('train')]
        baseline_loss = joint_ce(train.class_id.to_numpy(), train[[f'b{c}' for c in range(8)]].to_numpy())
        z, end = g.iloc[0], g.iloc[-1]
        chosen = g[g.epoch.eq(sel.epoch)].iloc[0]
        row = dict(seed=seed, arm=arm, selected_epoch=int(sel.epoch),
                   train_initial_jointCE_float64=baseline_loss,
                   train_last_logged_jointCE_pre_step200=end.train_loss,
                   train_logged_change=end.train_loss-baseline_loss,
                   validation_checks_above_initial_J=int((g.J > z.J + 1e-14).sum()),
                   validation_checks_equal_initial_J=int(np.isclose(g.J, z.J, atol=1e-14, rtol=0).sum()),
                   validation_checks_below_initial_J=int((g.J < z.J - 1e-14).sum()))
        for metric in ['OA', 'BA', 'J', 'NLL', 'balanced_NLL', 'Brier']:
            for name, point in [('initial', z), ('terminal', end), ('selected', chosen)]:
                row[f'val_{name}_{metric}'] = point[metric]
            row[f'val_terminal_delta_{metric}'] = end[metric] - z[metric]
        row['val_initial_jointCE'] = .5*(z.NLL+z.balanced_NLL)
        row['val_terminal_jointCE'] = .5*(end.NLL+end.balanced_NLL)
        rows.append(row)
        g['is_selected'] = g.epoch.eq(sel.epoch)
        g['train_loss_parameter_step'] = np.where(g.epoch.eq(0), 0, g.epoch-1)
        g['train_jointCE_logged_or_baseline'] = g.train_loss.fillna(baseline_loss)
        g['train_loss_origin'] = np.where(g.epoch.eq(0), 'recomputed_B_float64', 'original_pre_update_float32')
        g['val_jointCE'] = .5*(g.NLL+g.balanced_NLL)
        for metric in ['OA', 'BA', 'J', 'NLL', 'balanced_NLL']:
            g[f'val_delta_{metric}'] = g[metric] - z[metric]
        complete.append(g)
    result = pd.DataFrame(rows)
    result.to_csv(HERE / 'terminal_vs_selection.csv', index=False)
    pd.concat(complete, ignore_index=True).to_csv(HERE / 'curves_annotated.csv', index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
