"""Explain source-selected zero steps from existing trajectories, without refits."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = [f'b{i}' for i in range(8)]


def main():
    pred = pd.read_csv(ROOT / 'source_predictions.csv',
                       usecols=['fold', 'seed', 'arm', 'role', 'class_id'] + BASE)
    curves = pd.read_csv(ROOT / 'selection_curves.csv')
    selection = pd.read_csv(ROOT / 'selections.csv')
    rows = []
    for (fold, seed, arm), g in curves.groupby(['fold', 'seed', 'arm']):
        g = g.sort_values('epoch')
        train = pred.loc[pred.fold.eq(fold) & pred.seed.eq(seed) & pred.arm.eq(arm) & pred.role.eq('train')]
        y, s = train.class_id.to_numpy(int), train[BASE].to_numpy()
        loss = logsumexp(s, axis=1) - s[np.arange(len(y)), y]
        initial_train = .5 * loss.mean() + .5 * np.mean([loss[y == c].mean() for c in range(8)])
        first, last = g.iloc[0], g.iloc[-1]
        chosen = selection.loc[selection.fold.eq(fold) & selection.seed.eq(seed) & selection.arm.eq(arm)].iloc[0]
        risks = (g.NLL + g.balanced_NLL) / 2
        rows.append(dict(fold=fold, seed=seed, arm=arm, selected_epoch=int(chosen.epoch),
            baseline_train_loss=initial_train, step200_preupdate_train_loss=last.train_loss,
            train_loss_change=last.train_loss-initial_train,
            baseline_validation_loss=float(risks.iloc[0]), step200_validation_loss=float(risks.iloc[-1]),
            validation_loss_change=float(risks.iloc[-1]-risks.iloc[0]),
            minimum_validation_loss=float(risks.min()),
            step200_delta_OA=float(last.OA-first.OA), step200_delta_BA=float(last.BA-first.BA),
            any_checkpoint_improves_validation_risk=bool((risks < risks.iloc[0]).any())))
    result = pd.DataFrame(rows)
    result.to_csv(HERE / 'source_learning_diagnosis.csv', index=False)
    summary = result.groupby('arm').agg(
        heads=('seed', 'size'), zero_selected=('selected_epoch', lambda x: int(x.eq(0).sum())),
        train_loss_improves=('train_loss_change', lambda x: int(x.lt(0).sum())),
        validation_loss_improves_final=('validation_loss_change', lambda x: int(x.lt(0).sum())),
        any_validation_risk_improves=('any_checkpoint_improves_validation_risk', 'sum'),
        mean_final_validation_OA_change=('step200_delta_OA', 'mean'),
        mean_final_validation_BA_change=('step200_delta_BA', 'mean'))
    summary.to_csv(HERE / 'source_learning_summary.csv')
    print(summary.to_string())
    print(result.loc[result.selected_epoch.eq(0)].to_string(index=False))


if __name__ == '__main__':
    main()
