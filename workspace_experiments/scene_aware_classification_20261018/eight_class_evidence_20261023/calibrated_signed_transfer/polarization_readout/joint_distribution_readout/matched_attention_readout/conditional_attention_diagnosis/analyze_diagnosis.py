"""Fixed original-chip examples for the label-free attention replacement."""
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_diagnosis import HERE, run


def main():
    pred = pd.read_csv(HERE/'predictions.csv', dtype={'mmsi': str})
    sys.path.insert(0, str(run.parent.HERE))
    import analyze_results as analysis
    intervals = []
    for seed, g in pred[pred.role.eq('test')].groupby('seed'):
        intervals.extend(dict(seed=seed, **r) for r in
            analysis.intervals(g, g.net_own_vs_borrowed.to_numpy(), 'own-borrowed'))
    pd.DataFrame(intervals).to_csv(HERE/'intervals.csv', index=False)
    table = run.parent.load_table().set_index('instance_id')
    selected = pred[pred.role.eq('test') & pred.seed.eq(17)]
    (HERE/'panels').mkdir(exist_ok=True)
    cases = []
    for outcome in ['rescued', 'harmed', 'CC', 'WW']:
        group = selected[selected.outcome.eq(outcome)].sort_values('instance_id')
        if group.empty:
            continue
        q = group.iloc[0]
        b = q[[f'b{c}' for c in range(8)]].to_numpy(float)
        own = q[[f'own{c}' for c in range(8)]].to_numpy(float)
        borrow = q[[f'borrowed{c}' for c in range(8)]].to_numpy(float)
        y = int(q.class_id)
        masked = borrow.copy()
        masked[y] = -np.inf
        rival = int(masked.argmax())
        rec = dict(outcome=outcome, query=q.instance_id, donor=q.donor_id,
            fold=int(q.fold), seed=17, epoch=int(q.epoch), true_class=y,
            donor_class=int(q.donor_class), base_prediction=int(b.argmax()),
            own_prediction=int(own.argmax()), borrowed_prediction=int(borrow.argmax()),
            fixed_rival=rival, own_margin=float(own[y]-own[rival]),
            borrowed_margin=float(borrow[y]-borrow[rival]), B_mean_distance=q.B_mean_distance)
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
        for col, (label, ident) in enumerate([('Query', q.instance_id), ('Donor', q.donor_id)]):
            row = table.loc[ident]
            for j, pol in enumerate(['vv', 'vh']):
                path = row[f'image_{pol}']
                with Image.open(path) as im:
                    raw = np.asarray(im.convert('L'))
                axes[j, col].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                axes[j, col].set_title(f'{label} {pol.upper()} {raw.shape[1]}x{raw.shape[0]} px\n'
                    f'{ident} / class {row.class_id}')
                rec[f'{label.lower()}_{pol}_path'] = str(path)
        x = np.arange(8)
        for ax in axes[:, 2]:
            ax.set_xticks(x)
            ax.axhline(0, color='black', lw=.5)
        axes[0, 2].bar(x-.25, b, .25, color='#757575', label='B fixed')
        axes[0, 2].bar(x, own, .25, color='#137e65', label='Own attention')
        axes[0, 2].bar(x+.25, borrow, .25, color='#bd6c1d', label='Borrowed attention')
        axes[0, 2].legend(fontsize=8)
        axes[0, 2].set_title('Only h,b replaced; B and mean unchanged')
        axes[1, 2].bar(x, own-borrow, color='#167b9b')
        axes[1, 2].set_title('Own minus borrowed class scores')
        fig.suptitle(f'{outcome}: own vs borrowed (not vs B); fold{q.fold} seed17 epoch{q.epoch}\n'
            f'Different vessel/pass; no class used in matching; distance {q.B_mean_distance:.4f}\n'
            f'Fixed rival {rival}: margin {rec["borrowed_margin"]:+.4f} -> {rec["own_margin"]:+.4f}')
        path = HERE/'panels'/f'{outcome}.png'
        fig.savefig(path, dpi=110)
        plt.close(fig)
        rec['panel'] = str(path)
        cases.append(rec)
    pd.DataFrame(cases).to_csv(HERE/'review_cases.csv', index=False)
    print(pd.DataFrame(cases)[['outcome', 'query', 'donor', 'true_class', 'donor_class',
        'epoch', 'borrowed_margin', 'own_margin']].to_string(index=False))


if __name__ == '__main__':
    main()
