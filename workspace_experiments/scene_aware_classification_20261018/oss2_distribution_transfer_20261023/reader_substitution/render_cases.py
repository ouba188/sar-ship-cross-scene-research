"""Native source-validation cases for fixed local-reader substitution."""
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    t = pd.read_csv(HERE / 'review_cases.csv')
    (HERE / 'panels').mkdir(exist_ok=True)
    rows = []
    for q in t.itertuples():
        old = np.array([getattr(q, f'o{c}') for c in range(8)])
        new = np.array([getattr(q, f's{c}') for c in range(8)])
        y = q.class_id
        rival = np.argmax(np.where(np.arange(8) == y, -np.inf, new))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
        dimensions = []
        for j, pol in enumerate(['vv','vh']):
            with Image.open(getattr(q, f'chip_path_{pol}')) as im:
                raw = np.array(im)
            axes[j].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            axes[j].set_title(f'{pol.upper()} native {raw.shape[1]}x{raw.shape[0]} unchanged')
            dimensions.append(f'{pol}:{raw.shape[1]}x{raw.shape[0]}')
        ticks = np.arange(8)
        axes[2].bar(ticks-.2,old,width=.4,label='original outer moments')
        axes[2].bar(ticks+.2,new,width=.4,label='mean inner moments')
        axes[2].set_xticks(ticks)
        axes[2].legend(fontsize=8)
        axes[2].set_title(f'True {y}, fixed final rival {rival}\n'
                          f'margin {old[y]-old[rival]:+.4f} -> {new[y]-new[rival]:+.4f}')
        fig.suptitle(f'SOURCE VALIDATION: {q.sample_id} / {q.arm} / {q.outcome}\n'
                     f'fold{q.fold} seed{q.seed} epoch{q.epoch}; only local moments changed; no physical attribution')
        path = HERE / 'panels' / f'{q.arm}_{q.outcome}.png'
        fig.savefig(path,dpi=100)
        plt.close(fig)
        rows.append(dict(sample_id=q.sample_id,arm=q.arm,outcome=q.outcome,fold=q.fold,
            seed=q.seed,epoch=q.epoch,true_class=y,old_prediction=q.old_prediction,
            new_prediction=q.new_prediction,rival=int(rival),dimensions=';'.join(dimensions),
            old_fixed_margin=old[y]-old[rival],new_margin=new[y]-new[rival],
            true_delta=new[y]-old[y],rival_delta=new[rival]-old[rival],panel=str(path)))
    pd.DataFrame(rows).to_csv(HERE / 'case_details.csv',index=False)
    print(f'{len(rows)} panels, {t.sample_id.nunique()} unique source observations')


if __name__=='__main__':main()
