"""Actual queries and largest signed support-solve contributors, not nearest neighbors."""
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE, MANIFEST


def main():
    manifest = pd.read_csv(MANIFEST)
    cases = pd.read_csv(HERE / 'MECHANISM_CASES.csv')
    terms = pd.read_csv(HERE / 'SUPPORT_SCORE_CONTRIBUTIONS.csv')
    fig, axes = plt.subplots(len(cases), 6, figsize=(16, 6), layout='constrained')
    for i, case in cases.iterrows():
        row = int(case.global_row)
        values = terms[terms.query_row.eq(row)].sort_values('total')
        selected = [(row, 'Query'), (int(values.iloc[-1].support_row),
                    f'Largest + term {values.iloc[-1].total:+.3f}'),
                    (int(values.iloc[0].support_row), f'Largest - term {values.iloc[0].total:+.3f}')]
        for k, (index, label) in enumerate(selected):
            for j, pol in enumerate(('vv', 'vh')):
                ax = axes[i, 2*k+j]
                with Image.open(manifest.loc[index, 'chip_path_'+pol]) as im:
                    pixels = np.asarray(im.convert('L'))
                ax.imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                ax.set_title(f'{label} | {index} {pol.upper()}\n{manifest.loc[index, "class_name"]}', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Outcome-selected cases; signed true-minus-rival expert score changes\n'
                 'Joint residual solve: contributor class need not match query; not physical causal attribution',
                 fontsize=11)
    fig.savefig(HERE / 'query_support_cases.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
