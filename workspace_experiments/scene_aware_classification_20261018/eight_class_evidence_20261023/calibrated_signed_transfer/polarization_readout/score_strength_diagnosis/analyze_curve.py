"""Five predeclared strengths of an existing score-only head; no fitting."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from run_training import assessment

STRENGTHS = [0., .25, .5, .75, 1.]


def main():
    p = pd.read_csv(HERE.parent/'full_model_integration/predictions.csv')
    records, comparison = [], []
    for (role, seed), g in p[p.arm.eq('score_only')].groupby(['role', 'seed']):
        key = ['fold', 'instance_id']
        g = g.set_index(key).sort_index()
        real = p[p.role.eq(role)&p.seed.eq(seed)&p.arm.eq('free_real')].set_index(key).loc[g.index]
        y = g.class_id.to_numpy()
        base = g[[f'b{c}' for c in range(8)]].to_numpy()
        scores = g[[f's{c}' for c in range(8)]].to_numpy()
        np.testing.assert_array_equal(y, real.class_id)
        np.testing.assert_array_equal(base, real[[f'b{c}' for c in range(8)]])
        rm = assessment(y, real[[f's{c}' for c in range(8)]].to_numpy())
        for strength in STRENGTHS:
            value = base+strength*(scores-base)
            if strength == 0:
                np.testing.assert_array_equal(value, base)
            m = assessment(y, value)
            records.append(dict(role=role, seed=seed, strength=strength, **m))
            comparison.append(dict(role=role, seed=seed, strength=strength,
                                   delta_OA=m['OA']-rm['OA'], delta_BA=m['BA']-rm['BA'],
                                   both_at_least=(m['OA']>=rm['OA'] and m['BA']>=rm['BA'])))
    pd.DataFrame(records).to_csv(HERE/'curve.csv', index=False)
    pd.DataFrame(comparison).to_csv(HERE/'versus_real.csv', index=False)
    curve = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    colors = {17:'#146c94',29:'#b74359',43:'#26824b'}
    for ax, role in zip(axes, ['validation', 'test']):
        for seed, g in curve[curve.role.eq(role)].groupby('seed'):
            ax.plot(g.OA, g.BA, '-o', color=colors[seed], label=f'seed {seed} score scaling')
            for q in g.itertuples():
                ax.annotate(f'{q.strength:g}', (q.OA,q.BA), xytext=(3,5), textcoords='offset points', fontsize=8)
            r = p[p.role.eq(role)&p.seed.eq(seed)&p.arm.eq('free_real')]
            m = assessment(r.class_id.to_numpy(), r[[f's{c}' for c in range(8)]].to_numpy())
            ax.scatter([m['OA']],[m['BA']],s=90,marker='X',color=colors[seed])
        ax.set(xlabel='OA (%)',ylabel='BA (%)',title=f'{role}: X = real-polarization residual')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Fixed five strengths, frozen heads; descriptive curves, no test-selected model')
    fig.savefig(HERE/'tradeoff.png',dpi=120)
    plt.close(fig)
    cases = pd.read_csv(HERE.parent/'full_model_integration/review_cases.csv')
    cases = cases[cases.arm.eq('free_real') & cases.seed.eq(17)]
    case_rows = []
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 12), layout='constrained')
    for row_index, case in enumerate(cases.itertuples()):
        g = p[p.role.eq('test') & p.seed.eq(17) & p.instance_id.eq(case.instance_id)]
        ordinary = g[g.arm.eq('score_only')].iloc[0]
        real = g[g.arm.eq('free_real')].iloc[0]
        base = ordinary[[f'b{c}' for c in range(8)]].to_numpy(dtype=float)
        delta = ordinary[[f's{c}' for c in range(8)]].to_numpy(dtype=float)-base
        for column, pol in enumerate(['vv', 'vh']):
            with Image.open(getattr(case, f'image_{pol}')) as im:
                pixels = np.array(im)
            axes[row_index,column].imshow(pixels, cmap='gray', interpolation='nearest')
            axes[row_index,column].set_title(f'{case.instance_id} {case.outcome} {pol.upper()} {pixels.shape[1]}x{pixels.shape[0]}', fontsize=9)
            axes[row_index,column].axis('off')
        margins = []
        for strength in STRENGTHS:
            value = base+strength*delta
            margin = value[case.class_id]-np.max(np.delete(value, case.class_id))
            margins.append(margin)
            case_rows.append(dict(instance_id=case.instance_id, outcome=case.outcome,
                                  class_id=case.class_id, strength=strength,
                                  prediction=int(value.argmax()), margin=margin,
                                  image_vv=case.image_vv, image_vh=case.image_vh))
        rv = real[[f's{c}' for c in range(8)]].to_numpy(dtype=float)
        rm = rv[case.class_id]-np.max(np.delete(rv, case.class_id))
        ax = axes[row_index,2]
        ax.plot(STRENGTHS, margins, '-o', label='Frozen score-only scaling')
        ax.axhline(rm, color='#b74359', label='Real local readout')
        ax.axhline(0, color='black', linewidth=.7)
        ax.set(xlabel='Score-only strength', ylabel='True-class margin')
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle('Same four previously reviewed raw cases; no new case selection or training')
    fig.savefig(HERE/'raw_case_comparison.png', dpi=120)
    plt.close(fig)
    pd.DataFrame(case_rows).to_csv(HERE/'case_curve.csv', index=False)
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
