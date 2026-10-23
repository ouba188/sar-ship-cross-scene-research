"""Redraw only six source panels, hiding zero-pool-weight grid locations."""
import run_probe as p


def main():
    selection = p.json.loads((p.HERE / 'selection_frozen.json').read_text(encoding='utf-8'))
    predictions = p.pd.read_csv(p.HERE / 'predictions.csv')
    paired = predictions.pivot(index=['c', 'd', 'sample_id', 'true_class'], columns='arm',
                              values='prediction').reset_index()
    rescued = paired[(paired.same_position == paired.true_class) &
                     (paired.marginal != paired.true_class)].sort_values(['c', 'd', 'sample_id'])
    cases = selection['cases'].copy()
    if len(rescued):
        q = rescued.iloc[0]
        cases.append(dict(c=int(q.c), d=int(q.d), sample_id=q.sample_id,
                          review_only='first rescued vs marginal; not used for fitting or selection'))
    table = p.pd.read_csv(p.ROOT / 'cohort.csv')
    ids = list(dict.fromkeys(row['sample_id'] for row in cases))
    selected = table.set_index('sample_id').loc[ids].reset_index()
    model = p.network()
    with p.np.load(p.ROOT / 'readers/models/fold0_outer.npz') as z:
        w = p.torch.tensor(z['W'].reshape(8, 2, 1024).transpose(1, 0, 2),
                           device='cuda', dtype=p.torch.float64)
    with p.torch.inference_mode():
        x, mask = p.load_batch(selected, 0, len(selected))
        z = p.feature_map(model, x, 0)
        weight = p.spatial_weights(mask, z, 0).flatten(2).double()
        response = p.torch.einsum('bck,bku->bcu', w[p.np.tile([0, 1], len(selected))],
                                  z.flatten(2).double())
        _, _, residual = p.weighted_moments(response, weight)
        maps = residual.cpu().numpy().reshape(len(selected), 2, 8, 14, 14)
        valid = weight.cpu().numpy().reshape(len(selected), 2, 14, 14) > 0
    with p.np.load(p.HERE / 'source_moments.npz') as z:
        base = dict(zip(z['sample_ids'], z['base']))
    reviews = []
    for case in cases:
        i, c, d = ids.index(case['sample_id']), case['c'], case['d']
        row = selected.iloc[i]
        fig, axes = p.plt.subplots(2, 4, figsize=(15, 7), layout='constrained')
        for pol, polname in enumerate(['vv', 'vh']):
            with p.Image.open(row[f'chip_path_{polname}']) as im:
                raw = p.np.array(im)
            axes[pol, 0].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            axes[pol, 0].set_title(f'{polname.upper()} native {raw.shape[1]}x{raw.shape[0]}')
            values = [maps[i, pol, c], maps[i, pol, d], maps[i, pol, c]-maps[i, pol, d]]
            values = [p.np.where(valid[i, pol], v, p.np.nan) for v in values]
            limit = max(float(p.np.nanmax(p.np.abs(v))) for v in values)
            for j, (value, title) in enumerate(zip(values, [f'class {c}', f'class {d}', f'{c} minus {d}']), 1):
                ax = axes[pol, j]
                im = ax.imshow(value, cmap='RdBu_r', vmin=-limit, vmax=limit,
                               interpolation='nearest', extent=[0, 224, 224, 0])
                ax.set_facecolor('.85')
                ax.set_title(f'Valid-grid centered response: {title}')
                h, width = raw.shape
                ax.add_patch(p.plt.Rectangle(((224-width)//2, (224-h)//2), width, h,
                                             fill=False, edgecolor='black'))
                fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
        fig.suptitle(f'SOURCE VALIDATION: {row.sample_id}; true={row.class_id}; B={base[row.sample_id].argmax()}\n'
                     'Grey=zero pool weight, excluded; rotation0 only; coarse RF, NOT physical pixel causes')
        suffix = '_rescued_review' if 'review_only' in case else ''
        panel = p.HERE / 'panels' / f'pair{c}_{d}_class{row.class_id}{suffix}.png'
        fig.savefig(panel, dpi=100)
        p.plt.close(fig)
        q = paired[(paired.c == c) & (paired.d == d) & (paired.sample_id == row.sample_id)].iloc[0]
        reviews.append(dict(c=c, d=d, sample_id=row.sample_id, true_class=int(row.class_id),
                            B=int(base[row.sample_id].argmax()), global_prediction=int(q['global']),
                            marginal=int(q.marginal), same_position=int(q.same_position),
                            panel=str(panel), selection=case.get('review_only', 'pre-fit source B error, then ID')))
    p.pd.DataFrame(reviews).to_csv(p.HERE / 'review_cases.csv', index=False)
    print(f'Redrawn {len(cases)} panels / {len(ids)} unique source observations; no fitting or metric changes.')


if __name__ == '__main__':
    main()
