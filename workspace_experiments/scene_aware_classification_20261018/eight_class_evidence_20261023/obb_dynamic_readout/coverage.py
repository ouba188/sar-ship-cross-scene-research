"""Source-only coverage preparation, no labels used to choose geometry."""
from common import *
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    tick = time.perf_counter()
    table, valid, snapshots = load_data()
    roles, take = required('source', len(table))
    chosen = np.flatnonzero(take.any(0))
    geoms = np.full((len(table), 2, 6), np.nan)
    records, placements = [], []
    footprints = {}
    for count, i in enumerate(chosen):
        for p in range(2):
            _, fp, g, place = native(table, snapshots, i, p, pixels=False)
            placements.append(dict(row_index=int(i), instance_id=table.iloc[i].instance_id, pol=POL[p], **place))
            if g is not None:
                geoms[i, p] = g
            if not valid[i, p]:
                continue
            q = region_quality(soft_regions(g), fp, g)
            for j, name in enumerate(REGIONS):
                records.append(dict(row_index=int(i), instance_id=table.iloc[i].instance_id, pol=POL[p], region=name,
                    geometry_valid=g is not None, length=g[2] if g is not None else 0,
                    width=g[3] if g is not None else 0,
                    **{k:float(v[j]) for k,v in q.items()}))
        if count % 1500 == 0:
            print('coverage', count, '/', len(chosen), round(time.perf_counter()-tick,1), flush=True)
    templates = []
    for ri, r in enumerate(roles):
        for p in range(2):
            ids = r['fit'][valid[r['fit'], p] & np.isfinite(geoms[r['fit'], p, 2])]
            assert len(ids)
            lw = np.median(geoms[ids, p, 2:4], axis=0)
            templates.append(dict(reader=r['name'], pol=POL[p], length=lw[0], width=lw[1], n=len(ids)))
    result = pd.DataFrame(records)
    result.to_csv(HERE/'source_region_coverage.csv', index=False)
    pd.DataFrame(templates).to_csv(HERE/'source_templates.csv', index=False)
    pd.DataFrame(placements).to_csv(HERE/'source_placements.csv', index=False)
    np.savez_compressed(HERE/'source_geometry.npz', instance_ids=table.instance_id.to_numpy(dtype=str),
        geometry=geoms, valid=valid, source_indices=chosen)
    summary = dict(source_objects=len(chosen), valid_pol=valid[chosen].sum(0).tolist(),
        geometry_missing_valid=int((valid[chosen] & ~np.isfinite(geoms[chosen,:,2])).sum()),
        parameters=dict(rhoL=.25,rhoW=.5,soft_edge_pixels=1),
        minimum_condition_not_yet_frozen=True, regions={})
    for name,g in result.groupby('region'):
        summary['regions'][name] = {k:g[k].quantile([0,.01,.05,.5,.95,1]).to_dict() for k in ['area','coverage','min_neff','nodes']}
        summary['regions'][name]['below_neff4'] = int(g.min_neff.lt(4).sum())
        summary['regions'][name]['zero_mass'] = int(g.area.eq(0).sum())
    checks = structural_checks()
    save_json(HERE/'geometry_checks.json', checks)
    # Fixed source quantile/extreme cases, no outcomes or test labels consulted.
    body = result.query("pol=='vv' and region=='body'").set_index('row_index')
    criteria = [('small', body.area), ('large', -body.area), ('slender', -body.length/body.width),
                ('square', body.length/body.width), ('clipped', body.coverage)]
    selected = []
    for label, series in criteria:
        i = next(int(j) for j in series.sort_values(kind='stable').index if j not in [x[1] for x in selected])
        selected.append((label, i))
    (HERE/'coverage_panels').mkdir(exist_ok=True)
    cases=[]
    template = pd.DataFrame(templates).set_index(['reader','pol'])
    for label, i in selected:
        fig, axes = plt.subplots(2,3,figsize=(12,8))
        ri = int(np.flatnonzero(take[:, i])[0]); reader=roles[ri]['name']
        for p in range(2):
            pix, fp, g, place = native(table,snapshots,i,p)
            lw = template.loc[(reader,POL[p]), ['length','width']].to_numpy(float)
            for col, (title, masks) in enumerate([('native160', None),('actual OBB',soft_regions(g)),('fit template',soft_regions(g,lw))]):
                ax=axes[p,col]; ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if masks is not None:
                    for j,color in enumerate(['lime','cyan','magenta']):
                        if (masks[j]*fp).max()>.1:
                            ax.contour(masks[j]*fp, levels=[.5], colors=[color],linewidths=.8)
                ax.set_title(f'{POL[p].upper()} {title}; valid={bool(valid[i,p])}')
                ax.set_xlim(-.5,159.5); ax.set_ylim(159.5,-.5)
        row=table.iloc[i]
        fig.suptitle(f'{label}: {row.instance_id}; source reader {reader}\nbody=green, end=cyan, side=magenta; pixels unchanged')
        fig.tight_layout(rect=(0,0,1,.93))
        path=HERE/'coverage_panels'/f'{label}.png'; fig.savefig(path,dpi=130); plt.close(fig)
        cases.append(dict(case=label,row_index=i,instance_id=row.instance_id,reader=reader,path=str(path)))
    pd.DataFrame(cases).to_csv(HERE/'coverage_cases.csv',index=False)
    summary['seconds']=time.perf_counter()-tick
    save_json(HERE/'source_coverage_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
