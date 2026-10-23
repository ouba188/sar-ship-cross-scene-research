"""CPU-only source footprint/adjacency coverage; no images through a network."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def footprint(height, width):
    top, left = (224-height)//2, (224-width)//2
    start = np.arange(14)*16
    yy = np.maximum(0, np.minimum(start+16, top+height)-np.maximum(start, top))
    xx = np.maximum(0, np.minimum(start+16, left+width)-np.maximum(start, left))
    grid = yy[:, None]*xx[None, :]/256.
    # Integer 16x16 cells reproduce the existing area footprint exactly.
    native = np.zeros((224,224));native[top:top+height,left:left+width] = 1
    for rotation in range(4):
        actual = np.rot90(native, rotation).reshape(14,16,14,16).mean((1,3))
        np.testing.assert_array_equal(actual, np.rot90(grid, rotation))
    return grid


def graph(grid):
    nodes = np.flatnonzero(grid.ravel()>0)
    index = {int(v):i for i,v in enumerate(nodes)}
    edges = []
    for i,node in enumerate(nodes):
        row,col = divmod(int(node),14)
        for nxt,allowed in [(node+1,col<13),(node+14,row<13)]:
            if allowed and int(nxt) in index:edges.append((i,index[int(nxt)]))
    return nodes, np.asarray(edges,dtype=int).reshape(-1,2)


def geometry(height,width):
    grid = footprint(height,width)
    nodes,edges = graph(grid)
    weights = grid.ravel()[nodes]
    degree = np.bincount(edges.ravel(),minlength=len(nodes))
    permutation = np.arange(len(nodes))
    rng = np.random.default_rng(np.random.SeedSequence([20261023,height,width]))
    for value in np.unique(degree):
        subset = np.flatnonzero(degree==value)
        permutation[subset] = rng.permutation(subset)
    rewired = np.sort(permutation[edges],axis=1)
    np.testing.assert_array_equal(np.bincount(rewired.ravel(),minlength=len(nodes)),degree)
    assert len(np.unique(rewired,axis=0)) == len(edges)
    alpha = weights[edges[:,0]]*weights[edges[:,1]]
    null_alpha = weights[rewired[:,0]]*weights[rewired[:,1]]
    weighted_degree = np.bincount(edges.ravel(),weights=np.repeat(alpha,2),minlength=len(nodes))
    null_degree = np.bincount(rewired.ravel(),weights=np.repeat(null_alpha,2),minlength=len(nodes))
    overlap = len(set(map(tuple,edges)) & set(map(tuple,rewired)))
    total = weights.sum()
    rec = dict(height=height,width=width,n_nodes=len(nodes),n_full=int((weights==1).sum()),
               n_fractional=int((weights<1).sum()),mass=total,n_eff_nodes=total**2/(weights@weights),
               fractional_mass_share=float(weights[weights<1].sum()/total),
               n_edges=len(edges),edge_mass=float(alpha.sum()),
               n_eff_edges=float(alpha.sum()**2/(alpha@alpha)) if len(alpha) else 0.,
               effective_edge_sample_independent=False,
               pair_independent_self_mass=float((weights@weights)/total**2),
               finite_offdiagonal_factor=float(1/(1-(weights@weights)/total**2)) if len(nodes)>1 else np.nan,
               count_ge4=len(nodes)>=4,effective_ge4=total**2/(weights@weights)>=4,
               n_horizontal=int(sum(nodes[j]-nodes[i]==1 for i,j in edges)),
               n_vertical=int(sum(nodes[j]-nodes[i]==14 for i,j in edges)),
               null_edge_overlap=overlap/len(edges) if len(edges) else 1.,
               null_changed_edges=len(edges)-overlap,
               null_fixed_nodes=int((permutation==np.arange(len(nodes))).sum()),
               null_edge_mass_ratio=float(null_alpha.sum()/alpha.sum()) if len(edges) else 1.,
               null_weighted_degree_l1=float(abs(null_degree-weighted_degree).sum()/max(weighted_degree.sum(),1e-12)))
    # These arrays specify one null graph, not a search over graph seeds.
    config=dict(nodes=nodes.tolist(),edges=edges.tolist(),permutation=permutation.tolist(),
                null_edges=rewired.tolist(),weights=weights.tolist())
    return rec,config


def main():
    with np.load(ROOT/'pair_local_evidence/source_moments.npz') as z:
        ids,tr,va,reader = [z[k] for k in ['sample_ids','train','validation','reader']]
    assert len(ids)==1821 and len(tr)==1542 and len(va)==279
    cohort=pd.read_csv(ROOT/'cohort.csv',dtype={'mmsi':str}).set_index('sample_id')
    geom=pd.read_csv(ROOT/'native_geometry.csv').set_index(['sample_id','pol'])
    mapping=pd.Series(np.arange(len(cohort)),index=cohort.index).loc[ids].to_numpy()
    with np.load(ROOT/'readers/fold0_inputs.npz') as z:
        np.testing.assert_array_equal(mapping[tr],z['train'])
        np.testing.assert_array_equal(mapping[va],z['validation'])
    assigned=np.zeros(len(ids),int)
    for k in range(4):
        with np.load(ROOT/'readers/models'/f'fold0_inner{k}.npz') as z:
            held,fit=z['held'],z['fit']
        for column in ['mmsi','pass_group']:
            assert not set(cohort.iloc[held][column]) & set(cohort.iloc[fit][column])
        assigned[np.isin(mapping,held)] = k+1
    np.testing.assert_array_equal(assigned,reader)
    assert (reader[tr]>0).all() and (reader[va]==0).all()
    configs,cache,rows={}, {}, []
    for i,sample in enumerate(ids):
        for pol in ['vv','vh']:
            g=geom.loc[(sample,pol)]
            height,width=int(g.height),int(g.width)
            key=f'{height}x{width}'
            if key not in cache:cache[key],configs[key]=geometry(height,width)
            c=cohort.loc[sample]
            rows.append(dict(sample_id=sample,source_row=i,global_row=int(mapping[i]),
                role='validation' if reader[i]==0 else 'train',reader=int(reader[i]),
                pol=pol,class_id=int(c.class_id),class_name=c.class_name,mmsi=c.mmsi,
                pass_group=c.pass_group,path=g.path,geometry=key,**cache[key]))
    df=pd.DataFrame(rows)
    df.to_csv(HERE/'coverage.csv',index=False)
    pd.DataFrame([dict(geometry=k,**v) for k,v in cache.items()]).to_csv(HERE/'geometries.csv',index=False)
    (HERE/'null_graphs.json').write_text(json.dumps(configs,separators=(',',':')),encoding='utf-8')
    metrics=['n_nodes','n_eff_nodes','n_edges','n_eff_edges','fractional_mass_share',
             'finite_offdiagonal_factor','null_edge_overlap','null_edge_mass_ratio','null_weighted_degree_l1']
    summary=[]
    for keys in [['role','pol'],['role','pol','class_id'],['role','pol','reader']]:
        for values,g in df.groupby(keys):
            rec=dict(grouping='/'.join(keys),**dict(zip(keys,values)),n=len(g),
                     nodes_lt4=int((~g.count_ge4).sum()),neff_lt4=int((~g.effective_ge4).sum()),
                     no_edge=int(g.n_edges.eq(0).sum()),null_unchanged=int(g.null_changed_edges.eq(0).sum()))
            for m in metrics:
                for label,value in zip(['min','q25','median','q75','max'],g[m].quantile([0,.25,.5,.75,1])):
                    rec[f'{m}_{label}']=float(value)
            summary.append(rec)
    pd.DataFrame(summary).to_csv(HERE/'coverage_summary.csv',index=False)
    info=dict(source_rows=len(ids),pol_rows=len(df),train=len(tr),validation=len(va),unique_geometry=len(cache),
              GPU_used=False,training=False,pixel_network_extraction=False,source_mapping_exact=True,
              C4_weight_geometry_exact=True,null_unweighted_per_node_degree_exact=True,
              nodes_below4=int((~df.count_ge4).sum()),neff_below4=int((~df.effective_ge4).sum()),
              null_unchanged_pol_rows=int(df.null_changed_edges.eq(0).sum()))
    (HERE/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info))
    print(pd.DataFrame(summary).query("grouping=='role/pol'").to_string(index=False))


if __name__=='__main__':main()
