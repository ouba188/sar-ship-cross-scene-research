"""Independent saved-map weighted std checks, including class-specific padding."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def std(v,w):
    r=v-v.mean(-2,keepdims=True)
    w=w/w.sum(-1,keepdims=True)
    r=r-(r*w).sum(-1,keepdims=True)
    return np.sqrt((r*r*w).sum(-1))


def main():
    rows=[]
    with np.load(HERE/'validation_maps.npz') as z:
        for name in z.files:
            if not name.endswith('_v'):
                continue
            v=z[name].reshape(3,8,-1)
            w=z[name[:-2]+'_weight'].reshape(1,1,-1)
            result=std(v,w)
            offsets=np.arange(1,9)[None,:,None]*123.
            modified=v+(w==0)*offsets
            perm=np.random.default_rng(20261023).permutation(v.shape[-1])
            field=np.linspace(-1,1,v.shape[-1])
            row=dict(map=name,zero_weight_cells=int((w==0).sum()),
                class_specific_padding_error=float(abs(std(modified,w)-result).max()),
                permutation_error=float(abs(std(v[...,perm],w[...,perm])-result).max()),
                common_field_error=float(abs(std(v+field,w)-result).max()),
                constant_response_error=float(abs(std(np.broadcast_to(offsets,v.shape),w)).max()))
            assert max(row[k] for k in row if k.endswith('_error'))<1e-10
            rows.append(row)
    with np.load(HERE/'std_features.npz') as z:
        valid=z['valid'];d=z['delta_pol'];m=z['mean_pol']
        assert not d[:,~valid].any() and not m[:,~valid].any()
        np.testing.assert_array_equal(d.sum(2),z['delta'])
    assert any(r['zero_weight_cells']>0 for r in rows)
    pd.DataFrame(rows).to_csv(HERE/'independent_map_replay.csv',index=False)
    print(pd.DataFrame(rows).drop(columns='map').max().to_string())


if __name__=='__main__':
    main()
