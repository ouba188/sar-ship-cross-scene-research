"""Check actual cached readouts before fitting heads; no model fitting."""
from common import *
from run_heads import load_info, summary_checks, raw_inputs
import argparse


def main(stage):
    table,valid,_=load_data()
    if stage=='source':
        with np.load(HERE/'features_source.npz') as z: features=z['layer1'].reshape(len(table),2,256)
    rows=[]; differences=[]
    for fold in range(3):
        for role in (['train','validation'] if stage=='source' else ['test']):
            info=load_info(fold,role,table)
            differences.extend(summary_checks(info,fold,role))
            dummy=np.zeros((len(info['indices']),8))
            common=None
            for arm in ['global','fixed','dynamic','merged']:
                x=raw_inputs(dummy,info,arm)
                if common is not None:np.testing.assert_array_equal(x[...,-32:],common)
                common=x[...,-32:]
            assert not info['summary'][~info['region_valid']].any()
            local=info['summary'][...,[1,2]]
            assert (local[...,0]>=-1e-10).all()
            assert (abs(local[...,1])<=local[...,0]+1e-10).all()
            rows.append(dict(fold=fold,role=role,n=len(dummy),common_quality_all_four_exact=True,
                disabled_summaries_zero=True,h_nonnegative_absb_bounded=True))
    closure=[]
    if stage=='source':
        for r in readers():
            with np.load(HERE/'summaries_source'/f"{r['name']}.npz") as z:
                ix=z['indices']; observed=z['summary'][:,:,0,:,0]
            with np.load(HERE/'readers'/f"{r['name']}.npz") as z: W=z['W'].reshape(8,2,256).transpose(1,0,2)
            expected=np.einsum('npc,pkc->npk',features[ix].astype(float),W)
            error=float(abs(expected-observed).max())
            assert error<1e-4
            closure.append(dict(reader=r['name'],max_mu_vs_WGAP_error=error,n=len(ix)))
    pd.DataFrame(rows).to_csv(HERE/f'{stage}_summary_checks.csv',index=False)
    difference=pd.DataFrame(differences)
    assert difference.mean_abs.gt(1e-8).all()
    difference.to_csv(HERE/f'{stage}_fixed_dynamic_summary_difference.csv',index=False)
    if closure:pd.DataFrame(closure).to_csv(HERE/'source_global_mean_closure.csv',index=False)
    save_json(HERE/f'{stage}_summaries_checked.json',dict(completed_utc=now(),rows=len(rows),
        summary_difference_mean_abs_min=float(difference.mean_abs.min()),summary_difference_mean_abs_max=float(difference.mean_abs.max()),
        global_mu_WGAP_max_error=max([x['max_mu_vs_WGAP_error'] for x in closure],default=None)))
    print(f'{stage} summary checks passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['source','test']);main(parser.parse_args().stage)
