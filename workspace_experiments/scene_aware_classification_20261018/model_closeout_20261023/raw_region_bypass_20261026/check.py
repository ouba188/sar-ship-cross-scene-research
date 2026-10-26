"""Check that auxiliary initialization, source row order and capacity controls match the design."""
import numpy as np
import pandas as pd
import torch
import run


def main():
    checks=[]
    for source,other in [('oss1','oss2'),('oss2','oss1')]:
        table=pd.read_csv(run.ROOT/f'{source}_m123_replication_20261026/stack_table.csv')
        combined=pd.read_csv(run.REPAIR/f'{source}-to-{other}/table.csv')
        np.testing.assert_array_equal(combined.loc[combined.dataset_side.eq('source'),'original_obs_id'],table.obs_id)
        for seed in run.base.SEEDS:
            states={arm:torch.load(run.HERE/source/f'{arm}_seed{seed}.pt',weights_only=False)['initial'] for arm in ('raw_only','raw_summary','anchored_raw_summary')}
            a,b,c=states['raw_only'],states['raw_summary'],states['anchored_raw_summary']
            assert torch.equal(a['0.weight'],b['0.weight'][:,:1538])
            assert torch.count_nonzero(b['0.weight'][:,1538:])==0
            for key in ('0.bias','2.weight','2.bias'):
                assert torch.equal(a[key],b[key])
            for key in b:
                assert torch.equal(b[key],c[key])
            checks.append(dict(source=source,seed=seed,shared_initial_raw_weights=True,summary_initial_weights_zero=True,anchored_initial_weights_match=True,source_row_order_verified=True))
        wide=pd.read_csv(run.HERE/'wide_control'/source/'training.csv')
        assert set(wide.parameters)=={211395}
        assert (wide.parameters>210820).all()
    run.base.write_json(run.HERE/'verification.json',dict(checks=checks,raw_replay='see oss1/oss2 checks.json',capacity_control='raw137=211395 slightly larger than raw+summary128=210820'))
    print('PASS: six shared initialization and source-order checks; wider control capacity verified')


if __name__=='__main__':main()
