"""Separate per-target image and layout information, retaining matched training."""
import time
import numpy as np
import pandas as pd
import torch
from train_expanded import HERE, Relation, inputs_for, SEEDS, weights, risk, metrics

ARMS = ['image_constant_geometry_real', 'image_real_geometry_constant']


def main():
    torch.set_num_threads(2)
    t = pd.read_csv(HERE/'readout/targets.csv', dtype={'mmsi':str})
    y, ids = t.class_id.to_numpy(), t.mmsi.to_numpy()
    records, predictions = [], []
    started = time.perf_counter()
    for fold in range(6):
        d = dict(np.load(HERE/f'fold{fold}_inputs.npz'))
        train, val, test = [d[k] for k in ['train','validation','test']]
        source = train[d['m'][train].any(1)]
        node_mean = d['split'][train][d['m'][train]].mean(0)
        stat_mean = d['stats'][source].mean(0)
        yy = torch.tensor(y, device='cuda')
        oo = torch.tensor(d['oof'], dtype=torch.float32, device='cuda')
        w = torch.tensor(weights(y[train], ids[train]), dtype=torch.float32, device='cuda')
        for arm in ARMS:
            x = inputs_for('split_unit', d, 'cuda')
            if arm == 'image_constant_geometry_real':
                x[1][:] = torch.as_tensor(node_mean, device='cuda')
            else:
                x[3][:] = torch.as_tensor(stat_mean, device='cuda')
            for seed in SEEDS:
                torch.manual_seed(seed)
                model = Relation('split_unit').cuda()
                opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                br, be = risk(y[val],d['base'][val]), 0
                for epoch in range(1,201):
                    opt.zero_grad()
                    score = oo[train]+model(*[a[train] for a in x])
                    loss = (torch.nn.functional.cross_entropy(score,yy[train],reduction='none')*w).mean()
                    loss.backward()
                    opt.step()
                    if epoch%5 == 0:
                        with torch.no_grad():
                            s = d['base'][val]+model(*[a[val] for a in x]).cpu().numpy()
                        r = risk(y[val],s)
                        if r < br-1e-6:
                            br, be = r, epoch
                            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(best)
                with torch.no_grad():
                    delta = model(*x).cpu().numpy()
                assert not delta[~d['m'].any(1)].any()
                s = d['base']+delta
                torch.save(best,HERE/'models'/f'fold{fold}_{arm}_{seed}.pt')
                records.append(dict(fold=fold,seed=seed,arm=arm,epoch=be,val_nll=br,**metrics(y[test],s[test])))
                for i in test:
                    predictions.append(dict(fold=fold,seed=seed,arm=arm,instance_id=t.iloc[i].instance_id,class_id=int(y[i]),
                        prediction=int(s[i].argmax()),base_prediction=int(d['base'][i].argmax()),
                        **{f's{c}':float(s[i,c]) for c in range(3)}))
        print('information fold',fold,'complete',flush=True)
    p = pd.DataFrame(predictions)
    pd.DataFrame(records).to_csv(HERE/'information_source_metrics.csv',index=False)
    p.to_csv(HERE/'information_source_predictions.csv',index=False)
    pooled=[]
    for (arm,seed),q in p.groupby(['arm','seed']):
        pooled.append(dict(arm=arm,seed=seed,**metrics(q.class_id.to_numpy(),q[['s0','s1','s2']].to_numpy()),
            net=int((q.prediction==q.class_id).sum()-(q.base_prediction==q.class_id).sum())))
    pd.DataFrame(pooled).to_csv(HERE/'information_source_pooled.csv',index=False)
    p[p.instance_id.isin(pd.read_csv(HERE/'review_cases.csv').instance_id.unique())].to_csv(HERE/'review_information_source.csv',index=False)
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    print('seconds',time.perf_counter()-started,flush=True)


if __name__ == '__main__':
    main()
