"""Frozen-model interventions, not retraining or evaluation-based selection."""
import json
import argparse
import numpy as np
import pandas as pd
import torch
import train
from operators import Candidate


@torch.inference_mode()
def operator_activity(model, data):
    y = data.table.class_id.to_numpy(int)
    ids = np.concatenate([data.tr[y[data.tr]==c][:3] for c in range(8)])
    rows = []
    for j in ids:
        b = data.batch([j],model.arm)
        x = b['x'].flatten(0,2)
        region = b['region'].flatten(0,2)
        eligible = region.sum(1)>0
        item = dict(up_index=int(data.up[j]),class_id=int(y[j]))
        if model.conv is not None:
            d = model.conv(x,region,b['theta'].reshape(-1))
            item['conv_relative_l2'] = float(d.norm()/x.norm().clamp_min(1e-10))
            item['conv_rms_on_region'] = float(d.movedim(1,-1)[eligible].square().mean().sqrt())
            pol = torch.arange(2,device='cuda')[:,None].expand(2,4).reshape(-1)
            dq = torch.einsum('bcd,bdhw->bchw',model.wproject[pol],d)
            item['class_response_delta_rms'] = float(dq.movedim(1,-1)[eligible].square().mean().sqrt())
            x = x+d
        if model.pool is not None:
            aa = model.pool.assignment(x,b['a0'].flatten(0,2))
            l1, errors = [], []
            for a,row,col,prior in aa:
                valid = row.sum(1)>0
                l1.extend((a-prior).abs().sum((1,2))[valid].cpu().tolist())
                errors.append(float((a.sum(-1)-row).abs().max()))
            item['allocation_L1_mean'] = float(np.mean(l1))
            item['allocation_row_error_max'] = max(errors)
        rows.append(item)
    return rows


def main(compression_only=False):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    done = json.loads((train.HERE/'complete.json').read_text())
    metrics, activity, changes = [],[],[]
    for seed in done['seeds']:
        data = train.Data(seed)
        ids = np.concatenate((data.va,data.te))
        if seed==done['seeds'][0]:
            static = Candidate('static_pool',data.ck,data.wproject,data.tau).cuda()
            _,summary = train.predict(static,data,np.arange(len(data.table)),summaries=True)
            compression = []
            for role,ii in [('utility_train',data.tr),('selection',data.va),('evaluation',data.te)]:
                for p in range(2):
                    for r in range(3):
                        old = data.original_summary[ii,p,r]
                        new = summary[ii,p,r]
                        ok = data.flags[ii,p,r].astype(bool)
                        ratios = new[ok,:,1]/np.maximum(old[ok,:,1],1e-20)
                        ratios = ratios[old[ok,:,1]>1e-5]
                        compression.append(dict(role=role,polarization=('VV','VH')[p],region=('body','end','side')[r],n=int(ok.sum()),
                                                mean_error_max=float(abs(old[ok,:,0]-new[ok,:,0]).max()),
                                                highlow_ratio_median=float(np.median(ratios)),
                                                highlow_lower_fraction=float((ratios<1).mean())))
            pd.DataFrame(compression).to_csv(train.HERE/'compression.csv',index=False)
            del static
            if compression_only:
                print(pd.DataFrame(compression).to_string(index=False),flush=True)
                return
        for arm in ('conv_grid','learned_pool','conv_learned_pool'):
            if arm not in done['arms']:
                continue
            ck = torch.load(train.HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
            if ck['receipt']['selected_step']<0:
                changes.append(dict(seed=seed,arm=arm,fallback=True))
                continue
            model = Candidate(arm,data.ck,data.wproject,data.tau).cuda()
            model.load_state_dict(ck['state'])
            activity.extend([dict(seed=seed,arm=arm,**r) for r in operator_activity(model,data)])
            baseline = np.load(train.HERE/'results'/f'{arm}_{seed}.npz')['score'][ids]
            interventions = ['full']
            if model.conv is not None:
                interventions.append('conv_zero')
            if model.pool is not None:
                interventions.append('pool_static')
            if model.conv is not None and model.pool is not None:
                interventions.append('both_zero')
            for mode in interventions:
                model.load_state_dict(ck['state'])
                with torch.no_grad():
                    if mode in ('conv_zero','both_zero'):
                        model.conv.expand.weight.zero_()
                    if mode in ('pool_static','both_zero'):
                        model.pool.score[-1].weight.zero_()
                score = baseline if mode=='full' else train.predict(model,data,ids)
                if mode!='full':
                    np.savez_compressed(train.HERE/'results'/f'intervention_{arm}_{seed}_{mode}.npz',score=score,index=ids)
                changes.append(dict(seed=seed,arm=arm,mode=mode,fallback=False,
                                    score_difference_max=float(abs(score-baseline).max()),
                                    score_difference_mean=float(abs(score-baseline).mean()),
                                    decisions_changed=int((score.argmax(1)!=baseline.argmax(1)).sum())))
                for role,sl in [('selection',slice(0,len(data.va))),('evaluation',slice(len(data.va),None))]:
                    y = data.table.iloc[ids[sl]].class_id.to_numpy(int)
                    metrics.append(dict(seed=seed,arm=arm,mode=mode,role=role,**train.metrics(y,score[sl])))
            del model
        del data
        torch.cuda.empty_cache()
    pd.DataFrame(metrics).to_csv(train.HERE/'intervention_metrics.csv',index=False)
    pd.DataFrame(changes).to_csv(train.HERE/'intervention_changes.csv',index=False)
    pd.DataFrame(activity).to_csv(train.HERE/'operator_activity.csv',index=False)
    print(pd.DataFrame(changes).to_string(index=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compression-only',action='store_true')
    main(parser.parse_args().compression_only)
