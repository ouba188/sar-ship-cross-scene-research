"""Fixed-head on/off diagnostic for all selected axis models; no training."""
import json
import time
import numpy as np
import pandas as pd
import torch
import train_projection as run

tr = run.tr


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = tr.Data()
    reader = dict(np.load(tr.HERE/'reader.npz'))
    y = data.table.iloc[data.te].class_id.to_numpy(int)
    rows, records = [], []
    for seed in tr.SEEDS:
        ck = torch.load(run.HERE/'models'/f'axis_moment_{seed}.pt', map_location='cpu', weights_only=False)
        model = run.Model('axis_moment', reader).cuda()
        model.load_state_dict(ck['state'])
        model.eval().requires_grad_(False)
        full = np.load(run.HERE/'results'/f'axis_moment_{seed}.npz')['score'][data.te]
        replay = tr.predict(model,data,data.te[:2],data.baselines[seed])
        np.testing.assert_allclose(replay,full[:2],atol=3e-5,rtol=0)
        with torch.no_grad():
            model.conv.output.zero_()
        start = time.perf_counter()
        off = tr.predict(model,data,data.te,data.baselines[seed])
        p, q = full.argmax(1), off.argmax(1)
        rescued, harmed = (p==y)&(q!=y), (p!=y)&(q==y)
        a, b = tr.metrics(y,full), tr.metrics(y,off)
        rows.append(dict(seed=seed,selected_step=ck['receipt']['selected_step'],queries=len(y),
                         changed=int((p!=q).sum()),rescued=int(rescued.sum()),harmed=int(harmed.sum()),
                         wrong_to_wrong=int(((p!=q)&(p!=y)&(q!=y)).sum()),
                         OA_on=a['OA'],OA_off=b['OA'],BA_on=a['BA'],BA_off=b['BA'],
                         OA_pp=100*(a['OA']-b['OA']),BA_pp=100*(a['BA']-b['BA']),
                         max_score_change=float(np.abs(full-off).max()),seconds=time.perf_counter()-start,
                         replay_error=float(np.abs(replay-full[:2]).max())))
        detail = data.table.iloc[data.te][['obs_id','mmsi','pass_group','class_id','class_name','up_index']].copy()
        detail['seed'],detail['on_prediction'],detail['off_prediction']=seed,p,q
        detail['rescued'],detail['harmed'],detail['changed']=rescued,harmed,p!=q
        for c in range(8):
            detail[f'on_score_{c}'],detail[f'off_score_{c}']=full[:,c],off[:,c]
        records.append(detail)
        print(rows[-1],flush=True)
        del model
    pd.DataFrame(rows).to_csv(run.HERE/'conv_switch_summary.csv',index=False)
    pd.concat(records).to_csv(run.HERE/'conv_switch_predictions.csv.gz',index=False,compression='gzip')
    (run.HERE/'conv_switch_receipt.json').write_text(json.dumps(dict(
        training=False,queries=len(y),seeds=list(tr.SEEDS),
        interpretation='Same selected model, only conv.output zeroed. Conditional model reliance, not matched retraining or natural physical intervention.',
        no_target_selection=True),indent=2))


if __name__ == '__main__':
    main()
