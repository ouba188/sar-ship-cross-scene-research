"""Source-only check of executable single-action headroom before refitting gates."""
import train as run
import numpy as np
import pandas as pd
import torch


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    data=run.Data(); reader=run.Reader(data,17)
    ids=np.r_[data.tr,data.va]
    y=data.table.iloc[ids].class_id.to_numpy(int)
    b=reader.baseline[ids]; correct=b.argmax(1)==y
    rows=[]
    for dose in (1.,):
        run.DOSE=dose
        e,s=reader.teaching(ids,return_scores=True)
        np.savez_compressed(run.HERE/'source_full_endpoint.npz',ids=ids,effect=e,score=s)
        for role in ('utility_train','selection'):
            take=data.table.iloc[ids].role.eq(role).to_numpy()
            good=s.argmax(2)==y[:,None]
            pick=e.argmax(1); execute=e.max(1)>0
            oracle=b.copy(); oracle[execute]=s[np.flatnonzero(execute),pick[execute]]
            rows.append(dict(dose=dose,role=role,n=int(take.sum()),effect_rms=float(np.sqrt(np.mean(e[take]**2))),
                max_effect=float(abs(e[take]).max()),
                single_rescuable=int((~correct&good.any(1)&take).sum()),
                single_harmable=int((correct&(~good).any(1)&take).sum()),
                old_OA=float(correct[take].mean()),
                label_oracle_OA=float((oracle[take].argmax(1)==y[take]).mean()),
                oracle_note='Posthoc source-label choice of one positive fixed-rival-margin action; not deployable, not combination bound.'))
    pd.DataFrame(rows).to_csv(run.HERE/'source_endpoint_probe.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
