"""Describe source intervention headroom without using development labels."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def main():
    table=pd.read_csv(HERE.parent/'native_validation/table.csv',dtype={'mmsi':str})
    rows=[];coverage=[];flips=[]
    for seed in (17,29,43):
        z=np.load(HERE/'results'/f'source_interventions_{seed}.npz')
        ids,e,score=z['ids'],z['effect'],z['score']
        base=np.load(HERE.parent/'anchored_m3_20261025/results'/f'old_full_{seed}.npz')['score'][ids]
        f=table.iloc[ids].reset_index(drop=True);y=f.class_id.to_numpy(int)
        correct=base.argmax(1)==y;new=score.argmax(2)==y[:,None]
        old_margin=base[np.arange(len(y)),y]-np.where(np.arange(8)[None]==y[:,None],-np.inf,base).max(1)
        for role in ('utility_train','selection'):
            take=f.role.eq(role).to_numpy()
            for cls in (-1,*range(8)):
                subset=take if cls==-1 else take&(y==cls)
                if not subset.any():continue
                coverage.append(dict(seed=seed,role=role,class_id=cls,n=int(subset.sum()),
                    ships=f.loc[subset,'mmsi'].nunique(),passes=f.loc[subset,'pass_group'].nunique(),
                    old_wrong=int((subset&~correct).sum()),
                    wrong_within_001=int((subset&~correct&(old_margin>=-.01)).sum()),
                    wrong_within_01=int((subset&~correct&(old_margin>=-.1)).sum()),
                    single_rescuable=int((subset&~correct&new.any(1)).sum()),
                    single_harmable=int((subset&correct&(~new).any(1)).sum()),
                    effect_rms=float(np.sqrt(np.mean(e[subset]**2))),
                    maximum_abs_effect=float(abs(e[subset]).max())))
            for t in range(18):
                p,r,b=t//9,(t%9)//3,t%3
                rows.append(dict(seed=seed,role=role,type=t,polarization=('VV','VH')[p],
                    region=('body','sides','ends')[r],wavelength_px=(4,8,16)[b],
                    mean_effect=float(e[take,t].mean()),effect_rms=float(np.sqrt(np.mean(e[take,t]**2))),
                    positive_fraction=float((e[take,t]>1e-6).mean()),negative_fraction=float((e[take,t]<-1e-6).mean()),
                    rescued=int((take&~correct&new[:,t]).sum()),harmed=int((take&correct&~new[:,t]).sum())))
        for j,t in zip(*np.where(new!=correct[:,None])):
            flips.append(dict(seed=seed,obs_id=int(f.iloc[j].obs_id),role=f.iloc[j].role,
                class_id=int(y[j]),type=int(t),old_prediction=int(base[j].argmax()),
                new_prediction=int(score[j,t].argmax()),kind='rescue' if new[j,t] else 'harm',
                old_margin=float(old_margin[j]),fixed_rival_effect=float(e[j,t]),
                mmsi=f.iloc[j].mmsi,pass_group=f.iloc[j].pass_group))
    pd.DataFrame(rows).to_csv(HERE/'source_action_effects.csv',index=False)
    pd.DataFrame(coverage).to_csv(HERE/'source_action_coverage.csv',index=False)
    pd.DataFrame(flips).to_csv(HERE/'source_action_flips.csv',index=False)
    print(pd.DataFrame(coverage).query('class_id==-1').to_string(index=False))


if __name__=='__main__':main()
