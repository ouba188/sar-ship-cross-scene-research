"""Trace class-mean gains and the preserved pilot counterexample to vessels."""
import json
import numpy as np
import pandas as pd
from expand import OUT,metadata,write


def main():
    d=metadata();plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    events=[];badscene=[];shiprows=[]
    for e in plans:
        q=np.array(e['query']);y=d.iloc[q].class_id.to_numpy(int)
        for budget in [16,64]:
            p=np.load(OUT/'predictions'/f"episode_{e['episode']:02d}_b{budget}.npz")
            for draw in range(5):
                for ci,cell in enumerate(['same','other']):
                    raw=p['probabilities'][draw,ci,0,1];candidate=p['probabilities'][draw,ci,1,1]
                    a=raw.argmax(1);b=candidate.argmax(1);ca=a==y;cb=b==y
                    sy=d.iloc[p['support_rows'][draw,ci]].class_id.to_numpy(int)
                    for j,row in enumerate(q):
                        shiprows.append(dict(cohort=e['cohort'],budget=budget,cell=cell,port=e['port'],
                            ship=d.iloc[row].mmsi,class_id=int(y[j]),old_correct=float(ca[j]),new_correct=float(cb[j])))
                    for j in np.flatnonzero(ca!=cb):
                        events.append(dict(episode=e['episode'],cohort=e['cohort'],port=e['port'],budget=budget,draw=draw,
                            cell=cell,row=int(q[j]),ship=d.iloc[q[j]].mmsi,class_id=int(y[j]),
                            outcome='rescue' if cb[j] else 'harm',old_prediction=int(a[j]),new_prediction=int(b[j]),
                            own_support_count=int(np.sum(sy==y[j])),old_true_probability=float(raw[j,y[j]]),
                            new_true_probability=float(candidate[j,y[j]])))
                    if e['pilot_episode']==5 and budget==64 and cell=='other':
                        for c in np.unique(y):
                            mask=y==c
                            badscene.append(dict(draw=draw,class_id=int(c),query_count=int(mask.sum()),support_count=int(np.sum(sy==c)),
                                old_correct=int(ca[mask].sum()),new_correct=int(cb[mask].sum()),
                                contribution_ba=100*(cb[mask].mean()-ca[mask].mean())/len(np.unique(y))))
    df=pd.DataFrame(events);df.to_csv(OUT/'class_mean_events.csv',index=False)
    pd.DataFrame(badscene).to_csv(OUT/'pilot_counterexample.csv',index=False)
    new=df[(df.cohort=='new_acquisition')&(df.budget==64)&(df.cell=='other')]
    counts=new.groupby(['class_id','outcome']).agg(events=('row','size'),observations=('row','nunique'),ships=('ship','nunique'))
    counts.to_csv(OUT/'class_mean_event_scope.csv')
    ss=pd.DataFrame(shiprows).groupby(['cohort','budget','cell','ship','class_id'])[['old_correct','new_correct']].mean().reset_index()
    s=[]
    for (cohort,b,cell),g in ss.groupby(['cohort','budget','cell']):
        byclass=g.groupby('class_id')[['old_correct','new_correct']].mean()
        s.append(dict(cohort=cohort,budget=b,cell=cell,ships=len(g),
            ship_oa_delta=100*float((g.new_correct-g.old_correct).mean()),
            ship_ba_delta=100*float((byclass.new_correct-byclass.old_correct).mean())))
    pd.DataFrame(s).to_csv(OUT/'class_mean_vessel_summary.csv',index=False)
    print('NEW B64 CLASS MEAN EVENTS\n',counts.to_string(),flush=True)
    print('PILOT COUNTEREXAMPLE\n',pd.DataFrame(badscene).groupby('class_id').mean().to_string(),flush=True)
    print('SHIP BALANCED\n',pd.DataFrame(s).query("cohort=='new_acquisition' and cell=='other'").to_string(index=False),flush=True)


if __name__=='__main__':main()
