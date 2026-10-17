"""Fixed-task paired accounting; subgroup contributions retain full-task weights."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'analysis'
MET=['oa','ba','macro_nll','macro_brier']
UNIT=['seed','port','origin','acquisition','fold','budget','draw']
KEY=UNIT+['sample_id']
P=[f'p{i}' for i in range(8)]


def aggregate(frame,values,extra=()):
    x=frame.groupby([*extra,'seed','port','origin','acquisition'])[values].mean()
    return x.groupby([*extra,'seed','port','origin']).mean()


def md(frame):
    lines=['| '+' | '.join(frame.columns)+' |','|'+'---|'*len(frame.columns)]
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    met=pd.concat([pd.read_csv(p) for p in (HERE/'evaluation').glob('*/*_metrics.csv')],ignore_index=True)
    pred=pd.concat([pd.read_csv(p,dtype={'mmsi':str}) for p in (HERE/'evaluation').glob('*/*_predictions.csv')],ignore_index=True)
    arms={'B','strict_old','strict_input','strict_new','optional_old','optional_input','optional_new'}
    counts=met.loc[met.scope=='full'].groupby(['arm','seed']).size()
    assert set(counts.index)=={(a,s) for a in arms for s in range(3)}
    assert counts.eq(190).all()
    assert not pred.duplicated(KEY+['arm']).any()
    full=met.loc[met.scope=='full']
    port=aggregate(full,MET,['arm']).reset_index()
    seed=port.groupby(['arm','seed','origin'])[MET].mean().reset_index()
    summary=seed.groupby(['origin','arm'])[MET].agg(['mean','std'])
    port.to_csv(OUT/'per_port.csv',index=False)
    seed.to_csv(OUT/'seed_means.csv',index=False)
    summary.to_csv(OUT/'summary.csv')
    draw=aggregate(full,MET,['arm','draw']).groupby(['arm','draw','seed','origin']).mean().reset_index()
    draw.to_csv(OUT/'support_draw_means.csv',index=False)
    events=[]; contributions=[]; event_summary=[]; contrasts=[]
    for family in ('strict','optional'):
        old=pred.loc[pred.arm==family+'_old']
        new=pred.loc[pred.arm==family+'_new']
        joined=new.merge(old[KEY+['pred','correct',*P]],on=KEY,suffixes=('','_old'),validate='one_to_one')
        assert len(joined)==len(new)==len(old)
        joined['family']=family
        joined['delta_correct']=joined.correct.astype(int)-joined.correct_old.astype(int)
        joined['rescue']=joined.delta_correct.eq(1); joined['harm']=joined.delta_correct.eq(-1)
        joined['group']=np.where(joined.old_enabled,'common',np.where(joined.new_enabled,'newly_enabled','inactive'))
        events.append(joined)
        for keys,g in joined.groupby(['seed','port','origin']):
            changed=g.loc[g.rescue|g.harm]
            event_summary.append(dict(family=family,**dict(zip(['seed','port','origin'],keys)),
                rescues=int(g.rescue.sum()),harms=int(g.harm.sum()),changed_samples=changed.sample_id.nunique(),
                changed_ships=changed.mmsi.nunique(),rescue_ships=g.loc[g.rescue,'mmsi'].nunique(),
                harm_ships=g.loc[g.harm,'mmsi'].nunique()))
        for keys,g in joined.groupby(UNIT):
            n=len(g); class_n=g.truth.value_counts(); nclasses=len(class_n)
            for group in ('common','newly_enabled','inactive'):
                for cls in range(8):
                    part=g.loc[(g.group==group)&(g.truth==cls)]
                    change=int(part.delta_correct.sum())
                    contributions.append(dict(family=family,**dict(zip(UNIT,keys)),group=group,truth=cls,
                        delta_oa=100*change/n,
                        delta_ba=100*change/(nclasses*class_n[cls]) if cls in class_n else 0.,
                        n=len(part),rescues=int(part.rescue.sum()),harms=int(part.harm.sum())))
        for kind,ref in [('input','old'),('new','input'),('new','old')]:
            left=seed.loc[seed.arm==family+'_'+kind]
            right=seed.loc[seed.arm==family+'_'+ref]
            m=left.merge(right,on=['seed','origin'],suffixes=('','_ref'),validate='one_to_one')
            for metric in MET:
                m['delta_'+metric]=m[metric]-m[metric+'_ref']
            m['comparison']=family+'_'+kind+' - '+ref
            contrasts.append(m[['comparison','seed','origin']+['delta_'+c for c in MET]])
        input_pred=pred.loc[pred.arm==family+'_input']
        m=new.merge(input_pred[KEY+P],on=KEY,suffixes=('','_input'),validate='one_to_one')
        np.testing.assert_array_equal(m[P].to_numpy(),m[[c+'_input' for c in P]].to_numpy())
    events=pd.concat(events,ignore_index=True)
    con=pd.DataFrame(contributions)
    con.to_csv(OUT/'task_class_group_contributions.csv',index=False)
    totals=con.groupby(['family',*UNIT])[['delta_oa','delta_ba']].sum().reset_index()
    measured=aggregate(totals,['delta_oa','delta_ba'],['family']).reset_index()
    for row in measured.itertuples():
        a=port.loc[(port.arm==row.family+'_new')&(port.seed==row.seed)&(port.port==row.port)&(port.origin==row.origin)].iloc[0]
        b=port.loc[(port.arm==row.family+'_old')&(port.seed==row.seed)&(port.port==row.port)&(port.origin==row.origin)].iloc[0]
        assert abs(row.delta_ba-(a.ba-b.ba))<1e-9
        assert abs(row.delta_oa-(a.oa-b.oa))<1e-9
    group_task=con.groupby(['family',*UNIT,'group'])[['delta_oa','delta_ba']].sum().reset_index()
    group_port=aggregate(group_task,['delta_oa','delta_ba'],['family','group']).reset_index()
    group_port.to_csv(OUT/'group_contributions.csv',index=False)
    class_task=con.groupby(['family',*UNIT,'truth'])[['delta_oa','delta_ba']].sum().reset_index()
    aggregate(class_task,['delta_oa','delta_ba'],['family','truth']).reset_index().to_csv(OUT/'class_contributions.csv',index=False)
    changed=events.loc[events.rescue|events.harm]
    changed.to_csv(OUT/'changed_cases.csv',index=False)
    pd.DataFrame(event_summary).to_csv(OUT/'event_counts.csv',index=False)
    contrasts=pd.concat(contrasts,ignore_index=True)
    contrasts.to_csv(OUT/'paired_contrasts.csv',index=False)
    replay=json.loads((HERE/'REPLAY_CHECK.json').read_text())
    assert replay['max_error']<1e-10
    checks=dict(tasks_per_arm_seed=190,prediction_rows=len(pred),historical_replays=len(replay['checks']),
        max_replay_error=replay['max_error'],full_task_class_group_accounting=True,new_input_equals_refit=True)
    (OUT/'RESULT_CHECK.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    overview=seed.groupby(['origin','arm'])[MET].mean().reset_index()
    independent=[]
    for (family,origin),g in changed.groupby(['family','origin']):
        independent.append(dict(family=family,origin=origin,changed_samples=g.sample_id.nunique(),
                                changed_ships=g.mmsi.nunique(),rescue_events=int(g.rescue.sum()),harm_events=int(g.harm.sum())))
    independent=pd.DataFrame(independent)
    independent.to_csv(OUT/'independent_changed_counts.csv',index=False)
    lines=['# 恢复几何后的配对贡献汇总','','固定五留港研发任务，S32、3种子、5支持抽样。全体任务是主口径，不把有效子集BA冒充总体BA。',
        '',md(overview),'','## 相对自身旧版本的独立改判样本与重复事件','',md(independent),
        '','事件包含种子/支持抽样重复，不能当独立船数。详细变化见 changed_cases.csv。',
        '','## 配对差值','',md(contrasts.groupby(['origin','comparison'])[['delta_oa','delta_ba','delta_macro_nll','delta_macro_brier']].mean().reset_index()),
        '','class_contributions.csv 和 group_contributions.csv 保持全任务的类别与样本分母，所有类别/组贡献之和已与主表差值核对。',
        '','新增数据作用范围、参数一致性与固定新覆盖的角度屏蔽诊断见 ../FINDINGS_ZH.md 和 ../diagnostics/。',
        '','## 检查','',json.dumps(checks,ensure_ascii=False)]
    (OUT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(independent.to_string(index=False)); print(json.dumps(checks))
    print(contrasts.loc[contrasts.origin=='cross'].to_string(index=False))


if __name__=='__main__':
    main()
