"""Inspect a rescue in the class contributing most of the observed net gain."""
import analyze_pilot as a
import pandas as pd

f=pd.read_csv(a.ROOT/'cohort.csv',dtype={'mmsi':str}).set_index('obs_id')
p=pd.read_csv(a.ROOT/'predictions.csv',dtype={'mmsi':str})
b=p[p.arm.eq('image')].set_index('obs_id')
n=p[p.arm.eq('interaction')].set_index('obs_id')
eligible=b[b.class_id.eq(3)&~b.correct&n.correct].copy()
eligible['gain']=n.margin-b.margin
oid=int(eligible.sort_values('gain',ascending=False).index[0])
ids=[oid]
mates=b[b.mmsi.eq(b.loc[oid,'mmsi'])&b.index.to_series().ne(oid)]
if len(mates): ids.append(int(mates.sort_values('margin',ascending=False).index[0]))
classes=f[['class_id','class_name']].drop_duplicates().set_index('class_id').class_name.to_dict()
rows=a.render(ids,'main_gain_class_rescue',f,p,classes)
pd.DataFrame(rows).to_csv(a.ROOT/'review_gain_class.csv',index=False)
inter=pd.read_csv(a.ROOT/'fixed_condition_predictions.csv')
decomp=pd.read_csv(a.ROOT/'score_decomposition.csv')
print('CASE',ids)
print(f.loc[ids,['mmsi','class_name','incidence','axis_deg','los_axis_deg']].to_string())
print(inter[inter.obs_id.isin(ids)].to_string(index=False))
print(decomp[decomp.obs_id.isin(ids)].to_string(index=False))
