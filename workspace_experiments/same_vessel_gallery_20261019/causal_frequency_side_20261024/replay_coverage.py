
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import pandas as pd,numpy as np,json,itertools
R=Path('E:/临时会话/same_vessel_gallery_20261019')
f1=pd.read_csv(R/'condition_morphometry_20261021/measurements.csv',dtype={'mmsi':str})
f2=pd.read_csv(R/'multitemporal_20261021/measurements.csv',dtype={'mmsi':str})
shared=f1.merge(f2,on=['obs_id','pol'],suffixes=('_a','_b'))
for col in ['incidence','cos2phi','sin2phi','r3_length_m','r3_area_m2']:
 assert np.allclose(shared[col+'_a'],shared[col+'_b'],equal_nan=True),col
f=pd.concat([f1,f2]).drop_duplicates(['obs_id','pol']).copy()
f=f[f.obs_id.ne(31237)]
f['timestamp']=pd.to_datetime(f.time)
f['overpass']=f['product'].map(lambda x:'_'.join([x.split('_')[0],x.split('_')[6],x.split('_')[7]]))
print('UNION',json.dumps(dict(observations=int(f.obs_id.nunique()),ships=int(f.mmsi.nunique()),views=len(f),overlap_views=len(shared))),flush=True)
rows=[]
for (mmsi,pol),g in f.groupby(['mmsi','pol']):
 for a,b in itertools.combinations(g.sort_values(['timestamp','obs_id']).to_dict('records'),2):
  if a['overpass']==b['overpass']:continue
  delta=np.degrees(.5*np.arccos(np.clip(a['sin2phi']*b['sin2phi']+a['cos2phi']*b['cos2phi'],-1,1)))
  rows.append(dict(mmsi=mmsi,pol=pol,a=int(a['obs_id']),b=int(b['obs_id']),port=a['port'],
   sameport=a['port']==b['port'],sameplatform=a['platform']==b['platform'],samepass=a['pass_name']==b['pass_name'],
   days=(b['timestamp']-a['timestamp']).total_seconds()/86400,angle_gap=float(delta),
   incidence_gap=abs(a['incidence']-b['incidence']),valid=bool(a['r3_valid'] and b['r3_valid']),
   incidence_a=a['incidence'],incidence_b=b['incidence'],class_id=a['class_id'],
   time_a=a['time'],time_b=b['time'],product_a=a['product'],product_b=b['product'],
   image_vv_a=a['image_path'],image_vv_b=b['image_path']))
p=pd.DataFrame(rows)

from math import radians,sin,cos,asin,sqrt
good=p[p.sameport&p.sameplatform&p.samepass&(p.angle_gap<=10)&(p.incidence_gap>=2)&(p.days<=60)&p.valid]
geo=pd.read_csv(R/'radar_geometry/per_view_geometry.csv',usecols=['obs_id','pol','lat','lon','source_tiff','annotation_path','valid'])
geo=geo.set_index(['obs_id','pol'])
def dist(a,b):
 la,lb=radians(a.lat),radians(b.lat);dl=radians(b.lon-a.lon)
 return 6371*2*asin(min(1,sqrt(sin((lb-la)/2)**2+cos(la)*cos(lb)*sin(dl/2)**2)))
print('UNIQUE_PAIRS',good[['a','b']].drop_duplicates().shape[0],'UNIQUE_SHIPS',good.mmsi.nunique())
for r in good.itertuples():
 a,b=geo.loc[(r.a,r.pol)],geo.loc[(r.b,r.pol)]
 print(json.dumps(dict(pol=r.pol,a=r.a,b=r.b,mmsi=r.mmsi,port=r.port,
  days=round(r.days,2),angle_gap=round(r.angle_gap,2),incidence_gap=round(r.incidence_gap,2),
  location_distance_km=round(dist(a,b),2),lat_a=a.lat,lon_a=a.lon,lat_b=b.lat,lon_b=b.lon,
  source_tiffs_available=bool(Path(str(a.source_tiff)).is_file() and Path(str(b.source_tiff)).is_file()),
  annotations_available=bool(Path(str(a.annotation_path)).is_file() and Path(str(b.annotation_path)).is_file())),ensure_ascii=False),flush=True)
print('VIEW_PATHS',f[f.obs_id.isin([6351,6352,5207,5208])][['obs_id','pol','image_path']].to_json(orient='records',force_ascii=False))

SAVE=Path(__file__).resolve().parent/'round03_coverage'
SAVE.mkdir(parents=True,exist_ok=True)
p.to_csv(SAVE/'all_pairs.csv',index=False)
records=[]
for r in good.itertuples():
 a,b=geo.loc[(r.a,r.pol)],geo.loc[(r.b,r.pol)]
 records.append(dict(r._asdict(),location_distance_km=dist(a,b),
 source_tiffs_available=bool(Path(str(a.source_tiff)).is_file() and Path(str(b.source_tiff)).is_file()),
 annotations_available=bool(Path(str(a.annotation_path)).is_file() and Path(str(b.annotation_path)).is_file())))
pd.DataFrame(records).to_csv(SAVE/'eligible_pairs.csv',index=False)

