"""Test readout-center dependence of the calibrated VH relationship."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from PIL import Image,ImageDraw,ImageFont
import raw_pair_probe as raw

ROOT=raw.ROOT
OUT=ROOT/'round17_center_sensitivity'
CAL=ROOT/'round16_calibrated_relation'


def world_point(p,xy):
    g=p['scale']['transform'];x,y=xy
    return np.array([g[0]+x*g[1]+y*g[2],g[3]+x*g[4]+y*g[5]])


def vh_from_vv(pv,ph):
    assert pv['scale']['status']==ph['scale']['status']=='valid_native_png_original_crop_grid'
    # Both views belong to the same product/crop CRS in this cohort; validated below.
    point=world_point(pv,pv['ship_axis']['center']);g=ph['scale']['transform']
    return np.linalg.solve(np.array([[g[1],g[2]],[g[4],g[5]]]),point-np.array([g[0],g[3]]))


def main():
    OUT.mkdir(exist_ok=True)
    c=pd.read_csv(raw.OUT/'cohort.csv',dtype={'mmsi':str})
    g=json.loads((ROOT.parent/'gallery_data.js').read_text(encoding='utf8').split('=',1)[1].strip().removesuffix(';'))['observations']
    views=pd.read_csv(raw.WORK/'chip_quality_20261017/ais_metadata/view_metadata.csv',low_memory=False).set_index(['instance_id','pol'])
    rows=[]
    for pair in c.itertuples():
        for oid in (pair.obs_id_a,pair.obs_id_b):
            o=g[str(oid)];ph=o['images']['vh']['priors'];pv=o['images']['vv']['priors']
            if not ph.get('ship_axis'):continue
            centers={'anchor':np.array(ph['anchor']),'vh_obb':np.array(ph['ship_axis']['center'])}
            if pv.get('ship_axis'):
                assert views.loc[(o['instance_id'],'vv'),'epsg']==views.loc[(o['instance_id'],'vh'),'epsg']
                centers['vv_obb_projected']=vh_from_vv(pv,ph)
            im=np.load(CAL/'private_windows'/f'{oid}_vh.npz')['sigma_noise_amp'];yy,xx=np.indices(im.shape)
            valid=(xx>=6)&(xx<im.shape[1]-6)&(yy>=6)&(yy<im.shape[0]-6)
            for name,center in centers.items():
                mask=((xx+.5-center[0])**2+(yy+.5-center[1])**2<=36)&valid
                usable=bool(mask.sum()>=100)
                z=raw.measure(im,mask) if usable else None
                rows.append(dict(obs_id=oid,mmsi=pair.mmsi,category=pair.class_name_a,center_kind=name,
                    x=center[0],y=center[1],anchor_offset_m=float(np.linalg.norm(center-centers['anchor'])*10),
                    cos2relative=float(np.cos(2*(ph['ship_axis']['angle']-ph['radar_axis']))),
                    usable=usable,n_pixels=int(mask.sum()),log_ratio=z['log_ratio'] if z else np.nan))
    f=pd.DataFrame(rows);f.to_csv(OUT/'endpoint_centers.csv',index=False)
    rows=[]
    for pair in c.itertuples():
        for kind,z in f[f.mmsi.eq(pair.mmsi)].groupby('center_kind'):
            z=z[z.usable]
            if len(z)!=2:continue
            z=z.set_index('obs_id');a,b=z.loc[pair.obs_id_a],z.loc[pair.obs_id_b]
            rows.append(dict(mmsi=pair.mmsi,category=pair.class_name_a,center_kind=kind,
                delta_cos=b.cos2relative-a.cos2relative,delta_ratio=b.log_ratio-a.log_ratio,
                delta_offset=b.anchor_offset_m-a.anchor_offset_m))
    pairs=pd.DataFrame(rows);pairs.to_csv(OUT/'paired_centers.csv',index=False)
    common=set(pairs[pairs.center_kind.eq('vv_obb_projected')].mmsi)
    summary=[]
    for scope,d in [('all_available',pairs),('common_vv_vh',pairs[pairs.mmsi.isin(common)])]:
        for kind,z in d.groupby('center_kind'):
            x,y=z.delta_cos.to_numpy(),z.delta_ratio.to_numpy()
            loo=[float(spearmanr(np.delete(x,i),np.delete(y,i)).statistic) for i in range(len(z))]
            base=d[d.center_kind.eq('anchor')].set_index('mmsi').loc[z.mmsi,'delta_ratio'].to_numpy()
            summary.append(dict(scope=scope,center_kind=kind,n=len(z),rho=float(spearmanr(x,y).statistic),
                loo_min=min(loo),loo_max=max(loo),median_abs_difference_from_anchor=float(np.median(abs(y-base))),
                offset_delta_rho=float(spearmanr(z.delta_offset,y).statistic) if z.delta_offset.std()>0 else None))
    result=pd.DataFrame(summary);result.to_csv(OUT/'summary.csv',index=False)
    calpairs=pd.read_csv(CAL/'paired_changes.csv',dtype={'mmsi':str})
    base=calpairs[calpairs.pol.eq('vh')&calpairs.region.eq('disk60')&calpairs.representation.eq('sigma_noise_amp')]
    corrected=[]
    for scope,z in [('common_geographic_overlap',base[base.mmsi.isin(common)]),
                    ('exclude_crosspol_mismatch',base[base.mmsi.ne('538003181')])]:
        for col in ('delta_cos','delta_vv_cos'):
            h=z.dropna(subset=[col,'delta_ratio'])
            corrected.append(dict(scope=scope,angle=col,n=len(h),rho=float(spearmanr(h[col],h.delta_ratio).statistic)))
    pd.DataFrame(corrected).to_csv(OUT/'geographically_corrected_associations.csv',index=False)
    mismatches=f[~f.usable].copy()
    mismatches.to_csv(OUT/'unusable_projection.csv',index=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    # Include largest actual offset and retained representative states, not chosen by rho.
    selected=[(25673,25675),(8368,8369),(25750,25752),(7324,7326),(25819,25820)]
    for a,b in selected:
        canvas=Image.new('RGB',(820,400),'white');draw=ImageDraw.Draw(canvas)
        vmax=max(np.load(CAL/'private_windows'/f'{oid}_vh.npz')['sigma_noise_amp'].max() for oid in (a,b))
        for i,oid in enumerate((a,b)):
            im=np.load(CAL/'private_windows'/f'{oid}_vh.npz')['sigma_noise_amp']
            tile=Image.fromarray((im/vmax*255).round().astype('uint8')).convert('RGB');td=ImageDraw.Draw(tile)
            y=30+i*180
            for j,r in enumerate(f[f.obs_id.eq(oid)].itertuples()):
                color={'anchor':(255,160,0),'vh_obb':(20,230,120),'vv_obb_projected':(70,170,255)}[r.center_kind]
                td.ellipse((r.x-6,r.y-6,r.x+6,r.y+6),outline=color)
                draw.text((200,y+j*30),f'{r.center_kind}: log ratio {r.log_ratio:+.4f}, offset {r.anchor_offset_m:.1f} m',fill=color,font=font)
            canvas.paste(tile,((180-tile.width)//2,y+20));draw.text((8,y),f'{oid} VH',font=font,fill='black')
        canvas.save(OUT/f'case_{a}_{b}.png')
    print(result.to_string(index=False))


if __name__=='__main__':main()
