"""Disentangle intensity clipping operators on fixed native DN and masks."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from PIL import Image, ImageDraw, ImageFont
import raw_pair_probe as raw

ROOT = raw.ROOT
OUT = ROOT / 'round15_clipping_factorial'


def split(v00, v10, v01, v11):
    a = .5*((v10-v00)+(v11-v01))
    b = .5*((v01-v00)+(v11-v10))
    assert abs(a+b-(v11-v00)) < 1e-12
    return a, b


def main():
    OUT.mkdir(exist_ok=True)
    cohort = pd.read_csv(raw.OUT/'cohort.csv', dtype={'mmsi':str})
    bounds = pd.read_csv(raw.OUT/'common_display_bounds.csv', dtype={'mmsi':str}).set_index(['mmsi','pol'])
    gallery = json.loads((ROOT.parent/'gallery_data.js').read_text(encoding='utf8').split('=',1)[1].strip().removesuffix(';'))['observations']
    rows, spatial, scene = [], [], []
    for pair in cohort.itertuples():
        for pol in ('vv','vh'):
            lo, hi = bounds.loc[(pair.mmsi,pol), ['low','high98']]
            for oid in (pair.obs_id_a,pair.obs_id_b):
                dn = np.load(raw.OUT/'private_windows'/f'{oid}_{pol}.npz')['dn']
                p = gallery[str(oid)]['images'][pol]['priors']
                reps = [dn, np.maximum(dn,lo), np.minimum(dn,hi), np.clip(dn,lo,hi)]
                masks = raw.masks(dn,p)
                for region,mask in masks.items():
                    v = [raw.measure(im,mask)['log_ratio'] for im in reps]
                    low, high = split(*v)
                    rows.append(dict(mmsi=pair.mmsi,obs_id=oid,pol=pol,region=region,
                        raw=v[0],full=v[3],lower_effect=low,upper_effect=high,total_effect=v[3]-v[0],
                        upper_fraction_inside_readout=float((dn[mask]>hi).mean())))
                if p.get('ship_axis'):
                    s=p['ship_axis']
                    body=raw.jp.masks_for(dn,s,s['length_px'],s['width_px'])['body']
                    capped=np.minimum(dn,hi)
                    reps2=[dn,np.where(body,capped,dn),np.where(body,dn,capped),capped]
                    for region,mask in masks.items():
                        v=[raw.measure(im,mask)['log_ratio'] for im in reps2]
                        inside,outside=split(*v)
                        spatial.append(dict(mmsi=pair.mmsi,obs_id=oid,pol=pol,region=region,
                            inside_effect=inside,outside_effect=outside,total_effect=v[3]-v[0]))
                scene.append(dict(mmsi=pair.mmsi,obs_id=oid,pol=pol,incidence=p.get('radar_incidence'),
                    ship_axis=p.get('ship_axis',{}).get('angle'),radar_axis=p.get('radar_axis')))
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'endpoint_factorial.csv',index=False)
    pd.DataFrame(spatial).to_csv(OUT/'spatial_factorial.csv',index=False)
    pd.DataFrame(scene).to_csv(OUT/'conditions.csv',index=False)
    changes=[]
    for pair in cohort.itertuples():
        for (pol,region),g in frame[frame.mmsi.eq(pair.mmsi)].groupby(['pol','region']):
            if len(g)!=2:continue
            g=g.set_index('obs_id');a,b=g.loc[pair.obs_id_a],g.loc[pair.obs_id_b]
            changes.append(dict(mmsi=pair.mmsi,pol=pol,region=region,delta_raw=b.raw-a.raw,
                delta_full=b.full-a.full,delta_lower=b.lower_effect-a.lower_effect,
                delta_upper=b.upper_effect-a.upper_effect))
    changes=pd.DataFrame(changes)
    changes.to_csv(OUT/'paired_factorial.csv',index=False)
    summary=[]
    for (pol,region),g in changes.groupby(['pol','region']):
        err=g.delta_full-g.delta_raw
        summary.append(dict(pol=pol,region=region,n=len(g),
            median_abs_lower=float(g.delta_lower.abs().median()),
            median_abs_upper=float(g.delta_upper.abs().median()),
            median_abs_total=float(err.abs().median()),
            upper_larger_count=int((g.delta_upper.abs()>g.delta_lower.abs()).sum()),
            raw_clip_sign_reversed=int((g.delta_raw*g.delta_full<0).sum())))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'summary.csv',index=False)
    spatial_summary=[]
    for (pol,region),g in pd.DataFrame(spatial).groupby(['pol','region']):
        spatial_summary.append(dict(pol=pol,region=region,views=len(g),
            median_abs_inside=float(g.inside_effect.abs().median()),
            median_abs_outside=float(g.outside_effect.abs().median()),
            inside_larger_count=int((g.inside_effect.abs()>g.outside_effect.abs()).sum())))
    pd.DataFrame(spatial_summary).to_csv(OUT/'spatial_summary.csv',index=False)
    # Post-hoc association check: same measured angles, change only radiometric readout.
    paired=pd.read_csv(raw.OUT/'paired_changes.csv',dtype={'mmsi':str})
    associations=[]
    for (pol,rep),g in paired[paired.region.eq('anchor_disk60m') & paired.representation.isin(['dn','clip98','historical_png'])].groupby(['pol','representation']):
        g=g.dropna(subset=['delta_cos2relative','delta_log_ratio'])
        rho=float(spearmanr(g.delta_cos2relative,g.delta_log_ratio).statistic)
        leave=[]
        for i in g.index:
            h=g.drop(index=i)
            leave.append(float(spearmanr(h.delta_cos2relative,h.delta_log_ratio).statistic))
        associations.append(dict(pol=pol,representation=rep,n=len(g),rho=rho,
            leave_one_ship_out_min=min(leave),leave_one_ship_out_max=max(leave),
            median_abs_incidence_delta=float(g.delta_incidence.abs().median()),
            max_abs_incidence_delta=float(g.delta_incidence.abs().max()),
            status='post_hoc_observational_not_causal_angle_effect'))
    pd.DataFrame(associations).to_csv(OUT/'angle_associations_exploratory.csv',index=False)
    # Representative pre-existing states: wrong/correct, correct/wrong, both correct, both wrong.
    chosen=[(8368,8369),(25750,25752),(28337,28338),(7324,7326)]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for a,b in chosen:
        pair=cohort[cohort.obs_id_a.eq(a)].iloc[0]
        canvas=Image.new('RGB',(1200,720),'white');draw=ImageDraw.Draw(canvas)
        for ri,(oid,pol) in enumerate([(a,'vv'),(b,'vv'),(a,'vh'),(b,'vh')]):
            dn=np.load(raw.OUT/'private_windows'/f'{oid}_{pol}.npz')['dn']
            p=gallery[str(oid)]['images'][pol]['priors']
            lo,hi=bounds.loc[(pair.mmsi,pol),['low','high98']]
            # All displays here use one maximum, so the capped peaks visibly flatten.
            pooled=[np.load(raw.OUT/'private_windows'/f'{i}_{pol}.npz')['dn'].max() for i in (a,b)]
            vmax=max(pooled)
            cap=np.minimum(dn,hi)
            removed=np.maximum(dn-hi,0)
            images=[dn/vmax,cap/vmax,removed/vmax]
            for ci,im in enumerate(images):
                tile=Image.fromarray((np.clip(im,0,1)*255).round().astype('uint8')).convert('RGB')
                td=ImageDraw.Draw(tile);cx,cy=p['anchor']
                td.ellipse((cx-6,cy-6,cx+6,cy+6),outline=(255,160,0))
                canvas.paste(tile,(ci*200+(200-tile.width)//2,ri*180+25))
                draw.text((ci*200+4,ri*180+3),f'{oid} {pol} '+['DN','upper capped','removed peaks'][ci],font=font,fill='black')
            z=frame[(frame.obs_id==oid)&frame.pol.eq(pol)&frame.region.eq('anchor_disk60m')].iloc[0]
            draw.text((610,ri*180+10),f'log ratio: DN {z.raw:+.4f} -> clipped {z.full:+.4f}',font=font,fill='black')
            draw.text((610,ri*180+36),f'lower {z.lower_effect:+.4f}; upper {z.upper_effect:+.4f}',font=font,fill='black')
            draw.text((610,ri*180+62),f'pixels above upper limit in disk {z.upper_fraction_inside_readout:.1%}',font=font,fill='black')
        canvas.save(OUT/f'case_{a}_{b}.png')
    print(summary.to_string(index=False))
    print(pd.DataFrame(spatial_summary).to_string(index=False))


if __name__=='__main__':main()
