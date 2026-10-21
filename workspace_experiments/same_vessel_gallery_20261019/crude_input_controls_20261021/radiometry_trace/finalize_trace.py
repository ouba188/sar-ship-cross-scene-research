"""Summarize only existing trace outputs and source metadata; no model or remote IO."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from trace_radiometry import OUT, ROOT, GALLERY, FOCUS, read_csv, save_csv, save_json, polygon


def metadata_and_code():
    views = read_csv(OUT/'view_trace.csv')
    inventory = read_csv(ROOT/'visual_reliable_baseline/artifacts/safe841_metadata_20260917/metadata_inventory.csv')
    inventory = inventory[inventory['product'].isin(views['product'])].copy()
    rows = []
    for r in inventory.to_dict('records'):
        p = Path(r['metadata_dir'])
        measurements = list((p/'measurement').glob('*.tif*'))
        r.update(directory_exists=p.is_dir(), measurement_files=len(measurements),
                 measurement_bytes=sum(x.stat().st_size for x in measurements),
                 calibration_xml_present=len(list((p/'annotation/calibration').glob('calibration*.xml'))))
        rows.append(r)
    save_csv('safe_metadata_coverage.csv', rows)
    json_rows = []
    rx = re.compile(r'stretch|percentile|scale_min|scale_max|radiometr|sigma0|calibrat',re.I)
    def keys(value, prefix=''):
        if isinstance(value,dict):
            for k,v in value.items():
                full = prefix+'.'+k
                if rx.search(k):
                    yield full
                yield from keys(v,full)
        elif isinstance(value,list):
            for x in value:
                yield from keys(x,prefix+'[]')
    for p in sorted(set(views.object_local_json)):
        file = Path(p)
        obj = json.loads(file.read_text(encoding='utf-8-sig'))
        json_rows.append(dict(path=p,root_keys=';'.join(obj),radiometry_key_matches=';'.join(sorted(set(keys(obj))))))
    save_csv('source_json_radiometry_search.csv',json_rows)
    sources = [
        (ROOT/'asf_ais_first_screen_20260913/server_zip_pipeline.sh',[(4,8),(54,74)]),
        (ROOT/'asf_ais_first_screen_20260913/server2_zip_pipeline.sh',[(4,11),(76,96)]),
        (ROOT/'v5_server_toolkit_20260912/scripts/sentinel1_geocode_manifest.py',[(84,128),(148,197)]),
        (ROOT/'v5_server_toolkit_20260912/scripts/tile_sarlabelimg_quadrants.py',[(233,254),(318,322)]),
        (ROOT/'build_v5_context_chips.py',[(265,284)]),
        (ROOT/'build_v5_context_chips_gdal.py',[(88,120),(169,184)]),
    ]
    out = ['# Source Code Evidence', '', 'Read-only excerpts of local source files. These document the stored pipeline; no remote runtime version or per-scene stretch limits are claimed.', '']
    for path,ranges in sources:
        lines=path.read_text(encoding='utf-8-sig').splitlines()
        out += ['## '+str(path),'']
        for start,end in ranges:
            out += ['```text']+[f'{n}: {lines[n-1]}' for n in range(start,min(end,len(lines))+1)]+['```','']
    (OUT/'CODE_EVIDENCE.md').write_text('\n'.join(out),encoding='utf-8')


def coverage():
    views=read_csv(OUT/'view_trace.csv')
    checks=read_csv(OUT/'source_pixel_comparisons.csv')
    cohort=read_csv(OUT/'endpoint_cohort.csv')
    pairs=read_csv(GALLERY/'condition_outcomes_20261021/crude_pair_details.csv')
    rows=[]
    for v in views.itertuples():
        c=checks[checks.obs_id.eq(v.obs_id)&checks.pol.eq(v.pol)]
        display=c[c.type.eq('geocoded_uint8_window')]
        raw=c[c.type.eq('raw_SAFE_GCP_nearest')]
        exact=bool(display.exact_pixels.eq(True).any())
        state=('uint8_TIFF_exact_pixels' if exact else 'uint8_TIFF_sparse_mismatch' if len(display)
               else 'uint16_SAFE_GCP_window_candidate' if len(raw) else 'PNG_and_metadata_only')
        pipeline=('server_zip_pipeline.sh' if '/sar_download_v1/' in v.source_tiff else
                  'server2_zip_pipeline.sh' if '/asf_download_v2_' in v.source_tiff else '')
        rows.append(dict(obs_id=v.obs_id,pol=v.pol,instance_id=v.instance_id,product=v.product,port=v.port,
            image_path=v.image_path,original_source_tiff=v.source_tiff,source_crop_left=v.crop_left,
            source_crop_top=v.crop_top,source_crop_width=int(v.crop_right-v.crop_left),
            source_crop_height=int(v.crop_bottom-v.crop_top),source_window_manifest_known=True,
            png_dtype=v.png_dtype,png_alias_pixel_verified=True,state=state,
            local_display_window_read=bool(len(display)),display_exact_pixels=exact,
            local_raw_measurement_present=bool(len(raw)),raw_window_read=bool(len(raw)),
            raw_original_warp_exactly_replayed=False,prestretch_float_chip_found=False,
            calibrated_backscatter_recovered=False,actual_stretch_limits_recovered=False,
            stored_pipeline=pipeline,stretch_scope='scene_per_polarization_before_tiling',
            configured_clip_percent_per_tail=2 if pipeline else '',
            actual_scene_low='',actual_scene_high='',
            sources=';'.join(sorted(set(c.path))),raw_rank_rho=raw.iloc[0].rank_rho if len(raw) else '',
            evidence_limit=('Raw UInt16 DN only; original warp and scene limits unresolved' if len(raw) else
                            'Already stretched uint8; no physical radiometry' if len(display) else
                            'Remote source current state unavailable; metadata is not measurement data')))
    vc=pd.DataFrame(rows)
    save_csv('view_coverage_400.csv',vc)
    endpoint=[]
    for c in cohort.itertuples():
        v=vc[vc.obs_id.eq(c.obs_id)]
        endpoint.append(dict(obs_id=c.obs_id,instance_id=c.instance_id,mmsi=c.mmsi,port=c.port,product=c.product,
            png_views=len(v),crop_manifest_views=int(v.source_window_manifest_known.sum()),
            display_read_views=int(v.local_display_window_read.sum()),display_exact_views=int(v.display_exact_pixels.sum()),
            raw_DN_candidate_views=int(v.raw_window_read.sum()),float_chip_views=0,calibrated_views=0,
            actual_stretch_limits_views=0,state=';'.join(sorted(set(v.state)))))
    ep=pd.DataFrame(endpoint).set_index('obs_id',drop=False)
    save_csv('endpoint_coverage_200.csv',ep)
    pair_rows=[]
    for p in pairs.itertuples():
        a,b=ep.loc[p.obs_id_a],ep.loc[p.obs_id_b]
        pair_rows.append(dict(pair_id=p.pair_id,mmsi=p.mmsi,obs_id_a=p.obs_id_a,obs_id_b=p.obs_id_b,
            state_a=a.state,state_b=b.state,display_exact_views=int(a.display_exact_views+b.display_exact_views),
            raw_DN_candidate_views=int(a.raw_DN_candidate_views+b.raw_DN_candidate_views),
            both_endpoints_raw_DN_candidates=bool(a.raw_DN_candidate_views==2 and b.raw_DN_candidate_views==2),
            complete_prestretch_verified_pair=False))
    save_csv('pair_coverage_100.csv',pair_rows)
    product_rows=[]
    for product,v in vc.groupby('product'):
        product_rows.append(dict(product=product,ports=';'.join(sorted(set(v.port))),endpoints=v.obs_id.nunique(),views=len(v),
            display_read_views=int(v.local_display_window_read.sum()),display_exact_views=int(v.display_exact_pixels.sum()),
            raw_DN_candidate_views=int(v.raw_window_read.sum()),actual_stretch_limits_known=False))
    save_csv('product_coverage_43.csv',product_rows)
    stats=read_csv(OUT/'png_statistics.csv')
    focus=stats[stats.obs_id.isin(FOCUS)].pivot(index=['obs_id','pol'],columns='region',values=['min','max','mean','std','at255_fraction'])
    focus.columns=['_'.join(c) for c in focus.columns]
    focus=focus.reset_index()
    focus['target_minus_rest']=focus.mean_target-focus.mean_rest
    focus['source_status']='PNG_and_metadata_only'
    save_csv('focus_statistics.csv',focus)
    raw=checks[checks.type.eq('raw_SAFE_GCP_nearest')]
    summary=dict(pairs=100,endpoints=200,views=400,products=43,ports=7,
        view_states=vc.state.value_counts().to_dict(),endpoint_states=ep.state.value_counts().to_dict(),
        crop_manifest_views=400,png_alias_exact_views=400,
        local_display_views=int(vc.local_display_window_read.sum()),local_display_exact_views=int(vc.display_exact_pixels.sum()),
        local_raw_uint16_candidate_views=int(vc.raw_window_read.sum()),
        prestretch_float_views=0,calibrated_views=0,actual_stretch_limits_views=0,
        pair_both_raw_candidates=sum(x['both_endpoints_raw_DN_candidates'] for x in pair_rows),
        complete_verified_prestretch_pairs=0,
        raw_rank_rho_min=float(raw.rank_rho.min()),raw_rank_rho_max=float(raw.rank_rho.max()),
        raw_requested_pixels=int(raw.source_requested_pixels.sum()),
        raw_max_window_width=int(raw.width.max()),raw_max_window_height=int(raw.height.max()),
        raw_estimated_strip_decode_bytes=int(raw.estimated_strip_decode_bytes.sum()),
        focus_raw_views=0,focus_display_source_views=0,
        pipeline_mapped_views=int(vc.stored_pipeline.ne('').sum()),
        classifier_runs=0,network_data_download_bytes=0,whole_scene_decodes=0,
        intervention_scope='Main task +/-0.05 in PNG target/rest is a display-domain functional diagnostic, not physical radiometry.')
    save_json('SUMMARY.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def figures_and_mismatch():
    views=read_csv(OUT/'view_trace.csv').set_index(['obs_id','pol'])
    stats=read_csv(OUT/'png_statistics.csv').set_index(['obs_id','pol','region'])
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    panel=Image.new('RGB',(1120,750),'white');draw=ImageDraw.Draw(panel)
    draw.text((18,12),'Original uint8 PNG; fixed 0..255 display; nearest 3x; no per-chip contrast stretch',fill='black',font=font)
    for j,oid in enumerate([32042,32043,30233,30237]):
        for i,pol in enumerate(['vv','vh']):
            r=views.loc[(oid,pol)];im=Image.open(r.image_path).convert('L')
            x,y=15+j*280,55+i*340
            draw.text((x,y),f'{oid} {pol.upper()} | {im.width}x{im.height}',fill='black',font=font)
            panel.paste(im.resize((im.width*3,im.height*3),Image.Resampling.NEAREST).convert('RGB'),(x,y+28))
            t=stats.loc[(oid,pol,'target')];rest=stats.loc[(oid,pol,'rest')]
            draw.text((x,y+275),f'T={t["mean"]:.2f}, R={rest["mean"]:.2f}',fill='black',font=small)
            draw.text((x,y+296),f'T-R={t["mean"]-rest["mean"]:.2f}; T@255={t.at255_fraction:.1%}',fill='black',font=small)
    panel.save(OUT/'focus_native_pixels.png')
    # Diagnostic examples fixed by highest/lowest rank correspondence and exact
    # uint8 case; the array display below explicitly uses newly stated limits.
    panel=Image.new('RGB',(960,640),'white');draw=ImageDraw.Draw(panel)
    draw.text((15,10),'SAFE DN candidate vs PNG | diagnostic display only; no calibration',fill='black',font=font)
    for j,oid in enumerate([21799,32937]):
        for i,pol in enumerate(['vv','vh']):
            x,y=15+j*480,50+i*290
            r=views.loc[(oid,pol)];png=np.asarray(Image.open(r.image_path))
            raw=np.load(OUT/'recovered_windows'/f'{oid}_{pol}_dn_nearest.npy')
            lo,hi=np.quantile(raw,[.02,.98])
            display=np.clip((raw.astype(float)-lo)/(hi-lo)*255,0,255).astype(np.uint8)
            draw.text((x,y),f'{oid} {pol.upper()} PNG / mapped UInt16 DN',fill='black',font=font)
            for k,a in enumerate([png,display]):
                panel.paste(Image.fromarray(a).resize((a.shape[1]*3,a.shape[0]*3),Image.Resampling.NEAREST).convert('RGB'),(x+230*k,y+30))
            draw.text((x,y+230),f'DN shown using NEW local p02={lo:.1f}, p98={hi:.1f}',fill='black',font=small)
            draw.text((x,y+251),'These are NOT historical scene stretch limits.',fill='black',font=small)
    panel.save(OUT/'raw_candidate_pixels.png')
    snapshot={x['obs_id']:x for x in json.loads((GALLERY/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if x['obs_id']==27910}
    differences=[]
    for pol in ['vv','vh']:
        r=views.loc[(27910,pol)];png=np.asarray(Image.open(r.image_path))
        source=np.load(OUT/'recovered_windows'/f'27910_{pol}_uint8.npy')
        pr=snapshot[27910]['images'][pol]['priors']
        t=polygon((png.shape[1],png.shape[0]),pr['detections'][pr['target_index']]['corners'])
        other=np.zeros_like(t)
        for j,d in enumerate(pr['detections']):
            if j!=pr['target_index'] and d.get('shape_usable') and d['confidence']>=.25:
                other|=polygon((png.shape[1],png.shape[0]),d['corners'])
        for y,x in np.argwhere(png!=source):
            differences.append(dict(obs_id=27910,pol=pol,x=int(x),y=int(y),png=int(png[y,x]),
                cached_uint8_TIFF=int(source[y,x]),region='target' if t[y,x] else 'other' if other[y,x] else 'rest',
                cause='unresolved; both overlapping cached TIFF windows agree, remote original inaccessible'))
    save_csv('sparse_pixel_mismatches.csv',differences)


if __name__=='__main__':
    metadata_and_code()
    coverage()
    figures_and_mismatch()
