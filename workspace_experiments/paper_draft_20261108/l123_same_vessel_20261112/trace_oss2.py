"""Trace fixed M3 rescue records to independent GRD repeat observations."""
from pathlib import Path
import json
from html import escape
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
WORK=OUT.parents[1]
CF=OUT.parent/'m3_counterfactual_20261112'
RAW=Path('D:/Documents/project_root/project_data/public_opensarship2/opensarship2/metadata/opensarship2_manifest_native.csv')


def pass_group(scene):
    fields=scene.split('_')
    return fields[0]+'_'+fields[6]+'_'+fields[7]


def main():
    private=OUT/'private'; private.mkdir(exist_ok=True)
    gallery=OUT/'oss2_gallery';gallery.mkdir(exist_ok=True)
    rows=json.loads((WORK/'m1_coverage_20261101/manifest.json').read_text(encoding='utf-8'))
    events=pd.read_csv(CF/'private/events.csv')
    raw=pd.read_csv(RAW,dtype={'mmsi_or_ship_id':str,'imo':str})
    raw=raw[raw.scene_id.str.contains('_IW_GRDH_')].copy()
    raw['pass_group']=raw.scene_id.map(pass_group)
    result=[]
    for e in events[events.kind=='rescued'].to_dict('records'):
        q=rows[e['index']]
        same=raw[raw.mmsi_or_ship_id==q['mmsi']]
        query=same[same.sample_id==q['id']].iloc[0]
        other=same[same.pass_group!=q['pass_group']]
        for r in other.to_dict('records'):
            d=dict(query_id=q['id'],seed=e['seed'],mmsi=q['mmsi'],query_scene=q['scene_id'],
                   partner_id=r['sample_id'],partner_scene=r['scene_id'],ship_name=r['ship_name'],
                   same_name=str(query.ship_name).strip()==str(r['ship_name']).strip(),
                   same_imo=(str(query.imo)==str(r['imo']) if str(query.imo) not in ('0','nan','') and str(r['imo']) not in ('0','nan','') else None),query_imo=query.imo,partner_imo=r['imo'],
                   same_size=query.length_m==r['length_m'] and query.width_m==r['width_m'],
                   no_m3_margin=e['no_m3'],factual_margin=e['factual'],simple_mean_margin=e['simple_mean'],
                   query_vv=query.chip_path_vv,query_vh=query.chip_path_vh,
                   partner_vv=r['chip_path_vv'],partner_vh=r['chip_path_vh'],
                   partner_has_existing_prediction=r['sample_id'] in {a['id'] for a in rows})
            d['all_raw_images_exist']=all(Path(d[k]).is_file() for k in ('query_vv','query_vh','partner_vv','partner_vh'))
            result.append(d)
    pd.DataFrame(result).to_csv(private/'oss2_grd_repeat_candidates.csv',index=False,encoding='utf-8-sig')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    sections=[]
    for d in result:
        im=Image.new('RGB',(1000,500),'white');dr=ImageDraw.Draw(im)
        dr.text((14,10),f"{d['query_id']} / seed {d['seed']} | MMSI {d['mmsi']} | {d['ship_name']}",font=font,fill='black')
        dr.text((14,40),f"M3 margin {d['no_m3_margin']:+.4f} -> {d['factual_margin']:+.4f} | partner prediction not yet available",font=font,fill='black')
        for i,key in enumerate(('query_vv','query_vh','partner_vv','partner_vh')):
            tile=Image.open(d[key]).convert('RGB')
            assert max(tile.size)<=180
            x=15+i*245
            dr.text((x,85),key,font=font,fill='black')
            dr.text((x,112),f'{tile.width} x {tile.height} native pixels',font=font,fill='black')
            # Uniform 2x nearest-neighbor display, no per-image normalization.
            tile=tile.resize((tile.width*2,tile.height*2),Image.Resampling.NEAREST)
            im.paste(tile,(x,155))
        name=d['query_id'].split('_')[-1]+'_'+d['partner_id'].split('_')[-1]+'.png'
        im.save(gallery/name)
        sections.append(f'<section><h2>{escape(d["query_id"])} / seed{d["seed"]}</h2><p>{escape(d["ship_name"])}：{escape(d["partner_id"])} 是另一次GRD过境观测。伙伴当前无本轮预测，也尚无此查询的配套L2输入干预记录。</p><img src="oss2_gallery/{name}" alt="同船两次观测的原始VV/VH"></section>')
    summary=dict(rescue_events=23,grd_repeat_vessels=len({d['mmsi'] for d in result}),
                 candidate_pairs=len(result),raw_manifest=str(RAW),
                 note='GRD only; one identical MMSI is a candidate, metadata and images must also be checked. No model run, no inferred partner labels/predictions.')
    (OUT/'oss2_search_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    page='<!doctype html><meta charset="utf-8"><title>OSS2 M3救回同船追溯</title><style>body{max-width:1100px;margin:35px auto;padding:0 20px;font:16px/1.8 system-ui;color:#25323a}section{border-top:1px solid #ccd4d9;margin:32px 0}img{max-width:100%;height:auto}h1{font-size:28px}h2{font-size:20px}</style><h1>从M3救回反查同船跨场景观测</h1><p>23个救回对象中找到6艘船、7对跨过境GRD原图。本页不是已完成的三层证据链：伙伴预测与输入干预仍需补充。图像统一2倍显示，无灰度拉伸。<a href="OSS2_REVIEW_ZH.md">查阅口径说明</a></p>'
    (OUT/'oss2_candidates.html').write_text(page+''.join(sections),encoding='utf-8')
    print(json.dumps(summary))
    print(pd.DataFrame(result)[['query_id','partner_id','same_name','same_imo','same_size','all_raw_images_exist']].to_string(index=False))


if __name__=='__main__':main()
