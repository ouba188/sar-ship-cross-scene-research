"""Summaries and bounded checks for the specific evidence used in this study."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import ROOT,MANIFEST,WORK,csv,write

def main():
    rows=json.loads(MANIFEST.read_text(encoding='utf-8'));lookup={r['id']:r for r in rows}
    met=pd.read_csv(ROOT/'native_metrics_local.csv',dtype={'mmsi':str})
    pol=pd.read_csv(ROOT/'polar_profile_local.csv',dtype={'mmsi':str})
    source=pd.read_csv(ROOT/'source_observations_local.csv',dtype={'mmsi':str}).set_index('sample_id')
    native=met[met.pol=='vv']
    stats=native.groupby('class_name',as_index=False).agg(vessel_area_cells=('sample_id','size'),length_q10=('length_px',lambda x:x.quantile(.1)),length_median=('length_px','median'),length_q90=('length_px',lambda x:x.quantile(.9)),width_q10=('width_px',lambda x:x.quantile(.1)),width_median=('width_px','median'),width_q90=('width_px',lambda x:x.quantile(.9)),median_top2_mass=('top2_mass','median'))
    csv('native_extent_by_class.csv',stats)
    pol['both_model_pols']=[bool(lookup[k].get('vv') and lookup[k].get('vh')) for k in pol.sample_id]
    ps=pol[pol.both_model_pols].groupby('class_name',as_index=False).agg(vessel_area_cells=('sample_id','size'),median_vv_vh_tv=('vv_vh_tv','median'),top_bin_different_pct=('top_bin_diff',lambda x:x.mean()*100))
    csv('polar_model_available_summary.csv',ps)
    # Need this distinction because native TIFF can exist while the model mask rejects one branch.
    availability={p:sum(bool(lookup[k].get(p)) for k in pol.sample_id) for p in ['vv','vh']}
    availability.update(raw_pairs=len(pol),both_model_pols=int(pol.both_model_pols.sum()),both_model_pols_native_png_verified_only_in_gallery=True)
    write('model_polarization_availability.json',availability)
    pred=pd.read_csv(ROOT/'cached_prediction_cases_local.csv',dtype={'mmsi':str}).sort_values('sample_id').drop_duplicates('mmsi')
    refs=pd.read_csv(ROOT/'references_local.csv',dtype={'mmsi':str})
    out=[]
    for cls,g in pred.groupby('class_name'):
        near=[]
        for r in g.itertuples():
            rr=refs[(refs.class_name==cls)&(refs.mmsi!=r.mmsi)]
            factor=np.maximum.reduce([r.ais_length_m/rr.ais_length_m.to_numpy(),rr.ais_length_m.to_numpy()/r.ais_length_m,r.ais_width_m/rr.ais_width_m.to_numpy(),rr.ais_width_m.to_numpy()/r.ais_width_m])
            near.append(factor.min()<=1.25)
        q=g.copy();q['size_covered']=near
        for covered,z in q.groupby('size_covered'):
            out.append(dict(class_name=cls,joint_dimension_covered=covered,ships=len(z),cached_errors=int((~z.correct).sum()),note='In-sample baseline description; not a causal attribution or test accuracy.'))
    csv('reference_coverage_and_cached_errors.csv',pd.DataFrame(out))
    checks=[]
    # These checks catch repeated-identity inflation, wrong joins, and mismatched size controls.
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    check('native cell/polar keys unique',not met.duplicated(['class_name','port_proxy','mmsi','pol']).any())
    check('profiles normalized',np.allclose(met[[f'q{i}' for i in range(6)]].sum(axis=1),1))
    matches=pd.read_csv(ROOT/'size_matched_pairs_local.csv')
    unique=matches[['class_name','sample_a','sample_b']].drop_duplicates()
    for cls,g in unique.groupby('class_name'):
        check(cls+' pairs reuse no identity',len(set(source.loc[g.sample_a,'mmsi'])|set(source.loc[g.sample_b,'mmsi']))==2*len(g))
    for r in unique.itertuples():
        a,b=source.loc[r.sample_a],source.loc[r.sample_b]
        assert a.class_name==b.class_name==r.class_name and a.port_proxy!=b.port_proxy
        assert max(a.ais_length_m/b.ais_length_m,b.ais_length_m/a.ais_length_m,a.ais_width_m/b.ais_width_m,b.ais_width_m/a.ais_width_m)<=1.1+1e-10
    check('all 84 pairs same class/different areas and dimensions within 10%',len(unique)==84)
    cov=json.loads((ROOT/'image_coverage.json').read_text())
    check('inspected native TIFF and consumed PNG agree',cov['source_png_mismatches']==0)
    check('primary counts match direct cells',len(native)==661 and native.mmsi.nunique()==656)
    csv('checks.csv',pd.DataFrame(checks))
    plt.rcParams.update({'font.family':'Microsoft YaHei','svg.fonttype':'none','axes.unicode_minus':False,'font.size':11})
    cn={'bulk_carrier':'散货船','container_ship':'集装箱船','general_cargo':'普通货船','product_chemical_tanker':'化学品/成品油轮'}
    order=list(cn)
    fig,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for i,cls in enumerate(order):
        q=native[native.class_name==cls].length_px
        axs[0].boxplot(q,positions=[i],widths=.6,showfliers=False,patch_artist=True,boxprops={'facecolor':'#A4C2C5'},medianprops={'color':'#344C5C'})
    axs[0].set_xticks(range(4),[cn[c] for c in order],rotation=25,ha='right');axs[0].set_ylabel('检测框长轴 / 原始像素');axs[0].set_title('同类内部的目标尺度并不固定')
    v=pol.groupby('class_name').top_bin_diff.mean().reindex(order)*100
    axs[1].bar(range(4),v,color=['#86B0B5','#D5AA80','#A5B491','#B6A3BE'])
    for i,y in enumerate(v):axs[1].text(i,y+1,f'{y:.1f}%',ha='center')
    axs[1].set_xticks(range(4),[cn[c] for c in order],rotation=25,ha='right');axs[1].set_ylim(0,60);axs[1].set_ylabel('VV/VH 最强纵向区段不同的单元比例');axs[1].set_title('极化间响应位置存在差异')
    q=pd.read_csv(ROOT/'selected_pairs.csv').set_index('class_name');e=pd.read_csv(ROOT/'extended_selected_pair_sensitivity.csv');e=e[e.scope=='extended_GRDH'].set_index('class_name')
    x=np.arange(4);axs[2].bar(x-.18,q.loc[order,'gap_m'],width=.36,label='实际训练侧',color='#86B0B5');axs[2].bar(x+.18,e.loc[order,'gap_m'],width=.36,label='扩展元数据',color='#D5AA80')
    axs[2].axhline(0,color='#777777',lw=.7);axs[2].set_xticks(x,[cn[c] for c in order],rotation=25,ha='right');axs[2].set_ylabel('已选水域对的长度中位数差 / m');axs[2].set_title('集装箱差异保留；其他类别有边界');axs[2].legend(frameon=False,fontsize=9)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('OSS2：真实数据与原图支持研究问题，尚不等于模块收益已获验证',fontsize=14)
    for ext in ['png','svg']:fig.savefig(ROOT/f'evidence_summary.{ext}',dpi=170)
    plt.close(fig)
    print(stats.to_string(index=False));print(availability);print(pd.DataFrame(out).to_string(index=False));print(pd.DataFrame(checks).to_string(index=False))

if __name__=='__main__':main()
