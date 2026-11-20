"""Audit-label subtype composition; do not replace frozen model labels."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import ROOT, WORK, CLASSES, META_CLASSES, csv, write
from images import PAIRS


def main():
    v=pd.read_csv(ROOT/'source_vessels_local.csv',dtype={'mmsi':str})
    mapping=pd.read_csv(WORK/'visual_reliable_baseline/artifacts/safe841_mmsi_mapping_20260917/all_mmsi_mapping_coverage.csv',
        usecols=['mmsi','selected_mapping_type','selected_mapping_layer','mapping_status','final_class'],dtype={'mmsi':str})
    assert not mapping.mmsi.duplicated().any()
    v=v.merge(mapping,on='mmsi',how='left',validate='many_to_one')
    v['class_agrees']=v.final_class.eq(v.class_name.map(dict(zip(CLASSES,META_CLASSES))))
    v['audit_resolved']=v.class_agrees & v.mapping_status.eq('fine_class_mapped')
    v['subtype']=v.selected_mapping_type.where(v.audit_resolved,'UNRESOLVED_OR_RULE_DIFFERENCE').fillna('MISSING')
    tab=v.groupby(['class_name','port','subtype'],as_index=False).agg(vessels=('mmsi','size'),
        length_median=('length','median'),beam_median=('beam','median'))
    csv('fine_label_composition.csv',tab)
    csv('label_mapping_coverage.csv',v.groupby(['class_name','port'],as_index=False).agg(
        vessels=('mmsi','size'),audit_resolved=('audit_resolved','sum'),class_agrees=('class_agrees','sum')))
    csv('subtype_vessels_local.csv',v)
    out=[]
    for name,a,b in PAIRS:
        sub=v[(v.class_name==name)&v.port.isin([a,b])]
        shared=set(sub[sub.port==a].mmsi)&set(sub[sub.port==b].mmsi)
        sub=sub[~sub.mmsi.isin(shared)]
        for subtype,part in sub.groupby('subtype'):
            x,y=part[part.port==a],part[part.port==b]
            out.append(dict(class_name=name,port_a=a,port_b=b,subtype=subtype,
                n_a=len(x),n_b=len(y),length_a=x.length.median(),length_b=y.length.median(),
                beam_a=x.beam.median(),beam_b=y.beam.median()))
    csv('within_subtype_port_comparison.csv',pd.DataFrame(out))
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    refs=pd.read_csv(ROOT/'reference_size_coverage.csv').query("scope=='all_source'")
    for (name,a,b),ax in zip(PAIRS,axes.flat):
        sub=v[(v.class_name==name)&v.port.isin([a,b])]
        shared=set(sub[sub.port==a].mmsi)&set(sub[sub.port==b].mmsi)
        sub=sub[~sub.mmsi.isin(shared)]
        for port,color in [(a,'#147d92'),(b,'#b23b60')]:
            x=np.sort(sub[sub.port==port].length)
            ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=color,label=f'{port} (n={len(x)})')
        lengths=str(refs[refs.class_name==name].iloc[0].reference_lengths_m)
        for i,value in enumerate(lengths.split(';')):
            ax.axvline(float(value),color='#777777',ls='--',lw=1,label='Fixed same-class references' if i==0 else None)
        ax.set_title(name);ax.set_xlabel('Associated AIS length (m)');ax.set_ylabel('Fraction of vessels')
        ax.set_ylim(0,1.02);ax.grid(alpha=.15);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Within-class port populations and actual reference-size coverage',fontsize=13)
    fig.savefig(ROOT/'selected_ports_ecdf.png',dpi=160)
    fig.savefig(ROOT/'selected_ports_ecdf.svg')
    plt.close(fig)
    print(pd.DataFrame(out).query("class_name=='chemical_tanker'").to_string(index=False))


if __name__=='__main__':main()
