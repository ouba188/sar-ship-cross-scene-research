"""Source-only, vessel-weighted class/port size study. No model fitting."""
from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
STORE = Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')
META = WORK/'chip_quality_20261017/ais_metadata/ais_metadata_sidecar.csv'
CLASSES = ['bulk_carrier', 'fishing', 'general_cargo', 'chemical_tanker',
           'container', 'crude_oil_tanker', 'tug', 'offshore']
META_CLASSES = ['bulk_carrier', 'fishing_vessel', 'general_cargo',
                'product_chemical_tanker', 'container_ship', 'crude_oil_tanker',
                'tug_towing', 'offshore_supply']


def write(name, value):
    (ROOT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def csv(name, frame):
    frame.to_csv(ROOT/name, index=False, encoding='utf-8-sig')


def load():
    rows = json.loads((STORE/'manifest.json').read_text(encoding='utf-8'))
    groups = json.loads((STORE/'groups.json').read_text(encoding='utf-8'))
    idx = sorted(set(groups['source_fit']+groups['source_adapt']))
    assert not set(idx).intersection(groups['validation']+groups['evaluation'])
    columns = ['global_row','mmsi','port','product','class_name','ais_length_m',
        'ais_width_m','ais_size_valid','ais_size_status','ais_size_scope',
        'vessel_static_conflict','same_product_imo_conflict','vessel_imo_variants_json',
        'pixel_x_m_vv','pixel_y_m_vv','pixel_x_m_vh','pixel_y_m_vh',
        'scale_valid_vv','scale_valid_vh','identity_link_quality']
    meta = pd.read_csv(META, usecols=columns, dtype={'mmsi':str}).set_index('global_row')
    data = []
    for i in idx:
        row = rows[i]
        m = meta.loc[row['corpus_global_row']]
        assert m.mmsi == row['mmsi'] and m.port == row['port'] and m['product'] == row['product']
        assert m.class_name == META_CLASSES[row['label']]
        record = m.to_dict()
        record.update(index=i, id=row['id'], label=row['label'], class_name=CLASSES[row['label']],
                      pass_id='_'.join(row['pass_group'].split('_')[:2]),
                      quarter=pd.Timestamp(row['product'].split('_')[4][:8]).quarter)
        data.append(record)
    frame = pd.DataFrame(data)
    # Scope is associated AIS dimensions, not surveyed hull measurements.
    conflicts = frame.groupby('mmsi').label.nunique()
    conflicted = set(conflicts[conflicts > 1].index)
    frame['label_consistent'] = ~frame.mmsi.isin(conflicted)
    frame['dimension_consistent'] = ~frame.vessel_static_conflict & ~frame.same_product_imo_conflict
    frame['valid'] = (frame.ais_size_valid & frame.label_consistent & frame.dimension_consistent &
        frame.mmsi.str.fullmatch(r'\d{9}').fillna(False))
    frame['plausible'] = (frame.ais_length_m.between(10,500) & frame.ais_width_m.between(2,100) &
        (frame.ais_length_m/frame.ais_width_m).between(1.5,15))
    split = json.loads((WORK/'paper_draft_20261108/pcea_routed_correction_20261118/split_local.json').read_text())
    frame['current_fit'] = frame['index'].isin(split['fit'])
    assert int(frame.current_fit.sum()) == 15316
    audit = dict(source_observations=len(frame), source_vessels=frame.mmsi.nunique(),
        source_ports=frame.port.nunique(), source_passages=frame.pass_id.nunique(),
        ais_valid_observations=int(frame.ais_size_valid.sum()),
        class_conflicted_source_vessels=len(conflicted),
        consistent_observations=int(frame.valid.sum()),
        consistent_vessels=int(frame.loc[frame.valid,'mmsi'].nunique()),
        plausibility_flag_observations=int((frame.valid & ~frame.plausible).sum()),
        dimension_scopes=frame.ais_size_status.value_counts().to_dict(),
        current_fit_observations=int(frame.current_fit.sum()),
        notes=['Training-side exploration only; no held test inspection or model modification.',
               'One vessel gets one vote per port/class, not one vote per observation.',
               'Port membership can repeat across ports; pair comparisons remove shared vessels.',
               'AIS size is associated metadata, not an independently verified SAR target measurement.',
               'Plausibility limits only sensitivity analysis; no source data are changed.'])
    write('coverage.json', audit)
    return rows, groups, frame


def vessel_cells(df):
    return df.groupby(['class_name','label','port','mmsi'], as_index=False).agg(
        length=('ais_length_m','median'), beam=('ais_width_m','median'),
        observations=('index','size'), passes=('pass_id','nunique'),
        exact=('ais_size_status',lambda x: bool((x == 'valid_exact_record').any())))


def summarize(df):
    v = vessel_cells(df)
    stats = v.groupby(['label','class_name','port'],as_index=False).agg(
        vessels=('mmsi','size'), observations=('observations','sum'),
        length_q10=('length',lambda x:x.quantile(.1)), length_q25=('length',lambda x:x.quantile(.25)),
        length_median=('length','median'), length_q75=('length',lambda x:x.quantile(.75)),
        length_q90=('length',lambda x:x.quantile(.9)), beam_median=('beam','median'),
        exact_vessels=('exact','sum'))
    return v, stats


def pair_stats(v):
    results = []
    for name, group in v.groupby('class_name'):
        cells = {port:q for port,q in group.groupby('port') if len(q)>=30}
        for a,b in itertools.combinations(sorted(cells),2):
            qa,qb = cells[a],cells[b]
            shared = set(qa.mmsi) & set(qb.mmsi)
            qa,qb = qa[~qa.mmsi.isin(shared)],qb[~qb.mmsi.isin(shared)]
            if min(len(qa),len(qb)) < 30:
                continue
            la,lb = qa.length.to_numpy(),qb.length.to_numpy()
            ba,bb = qa.beam.to_numpy(),qb.beam.to_numpy()
            low,high = (a,b) if np.median(la)<=np.median(lb) else (b,a)
            lo,hi = (la,lb) if low==a else (lb,la)
            results.append(dict(class_name=name,port_low=low,port_high=high,
                n_low=len(lo),n_high=len(hi),shared_vessels_removed=len(shared),
                length_low=float(np.median(lo)),length_high=float(np.median(hi)),
                median_gap_m=float(np.median(hi)-np.median(lo)),
                wasserstein_m=float(wasserstein_distance(la,lb)),
                log_length_wasserstein=float(wasserstein_distance(np.log(la),np.log(lb))),
                beam_low=float(np.median(ba if low==a else bb)),
                beam_high=float(np.median(bb if low==a else ba))))
    return pd.DataFrame(results)


def selected_controls(df, pairs):
    result=[]
    rng = np.random.default_rng(20261120)
    for name in CLASSES:
        p=pairs[pairs.class_name==name]
        if p.empty:
            continue
        best=p.sort_values(['median_gap_m','port_low','port_high'],ascending=[False,True,True]).iloc[0]
        a,b=best.port_low,best.port_high
        for scope,part in [('primary',df),('current_fit',df[df.current_fit]),
                           ('plausibility_sensitivity',df[df.plausible]),
                           ('exact_record_only',df[df.ais_size_status=='valid_exact_record'])]:
            sub=part[(part.class_name==name)&part.port.isin([a,b])]
            for quarter in [0,1,2,3,4]:
                if quarter and scope!='primary':
                    continue
                ss=sub if quarter==0 else sub[sub.quarter==quarter]
                v=vessel_cells(ss)
                x,y=v[v.port==a],v[v.port==b]
                shared=set(x.mmsi)&set(y.mmsi)
                x=x[~x.mmsi.isin(shared)].length.to_numpy()
                y=y[~y.mmsi.isin(shared)].length.to_numpy()
                if min(len(x),len(y))<5:
                    continue
                ci=np.array([np.median(rng.choice(y,len(y)))-np.median(rng.choice(x,len(x))) for _ in range(1000)])
                result.append(dict(class_name=name,port_low=a,port_high=b,scope=scope,quarter=quarter,
                    n_low=len(x),n_high=len(y),low_median=float(np.median(x)),high_median=float(np.median(y)),
                    gap_m=float(np.median(y)-np.median(x)), ci_low=float(np.quantile(ci,.025)),
                    ci_high=float(np.quantile(ci,.975)),
                    note='exploratory selected extreme; descriptive bootstrap, not confirmatory significance'))
    return pd.DataFrame(result)


def reference_coverage(df, groups):
    refs=df[df['index'].isin(groups['references'])]
    output=[]
    for name,part in df.groupby('class_name'):
        ref=refs[(refs.class_name==name)&refs.valid]
        vessels=part.drop_duplicates('mmsi')
        lengths=ref.ais_length_m.to_numpy()
        if not len(lengths):
            continue
        for scope, subset in [('all_source',vessels),('current_fit',part[part.current_fit].drop_duplicates('mmsi'))]:
            # Metadata diagnostic only; no true query class is fed to a model.
            dist=np.abs(np.log(subset.ais_length_m.to_numpy()[:,None]/lengths[None,:])).min(1)
            output.append(dict(class_name=name,scope=scope,reference_count=len(lengths),
                reference_lengths_m=';'.join(map(str,lengths)),
                reference_beams_m=';'.join(map(str,ref.ais_width_m.to_numpy())),
                vessels=len(subset), fraction_nearest_size_within_25pct=float((dist<=np.log(1.25)).mean()),
                fraction_nearest_size_more_than_1_5x=float((dist>np.log(1.5)).mean()),
                fraction_nearest_size_more_than_2x=float((dist>np.log(2)).mean())))
    return pd.DataFrame(output)


def plot(v,stats):
    fig,axes=plt.subplots(2,4,figsize=(20,11),layout='constrained')
    for name,ax in zip(CLASSES,axes.flat):
        cells=stats[(stats.class_name==name)&(stats.vessels>=30)].sort_values('length_median')
        yy=np.arange(len(cells))
        ax.errorbar(cells.length_median,yy,
            xerr=[cells.length_median-cells.length_q25,cells.length_q75-cells.length_median],
            fmt='o',color='#167d8d',capsize=2)
        ax.set_yticks(yy,[f'{r.port} (n={r.vessels})' for r in cells.itertuples()],fontsize=7)
        title = name + (' [label audit needed]' if name in ('tug','offshore') else '')
        ax.set_title(title,fontsize=10); ax.set_xlabel('AIS length (m), median and IQR',fontsize=8)
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Same class, different ports: one vote per vessel per port; cells with >=30 vessels',fontsize=14)
    fig.savefig(ROOT/'port_length_distributions.png',dpi=160)
    fig.savefig(ROOT/'port_length_distributions.svg')
    plt.close(fig)


def main():
    rows,groups,frame=load()
    valid=frame[frame.valid].copy()
    v,stats=summarize(valid)
    pairs=pair_stats(v)
    csv('port_class_sizes.csv',stats)
    csv('port_pair_size_differences.csv',pairs)
    csv('selected_pair_sensitivity.csv',selected_controls(valid,pairs))
    csv('reference_size_coverage.csv',reference_coverage(valid,groups))
    for scope,p in [('current_fit',valid[valid.current_fit]),
                    ('exact',valid[valid.ais_size_status=='valid_exact_record']),
                    ('plausible',valid[valid.plausible])]:
        _,s=summarize(p)
        csv(f'port_class_sizes_{scope}.csv',s)
    csv('source_observations_local.csv',frame)
    csv('source_vessels_local.csv',v)
    plot(v,stats)
    print(json.dumps(json.loads((ROOT/'coverage.json').read_text()),indent=2))
    print(pairs.sort_values('median_gap_m',ascending=False).groupby('class_name').head(1).to_string(index=False))


if __name__=='__main__':
    main()
