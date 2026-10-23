"""One fixed source case and per-query availability counts; no model changes."""
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
UP = PARENT.parent / 'regional_utility_pilot/upstream'
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
sys.path.insert(0,str(PARENT))
from extract_tokens import geo, slots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    table = pd.read_csv(UP/'table.csv',dtype={'mmsi':str,'instance_id':str})
    train = np.flatnonzero(table.role.eq('utility_train'))
    rows = []
    for layer in ('stem','layer1'):
        z = np.load(HERE/f'scores_current_{layer}.npz')
        count = z['anchored_count']
        for i in train:
            for p,pol in enumerate(('VV','VH')):
                for r,region in enumerate(('body','side','end')):
                    a = count[i,:,p,r]
                    rows.append(dict(layer=layer,source_index=int(i),obs_id=int(table.iloc[i].obs_id),
                        pol=pol,region=region,classes_count0=int((a==0).sum()),
                        classes_count1=int((a==1).sum()),classes_count2=int((a==2).sum()),
                        different_across_classes=bool(a.min()!=a.max()),
                        **{f'class{c}_available_refs':int(a[c]) for c in range(8)}))
    counts = pd.DataFrame(rows)
    counts.to_csv(HERE/'utility_query_availability.csv',index=False)
    grouped = []
    for key,g in counts.groupby(['layer','pol','region']):
        grouped.append(dict(zip(('layer','pol','region'),key),n=len(g),
            queries_different_across_classes=int(g.different_across_classes.sum()),
            queries_any_count1=int(g.classes_count1.gt(0).sum()),
            queries_all_classes_count0=int(g.classes_count0.eq(8).sum()),
            query_candidate_pairs_count1=int(g.classes_count1.sum())))
    pd.DataFrame(grouped).to_csv(HERE/'utility_query_summary.csv',index=False)

    i = int(np.flatnonzero(table.obs_id.eq(31024))[0])
    assert table.iloc[i].role=='utility_train'
    true = int(table.iloc[i].class_id)
    base = np.load(UP/'strategies_seed17.npz')['logits'][i,2].astype(np.float32)
    rival = int(np.argmax(np.where(np.arange(8)==true,-np.inf,base)))
    per = np.load(HERE/'per_reference_layer1.npz')
    old = np.load(HERE/'scores_current_layer1.npz')
    new = np.load(HERE/'scores_valid_layer1.npz')
    ref = per['refs'][i]
    ids = [i,*ref[true],*ref[rival]]
    names = ['QUERY','TRUE REF 1','TRUE REF 2','B-RIVAL REF 1','B-RIVAL REF 2']
    cache = np.load(PARENT/'tokens.npz')
    wanted = set(table.iloc[ids].obs_id)
    snapshots = {s['obs_id']:s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if s['obs_id'] in wanted}
    fig,axes = plt.subplots(2,5,figsize=(20,10))
    fig.subplots_adjust(left=.03,right=.99,top=.83,bottom=.19,wspace=.16,hspace=.35)
    images = []
    # Reuse external_cases.py's native rendering and region-mask conventions,
    # without its hard-coded external title, rival choice, or output directory.
    for p,pol in enumerate(('vv','vh')):
        for j,(idx,name) in enumerate(zip(ids,names)):
            row = table.iloc[int(idx)]
            pix,fp,g,place = geo.native(table,snapshots,int(idx),p)
            ax = axes[p,j]
            ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            ax.set(xlim=(-.5,159.5),ylim=(159.5,-.5),xticks=[0,80,159],yticks=[0,80,159],aspect='equal')
            valid = bool(cache['valid'][idx,p])
            if g is not None and valid:
                masks = slots(g,fp)
                for sl,color in [(slice(0,8),'#24a878'),(slice(8,16),'#2c84c6'),(slice(16,20),'#d39222')]:
                    field = masks[sl].sum(0)
                    if field.min()<.5<field.max():
                        ax.contour(field,levels=[.5],colors=[color],linewidths=.8)
            if not valid:
                ax.text(80,80,'INVALID LOCAL INPUT',ha='center',color='red',fontsize=9,
                    bbox=dict(facecolor='white',alpha=.85))
            ax.set_title(f'{pol.upper()} {name} | obs {row.obs_id}\n'+textwrap.fill(str(row.class_name),26)+
                f'\n{row.port} | valid={int(valid)}',fontsize=9)
            images.append(dict(panel_column=j,pol=pol,source_index=int(idx),obs_id=int(row.obs_id),
                instance_id=str(row.instance_id),mmsi=str(row.mmsi),role=str(row.role),
                image_path=str(row['image_'+pol]),local_input_valid=valid,
                raw_width=snapshots[int(row.obs_id)]['images'][pol]['w'],
                raw_height=snapshots[int(row.obs_id)]['images'][pol]['h'],placement=place))
    bm = float(base[true]-base[rival])
    lines = [f'SOURCE utility_train | obs31024 | true={true}, fixed full rival={rival}, full pred={base.argmax()} | full margin={bm:+.6f}',
             'Layer1 fixed references. No head replay, training, or predicted rescue is claimed.']
    fig.suptitle('\n'.join(lines),fontsize=13)
    evidence = []
    footer = []
    for r,region in enumerate(('body','side','end')):
        a = float(old['anchored'][i,true,1,r,0]-old['anchored'][i,rival,1,r,0])
        b = float(new['anchored'][i,true,1,r,0]-new['anchored'][i,rival,1,r,0])
        footer.append(f'VH {region} similarity-mass margin: mean2 {a:+.6f} -> valid-reference mean {b:+.6f}')
    fig.text(.035,.12,'\n'.join(footer),fontsize=11)
    fig.text(.035,.035,'Native 160 crop/pad, no input resize. Green body / blue side / amber end.\n'
        'Invalid input is not observed zero evidence; available but semantically rejected references remain in the denominator.\n'
        'These are descriptor margins, not new classification logits. Layer1 RF35 is not pixel-pure regional evidence.',fontsize=10)
    fig.savefig(HERE/'source_31024.png',dpi=140)
    plt.close(fig)
    for arm in ('pooled','independent','anchored','body_match'):
        for c in (true,rival):
            for k,idx in enumerate(ref[c]):
                for p,pol in enumerate(('VV','VH')):
                    for r,region in enumerate(('body','side','end')):
                        if arm=='body_match' and r>0:
                            continue
                        evidence.append(dict(arm=arm,query_obs=31024,candidate_class=c,reference_slot=k,
                            reference_obs=int(table.iloc[int(idx)].obs_id),pol=pol,region=region,
                            available=bool(per[f'{arm}_available'][i,c,k,p,r]),
                            original_similarity_mass=float(per[arm][i,c,k,p,r,0]),
                            original_matched_mass=float(per[arm][i,c,k,p,r,1]),
                            current_similarity_mass=float(old[arm][i,c,p,r,0]),
                            valid_similarity_mass=float(new[arm][i,c,p,r,0]),
                            current_matched_mass=float(old[arm][i,c,p,r,1]),
                            valid_matched_mass=float(new[arm][i,c,p,r,1])))
    pd.DataFrame(evidence).to_csv(HERE/'source_31024_evidence.csv',index=False)
    (HERE/'source_31024_images.json').write_text(json.dumps(dict(query_obs=31024,role='utility_train',
        seed=17,true_class=true,fixed_B_rival=rival,full_margin=bm,views=images),indent=2),encoding='utf-8')
    print(pd.DataFrame(grouped).to_string(index=False),flush=True)
    print('CASE',31024,'true',true,'rival',rival,'refs',table.iloc[ids].obs_id.tolist(),'full_margin',bm,flush=True)


if __name__=='__main__':
    main()
