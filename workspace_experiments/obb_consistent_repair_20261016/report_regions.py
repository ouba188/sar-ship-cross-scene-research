"""Image and cohort evidence, independent of classification outcomes."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

HERE=Path(__file__).resolve().parent


def main():
    meta=pd.read_csv(HERE/'inventory.csv').set_index('global_row')
    roles=pd.read_csv(HERE/'roles.csv')
    det=pd.DataFrame([json.loads(line) for line in (HERE/'detections.jsonl').read_text().splitlines()])
    det=det.merge(meta.reset_index()[['global_row','port','class_id']],on='global_row',validate='many_to_one')
    for arm in ('historical','wrong','fixed'):
        det[arm+'_bright_outside_fraction']=det[arm+'_bright_outside']/det.bright_count.replace(0,np.nan)
    task_rows=det.merge(roles,on='global_row',validate='many_to_many')
    aggregate={'global_row':'count','accepted':'mean','adapter_changes_box':'mean',
               **{a+'_bright_outside_fraction':'mean' for a in ('historical','wrong','fixed')}}
    task_rows.groupby(['split','role','pol']).agg(aggregate).rename(columns={'global_row':'images'}).to_csv(HERE/'geometry_by_role.csv')
    task_rows.groupby(['split','role','class_id']).agg(aggregate).rename(columns={'global_row':'images'}).to_csv(HERE/'geometry_by_class.csv')
    source=set(roles.loc[roles.split=='fit','global_row'])
    candidates=det.loc[det.global_row.isin(source)&det.adapter_changes_box&det.pol.eq('vv')].sort_values('global_row')
    selected=candidates.drop_duplicates('port').global_row.head(4).tolist()
    if len(selected)<4:
        selected+=candidates.loc[~candidates.global_row.isin(selected),'global_row'].head(4-len(selected)).tolist()
    selected += [24846,35475]
    figure,axes=plt.subplots(3,4,figsize=(13,10))
    selection=[]
    for i,grow in enumerate(selected):
        m=meta.loc[grow]
        for j,pol in enumerate(('vv','vh')):
            r=det.loc[(det.global_row==grow)&(det.pol==pol)].iloc[0]
            ax=axes.flat[2*i+j]
            with Image.open(m['image_relpath_'+pol]) as im:
                native=np.asarray(im.convert('L'))
            ax.imshow(native,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            for arm,color,style in (('historical','cyan','-'),('wrong','orange','--'),('fixed','lime','-')):
                x0,y0,x1,y1=r[arm+'_box']
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,
                                      linewidth=1,linestyle=style,label=arm))
            ax.set_title(f"{m.port} | row {grow} | {pol.upper()}\n"
                         f"{'source fit' if grow in source else 'previous query'}; detector={int(r.accepted)}",fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        selection.append(dict(global_row=grow,selection_reason='first changed source by port' if grow in source else 'previous diagnostic query'))
    axes.flat[0].legend(fontsize=7,loc='lower left')
    figure.suptitle('Native SAR pixels; original box / same fresh OBB through wrong vs fixed adapter',fontsize=11)
    figure.tight_layout(rect=(0,0,1,.97))
    figure.savefig(HERE/'region_examples.png',dpi=150)
    plt.close(figure)
    pd.DataFrame(selection).to_csv(HERE/'region_example_selection.csv',index=False)
    if (HERE/'FEATURES_DONE.json').is_file():
        with np.load(HERE/'features_wrong.npz') as a, np.load(HERE/'features_fixed.npz') as b:
            np.testing.assert_array_equal(a['rows'],b['rows'])
            changes=pd.DataFrame({'global_row':a['rows']})
            for key in ('ship_peak','tokens','img_tokens','ship_tokens'):
                x=a[key].reshape(len(a['rows']),-1).astype(float)
                y=b[key].reshape(len(b['rows']),-1).astype(float)
                changes[key+'_relative_change']=np.linalg.norm(y-x,axis=1)/np.linalg.norm(x,axis=1).clip(1e-12)
                changes[key+'_identical']=(x==y).all(1)
            changes.to_csv(HERE/'feature_changes.csv',index=False)
            changes.merge(roles,on='global_row').groupby(['split','role']).mean(numeric_only=True).drop(
                columns='global_row').to_csv(HERE/'feature_changes_by_role.csv')
    print(task_rows.groupby(['split','role']).agg(aggregate).to_string())


if __name__=='__main__':
    main()
