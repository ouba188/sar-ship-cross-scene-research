"""Eight native query pairs, selected by geometric risk without model scores."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from lxml import etree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = Path('D:/Documents/project_root/project_data/OpenSARShip_2')


def main():
    t = pd.read_csv(HERE / 'cohort.csv', dtype={'mmsi':str}).set_index('sample_id')
    g = pd.read_csv(HERE / 'native_geometry.csv')
    v = g[g.pol.eq('vv')].copy()
    v['endpoint_risk'] = ~v.estimated_head_tail_inside160
    selected = v.sort_values(['class_id','degenerate_dimension','endpoint_risk','max_dimension','sample_id'],
                            ascending=[True,False,False,False,True]).groupby('class_id',sort=True).head(1)
    (HERE / 'panels').mkdir(exist_ok=True)
    trees, rows = {}, []
    for r in selected.itertuples():
        q = t.loc[r.sample_id]
        xmlpath = ROOT / q.scene_id / 'Ship.xml'
        if str(xmlpath) not in trees:
            parser = etree.XMLParser(recover=True,encoding='utf-8')
            tree = etree.fromstring(xmlpath.read_bytes().decode('utf-8',errors='replace').encode('utf-8'),parser)
            trees[str(xmlpath)] = tree
        ships = [s for s in trees[str(xmlpath)].findall('ship')
                 if int(s.findtext('SARShipInformation/Center_x'))==r.center_x
                 and int(s.findtext('SARShipInformation/Center_y'))==r.center_y]
        assert len(ships)==1
        ship = ships[0]
        assert ship.findtext('AISShipInformation/MMSI').strip()==q.mmsi
        def value(name):
            return float(ship.findtext('SARShipInformation/'+name))
        xmin,xmax=sorted([value('UpperLeft_x'),value('LowerRight_x')])
        ymin,ymax=sorted([value('UpperLeft_y'),value('LowerRight_y')])
        hx,hy=value('Head_x')-xmin,value('Head_y')-ymin
        tx,ty=value('Tail_x')-xmin,value('Tail_y')-ymin
        rec=dict(sample_id=r.sample_id,row_index=r.row_index,class_id=r.class_id,class_name=q.class_name,
                 pass_group=q.pass_group,mmsi=q.mmsi,source_time=q.source_time,
                 width=r.width,height=r.height,xml_path=str(xmlpath),
                 xmin=xmin,ymin=ymin,xmax=xmax,ymax=ymax,head_x=hx,head_y=hy,tail_x=tx,tail_y=ty,
                 estimated_origin_error=max(abs(xmin-r.estimated_origin_x),abs(ymin-r.estimated_origin_y)),
                 original_head_tail_inside=(min(hx,tx)>=0 and max(hx,tx)<r.width and min(hy,ty)>=0 and max(hy,ty)<r.height),
                 selection='degenerate then endpoint risk then largest native then first ID; no model',
                 degenerate_dimension=r.degenerate_dimension)
        assert xmax-xmin+1==r.width and ymax-ymin+1==r.height
        fig, axes = plt.subplots(1,2,figsize=(10,5.5),layout='constrained')
        for pol,ax in zip(['vv','vh'],axes):
            path = Path(q['chip_path_'+pol])
            with Image.open(path) as im:
                a=np.asarray(im).copy()
            assert a.shape==(r.height,r.width)
            rec['image_'+pol]=str(path)
            rec['pixel_min_'+pol]=int(a.min())
            rec['pixel_max_'+pol]=int(a.max())
            ax.imshow(a,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            if r.width>160 or r.height>160:
                x,y=max((r.width-160)//2,0),max((r.height-160)//2,0)
                ax.add_patch(Rectangle((x-.5,y-.5),min(160,r.width),min(160,r.height),fill=False,
                                       edgecolor='red',linestyle='--',linewidth=1.2))
            ax.plot([hx,tx],[hy,ty],color='yellow',marker='o',markersize=3,linewidth=1)
            # Keep native extent: inconsistent endpoint metadata must not enlarge the displayed chip.
            ax.set_xlim(-.5,r.width-.5);ax.set_ylim(r.height-.5,-.5)
            ax.set_title(f'{pol.upper()} native {r.width}x{r.height}; range {a.min()}..{a.max()}',fontsize=11)
            ax.set_xlabel('native x');ax.set_ylabel('native y')
        fig.suptitle(f'{q.class_name} | {r.sample_id}\n{q.pass_group} | MMSI {q.mmsi}\n'
                     'Yellow: XML head/tail, NOT OBB. Red dashed: hypothetical 160 crop.',fontsize=10)
        path=HERE/'panels'/f'class{r.class_id}.png'
        fig.savefig(path,dpi=150)
        plt.close(fig)
        rec['panel']=str(path)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    summary=dict(cases=len(rows),unique_xml=len(trees),images_viewed=False,
                 maximum_origin_error=max(r['estimated_origin_error'] for r in rows),
                 native_pixels_modified=False,model_extraction=False)
    (HERE/'review_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows)[['sample_id','class_id','width','height','estimated_origin_error','original_head_tail_inside']].to_string(index=False))


if __name__=='__main__':
    main()
