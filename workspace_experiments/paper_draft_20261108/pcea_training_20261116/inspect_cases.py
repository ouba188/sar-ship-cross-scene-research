"""Actual-chip and fixed-weight withdrawal checks after paired training."""
import gc
import argparse
import html
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ROOT, Model, full, torch, np

NAMES=('Bulk','General cargo','Chemical/product tanker','Container')
plt.rcParams['svg.fonttype']='none'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scope',choices=('joint','branch'),default='joint')
    parser.add_argument('--checkpoint',choices=('best','last'),default='best')
    args=parser.parse_args()
    chosen=json.loads((ROOT/'selected_validation.json').read_text())['17']
    if args.scope=='branch':
        chosen.update(json.loads((ROOT/'selected_branch.json').read_text())['17'])
    suffix=f'{args.scope}_{args.checkpoint}'
    inputs=full.Inputs()
    for split in ('small','validation'): inputs.prepare_geometry(split)
    ids=inputs.groups['validation'];y=inputs.y[ids]
    scores={a:np.load(ROOT/d/('last_validation.npz' if args.checkpoint=='last' and a!='baseline' else 'validation.npz'))['score'] for a,d in chosen.items()}
    net=Model(17,'pcea')
    net.load_state_dict(torch.load(ROOT/chosen['pcea']/(args.checkpoint+'.pt'),map_location='cpu',weights_only=True,mmap=True))
    replay=full.predict(net,inputs,ids,'validation')
    error=float(abs(replay-scores['pcea']).max())
    assert error<1e-3,error
    net.exchange.disabled=True
    off=full.predict(net,inputs,ids,'validation')
    net.exchange.disabled=False;net.exchange.arm='gate'
    no_product=full.predict(net,inputs,ids,'validation')
    net.exchange.arm='pcea'
    np.savez_compressed(ROOT/('withdrawal_'+suffix+'.npz'),ids=ids,y=y,full=replay,off=off,no_product=no_product)
    report=dict(replay_error=error,full=full.metrics(y,replay),branch_off=full.metrics(y,off),
        product_off=full.metrics(y,no_product),note='Fixed-weight inference diagnostic, not matched retraining.')
    report.update(scope=args.scope,checkpoint=args.checkpoint)
    full.write(ROOT/('withdrawal_'+suffix+'.json'),report)
    b=scores['baseline'];p=scores['pcea'];pc,bc=p.argmax(1)==y,b.argmax(1)==y
    rival=b.copy();rival[np.arange(len(y)),y]=-np.inf;rival=rival.argmax(1)
    margin=lambda z:z[np.arange(len(y)),y]-z[np.arange(len(y)),rival]
    change=margin(p)-margin(b)
    states=dict(rescued=pc&~bc,harmed=~pc&bc,stable_correct=pc&bc,stable_wrong=~pc&~bc)
    selected=[]
    for state,eligible in states.items():
        ix=np.flatnonzero(eligible & inputs.available[ids].all(1))
        if len(ix):
            ix=ix[np.argsort(abs(change[ix]))]
            selected.append((state,int(ix[len(ix)//2])))
    folder=ROOT/('gallery_'+suffix);folder.mkdir(exist_ok=True)
    records=[]
    with torch.no_grad():
        reference=net.reference(inputs.data(inputs.groups['references'],'small'))
        for state,j in selected:
            idx=int(ids[j]);row=inputs.rows[idx]
            data=inputs.data([idx],'validation')
            net(data,[row],reference,inputs.reference_rows)
            trace=net.exchange.trace
            delta=trace['delta'][0].square().mean(1).sqrt().cpu().numpy()
            legal=full.staged.task.legal([row]+inputs.reference_rows,np.array([0]),np.arange(1,1+len(inputs.reference_rows)))[0]
            legal=legal.cpu().numpy() if torch.is_tensor(legal) else np.asarray(legal)
            refs=[]
            for klass in (int(y[j]),int(rival[j])):
                possible=[int(i) for i,ok in zip(inputs.groups['references'],legal) if ok and inputs.y[i]==klass]
                refs.append(next((i for i in possible if inputs.available[i].all()),possible[0] if possible else None))
            fig,axes=plt.subplots(2,4,figsize=(10.6,5.8))
            for pol in range(2):
                for ax in axes[pol]:ax.axis('off')
                axes[pol,0].imshow(inputs.pixels[idx,pol],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[pol,0].set_title(('VV' if pol==0 else 'VH')+' native chip',fontsize=9)
                axes[pol,1].imshow(delta[pol],cmap='magma',vmin=0,vmax=max(float(delta.max()),1e-8),interpolation='nearest')
                axes[pol,1].set_title('PCEA writeback RMS (10x10)',fontsize=9)
                for k,ref in enumerate(refs,2):
                    if ref is not None:
                        axes[pol,k].imshow(inputs.pixels[ref,pol],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                        axes[pol,k].set_title(('Same-class' if k==2 else 'Baseline rival')+' bank example\n'+inputs.rows[ref]['id'].split('_')[-1],fontsize=9)
            predictions={a:NAMES[z[j].argmax()] for a,z in scores.items()}
            margins={a:float(margin(z)[j]) for a,z in scores.items()}
            margins.update(branch_off=float(margin(off)[j]),product_off=float(margin(no_product)[j]))
            fig.suptitle(f"{row['id']} | {NAMES[y[j]]} | {state} | seed17\nBaseline margin {margins['baseline']:+.3f}; gate {margins['gate']:+.3f}; PCEA {margins['pcea']:+.3f}\nPCEA same weights: branch off {margins['branch_off']:+.3f}; product off {margins['product_off']:+.3f}",fontsize=10)
            fig.text(.5,.02,'Native160, fixed display scale. Reference examples are legal bank members, not ranked causal influences.',ha='center',fontsize=8)
            fig.subplots_adjust(left=.02,right=.98,top=.79,bottom=.06,wspace=.16,hspace=.3)
            stem=f'{state}_{idx}'
            fig.savefig(folder/(stem+'.png'),dpi=150);fig.savefig(folder/(stem+'.svg'));plt.close(fig)
            records.append(dict(id=row['id'],index=idx,state=state,true_class=NAMES[y[j]],rival=NAMES[rival[j]],
                predictions=predictions,margins=margins,branch_off_prediction=NAMES[off[j].argmax()],
                product_off_prediction=NAMES[no_product[j].argmax()],
                alpha=trace['alpha'][0].cpu().tolist(),beta=trace['beta'][0].cpu().tolist(),
                input_rms=trace['input_rms'][0].cpu().tolist(),update_rms=trace['update_rms'][0].cpu().tolist(),
                delta_rms=delta.tolist(),refs=[inputs.rows[i]['id'] if i is not None else None for i in refs],
                source_paths=row.get('source_paths'),image=stem+'.png',svg=stem+'.svg'))
    full.write(folder/'cases.json',records)
    body=['<!doctype html><meta charset="utf-8"><title>PCEA training cases</title>',
        '<style>body{font:16px system-ui;max-width:1300px;margin:24px auto}img{width:100%}section{border-top:1px solid #aaa;padding:20px 0}pre{white-space:pre-wrap}</style>',
        '<h1>PCEA: trained development cases / '+suffix+'</h1><p>Same-weight withdrawals are diagnostic; they do not replace paired training comparisons. Case selection uses median absolute margin change within each outcome group. Last-epoch figures, when requested, are not validation-selected gains.</p>']
    for r in records:
        body += [f'<section><h2>{html.escape(r["id"])} / {r["state"]}</h2><img src="{r["image"]}">',
            f'<a href="{r["svg"]}">Editable SVG</a><pre>'+html.escape(json.dumps({k:v for k,v in r.items() if k not in ('alpha','beta','delta_rms','source_paths')},ensure_ascii=False,indent=2))+'</pre></section>']
    (folder/'index.html').write_text('\n'.join(body),encoding='utf-8')
    print(json.dumps(report,indent=2));print('CASES',len(records))
    del net;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
