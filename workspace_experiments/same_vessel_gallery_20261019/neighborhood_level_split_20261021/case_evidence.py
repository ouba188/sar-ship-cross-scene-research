"""Fixed supplemental-head interventions on six already selected native cases."""
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
import run as exp

ROOT=exp.ROOT


def margin(score,truth):
    return float(score[truth]-np.max(np.delete(score,truth)))


def main():
    frame,views,disabled=exp.control.load_data()
    _,splits=exp.relation.load_data()
    cache=np.load(ROOT/'features.npz')
    selected=pd.read_csv(ROOT/'case_index.csv').drop_duplicates('obs_id')
    stem=exp.control.make_stem()
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    entries=[]
    for case in selected.itertuples():
        i=int(frame.index[frame.obs_id.eq(case.obs_id)][0]); row=frame.iloc[i]
        fold=int(row.fold); tr,val,test=splits[fold]
        matrix,_=exp.standardize(cache,tr)
        z=matrix['separated'][i].copy()
        no_variation=z.copy()
        norm=np.load(ROOT/f'normalization_f{fold}.npz')
        saved=torch.load(ROOT/f'head_f{fold}_separated_s0.pt',map_location='cpu',weights_only=False)
        head=exp.Residual().cuda().eval();head.load_state_dict(saved['state'])
        scores=np.load(ROOT/f'scores_f{fold}_separated_s0.npz')
        j=int(np.where(scores['rows']==i)[0][0])
        base=scores['base'][j]; full=scores['scores'][j]
        truth=int(row.class_id); full_margin=margin(full,truth)
        canvas=Image.new('RGB',(690,260),'white'); draw=ImageDraw.Draw(canvas)
        draw.text((8,5),f'{case.case} obs {case.obs_id}, class {truth}; fixed D head seed 0; baseline scores unchanged',font=font,fill='black')
        for pp,pol in enumerate(('vv','vh')):
            pix,masks,_,_,_=exp.control.prepare_view(row,views[i]['images'][pol],pol)
            raw=Image.fromarray(np.rint(pix[0]*255).astype(np.uint8)).convert('RGB')
            canvas.paste(raw,(8+340*pp,43))
            overlay=raw.copy(); od=ImageDraw.Draw(overlay)
            draw.text((8+340*pp,26),pol.upper()+' original / R tiles outlined by effect',font=font,fill='black')
            if not cache['usable'][i,pp]:
                continue
            x=torch.tensor(pix[None],device='cuda'); r=torch.tensor(masks[None,2:3],device='cuda')
            encoded,_=exp.encode(x,r,True)
            edits=[('no_R_variation',encoded*0+.5,-1,-1)]
            for a in range(4):
                for b in range(4):
                    changed=encoded.clone()
                    changed[:,:,a*40:(a+1)*40,b*40:(b+1)*40]=.5
                    edits.append(('tile',changed,a,b))
            with torch.inference_mode():
                original=exp.descriptor(stem,encoded,r).cpu().numpy()[0]
                np.testing.assert_allclose(original,cache['centered'][i,pp],rtol=2e-5,atol=1e-5)
                data=exp.descriptor(stem,torch.cat([e[1] for e in edits]),r.expand(len(edits),-1,-1,-1)).cpu().numpy()
                zz=np.repeat(z[None],len(edits),axis=0)
                zz[:,128*pp:128*(pp+1)]=(data-norm['centered_mean'][pp])/norm['centered_std'][pp]
                no_variation[128*pp:128*(pp+1)]=zz[0,128*pp:128*(pp+1)]
                edited=base+head(torch.from_numpy(zz).cuda()).cpu().numpy()
            for k,(kind,_,a,b) in enumerate(edits):
                difference=full_margin-margin(edited[k],truth)
                entries.append(dict(obs_id=int(case.obs_id),case=case.case,fold=fold,seed=0,pol=pol,
                    intervention=kind,tile_row=a,tile_col=b,true_class=truth,
                    full_prediction=int(full.argmax()),modified_prediction=int(edited[k].argmax()),
                    full_margin=full_margin,modified_margin=margin(edited[k],truth),
                    contribution_to_margin=difference,
                    affected_R_pixels=int(masks[2].sum() if a<0 else masks[2,a*40:(a+1)*40,b*40:(b+1)*40].sum())))
                if kind=='tile' and abs(difference)>.001:
                    color=(220,35,35) if difference>0 else (35,110,220)
                    od.rectangle((b*40,a*40,b*40+39,a*40+39),outline=color,width=2)
                    od.text((b*40+2,a*40+2),f'{difference:+.2f}',font=font,fill=color)
            canvas.paste(overlay,(174+340*pp,43))
        with torch.inference_mode():
            changed=z.copy();changed[256:258]=0
            result=base+head(torch.from_numpy(changed[None]).cuda()).cpu().numpy()[0]
        entries.append(dict(obs_id=int(case.obs_id),case=case.case,fold=fold,seed=0,pol='both',
            intervention='source_mean_level',tile_row=-1,tile_col=-1,true_class=truth,
            full_prediction=int(full.argmax()),modified_prediction=int(result.argmax()),
            full_margin=full_margin,modified_margin=margin(result,truth),
            contribution_to_margin=full_margin-margin(result,truth),affected_R_pixels=0))
        with torch.inference_mode():
            blank=base+head(torch.from_numpy(no_variation[None]).cuda()).cpu().numpy()[0]
        for label,score in (('both_no_R_variation',blank),('source_constant_residual',scores['source_constant'][j])):
            entries.append(dict(obs_id=int(case.obs_id),case=case.case,fold=fold,seed=0,pol='both',
                intervention=label,tile_row=-1,tile_col=-1,true_class=truth,
                full_prediction=int(full.argmax()),modified_prediction=int(score.argmax()),
                full_margin=full_margin,modified_margin=margin(score,truth),
                contribution_to_margin=full_margin-margin(score,truth),affected_R_pixels=0))
        draw.text((8,211),'Red: removing this R tile lowers true-class margin. Blue: removing it helps. Only new channel edited.',font=font,fill='black')
        draw.text((8,230),'Mean scalar, masks and original predictor are fixed. Artificial occlusion, not a physical scene intervention.',font=font,fill='black')
        canvas.save(ROOT/f'evidence_{case.case}.png')
    pd.DataFrame(entries).to_csv(ROOT/'case_interventions.csv',index=False)
    print(pd.DataFrame(entries).query("intervention!='tile'").to_string(index=False))


if __name__=='__main__':
    main()
