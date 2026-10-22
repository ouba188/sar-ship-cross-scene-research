"""Paired classification outcomes and native-image review for linear CE control."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run_anchored import EXP


def main():
    t = pd.read_csv(EXP/'readout/targets.csv', dtype={'mmsi': str}).set_index('instance_id')
    p = pd.read_csv(HERE/'predictions.csv')
    p['net'] = p.prediction.eq(p.class_id).astype(int)-p.base_prediction.eq(p.class_id).astype(int)
    p['outcome'] = np.select([p.net.eq(1), p.net.eq(-1), p.prediction.eq(p.class_id)],
                             ['rescue', 'harm', 'correct'], default='wrong')
    p = p.join(t[['mmsi','pass_group','port']], on='instance_id', validate='many_to_one')
    p.groupby(['cohort','pass_group','port']).agg(n=('net','size'),net=('net','sum')).to_csv(HERE/'acquisition_effects.csv')
    risk_rows, selected = [], []
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    for cohort, part in p.groupby('cohort'):
        y = part.class_id.to_numpy()
        for kind, columns in [('ridge',['b0','b1','b2']), ('linear_ce',['s0','s1','s2'])]:
            prob = softmax(part[columns].to_numpy(), axis=1)
            nll = -np.log(np.maximum(prob[np.arange(len(y)),y], 1e-15))
            brier = ((prob-np.eye(3)[y])**2).sum(1)
            risk_rows.append(dict(cohort=cohort,kind=kind,NLL=nll.mean(),Brier=brier.mean(),
                balanced_NLL=np.mean([nll[y==c].mean() for c in range(3)]),
                balanced_Brier=np.mean([brier[y==c].mean() for c in range(3)])))
        for outcome in ['rescue','harm','correct','wrong']:
            r = part[part.outcome.eq(outcome)].sort_values('instance_id').iloc[0]
            meta = t.loc[r.instance_id]
            selected.append(r.to_dict())
            canvas = Image.new('RGB',(760,400),'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8,8),f'{cohort} {r.instance_id} {meta.class_name} {meta.port} {outcome}',font=font,fill='black')
            for j,(name,cols) in enumerate([('ridge',['b0','b1','b2']),('linear CE',['s0','s1','s2'])]):
                score = r[cols].to_numpy(float)
                margin = score[int(r.class_id)]-np.delete(score,int(r.class_id)).max()
                draw.text((8,34+22*j),f'{name}: pred={score.argmax()} true margin={margin:+.4f}',font=font,fill='black')
            for j,pol in enumerate(['vv','vh']):
                image = Image.open(EXP/'readout/native_targets'/f'{r.instance_id}_{pol}.png').convert('RGB')
                draw.text((8+190*j,88),f'{pol.upper()} {image.width}x{image.height}',font=font,fill='black')
                canvas.paste(image,(8+190*j,112))
            draw.text((8,300),'Native pixels, unchanged inputs and backbone. No external neighbor input.',font=font,fill='black')
            draw.text((8,324),'0=bulk, 1=fishing, 2=general. Ridge and CE use different logit scales.',font=font,fill='black')
            draw.text((8,348),'Visible structures do not by themselves establish why either classifier errs.',font=font,fill='black')
            canvas.save(HERE/f'case_{cohort}_{outcome}_{r.instance_id}.png')
    pd.DataFrame(selected).to_csv(HERE/'review_cases.csv',index=False)
    pd.DataFrame(risk_rows).to_csv(HERE/'risk_metrics.csv',index=False)
    print(pd.DataFrame(risk_rows).to_string(index=False))
    print(pd.DataFrame(selected)[['cohort','instance_id','outcome','class_id','base_prediction','prediction']].to_string(index=False))


if __name__ == '__main__':
    main()
