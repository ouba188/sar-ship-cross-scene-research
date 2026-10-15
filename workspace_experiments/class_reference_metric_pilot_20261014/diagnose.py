"""Posthoc directional-risk and data-composition diagnosis; no new fits."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('metric_pilot_diagnostic', OUT/'run_pilot.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    d, f, plans, banks, cov, source = run.load_data()
    plan = {e['episode']: e for e in plans}
    enrolled = json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    tasks, condition, isolation = {}, [], []
    for held in sorted({e['held'] for e in enrolled}):
        tasks[held] = {}
        for role in ['train', 'held']:
            selected = [e for e in enrolled if e['held'] == held and e['role'] == role and e['cell'] == 'other_acquisition']
            byport = {}
            for e in selected:
                byport.setdefault(e['port'], []).append(e)
            task_list = []
            for port, es in byport.items():
                scale = 1/(len(byport)*len(es))
                for e in es:
                    t = run.make_task(d, f, plan[e['episode']], e['query'], e['support'],
                                      e['draw'], e['cell'], held, role)
                    task_list.append((scale, t))
                    present = np.unique(t['y'])
                    for c in range(8):
                        mask = t['y'] == c
                        condition.append(dict(held=held, role=role, port=port, episode=e['episode'], draw=e['draw'],
                            class_id=c, task_weight=scale, query_n=int(mask.sum()), support_n=int(t['counts'][c]),
                            class_risk_weight=scale/len(present) if mask.any() else 0,
                            query_exposure_weight=scale*mask.mean(),
                            no_support=bool(t['counts'][c] == 0)))
            tasks[held][role] = task_list
        train_rows = np.unique(np.concatenate([np.r_[t['q'], t['s']] for _, t in tasks[held]['train']]))
        held_rows = np.unique(np.concatenate([np.r_[t['q'], t['s']] for _, t in tasks[held]['held']]))
        isolation.append(dict(held=held, train_rows=len(train_rows), held_rows=len(held_rows),
            train_ships=d.iloc[train_rows].mmsi.nunique(), held_ships=d.iloc[held_rows].mmsi.nunique(),
            identity_overlap=len(set(d.iloc[train_rows].mmsi)&set(d.iloc[held_rows].mmsi)),
            acquisition_overlap=len(set(d.iloc[train_rows].acquisition)&set(d.iloc[held_rows].acquisition)),
            train_query_rows=len(set(np.concatenate([t['q'] for _,t in tasks[held]['train']]))),
            train_query_ships=d.iloc[np.unique(np.concatenate([t['q'] for _,t in tasks[held]['train']]))].mmsi.nunique(),
            held_query_rows=len(set(np.concatenate([t['q'] for _,t in tasks[held]['held']]))),
            held_query_ships=d.iloc[np.unique(np.concatenate([t['q'] for _,t in tasks[held]['held']]))].mmsi.nunique()))
    cond = pd.DataFrame(condition)
    cond['missing_risk_weight'] = cond.class_risk_weight*cond.no_support
    cond['weighted_support_n'] = cond.task_weight*cond.support_n
    cond['weighted_class_support_n'] = cond.class_risk_weight*cond.support_n
    cond.groupby(['held','role','class_id'])[['class_risk_weight','query_exposure_weight','missing_risk_weight',
        'weighted_support_n','weighted_class_support_n']].sum().reset_index().to_csv(OUT/'data_class_conditions.csv', index=False)
    pd.DataFrame(isolation).to_csv(OUT/'data_isolation.csv', index=False)
    gradients, risk0, output = {}, {}, []
    eye = torch.eye(8, dtype=run.DTYPE)
    for held in tasks:
        for anchor in run.ANCHORS:
            for role in ['train','held']:
                g = torch.zeros_like(eye)
                total = 0.
                for scale,t in tasks[held][role]:
                    m = eye.clone().requires_grad_(True)
                    loss = run.task_loss(t,anchor,m,'task')
                    g += scale*torch.autograd.grad(loss,m)[0]
                    total += scale*loss.item()
                gradients[(held,anchor,role)] = g
                risk0[(held,anchor,role)] = total
            gt,gh = [gradients[(held,anchor,role)] for role in ['train','held']]
            c = cov[anchor]
            pt = gt-c*(gt*c).sum()/(c*c).sum()
            ph = gh-c*(gh*c).sum()/(c*c).sum()
            cosine = (pt*ph).sum()/(torch.linalg.norm(pt)*torch.linalg.norm(ph))
            for objective in ['contrastive','task']:
                for seed in run.SEEDS:
                    key = f"{held.replace(' ','_')}_{anchor}_{objective}_{seed}"
                    with np.load(OUT/'models'/f'{key}.npz') as a:
                        m = run.tensor(a['metric'])
                    delta = m-eye
                    risks = {}
                    with torch.no_grad():
                        for role in ['train','held']:
                            risks[role] = sum(scale*run.task_loss(t,anchor,m,'task').item()
                                              for scale,t in tasks[held][role])
                    output.append(dict(held=held,anchor=anchor,objective=objective,seed=seed,
                        projected_gradient_cosine=cosine.item(),
                        train_initial_direction=(gt*delta).sum().item(),
                        held_initial_direction=(gh*delta).sum().item(),
                        train_nll_delta=risks['train']-risk0[(held,anchor,'train')],
                        held_nll_delta=risks['held']-risk0[(held,anchor,'held')],
                        mean_source_energy_error=(c*delta).sum().item(),
                        metric_change=torch.linalg.norm(delta).item()))
            print('DIAGNOSED',held,anchor,'gradient cosine',round(cosine.item(),4),flush=True)
    pd.DataFrame(output).to_csv(OUT/'gradient_transfer.csv',index=False)
    # Independently reconcile reconstructed held loss with saved endpoint metrics.
    metrics = pd.read_csv(OUT/'metrics.csv')
    frame = metrics[(metrics['mode']=='full') & (metrics.cell=='other_acquisition')]
    checks = []
    for record in output:
        key = {k:record[k] for k in ['held','anchor','objective','seed']}
        selector = np.ones(len(frame),dtype=bool)
        for name,value in key.items():
            selector &= frame[name].eq(value).to_numpy()
        expected = frame.loc[selector,'macro_nll'].mean()-risk0[(record['held'],record['anchor'],'held')]
        error=abs(expected-record['held_nll_delta'])
        assert error < 1e-12
        checks.append(dict(**key,error=error))
    pd.DataFrame(checks).to_csv(OUT/'diagnostic_checks.csv',index=False)
    run.write('DIAGNOSIS_COMPLETE.json',dict(seconds=time.perf_counter()-started,new_fits=0,
        interpretation='Posthoc gradient alignment and local directional derivative, not physical causal attribution or a selected new metric',
        query_labels='Held labels used only for diagnosis after frozen training; no deployment changes'))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
