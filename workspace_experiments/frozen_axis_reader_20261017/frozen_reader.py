"""Freeze a useful readout; learn a separate common local feature map."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'axis_access_margin_20261017'
sys.path.insert(0, str(PREVIOUS))
from model import prior, EvidenceReader, read, predict, loss_parts, retrieval_loss, decision_margin_loss
sys.path.insert(0, str(HERE.parent/'support_task_alignment_20261017'))
from sampling import training_schedule


@torch.no_grad()
def all_raw(reader, features, common=False):
    return torch.cat([read(reader, features, np.arange(i,min(i+256,len(features['body']))),common)[0]
                      for i in range(0,len(features['body']),256)])


def freeze_axis(reader, features):
    reader.eval().requires_grad_(False)
    return all_raw(reader, features).detach()


def task_loss(reader, features, task, labels, fixed_axis=None):
    if fixed_axis is None:
        return loss_parts(reader, features, task, labels, False, True)
    ids = task['ids']
    common = read(reader, features, ids, True)[0]
    p, h, valid = predict(fixed_axis[ids].detach(), features['valid'][ids], task,
                          common, features['common_valid'][ids], axis_weight=1.)
    truth = labels[task['q']]
    brier = prior.old.risk(p,truth)
    auxiliary = retrieval_loss(h,labels[ids],task['ns'],valid)
    margin = decision_margin_loss(p,truth)
    return brier+.05*auxiliary+.05*margin,brier,auxiliary,margin,p,h


@torch.no_grad()
def evaluate(reader, features, tasks, labels, fixed_axis=None):
    records = []
    for task in tasks:
        loss,brier,aux,margin,p,_ = task_loss(reader,features,task,labels,fixed_axis)
        scores = prior.old.scores(labels[task['q']].numpy(),p.numpy())
        records.append(dict(objective=float(loss),brier=float(brier),retrieval_ce=float(aux),
                            margin_loss=float(margin),oa=scores['oa'],ba=scores['ba']))
    return pd.DataFrame(records).mean().to_dict()


def fit(reader, features, fit_tasks, val_tasks, labels, seed, folder, fixed_axis=None):
    optimizer = torch.optim.AdamW([p for p in reader.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    schedule,_ = training_schedule(seed,len(fit_tasks))
    np.savez_compressed(folder/'schedule.npz',episode=schedule)
    history = []
    for step in range(201):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([task_loss(reader,features,fit_tasks[i],labels,fixed_axis)[0]
                                for i in schedule[step-1]]).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(reader.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
        if step in (0,50,100,200):
            for name,tasks in [('fit',fit_tasks),('selection',val_tasks)]:
                history.append(dict(step=step,evaluation=name,**evaluate(reader,features,tasks,labels,fixed_axis),
                    training_batch_loss=np.nan if not step else float(loss.detach()),
                    gradient_norm=np.nan if not step else float(norm)))
    torch.save(reader.state_dict(),folder/'model.pt')
    pd.DataFrame(history).to_csv(folder/'learning.csv',index=False)
    return reader
