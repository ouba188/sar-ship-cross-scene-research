"""Bounded task-dependent features and source-only hard-class supervision."""

import sys
from pathlib import Path
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"geometry_relation_slots_20261016"))
from slot_reader import SlotReader, centered_features


class OptionalGeometryReader(SlotReader):
    @staticmethod
    def angular(geometry):
        valid=geometry[:,3:6]>0
        angles=torch.where(valid,geometry[:,:3],torch.zeros_like(geometry[:,:3]))
        s,r,inc=angles.unbind(-1)
        both=valid[:,0]&valid[:,1]
        encoded=torch.stack((torch.sin(2*s),torch.cos(2*s),torch.sin(2*r),torch.cos(2*r),
                              torch.sin(2*(s-r)),torch.cos(2*(s-r)),torch.sin(inc),torch.cos(inc)),-1)
        masks=torch.stack((valid[:,0],valid[:,0],valid[:,1],valid[:,1],both,both,valid[:,2],valid[:,2]),-1)
        return encoded*masks


def bounded_features(raw,valid,ns):
    psi=centered_features(raw,valid,ns)
    return psi/torch.sqrt(1+psi.square().sum(-1,keepdim=True))


def bounded_predict(item,positions,raw,valid,coefficients):
    ns=len(item["s"])
    h=bounded_features(raw,valid,ns).double()
    ss=item["ss"]+.75*(h[:ns]@h[:ns].T)
    qs=item["qs"][positions]+.75*(h[ns:]@h[:ns].T)
    coef=coefficients(ss,item["prior_s"],item["sy"])
    score=item["prior_q"][positions]+qs@coef
    p=.5*(item["base_prob"][positions]+score.softmax(-1))
    active=valid[ns:]&(valid[:ns].sum()>=2)
    return torch.where(active[:,None],p,item["old_prob"][positions]),qs,h


def relation_loss(h,support_labels,query_labels,valid,old_prob):
    ns=len(support_labels)
    sim=h[ns:]@h[:ns].T
    scores=[]; classes=[]
    for cls in support_labels[valid[:ns]].unique():
        mask=(support_labels==cls)&valid[:ns]
        values=sim[:,mask]/.2
        scores.append(.2*(torch.logsumexp(values,dim=1)-torch.log(mask.sum().to(sim.dtype))))
        classes.append(cls)
    if len(classes)<2:
        return h.sum()*0.,0
    classes=torch.stack(classes)
    class_scores=torch.stack(scores,1)
    positive=query_labels[:,None]==classes[None]
    available=positive.any(1)&valid[ns:]
    if not available.any():
        return h.sum()*0.,0
    # Frozen full B picks a difficult supported wrong class; labels are source loss only.
    negatives=old_prob[:,classes].masked_fill(positive,-torch.inf).argmax(1)
    positives=positive.to(torch.long).argmax(1)
    rows=torch.arange(len(query_labels),device=h.device)
    loss=F.softplus((class_scores[rows,negatives]-class_scores[rows,positives]+.05)/.2)
    balanced=torch.stack([loss[available&(query_labels==c)].mean() for c in query_labels[available].unique()]).mean()
    return balanced,int(available.sum())
