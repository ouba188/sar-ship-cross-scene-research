"""Experimental independently placed M1/M2/M3 operators in frozen ResNet50.

This is not a checkpoint-compatible replacement of the terminal M123 system.
Reference features are recomputed through current M1/M2; references do not
recursively retrieve one another through M3.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .backbone_regions import RegionResNet50, RegionFeedback
from .backbone_attention import ClassResponseFeedback


class BackboneM123(RegionResNet50):
    def __init__(self, weight_path, classes=8, placements=((2,'m1'),(2,'m2'),(3,'m3')),
                 hidden=32, m2_update='broadcast', m2_gate='fixed', m2_merged=False,
                 m1_temperature=1., m3_eps=.1, m3_frequency=True):
        super().__init__(weight_path,classes=classes,mode='none')
        if any(stage not in (1,2,3,4) or name not in ('m1','m2','m3') for stage,name in placements):
            raise ValueError('Use stage 1..4 and operator m1/m2/m3.')
        self.placements=tuple(sorted(placements,key=lambda value:value[0]))
        self.first_stage=min((s for s,_ in self.placements),default=2)
        self.needs_reference=any(name=='m3' for _,name in self.placements)
        self.m2_merged=bool(m2_merged)
        channels={1:256,2:512,3:1024,4:2048}
        self.blocks=nn.ModuleDict()
        for index,(stage,name) in enumerate(self.placements):
            for p in range(2):
                if name=='m1':block=ClassResponseFeedback(channels[stage],classes=classes,hidden=hidden,temperature=m1_temperature)
                elif name=='m2':block=RegionFeedback(channels[stage],hidden=hidden,gate_mode=m2_gate,update_mode=m2_update)
                else:
                    from .backbone_matching import ReferenceMatchFeedback
                    block=ReferenceMatchFeedback(channels[stage],hidden=hidden,transport_eps=m3_eps,use_frequency=m3_frequency)
                self.blocks[f'{index}_{p}']=block

    def prefix(self,images,stop=None):
        stop=self.first_stage if stop is None else stop
        if any(stage<stop for stage,_ in self.placements):
            raise ValueError('Cached prefix would bypass a trainable operator.')
        return super().prefix(images,stop=stop)

    def forward_from(self,prefix,masks,valid,available,*,slot_masks=None,frequency=None,
                     reference=None,pair_valid=None,start=None,return_details=False):
        start=self.first_stage if start is None else start
        if any(stage<start for stage,_ in self.placements):
            raise ValueError('Cached entry would bypass a trainable operator.')
        if self.needs_reference and (reference is None or slot_masks is None or frequency is None):
            raise ValueError('M3 needs current reference features, native slots and frequency fields.')
        x=prefix
        valid=valid*available[...,None,None,None].to(valid)
        masks=masks*available[...,None,None,None].to(masks)
        if self.needs_reference:
            r=reference['prefix']
            ra=reference['available']
            rv=reference['valid']*ra[...,None,None,None].to(reference['valid'])
            rm=reference['masks']*ra[...,None,None,None].to(reference['masks'])
        else:r=None
        auxiliary,trace=[],{}
        for stage in range(start,5):
            if stage>start:
                layer=getattr(self.backbone.model,f'layer{stage}')
                z=layer(x.flatten(0,1))
                x=z.reshape(*x.shape[:2],*z.shape[1:])
                if r is not None:
                    z=layer(r.flatten(0,1))
                    r=z.reshape(*r.shape[:2],*z.shape[1:])
            for index,(where,name) in enumerate(self.placements):
                if where!=stage:continue
                updated,updated_refs,aux=[],[],[]
                for p in range(2):
                    block=self.blocks[f'{index}_{p}']
                    if name=='m1':
                        value,details=block(x[:,p],valid[:,p],return_details=True)
                        aux.append(details['aux_logits'])
                        if r is not None:updated_refs.append(block(r[:,p],rv[:,p]))
                    elif name=='m2':
                        mask=masks[:,p]
                        if self.m2_merged:mask=mask.sum(1,keepdim=True).clamp(0,1).expand(-1,3,-1,-1)
                        value,details=block(x[:,p],mask,valid[:,p],return_details=True)
                        if r is not None:
                            mask=rm[:,p]
                            if self.m2_merged:mask=mask.sum(1,keepdim=True).clamp(0,1).expand(-1,3,-1,-1)
                            updated_refs.append(block(r[:,p],mask,rv[:,p]))
                    else:
                        bank=dict(features=r[:,p],slot_masks=reference['slot_masks'][:,p],
                                  valid=rv[:,p],frequency=reference['frequency'][:,p])
                        value,details=block(x[:,p],slot_masks[:,p],valid[:,p],frequency[:,p],bank,
                                            pair_valid=pair_valid,return_details=True)
                    updated.append(value)
                    if return_details:trace[f'{index}_{name}_stage{stage}_{p}']=details
                x=torch.stack(updated,1)
                if updated_refs:r=torch.stack(updated_refs,1)
                if aux:
                    aux=torch.stack(aux,1)*available[...,None].to(x)
                    auxiliary.append(aux.sum(1)/available.sum(1,keepdim=True).clamp_min(1).to(x))
        footprint=F.interpolate(valid.flatten(0,1).to(x),size=x.shape[-2:],mode='area')
        footprint=footprint.reshape(*x.shape[:2],1,*x.shape[-2:])
        features=(x*footprint).sum((-2,-1))/footprint.sum((-2,-1)).clamp_min(1e-6)
        features=features*available[...,None].to(x)
        logits=self.head(torch.cat((features.flatten(1),available.to(x)),1))
        if return_details:return logits,dict(aux_logits=auxiliary,operators=trace)
        return logits

    def forward(self,images,masks,valid,available,*,slot_masks=None,frequency=None,
                reference=None,pair_valid=None,return_details=False):
        prefix=self.prefix(images)
        if self.needs_reference:
            if reference is None:raise ValueError('M3 requires a reference bank.')
            reference={**reference,'prefix':self.prefix(reference['images'])}
        return self.forward_from(prefix,masks,valid,available,slot_masks=slot_masks,frequency=frequency,
                                 reference=reference,pair_valid=pair_valid,return_details=return_details)
