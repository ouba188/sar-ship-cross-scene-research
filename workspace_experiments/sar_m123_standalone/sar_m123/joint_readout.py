"""Source-pilot composition: M1/M3 at stage3, M2 layer2+layer4 readout.

This internal M3 is a reference feature adapter, not the terminal B checkpoint.
Native slot and slot-energy integrals are supplied at the stage3 grid.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .backbone_attention import ClassResponseFeedback
from .backbone_matching import ReferenceMatchFeedback
from .pool_replacement import PoolReplacementResNet50, grid_occupancy, regional_mean


class IntegratedReferenceMatch(ReferenceMatchFeedback):
    def _describe(self, x, slot_masks, valid, frequency):
        size=x.shape[-2:]
        if slot_masks.shape != (len(x),20,*size) or frequency.shape != (len(x),20,3,*size):
            raise ValueError('Supply aligned native slot and slot-energy integrals.')
        occupancy=slot_masks.to(x).clamp(0,1)
        footprint=grid_occupancy(valid.to(x),size).clamp(0,1)
        masks=torch.minimum(occupancy,footprint)
        mass=masks.sum((-2,-1))
        den=mass.clamp_min(1e-8)
        z=F.layer_norm(x.permute(0,2,3,1),(self.channels,)).permute(0,3,1,2)
        z=self.project(z)
        mean=torch.einsum('bshw,bdhw->bsd',masks,z)/den[...,None]
        energy=frequency.to(x)*(masks/occupancy.clamp_min(1e-8))[:,:,None]
        band_mean=energy.sum((-2,-1))/den[...,None]
        cross=torch.einsum('bsfhw,bdhw->bsfd',energy,z)/den[...,None,None]
        covariance=cross-band_mean[...,None]*mean[:,:,None]
        return masks,mass,mean,covariance.flatten(-2)


class JointReadoutResNet50(PoolReplacementResNet50):
    def __init__(self, weight_path, classes=8, use_m1=True, use_m3=True,
                 merged=False, use_frequency=True, matching_mode=None):
        super().__init__(weight_path,classes=classes,
                         mode='readout24_merged' if merged else 'readout24_region')
        self.use_m1,self.use_m3,self.merged=use_m1,use_m3,merged
        self.m1=nn.ModuleList(ClassResponseFeedback(1024,classes) for _ in range(2)) if use_m1 else nn.ModuleList()
        self.m3=nn.ModuleList(IntegratedReferenceMatch(1024,use_frequency=use_frequency) for _ in range(2)) if use_m3 else nn.ModuleList()
        self.matching_mode=matching_mode
        if matching_mode is not None:
            if matching_mode not in ('separate','local','local_frequency') or not use_m3:
                raise ValueError('A separate-evidence mode requires M3.')
            from .multiscale_matching import SeparateEvidenceMatch
            self.m3=nn.ModuleList(SeparateEvidenceMatch(use_local=matching_mode!='separate',
                                      use_frequency=matching_mode=='local_frequency') for _ in range(2))

    def forward_from(self,prefix,masks,valid,available,start=2,*,slot_masks=None,frequency=None,
                     reference=None,pair_valid=None,return_details=False,
                     local_slot_masks=None,local_frequency=None):
        if start!=2:raise ValueError('Joint experiment starts from frozen layer2.')
        b,p=prefix.shape[:2]
        valid=valid*available[...,None,None,None].to(valid)
        masks=masks*available[...,None,None,None].to(masks)
        z=self.backbone.model.layer3(prefix.flatten(0,1))
        x=z.reshape(b,p,*z.shape[1:])
        r=None
        if self.use_m3:
            if reference is None or slot_masks is None or frequency is None or pair_valid is None:
                raise ValueError('M3 requires reference bank and explicit same-vessel exclusions.')
            ra=reference['available']
            rv=reference['valid']*ra[...,None,None,None].to(valid)
            rz=self.backbone.model.layer3(reference['prefix'].flatten(0,1))
            r=rz.reshape(len(ra),p,*rz.shape[1:])
        auxiliary,trace=[],{}
        if self.use_m1:
            result=[self.m1[pol](x[:,pol],grid_occupancy(valid[:,pol],x.shape[-2:]),True) for pol in range(p)]
            x=torch.stack([v[0] for v in result],1)
            aux=torch.stack([v[1]['aux_logits'] for v in result],1)*available[...,None].to(x)
            auxiliary.append(aux.sum(1)/available.sum(1,keepdim=True).clamp_min(1).to(x))
            if return_details:trace['m1']=[v[1] for v in result]
            if r is not None:
                r=torch.stack([self.m1[pol](r[:,pol],grid_occupancy(rv[:,pol],r.shape[-2:])) for pol in range(p)],1)
        if self.use_m3:
            result=[]
            for pol in range(p):
                bank=dict(features=r[:,pol],valid=rv[:,pol],
                          slot_masks=reference['slot_masks'][:,pol],frequency=reference['frequency'][:,pol])
                legal=pair_valid & available[:,pol,None].bool() & ra[None,:,pol].bool()
                extra={}
                if self.matching_mode in ('local','local_frequency'):
                    bank.update(local_features=reference['prefix'][:,pol],
                                local_slot_masks=reference['local_slot_masks'][:,pol],
                                local_frequency=reference['local_frequency'][:,pol])
                    extra=dict(local_features=prefix[:,pol],local_slot_masks=local_slot_masks[:,pol],
                               local_frequency=local_frequency[:,pol])
                result.append(self.m3[pol](x[:,pol],slot_masks[:,pol],valid[:,pol],frequency[:,pol],
                                          bank,pair_valid=legal,return_details=True,**extra))
            x=torch.stack([v[0] for v in result],1)
            if return_details:trace['m3']=[v[1] for v in result]
        z=self.backbone.model.layer4(x.flatten(0,1))
        global_token=regional_mean(z,valid.flatten(0,1),valid.flatten(0,1))[0][:,0].reshape(b,p,-1)
        region=masks.flatten(0,1)
        if self.merged:region=region.sum(1,keepdim=True).clamp(0,1).expand(-1,3,-1,-1)
        pooled,mass=regional_mean(prefix.flatten(0,1),region,valid.flatten(0,1))
        values=torch.cat((global_token,pooled.reshape(b,p,-1)),-1)*available[...,None].to(z)
        score=self.head(torch.cat((values.flatten(1),available.to(z)),1))
        if return_details:
            return score,dict(aux_logits=auxiliary,operators=trace,regional_tokens=pooled.reshape(b,p,3,-1),regional_mass=mass.reshape(b,p,3))
        return score

    def forward(self,images,masks,valid,available,*,slot_masks=None,frequency=None,
                reference=None,pair_valid=None,return_details=False,
                local_slot_masks=None,local_frequency=None):
        if self.use_m3 and reference is not None:
            reference={**reference,'prefix':self.prefix(reference['images'],stop=2)}
        return self.forward_from(self.prefix(images,stop=2),masks,valid,available,
                                 slot_masks=slot_masks,frequency=frequency,reference=reference,
                                 pair_valid=pair_valid,return_details=return_details,
                                 local_slot_masks=local_slot_masks,local_frequency=local_frequency)
