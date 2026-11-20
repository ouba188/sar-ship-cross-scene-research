"""Full local transport with late axis comparison and absolute rejection strength."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .selection import GROUPS, REGIONS, pack


def sinkhorn(cost, query_mass, reference_mass, allowed, steps=60, temperature=.1, dustbin=True):
    a = query_mass/query_mass.sum(-1,keepdim=True).clamp_min(1e-20)
    b = reference_mass/reference_mass.sum(-1,keepdim=True).clamp_min(1e-20)
    kernel = (cost/temperature).masked_fill(~allowed, -1e4)
    if dustbin:
        a = torch.cat((a,2-a.sum(-1,keepdim=True)),-1)
        b = torch.cat((b,2-b.sum(-1,keepdim=True)),-1)
        kernel = F.pad(kernel,(0,1,0,1))
    u,v = torch.zeros_like(a),torch.zeros_like(b)
    for _ in range(steps):
        u = a.clamp_min(1e-20).log()-torch.logsumexp(kernel+v[...,None,:],-1)
        v = b.clamp_min(1e-20).log()-torch.logsumexp(kernel+u[..., :,None],-2)
    plan = (kernel+u[..., :,None]+v[...,None,:]).exp()
    return (plan[...,:-1,:-1] if dustbin else plan)*allowed


def slot_geometry():
    fine = torch.tensor([(u,v) for u in (-.375,-.125,.125,.375) for v in (-.25,.25)]
                        +[(u,v) for u in (-.375,-.125,.125,.375) for v in (-.75,.75)]
                        +[(u,v) for u in (-.625,.625) for v in (-.25,.25)])
    extent = torch.tensor([.25,.5]).expand(20,2).clone()
    coarse = torch.stack([fine[list(g)].mean(0) for g in GROUPS])
    coarse_extent = torch.stack([(fine[list(g)]+extent[list(g)]/2).amax(0)
                                 -(fine[list(g)]-extent[list(g)]/2).amin(0) for g in GROUPS])
    return pack(fine,coarse,0),pack(extent,coarse_extent,0)


def masked_softmax(logits, allowed, dim):
    weights = logits.masked_fill(~allowed,-1e4).softmax(dim)*allowed
    return weights/weights.sum(dim,keepdim=True).clamp_min(1e-20)


def late_comparison(query, reference, plans, orientation, metric):
    """Compare per-reference axis hypotheses before averaging distances."""
    mass = plans.sum(-1)
    values = torch.einsum('brhij,rjd->brhid',plans,reference)/mass[...,None].clamp_min(1e-20)
    weights = orientation[...,None]*mass
    weights = weights/weights.sum(2,keepdim=True).clamp_min(1e-20)
    distance = ((query[:,None,None]-values).square()*metric[:,:,None,None]).sum(-1)
    return (weights*distance).sum(2)


def absolute_reliability(plan, input_mass):
    """Retain absolute real-match mass, independently of normalized values."""
    return (plan.sum(-1)/input_mass.clamp_min(1e-20)).clamp(0,1)*(input_mass>0)


class RegionalMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.channels = (256,512,1024) if config.third_layer else (256,512)
        self.layers = len(self.channels)
        width = config.match_dim
        self.project = nn.ModuleList(nn.ModuleList(nn.Sequential(nn.LayerNorm(c),nn.Linear(c,width,bias=False),
                                                       nn.LayerNorm(width)) for c in self.channels) for _ in range(2))
        merged = width*self.layers if config.layer_matching=='concat' else width
        self.key = nn.ModuleList(nn.Sequential(nn.Linear(merged,width,bias=False),nn.LayerNorm(width)) for _ in range(2))
        self.value = nn.ModuleList(nn.Sequential(nn.Linear(merged,width,bias=False),nn.LayerNorm(width)) for _ in range(2))
        self.metric = nn.Sequential(nn.Linear(3*width,16),nn.GELU(),nn.Linear(16,width))
        self.output = nn.ModuleList(nn.Conv2d(config.classes,1024,1,bias=False) for _ in range(2))
        # Small nonzero initialization lets classification reach matcher on step one.
        for layer in self.output:
            nn.init.normal_(layer.weight,std=.001)
        position,extent = slot_geometry()
        self.register_buffer('position',position)
        self.register_buffer('extent',extent)

    def encode(self, desc, pol):
        pieces = desc.split(self.channels,-1)
        projected = [layer(x) for layer,x in zip(self.project[pol],pieces)]
        joined = torch.cat(projected,-1) if self.config.layer_matching=='concat' else torch.stack(projected,1).flatten(1,2)
        return F.normalize(self.key[pol](joined),dim=-1),self.value[pol](joined)

    def forward(self, query, reference, labels, legal, details=False):
        cfg = self.config
        q,r = query['desc'],reference['desc']
        batch,count = len(q),len(r)
        class_labels = F.one_hot(labels.long(),cfg.classes).to(q)
        repeats = 1 if cfg.layer_matching=='concat' else self.layers
        positions = self.position.repeat(repeats,1)
        extents = self.extent.repeat(repeats,1)
        layer_id = torch.arange(repeats,device=q.device).repeat_interleave(28)
        updates,traces,all_reliability = [],[],[]
        all_evidence = q.new_zeros(batch,cfg.classes)
        for pol in range(2):
            kq,vq = self.encode(q[:,pol],pol)
            kr,vr = self.encode(r[:,pol],pol)
            qm = query['mass'][:,pol].repeat(1,repeats)/repeats
            rm = reference['mass'][:,pol].repeat(1,repeats)/repeats
            qp = query['token_prior'][:,pol].repeat(1,1,repeats)
            rp = reference['token_prior'][:,pol].repeat(1,1,repeats)
            qweighted = qm[:,None]*qp
            rweighted = rm*rp[torch.arange(count,device=q.device),labels]
            legal_pol = legal & query['available'][:,pol,None].bool() & reference['available'][None,:,pol].bool()
            records,axis_scores,axis_live = [],[],[]
            for region in REGIONS:
                ix = torch.cat([region.to(q.device)+28*l for l in range(repeats)])
                qa0 = qweighted[:,labels][:,:,ix]
                ra0 = rweighted[None,:,ix].expand(batch,-1,-1)
                live = legal_pol & (qa0.sum(-1)>0) & (ra0.sum(-1)>0)
                ref_mass = rm[:,ix]
                mass = live[...,None]*ref_mass[None]
                total = torch.einsum('brj,rc->bc',mass,class_labels)
                available = total>0
                mu = torch.einsum('brj,rc,rjd->bcd',mass,class_labels,kr[:,ix])/total[...,None].clamp_min(1e-20)
                nc = available.sum(-1,keepdim=True)
                other = (mu.sum(1,keepdim=True)-mu)/(nc-available.long()).clamp_min(1)[...,None]
                metric = 1+.5*self.metric(torch.cat((mu,other,mu-other),-1)).tanh()
                scale = metric[:,labels].sqrt()[:,:,None]
                sim = torch.einsum('brid,brjd->brij',F.normalize(kq[:,None,ix]*scale,dim=-1),
                                   F.normalize(kr[None,:,ix]*scale,dim=-1))
                qa = qa0/qa0.sum(-1,keepdim=True).clamp_min(1e-20)
                ra = ra0/ra0.sum(-1,keepdim=True).clamp_min(1e-20)
                allowed = live[...,None,None] & (qa>0)[...,None] & (ra>0)[...,None,:]
                if cfg.layer_matching=='same':
                    allowed = allowed & (layer_id[ix,None]==layer_id[None,ix])
                if cfg.class_competition:
                    best = torch.stack([sim.masked_fill(~(allowed & (labels==c)[None,:,None,None]),-1e4).amax((1,3))
                                        for c in range(cfg.classes)],1)
                    active_class = available[...,None] & (qweighted[:,:,ix]>0)
                    best = best*active_class
                    other_best = (best.sum(1,keepdim=True)-best)/(nc-available.long()).clamp_min(1)[...,None]
                    competition = (best-other_best)*active_class*(nc>1)[...,None]
                    conditioned = sim+competition[:,labels,:,None]
                else:
                    conditioned = sim
                costs,plans = [],[]
                for sign in ((1.,-1.) if cfg.axis_matching else (1.,)):
                    geometry = (positions[ix,None]-sign*positions[None,ix]).square().sum(-1)
                    extent_cost = (extents[ix,None].log()-extents[None,ix].log()).abs().sum(-1)
                    depth_cost = (layer_id[ix,None]-layer_id[None,ix]).abs()
                    cost = conditioned-cfg.geometry_weight*geometry-cfg.scale_weight*extent_cost-cfg.layer_weight*depth_cost
                    plan = sinkhorn(cost,qa,ra,allowed,cfg.sinkhorn_steps,cfg.match_temperature,cfg.dustbin)
                    if cfg.pair_matching=='aggregate':
                        row,col = plan.sum(-1),plan.sum(-2)
                        if cfg.layer_matching=='same':
                            aggregate = torch.zeros_like(plan)
                            for level in range(repeats):
                                member = layer_id[ix]==level
                                block_mass = row[...,member].sum(-1)
                                block = row[...,None]*col[...,None,:]/block_mass[...,None,None].clamp_min(1e-20)
                                aggregate = aggregate+block*(member[:,None]&member[None,:])
                            plan = aggregate
                        else:
                            plan = row[...,None]*col[...,None,:]/row.sum(-1)[...,None,None].clamp_min(1e-20)
                    costs.append(cost);plans.append(plan)
                plans = torch.stack(plans,2)
                costs = torch.stack(costs,2)
                mass_h = plans.sum((-1,-2))
                axis_scores.append((plans*costs).sum((-1,-2))/mass_h.clamp_min(1e-20))
                axis_live.append(mass_h>0)
                records.append((ix,qa,available,metric,sim,plans))
            scores,live_h = torch.stack(axis_scores),torch.stack(axis_live)
            scores = (scores*live_h).sum(0)/live_h.sum(0).clamp_min(1)
            orientation = masked_softmax(scores/cfg.match_temperature,live_h.any(0),-1)
            field = q.new_zeros(batch,28,cfg.classes)
            pol_trace,pol_rel = [],[]
            for ix,qa,available,metric,sim,plans in records:
                original = (plans*orientation[...,None,None]).sum(2)
                accepted = original.sum(-1)
                similarity = (original*sim).sum(-1)/accepted.clamp_min(1e-20)
                beta = masked_softmax(similarity/cfg.match_temperature,accepted>0,1)
                class_beta = beta[:,:,None]*class_labels[None,:,:,None]
                class_beta = class_beta/class_beta.sum(1,keepdim=True).clamp_min(1e-20)
                if cfg.port_aggregation=='hierarchical':
                    ports = reference['port_index']
                    hierarchical = torch.zeros_like(class_beta)
                    for c in range(cfg.classes):
                        port_scores,within,indices = [],[],[]
                        for port in ports[labels==c].unique():
                            members = torch.where((labels==c)&(ports==port))[0]
                            group_live = accepted[:,members]>0
                            group_score = similarity[:,members].masked_fill(~group_live,-1e4)
                            port_scores.append(group_score.amax(1))
                            within.append(masked_softmax(group_score/cfg.match_temperature,group_live,1))
                            indices.append(members)
                        if not indices:
                            continue
                        port_scores = torch.stack(port_scores,1)
                        port_weights = masked_softmax(port_scores/cfg.match_temperature,port_scores>-1e4,1)
                        for i,members in enumerate(indices):
                            hierarchical[:,members,c] = within[i]*port_weights[:,i,None]
                    class_beta = hierarchical
                distance_ref = late_comparison(vq[:,ix],vr[:,ix],plans,orientation,metric[:,labels])
                distance = torch.einsum('brci,bri->bci',class_beta,distance_ref)
                rel_ref = absolute_reliability(original,qa)
                reliability = torch.einsum('brci,bri->bci',class_beta,rel_ref)
                active = available[...,None] & (class_beta.sum(1)>0)
                evidence = -distance/(cfg.match_temperature*cfg.match_dim)*active
                centered = (evidence-evidence.sum(1,keepdim=True)/active.sum(1,keepdim=True).clamp_min(1))*active
                if cfg.rejection_gate:
                    centered = centered*reliability
                # Absolute prior region weight cannot disappear inside region-normalized OT.
                region_strength = qweighted[:,:,ix].sum(-1)/qweighted.sum(-1).clamp_min(1e-20)
                message = centered*region_strength[...,None]
                message = message.reshape(batch,cfg.classes,repeats,-1).mean(2)
                target = (ix[:len(ix)//repeats] % 28)
                field[:,target] = message.transpose(1,2)
                all_evidence = all_evidence+message.mean(-1)/2
                pol_rel.append(reliability.mean(-1))
                if details:
                    pol_trace.append(dict(transport=original,orientation=orientation,reliability=reliability,
                                          evidence=centered,region_strength=region_strength))
            mixed = torch.einsum('bnhw,bnc->bchw',query['write'][:,pol],field)
            updates.append(cfg.residual_scale*self.output[pol](mixed))
            all_reliability.append(torch.stack(pol_rel,1))
            if details: traces.append(pol_trace)
        return dict(delta=torch.stack(updates,1),evidence=all_evidence,
                    reliability=torch.stack(all_reliability,1),trace=traces)
