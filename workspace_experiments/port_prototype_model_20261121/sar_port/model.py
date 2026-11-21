"""Composable native-scale prototype model; query labels never enter forward."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from .backbone import ShipBackbone
from .priors import PriorEffectStudent, attention_from_response
from .data import pool_native
from .selection import ContentScaleSelector, GROUPS, parent_sum, pack
from .matching import RegionalMatcher


@dataclass
class ModelConfig:
    classes: int = 8
    m1: bool = True
    m2: bool = True
    guidance: bool = True
    pcea: bool = True
    m3: bool = True
    priors: bool = True
    effects: bool = True
    effect_target: str = 'utility'
    utility_samples: int = 2
    size_prior: bool = True
    dynamic_selection: bool = True
    third_layer: bool = False
    gate: bool = False
    m1_strength: float = .3
    m2_strength: float = 1.
    pcea_strength: float = .1
    prior_hidden: int = 64
    selector_hidden: int = 32
    gate_hidden: int = 16
    initial_gate: float = .95
    selection_mode: str = 'soft'
    selector_temperature: float = 1.
    selector_threshold: float = .5
    layer_matching: str = 'concat'
    pair_matching: str = 'paired'
    axis_matching: bool = True
    dustbin: bool = True
    rejection_gate: bool = True
    class_competition: bool = True
    port_aggregation: str = 'flat'
    geometry_weight: float = .25
    scale_weight: float = 0.
    layer_weight: float = 0.
    residual_scale: float = .1
    match_dim: int = 32
    match_temperature: float = .1
    sinkhorn_steps: int = 60
    aux_weight: float = .02
    prior_weight: float = 0.
    effect_weight: float = 0.
    consistency_weight: float = 0.
    consistency_temperature: float = 1.
    consistency_warmup_steps: int = 100
    consistency_ramp_steps: int = 200
    donor_policy: str = 'same_class'
    gate_fusion: str = 'fixed'
    intervention_strength: float = .5
    budget_weight: float = 0.

    def __post_init__(self):
        if self.effect_target not in ('utility','frequency'):
            raise ValueError('effect_target must be utility or frequency')
        if not isinstance(self.utility_samples,int) or not 1 <= self.utility_samples <= 16:
            raise ValueError('utility_samples must be an integer in [1,16]')
        if self.donor_policy not in ('same_class','cross_class','any'):
            raise ValueError('donor_policy must be same_class, cross_class, or any')
        if self.gate_fusion not in ('fixed','learned'):
            raise ValueError('gate_fusion must be fixed or learned')
        if any(not isinstance(value,int) or value < 0 for value in
               (self.consistency_warmup_steps,self.consistency_ramp_steps)):
            raise ValueError('Consistency warmup and ramp steps must be nonnegative integers')
        if self.layer_matching not in ('concat','same','cross'):
            raise ValueError('layer_matching must be concat, same, or cross')
        if self.pair_matching not in ('paired','aggregate'):
            raise ValueError('pair_matching must be paired or aggregate')
        if self.selection_mode not in ('soft','hard','fine','coarse'):
            raise ValueError('selection_mode must be soft, hard, fine, or coarse')
        if self.port_aggregation not in ('flat','hierarchical'):
            raise ValueError('port_aggregation must be flat or hierarchical')
        if min(self.match_temperature,self.selector_temperature,self.sinkhorn_steps,self.match_dim)<=0:
            raise ValueError('Temperatures, matching dimension and iterations must be positive')
        if self.consistency_temperature<=0 or not 0<self.intervention_strength<=1:
            raise ValueError('Consistency temperature must be positive; intervention strength in (0,1]')
        if not 0<self.initial_gate<1 or not 0<=self.selector_threshold<=1:
            raise ValueError('Gate initialization and selector threshold must be probabilities')
        if self.prior_weight and not self.priors:
            raise ValueError('prior_weight requires priors')
        if self.effect_weight and not self.effects:
            raise ValueError('effect_weight requires effects')
        if self.consistency_weight and not self.gate:
            raise ValueError('consistency_weight requires gate')

    def to_dict(self):
        return asdict(self)


class PortPrototypeModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.backbone = ShipBackbone(cfg.classes,use_m1=cfg.m1,use_m2=cfg.m2,use_guidance=cfg.guidance,
                                     m1_strength=cfg.m1_strength,m2_strength=cfg.m2_strength)
        self.student = PriorEffectStudent(classes=cfg.classes,hidden=cfg.prior_hidden,effect_target=cfg.effect_target)
        self.deep_context = nn.Linear(1024,512,bias=False) if cfg.third_layer else None
        channels = 1792 if cfg.third_layer else 768
        conditions = cfg.classes*(int(cfg.priors)+(1 if cfg.effect_target == 'utility' else 2)*int(cfg.effects))
        conditions += 5*int(cfg.size_prior)
        self.selector = ContentScaleSelector(channels,cfg.selector_hidden,conditions,cfg.selector_temperature,cfg.selector_threshold)
        self.matcher = RegionalMatcher(cfg) if cfg.m3 else None
        if cfg.gate:
            from .disentangle import PolarDisentangler
            self.disentangler = PolarDisentangler(256,cfg.gate_hidden,cfg.initial_gate,fusion=cfg.gate_fusion)
        else:
            self.disentangler = None
        if cfg.pcea:
            from .exchange import SourcePolarExchange
            self.exchange = SourcePolarExchange(cfg.classes,strength=cfg.pcea_strength)
        else:
            self.exchange = None

    def _early(self,data,mode='content',donor=None):
        transform = None
        if self.disentangler is not None:
            transform = lambda f,v,a:self.disentangler(f,v,a,mode=mode,donor=donor)
        return self.backbone.early(data['images'],data['valid'],data['available'],data['local_slots'],transform)

    def _exchange(self,source):
        if self.exchange is None or source is None:
            return None
        return lambda x,early,data:self.exchange(x,early,data,source)

    def _tokens(self,early,data,layer3=None,selection_override=None):
        cfg = self.config
        f2 = early['f2']
        if layer3 is not None and self.deep_context is not None:
            from .region_ops import grid_occupancy
            valid = grid_occupancy(data['valid'].flatten(0,1),(10,10)).reshape(-1,2,1,10,10)
            context = (layer3*valid).sum((-1,-2))/valid.sum((-1,-2)).clamp_min(1e-12)
            # Deep semantic context conditions the prior; no invented fine detail.
            f2 = f2+self.deep_context(context)[...,None,None]
        regions = parent_sum(data['read20'],2) if cfg.effect_target == 'utility' else early['masks']
        prior = self.student(f2,early['valid20'],regions,data['available'])
        uniform = attention_from_response(torch.zeros_like(prior['attention']),early['valid20'],data['available'])
        attention = prior['attention'] if cfg.priors else uniform
        pieces = [pool_native(early['f1'].flatten(0,1),data['read40'].flatten(0,1)),
                  pool_native(early['f2'].flatten(0,1),data['read20'].flatten(0,1))]
        if cfg.third_layer:
            pieces.append(pool_native(layer3.flatten(0,1),data['slots'].flatten(0,1)))
        fine = torch.cat(pieces,-1).reshape(len(f2),2,20,-1)
        mass = data['read40'].sum((-1,-2))*data['available'][...,None]
        # A stores probability mass (including fractional valid area). Convert
        # to density before integrating masks, so boundary area is counted once.
        attention_density = attention/early['valid20'].clamp_min(1e-12)
        density = pool_native(attention_density.flatten(0,1),data['read20'].flatten(0,1)).reshape(len(f2),2,20,cfg.classes)
        parent_mass = parent_sum(mass,2)
        coarse_prior = parent_sum(density*mass[...,None],2)/parent_mass[...,None].clamp_min(1e-12)
        cond=[]
        if cfg.priors:
            # Relative attention density avoids a 1/(H*W) feature scale.
            cond.append(coarse_prior*early['valid20'].sum((-1,-2)).unsqueeze(-1))
        if cfg.effects:
            if cfg.effect_target == 'utility':
                effect = prior['utility'].permute(0,1,3,2)
            else:
                parent_regions = torch.tensor([0,0,2,2,2,2,1,1],device=f2.device)
                effect = prior['effects'].index_select(3,parent_regions).permute(0,1,3,2,4).flatten(-2)
            cond.append(effect)
        if cfg.size_prior:
            cond.append(data['size_condition'].to(fine)*data['available'][...,None,None])
        chosen = self.selector(fine,mass,data['fine_write'],data['eligible'],
                               cfg.selection_mode if cfg.dynamic_selection else 'fine',
                               override=selection_override,condition=torch.cat(cond,-1) if cond else None)
        chosen['token_prior'] = pack(density,coarse_prior,2).permute(0,1,3,2)
        chosen['available'] = data['available']
        if 'port_index' in data:
            chosen['port_index'] = data['port_index']
        return chosen,prior

    def encode_reference(self,data,*,mode='content',selection_override=None):
        early = self._early(data,mode=mode)
        source = self.exchange.encode_source(early) if self.exchange is not None else None
        layer3 = self.backbone.middle(early,data,self._exchange(source)) if self.config.third_layer else None
        tokens,_ = self._tokens(early,data,layer3,selection_override)
        if source is not None:
            tokens['source'] = source
        return tokens

    def describe(self,data,descriptor='body'):
        """Source-only target descriptor; global GAP remains a matched control."""
        early = self._early(data)
        if descriptor == 'body':
            body = data['read20'][:,:,:8].sum(2,keepdim=True)
            vector = pool_native(early['f2'].flatten(0,1),body.flatten(0,1)).reshape(len(body),2,-1)
        elif descriptor == 'global':
            valid = early['valid20']
            vector = (early['f2']*valid).sum((-1,-2))/valid.sum((-1,-2)).clamp_min(1e-12)
        else:
            raise ValueError('descriptor must be body or global')
        vector = vector*data['available'][...,None]
        return torch.nn.functional.normalize(vector.flatten(1),dim=-1)

    def forward(self,data,reference=None,reference_labels=None,legal=None,*,m3_on=True,details=False,
                mode='content',donor=None,selection_override=None):
        early = self._early(data,mode,donor)
        source = reference.get('source') if reference is not None else None
        if self.config.third_layer:
            layer3 = self.backbone.middle(early,data,self._exchange(source))
            tokens,prior = self._tokens(early,data,layer3,selection_override)
        else:
            tokens,prior = self._tokens(early,data,selection_override=selection_override)
            layer3 = self.backbone.middle(early,data,self._exchange(source))
        match = None
        if self.matcher is not None and reference is not None and m3_on:
            if reference_labels is None:
                raise ValueError('Reference class labels are required for prototype matching')
            if legal is None:
                legal = torch.ones(len(layer3),len(reference_labels),dtype=torch.bool,device=layer3.device)
            match = self.matcher(tokens,reference,reference_labels,legal,details)
            layer3 = layer3+match['delta']
        logits = self.backbone.finish(layer3,early,data)
        return dict(logits=logits,auxiliary=early['auxiliary'],responses=early['responses'],**prior,
                    scale_cost=tokens['scale_cost'],scale_trace=tokens['scale_trace'],matching=match)

    def consistency_loss(self,data,reference,reference_labels,legal,clean_logits,*,query_rows,
                         seed=None,details=False):
        from .disentangle import donor_permutation,weighted_consistency
        if seed is None:
            seed = int(torch.randint(0,2**31-1,()).item())
        donor,eligible = donor_permutation(query_rows,data['available'],seed=seed,policy=self.config.donor_policy)
        loss = clean_logits.sum()*0
        if bool(eligible.any()):
            changed = self(data,reference,reference_labels,legal,mode='swap',donor=donor)['logits']
            loss = weighted_consistency(changed,clean_logits,eligible,self.config.consistency_temperature)
        return dict(loss=loss,eligible_donors=int(eligible.sum())) if details else loss
