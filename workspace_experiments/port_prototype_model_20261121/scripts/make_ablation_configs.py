"""Generate explicit single-module controls from a chosen JSON configuration."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
from sar_port.model import ModelConfig

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='configs/full.json')
    parser.add_argument('--output',default='configs/ablations')
    args=parser.parse_args()
    baseline=ModelConfig(**json.loads(Path(args.base).read_text(encoding='utf-8'))).to_dict()
    changes={
        'no_m1':dict(m1=False), 'no_m2':dict(m2=False), 'no_guidance':dict(guidance=False),
        'no_pcea':dict(pcea=False), 'no_m3':dict(m3=False),
        'no_prior':dict(priors=False,prior_weight=0), 'no_effect':dict(effects=False,effect_weight=0),
        'fixed_fine':dict(dynamic_selection=False,budget_weight=0),
        'fixed_coarse':dict(selection_mode='coarse',budget_weight=0),
        'no_third_layer':dict(third_layer=False), 'no_gate':dict(gate=False,consistency_weight=0),
        'same_layer':dict(layer_matching='same'), 'cross_layer':dict(layer_matching='cross'),
        'aggregate':dict(pair_matching='aggregate'), 'no_axis':dict(axis_matching=False),
        'no_dustbin':dict(dustbin=False), 'no_rejection_gate':dict(rejection_gate=False),
        'no_class_competition':dict(class_competition=False),
        'hierarchical_ports':dict(port_aggregation='hierarchical'),
        'hard_selection':dict(selection_mode='hard',budget_weight=0),
        'no_size_prior':dict(size_prior=False),
        'frequency_effect':dict(effect_target='frequency'),
        'gate_ce_only':dict(consistency_weight=0),
        'cross_class_donor':dict(donor_policy='cross_class'),
        'learned_fusion':dict(gate_fusion='learned'),
    }
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for name,patch in changes.items():
        value=ModelConfig(**dict(baseline,**patch)).to_dict()
        (output/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(changes)} explicit ablation configurations; no training started.')

if __name__=='__main__':main()
