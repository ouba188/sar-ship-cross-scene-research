"""Current integration: late-axis M3 plus existing CCREL/PCEA, soft selection."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1]/'paper_draft_20261108'
spec = importlib.util.spec_from_file_location('unified_intervention_late_axis',
    PAPER/'m3_late_axis_evidence_20261118/model.py')
late = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = late
spec.loader.exec_module(late)
torch, Inputs = late.torch, late.Inputs
RATES = dict(backbone=1e-6, modules=1e-6, reader=1e-5, exchange=3e-4, selector=1e-4)
SPEC = dict(matcher='late_axis', coupling='paired', selection='soft',
            native_readout=[40,20], pcea='cross_mean', pcea_fine_residual=False,
            auxiliary_targets=['ccrel','exchange'], budget_weight=0.)
SOURCE_FILES = [Path(__file__),
    PAPER/'m3_late_axis_evidence_20261118/model.py',
    PAPER/'m3_content_scale_20261118/model.py',
    PAPER/'m3_content_scale_20261118/content_selection.py',
    PAPER/'m3_content_scale_20261118/native_masks.py',
    PAPER/'m3_axis_coupling_20261118/axis_operator.py',
    PAPER/'m3_native_readout_20261118/model.py']


class Model(late.Model):
    def __init__(self):
        # Identical soft rule for frozen teacher, student training and evaluation.
        super().__init__(selection_mode='soft', comparison='late', weighting='content')


def groups_for(net):
    buckets = {key: [] for key in RATES}
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('net.base.backbone.'):
            group = 'backbone'
        elif name.startswith('net.'):
            group = 'modules'
        elif name.startswith('ccrel.'):
            group = 'reader'
        elif name.startswith('exchange.'):
            group = 'exchange'
        elif name.startswith('selector.'):
            group = 'selector'
        else:
            raise ValueError(name)
        buckets[group].append(parameter)
    return [dict(name=key, params=values, lr=RATES[key], base_lr=RATES[key])
            for key, values in buckets.items() if values]
