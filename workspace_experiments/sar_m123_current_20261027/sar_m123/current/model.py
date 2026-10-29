"""Full checkpoints and direct chip-to-logit inference; no experiment imports."""
from pathlib import Path

import torch

from ..bottleneck_evidence import BottleneckEvidenceResNet50
from .spatial_m12 import SpatialM12ResNet50
from .data import batch, pair_mask, prepare_record

CLASSES = ['bulk_carrier', 'fishing', 'general_cargo', 'chemical_tanker',
           'container', 'crude_oil_tanker', 'tug', 'offshore']
LIVE = tuple(f'backbone.model.layer3.5.conv{i}.weight' for i in (1, 2, 3))


def setup(device='cuda'):
    torch.set_num_threads(2)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; pass --device cpu.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def load_model(path, device='cpu'):
    blob = torch.load(path, map_location='cpu', weights_only=True)
    architecture = blob.get('architecture')
    if architecture == SpatialM12ResNet50.architecture:
        model = SpatialM12ResNet50(None, position='pre3x3', policy='local', classes=8)
    elif architecture == 'native160_pre3x3_m123':
        model = BottleneckEvidenceResNet50(None, position='pre3x3', policy='local', classes=8)
    else:
        raise ValueError('Expected a complete current-model checkpoint, not an adaptation delta.')
    model.load_state_dict(blob['state'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def save_model(model, path, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(architecture=getattr(model, 'architecture', 'native160_pre3x3_m123'), classes=CLASSES,
        state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        metadata=metadata), path)


def training_policy(model, stage):
    model.requires_grad_(False)
    if stage == 'fit':
        for name, parameter in model.named_parameters():
            if not name.startswith(('backbone.', 'anchor_last.')):
                parameter.requires_grad_(True)
        for operators in model.m3:
            for operator in operators:
                operator.reader2.requires_grad_(False)
                operator.reader3.requires_grad_(False)
    elif stage != 'adapt':
        raise ValueError(stage)
    for name, parameter in model.named_parameters():
        if name in LIVE:
            parameter.requires_grad_(True)
    # Frozen BN state is part of the actual trained protocol.
    model.eval()


def reference_bank(model, records, device):
    bank = batch([prepare_record(r) for r in records], device)
    with torch.no_grad():
        bank['prefix'] = model.prefix(bank['images'], stop=2)
    return bank


def forward_batch(model, values, references, legal, details=False):
    with torch.no_grad():
        prefix = model.prefix(values['images'], stop=2)
    return model.forward_from(prefix, values['masks'], values['valid'], values['available'],
        **{k: v for k, v in values.items() if k not in ('images', 'masks', 'valid', 'available')},
        reference=references, pair_valid=legal, return_details=details)


@torch.no_grad()
def predict_records(model, records, references, device='cpu', batch_size=16):
    bank = reference_bank(model, references, device)
    outputs = []
    for first in range(0, len(records), batch_size):
        rows = records[first:first+batch_size]
        values = batch([prepare_record(r) for r in rows], device)
        outputs.append(forward_batch(model, values, bank, pair_mask(rows, references, device)).cpu())
    return torch.cat(outputs)
