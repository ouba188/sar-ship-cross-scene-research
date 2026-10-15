"""Cache fixed R18 spatial features; run matched CE and utility-tail training."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18

from extract_opensarship2_transfer import WS, write_json
from neural_complement import ComplementTail, pilot_loss

PIXELS = WS/'artifacts/upper_bound_side_audit_20260916/eight_class_20260916/images_uint8.npy'
SEED = 20261007


def cache_prefix(out, prefix, pixels, name, device, channel=1):
    path = out/f'{name}_layer3.npy'
    if path.exists():
        raise ValueError(f'prefix cache already exists: {path}')
    result = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16,
                                     shape=(len(pixels), 256, 8, 8))
    started = time.monotonic()
    with torch.inference_mode():
        for lo in range(0, len(pixels), 128):
            hi = min(lo+128, len(pixels))
            x = torch.from_numpy(np.array(pixels[lo:hi, channel:channel+1])).to(device).float().repeat(1, 3, 1, 1)/255.
            z = prefix(x).cpu().numpy()
            if z.shape[1:] != (256, 8, 8) or not np.isfinite(z).all():
                raise ValueError('unexpected prefix output')
            result[lo:hi] = z
            if lo == 0 or lo % 8192 == 0 or hi == len(pixels):
                print(f'PREFIX {name} {hi}/{len(pixels)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    result.flush()
    return result


def extract_tail(out, model, source, target, device):
    model.eval()
    for name, spatial in [('source', source), ('target', target)]:
        features = np.lib.format.open_memmap(out/f'{name}_extra.npy', mode='w+',
            dtype=np.float32, shape=(len(spatial), 512))
        with torch.inference_mode():
            for lo in range(0, len(spatial), 256):
                hi = min(lo+256, len(spatial))
                x = torch.from_numpy(np.array(spatial[lo:hi])).to(device).float()
                values = model.features(x).float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError('nonfinite tail features')
                features[lo:hi] = values
        features.flush()
        print(f'FEATURES {out.name}/{name} rows={len(spatial)}', flush=True)
    write_json(out/'feature_recipe.json', dict(prefix='e206 frozen through layer3; float32 inference, float16 cache',
        tail='float32 layer4 inference and global average pooling', arm=out.name,
        source_rows=len(source), target_rows=len(target),
        base='original paired R50 unchanged', learned_attention=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if (out/'TRAINING_COMPLETE.json').exists():
        raise ValueError('completed pilot must not be overwritten')
    protocol = json.loads((out/'protocol.json').read_text())
    config = protocol['fixed_training']
    target_dir = Path(protocol['target'])
    with np.load(out/'source_episode_scores.npz') as archive:
        source_rows, teacher, eligible, labels = (archive[k] for k in
            ('source_rows', 'base_scores', 'eligible', 'labels'))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda')
    checkpoint = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    backbone = resnet18(weights=None)
    missing, unexpected = backbone.load_state_dict(checkpoint['backbone'], strict=False)
    if set(missing) != {'fc.weight', 'fc.bias'} or unexpected:
        raise ValueError('unexpected e206 prefix checkpoint')
    prefix = torch.nn.Sequential(*list(backbone.children())[:-3]).to(device).eval().requires_grad_(False)
    source_pixels = np.load(PIXELS, mmap_mode='r')
    target_pixels = np.load(target_dir/'inputs128.npy', mmap_mode='r')
    np.save(out/'target_feature_rows.npy', np.load(target_dir/'feature_rows.npy'))
    source = cache_prefix(out, prefix, source_pixels, 'source', device)
    target = cache_prefix(out, prefix, target_pixels, 'target', device)
    del prefix, backbone, source_pixels, target_pixels
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    model = ComplementTail(checkpoint['backbone'])
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    torch.save(initial, out/'initial_tail.pt')
    model = model.to(device)
    frozen = out/'frozen_replay'
    frozen.mkdir(exist_ok=True)
    extract_tail(frozen, model, source, target, device)
    counts = np.bincount(labels[source_rows], minlength=8)
    class_weights = torch.tensor(len(source_rows)/(8*counts), dtype=torch.float32, device=device)
    for arm in ('supervised', 'utility_tail'):
        arm_dir = out/arm
        arm_dir.mkdir(exist_ok=True)
        model.load_state_dict(initial)
        torch.manual_seed(SEED)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        history = []
        started = time.monotonic()
        for epoch in range(config['epochs']):
            order = np.random.default_rng(SEED+epoch).permutation(source_rows)
            model.train()
            totals = dict(classification=0., utility=0., tail=0.)
            samples = 0
            for lo in range(0, len(order), config['batch_size']):
                rows = order[lo:lo+config['batch_size']]
                x = torch.from_numpy(np.array(source[rows])).to(device).float()
                y = torch.as_tensor(labels[rows], dtype=torch.long, device=device)
                base = torch.from_numpy(teacher[rows]).to(device)
                allowed = torch.as_tensor(eligible[rows], dtype=torch.bool, device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, utility = model(x, base)
                    loss, parts = pilot_loss(logits, utility, y, base, allowed, class_weights,
                                            candidate=arm == 'utility_tail')
                if not torch.isfinite(loss):
                    raise ValueError(f'nonfinite {arm} training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                samples += len(rows)
                for key in totals:
                    totals[key] += float(parts[key])*len(rows)
            record = dict(epoch=epoch+1, **{k: v/samples for k, v in totals.items()},
                          elapsed_seconds=time.monotonic()-started)
            history.append(record)
            write_json(arm_dir/'training_history.json', history)
            print(f'TRAIN {arm} {record}', flush=True)
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save(state, arm_dir/'final_tail.pt')
        extract_tail(arm_dir, model, source, target, device)
    write_json(out/'TRAINING_COMPLETE.json', dict(arms=['frozen_replay', 'supervised', 'utility_tail'],
        epochs=config['epochs'], source_training_rows=len(source_rows), target_training_labels=0,
        pixel_cache=str(PIXELS), seed=SEED, same_initialization_and_minibatch_orders=True,
        gradient_clip_norm=5., frozen_prefix_shared=True))


if __name__ == '__main__':
    main()
