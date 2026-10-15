"""Learn paired spatial evidence through the retained support-adapted expert."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from aligned_residual_ops import FixedExpertGeometry, residual_scores
from polar_residual_expert import PolarResidualExpert
from support_visible_polar import SupportVisibleGeometry, SupportVisiblePolarExpert
from extract_opensarship2_transfer import WS, write_json


def tensor_batch(array, rows, device):
    return torch.from_numpy(np.array(array[rows])).to(device).float()


def extract_target(folder, model, vh, vv, xs, geometry, device, ablation_folder=None):
    model.eval()
    destination = np.lib.format.open_memmap(folder/'target_extra.npy', mode='w+',
        dtype=np.float32, shape=(len(vh), 512))
    ablated = None
    if ablation_folder is not None:
        ablation_folder.mkdir(exist_ok=True)
        ablated = np.lib.format.open_memmap(ablation_folder/'target_extra.npy', mode='w+',
            dtype=np.float32, shape=(len(vh), 512))
    totals = dict(residual_norm=0., contrast_norm=0., base_norm=0., spatial_feature_change=0.)
    with torch.inference_mode():
        for lo in range(0, len(vh), 256):
            hi = min(lo+256, len(vh))
            rows = slice(lo, hi)
            h, v = model.encode(tensor_batch(vh, rows, device), tensor_batch(vv, rows, device))
            x = tensor_batch(xs, rows, device)
            features, details = model.fuse(h, v, x, geometry, return_details=True)
            if not torch.isfinite(features).all():
                raise ValueError('nonfinite paired target features')
            destination[rows] = features.float().cpu().numpy()
            totals['residual_norm'] += float(details['residual_norm'].sum())
            totals['contrast_norm'] += float(details['contrast_norm'].sum())
            totals['base_norm'] += float(details['base_features'].norm(dim=1).sum())
            if ablated is not None:
                without = model.fuse(h, v, x, geometry, disable_contrast=True)
                ablated[rows] = without.float().cpu().numpy()
                totals['spatial_feature_change'] += float((features-without).norm(dim=1).sum())
            if lo == 0 or lo % 32768 == 0 or hi == len(vh):
                print(f'POLAR FEATURES {folder.name} {hi}/{len(vh)}', flush=True)
    destination.flush()
    if ablated is not None:
        ablated.flush()
    write_json(folder/'information_readout.json', dict(rows=len(vh),
        means={key: value/len(vh) for key, value in totals.items()},
        paired_vv_used=True, calibration='learned image features, not a physical backscatter ratio',
        interpretation='Activation/sensitivity only; accuracy benefit must be read from fixed predictions.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if (out/'TRAINING_COMPLETE.json').exists():
        raise ValueError('completed paired training exists')
    if not (out/'VV_CACHE_COMPLETE.json').exists():
        raise ValueError('missing completed VV prefix cache')
    protocol = json.loads((out/'protocol.json').read_text())
    config = protocol['fixed_training']
    preserving = protocol.get('prior_preserving', False)
    geometry_class = SupportVisibleGeometry if preserving else FixedExpertGeometry
    model_class = SupportVisiblePolarExpert if preserving else PolarResidualExpert
    original = Path(protocol['original'])
    source_h = np.load(original/'source_layer3.npy', mmap_mode='r')
    target_h = np.load(original/'target_layer3.npy', mmap_mode='r')
    vv_root = Path(protocol.get('vv_cache_root', str(out)))
    source_v = np.load(vv_root/'source_vv_layer3.npy', mmap_mode='r')
    target_v = np.load(vv_root/'target_vv_layer3.npy', mmap_mode='r')
    target_xs = np.load(out/'target_xs.npy', mmap_mode='r')
    with np.load(original/'source_episode_scores.npz') as saved:
        counts = np.bincount(saved['labels'][saved['source_rows']], minlength=8)
    descriptors = json.loads(Path(protocol['source_episodes']).read_text())['episodes']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    device = torch.device('cuda')
    checkpoint = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    initial_model = model_class(checkpoint['backbone'], mode='contrast')
    initial = {key: value.clone() for key, value in initial_model.state_dict().items()}
    del initial_model
    with np.load(out/'deployment_geometry.npz') as saved:
        deployment = geometry_class({key: saved[key] for key in saved.files}).to(device).float()
    episodes, geometries = [], []
    for descriptor in descriptors:
        with np.load(descriptor['path']) as saved:
            episode = {key: saved[key] for key in saved.files}
        episodes.append(episode)
        geometries.append(geometry_class(episode).to(device).float())
    weights = torch.as_tensor(counts.sum()/(8*counts), dtype=torch.float32, device=device)
    for arm in protocol['training_arms']:
        folder = out/arm
        folder.mkdir(exist_ok=True)
        model = model_class(checkpoint['backbone'], mode=arm.removeprefix('polar_'))
        model.load_state_dict(initial)
        model = model.to(device)
        optimizer = torch.optim.AdamW([
            dict(params=model.layer4.parameters(), lr=config['backbone_lr']),
            dict(params=[p for name, p in model.named_parameters() if not name.startswith('layer4.')],
                 lr=config['module_lr'])], weight_decay=config['weight_decay'])
        started = time.monotonic()
        history = []
        for epoch in range(config['epochs']):
            rng = np.random.default_rng(config['seed']+epoch)
            loss_total, samples, steps = 0., 0, 0
            model.train()
            for ei in rng.permutation(len(episodes)):
                episode, geometry = episodes[ei], geometries[ei]
                support = episode['support']
                query = rng.permutation(episode['query'])
                sy = torch.as_tensor(episode['labels'][support], dtype=torch.long, device=device)
                for lo in range(0, len(query), config['query_batch_size']):
                    q = query[lo:lo+config['query_batch_size']]
                    indices = np.r_[support, q]
                    rows = episode['rows'][indices]
                    vh, vv = tensor_batch(source_h, rows, device), tensor_batch(source_v, rows, device)
                    xs = torch.as_tensor(episode['xs'][indices], dtype=torch.float32, device=device)
                    labels = torch.as_tensor(episode['labels'][q], dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        h, v = model.encode(vh, vv)
                    # Frozen ridge geometry and residual solve stay in float32.
                    features = model.fuse(h.float(), v.float(), xs, geometry)
                    prior, phi = geometry(xs, features)
                    adapted = residual_scores(phi[:64], prior[:64], sy, phi[64:], prior[64:])
                    per_row = F.cross_entropy(adapted, labels, reduction='none')
                    loss = (per_row*weights[labels]).sum()/weights[labels].sum()
                    if not torch.isfinite(loss):
                        raise ValueError(f'nonfinite paired training loss: {arm}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
                    optimizer.step()
                    loss_total += float(loss.detach())*len(q)
                    samples += len(q)
                    steps += 1
            record = dict(epoch=epoch+1, loss=loss_total/samples, query_rows=samples,
                steps=steps, elapsed_seconds=time.monotonic()-started)
            history.append(record)
            write_json(folder/'training_history.json', history)
            print(f'POLAR TRAIN {arm} {record}', flush=True)
        torch.save(dict(state={key: value.detach().cpu() for key, value in model.state_dict().items()},
            mode=model.mode, prior_preserving=preserving), folder/'expert.pt')
        extract_target(folder, model, target_h, target_v, target_xs, deployment, device,
            ablation_folder=out/'contrast_no_spatial' if arm == 'polar_contrast' else None)
        del model, optimizer
    write_json(out/'TRAINING_COMPLETE.json', dict(training_arms=protocol['training_arms'],
        evaluation_arms=protocol['evaluation_arms'], actual_new_expert_input='VV spatial features',
        gradient_through_support_adaptation=True, target_labels_used_for_encoder=0,
        no_head_refit=True, prior_preserving=preserving, backbone_frozen=preserving,
        checkpoints='final fixed epoch', independent_confirmation=False))


if __name__ == '__main__':
    main()
