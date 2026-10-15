"""Matched aligned-only and complement-enabled deployment-aligned training."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from aligned_residual_ops import FixedExpertGeometry, residual_scores
from joint_residual_ops import ComplementExpertGeometry
from neural_complement import ComplementTail
from residual_expert import fit_correction
from train_neural_complement import extract_tail
from extract_opensarship2_transfer import WS, write_json


def numpy_geometry(xs, extra, base_discarded, arrays, complement):
    centered = extra-arrays['ssl_mean']
    projected = centered@arrays['ssl_components'].T
    xe = (np.column_stack((xs, projected))-arrays['expert_center'])/arrays['expert_scale']
    prior = xe@arrays['expert_coef'].T+arrays['expert_intercept']
    retained = xe/np.sqrt(xe.shape[1])
    if complement:
        residual = centered-projected@arrays['ssl_components']
        start = base_discarded.shape[1]
        normalized = (residual-arrays['discarded_center'][start:])/arrays['discarded_scale'][start:]
        discarded = np.column_stack((base_discarded, normalized))
        features = np.column_stack((retained/np.sqrt(2), discarded/np.sqrt(2*discarded.shape[1])))
    else:
        features = retained
    return prior, np.column_stack((features, np.ones(len(features))))


def check_operator(geometry, model, episode, source, device, complement):
    selected = np.r_[episode['support'], episode['query'][:32]]
    pixels = torch.from_numpy(np.array(source[episode['rows'][selected]])).to(device).float()
    with torch.no_grad():
        extra = model.features(pixels)
        xs = torch.as_tensor(episode['xs'][selected], device=device)
        bd = torch.as_tensor(episode['base_discarded'][selected], device=device)
        prior, phi = geometry(xs, extra, bd) if complement else geometry(xs, extra)
        support_y = episode['labels'][episode['support']]
        actual = residual_scores(phi[:64], prior[:64], torch.as_tensor(support_y, device=device), phi[64:], prior[64:])
    np_prior, np_phi = numpy_geometry(episode['xs'][selected].astype(float),
        extra.cpu().numpy().astype(float), episode['base_discarded'][selected].astype(float), episode, complement)
    expected = np_prior[64:]+np_phi[64:]@fit_correction(np_phi[:64], support_y, np_prior[:64])
    error = float(np.max(np.abs(actual.cpu().numpy()-expected)))
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=2e-4, rtol=2e-4)
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if (out/'TRAINING_COMPLETE.json').exists():
        raise ValueError('training already completed')
    protocol = json.loads((out/'protocol.json').read_text())
    config = protocol['fixed_training']
    previous = Path(protocol['previous'])
    source = np.load(previous/'source_layer3.npy', mmap_mode='r')
    target = np.load(previous/'target_layer3.npy', mmap_mode='r')
    with np.load(previous/'source_episode_scores.npz') as saved:
        counts = np.bincount(saved['labels'][saved['source_rows']], minlength=8)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    device = torch.device('cuda')
    checkpoint = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    model = ComplementTail(checkpoint['backbone'])
    model.load_state_dict(torch.load(previous/'initial_tail.pt', map_location='cpu', weights_only=True))
    model.classifier.requires_grad_(False)
    model.utility.requires_grad_(False)
    initial = {key: value.clone() for key, value in model.layer4.state_dict().items()}
    model = model.to(device)
    weights = torch.as_tensor(counts.sum()/(8*counts), dtype=torch.float32, device=device)
    descriptors = json.loads((out/'EPISODES_COMPLETE.json').read_text())['episodes']
    episodes = []
    for descriptor in descriptors:
        with np.load(descriptor['path']) as saved:
            episodes.append({key: saved[key] for key in saved.files})
    errors = {}
    for arm in protocol['trained_arms']:
        folder = out/arm
        folder.mkdir()
        complement = arm == 'joint'
        cls = ComplementExpertGeometry if complement else FixedExpertGeometry
        geometries = [cls(episode).to(device).float() for episode in episodes]
        model.layer4.load_state_dict(initial)
        model.eval()
        errors[arm] = check_operator(geometries[0], model, episodes[0], source, device, complement)
        optimizer = torch.optim.AdamW(model.layer4.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        history = []
        started = time.monotonic()
        for epoch in range(config['epochs']):
            rng = np.random.default_rng(config['seed']+epoch)
            total, count, steps = 0., 0, 0
            model.train()
            for ei in rng.permutation(len(episodes)):
                episode, geometry = episodes[ei], geometries[ei]
                support, query = episode['support'], rng.permutation(episode['query'])
                sy = torch.as_tensor(episode['labels'][support], dtype=torch.long, device=device)
                for lo in range(0, len(query), config['query_batch_size']):
                    q = query[lo:lo+config['query_batch_size']]
                    selected = np.r_[support, q]
                    pixels = torch.from_numpy(np.array(source[episode['rows'][selected]])).to(device).float()
                    xs = torch.as_tensor(episode['xs'][selected], dtype=torch.float32, device=device)
                    qy = torch.as_tensor(episode['labels'][q], dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        features = model.features(pixels)
                    if complement:
                        bd = torch.as_tensor(episode['base_discarded'][selected], dtype=torch.float32, device=device)
                        prior, phi = geometry(xs, features.float(), bd)
                    else:
                        prior, phi = geometry(xs, features.float())
                    adapted = residual_scores(phi[:64], prior[:64], sy, phi[64:], prior[64:])
                    per_row = F.cross_entropy(adapted, qy, reduction='none')
                    loss = (per_row*weights[qy]).sum()/weights[qy].sum()
                    if not torch.isfinite(loss):
                        raise ValueError(f'nonfinite loss in {arm}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.layer4.parameters(), config['gradient_clip'])
                    optimizer.step()
                    total += float(loss.detach())*len(q)
                    count += len(q)
                    steps += 1
            record = dict(epoch=epoch+1, classification=total/count, query_rows=count,
                steps=steps, elapsed_seconds=time.monotonic()-started)
            history.append(record)
            write_json(folder/'training_history.json', history)
            print(f'JOINT TRAIN {arm} {record}', flush=True)
        torch.save({key: value.detach().cpu() for key, value in model.layer4.state_dict().items()}, folder/'layer4.pt')
        extract_tail(folder, model, source, target, device)
        del geometries
    write_json(out/'TRAINING_COMPLETE.json', dict(arms=protocol['trained_arms'],
        parity_max_abs_errors=errors, actual_final_complement_operator_in_gradient=True,
        target_labels_for_encoder=0, epochs=config['epochs'], posttraining_geometry_refit=False))
    print('MATCHED JOINT TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
