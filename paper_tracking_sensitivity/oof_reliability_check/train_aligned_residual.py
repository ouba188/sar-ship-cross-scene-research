"""Train the actual support-adapted ridge expert; retain its head at deployment."""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from aligned_residual_ops import FixedExpertGeometry, residual_scores
from budgeted_residual_handoff import ranked_mask
from neural_complement import ComplementTail
from receiver_consistent_handoff import gate_scores
from residual_expert import fit_correction
from train_neural_complement import extract_tail
from extract_opensarship2_transfer import WS, write_json


def check_episode_operator(geometry, model, episode, source, device):
    selected = np.r_[episode['support'], episode['query'][:32]]
    rows = episode['rows'][selected]
    with torch.no_grad():
        features = model.features(torch.from_numpy(np.array(source[rows])).to(device).float())
        xs = torch.as_tensor(episode['xs'][selected], device=device)
        prior, phi = geometry(xs, features)
        y = torch.as_tensor(episode['labels'][episode['support']], device=device)
        actual = residual_scores(phi[:64], prior[:64], y, phi[64:], prior[64:])
    raw = features.cpu().numpy()
    z = (raw-episode['ssl_mean'])@episode['ssl_components'].T
    xe = (np.column_stack((episode['xs'][selected], z))-episode['expert_center'])/episode['expert_scale']
    np_prior = xe@episode['expert_coef'].T+episode['expert_intercept']
    np_phi = np.column_stack((xe/np.sqrt(xe.shape[1]), np.ones(len(xe))))
    expected = np_prior[64:]+np_phi[64:]@fit_correction(
        np_phi[:64], episode['labels'][episode['support']], np_prior[:64])
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=2e-4, rtol=2e-4)
    return float(np.max(np.abs(actual.cpu().numpy()-expected)))


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
    with (out/'source_model.pkl').open('rb') as stream:
        gate = pickle.load(stream)['gate']
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
    # These old pilot heads are never called or optimized; all losses use ridge.
    model.classifier.requires_grad_(False)
    model.utility.requires_grad_(False)
    initial = {key: value.clone() for key, value in model.layer4.state_dict().items()}
    model = model.to(device)
    weights = torch.as_tensor(counts.sum()/(8*counts), dtype=torch.float32, device=device)
    descriptors = json.loads((out/'EPISODES_COMPLETE.json').read_text())['episodes']
    episodes = []
    geometries = []
    for descriptor in descriptors:
        with np.load(descriptor['path']) as saved:
            episode = {key: saved[key] for key in saved.files}
        episodes.append(episode)
        geometries.append(FixedExpertGeometry(episode).to(device).float())
    model.eval()
    error = check_episode_operator(geometries[0], model, episodes[0], source, device)
    write_json(out/'OPERATOR_PARITY.json', dict(real_source_episode_max_abs_error=error,
        reference='unchanged numpy deployment fit_correction', support_rows=64, query_rows=32))
    for arm in protocol['arms']:
        folder = out/arm
        folder.mkdir(exist_ok=True)
        model.layer4.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.layer4.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        history = []
        started = time.monotonic()
        for epoch in range(config['epochs']):
            rng = np.random.default_rng(config['seed']+epoch)
            totals = dict(classification=0., handoff=0., selected_rows=0, query_rows=0, steps=0)
            model.train()
            for ei in rng.permutation(len(episodes)):
                episode, geometry = episodes[ei], geometries[ei]
                support = episode['support']
                query = rng.permutation(episode['query'])
                sy = torch.as_tensor(episode['labels'][support], dtype=torch.long, device=device)
                observed = np.bincount(episode['labels'][support], minlength=8) > 0
                for lo in range(0, len(query), config['query_batch_size']):
                    q = query[lo:lo+config['query_batch_size']]
                    indices = np.r_[support, q]
                    pixels = torch.from_numpy(np.array(source[episode['rows'][indices]])).to(device).float()
                    xs = torch.as_tensor(episode['xs'][indices], dtype=torch.float32, device=device)
                    qy = torch.as_tensor(episode['labels'][q], dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        features = model.features(pixels)
                    prior, phi = geometry(xs, features.float())
                    adapted = residual_scores(phi[:64], prior[:64], sy, phi[64:], prior[64:])
                    per_row = F.cross_entropy(adapted, qy, reduction='none')
                    classification = (per_row*weights[qy]).sum()/weights[qy].sum()
                    handoff = classification*0.
                    selected_count = 0
                    if arm == 'residual_handoff':
                        receiving = adapted.detach().cpu().numpy()
                        base = episode['adapted_base'][q]
                        score = gate_scores(base, receiving, gate)
                        bp, ep = base.argmax(1), receiving.argmax(1)
                        eligible = (bp != ep) & observed[ep] & (score > 0)
                        selected = ranked_mask(score, eligible, episode['rows'][q], int(np.floor(.05*len(q))))
                        selected_count = int(selected.sum())
                        if selected_count:
                            chosen = torch.as_tensor(np.flatnonzero(selected), device=device)
                            handoff = (per_row[chosen]*weights[qy[chosen]]).sum()/weights[qy[chosen]].sum()
                    loss = classification+handoff
                    if not torch.isfinite(loss):
                        raise ValueError(f'nonfinite loss in {arm}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.layer4.parameters(), config['gradient_clip'])
                    optimizer.step()
                    totals['classification'] += float(classification.detach())*len(q)
                    totals['handoff'] += float(handoff.detach())*len(q)
                    totals['query_rows'] += len(q)
                    totals['selected_rows'] += selected_count
                    totals['steps'] += 1
            record = dict(epoch=epoch+1, **totals, elapsed_seconds=time.monotonic()-started)
            for key in ('classification', 'handoff'):
                record[key] /= totals['query_rows']
            history.append(record)
            write_json(folder/'training_history.json', history)
            print(f'ALIGNED TRAIN {arm} {record}', flush=True)
        torch.save({key: value.detach().cpu() for key, value in model.layer4.state_dict().items()}, folder/'layer4.pt')
        extract_tail(folder, model, source, target, device)
    write_json(out/'TRAINING_COMPLETE.json', dict(arms=protocol['arms'],
        actual_residual_solve_in_gradient=True, auxiliary_heads_used=False,
        target_labels_for_encoder=0, no_geometry_or_head_refit_after_training=True,
        epochs=config['epochs'], operator_parity_max_abs_error=error))


if __name__ == '__main__':
    main()
