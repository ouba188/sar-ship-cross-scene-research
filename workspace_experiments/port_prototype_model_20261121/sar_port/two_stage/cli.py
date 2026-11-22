"""Source-only train/evaluate commands for the separate native two-stage path."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from .data import InputConfig, prepare_record, collate, background_targets
from .geometry import GeometryConfig
from .stage1 import Stage1, Stage1Config
from .h import HModel, HConfig
from .training import LossConfig, stage1_optimizers, stage1_step, base_step, matching_step
from ..protocol import split_lopo, legal_reference_mask, validate_bank_source

VERSION = 'native3_frozen_g_area_ot_bounded_a_v1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_records(path):
    data = read_json(path)
    rows = data if isinstance(data, list) else data['records']
    return [dict(r, label=str(r['label'])) for r in rows]


def protocol(rows, config):
    split = split_lopo(rows, config['target_ports'], config['validation_ports'])
    if split['missing_source_classes'] or any(len(split[k]) == 0 for k in ('train', 'validation', 'test')):
        raise ValueError('Closed-set training requires nonempty splits and source coverage of every class')
    return split


def batches(samples, indices, size, shuffle=False):
    groups = {}
    for i in indices:
        groups.setdefault(samples[i].geometry.canvas_size, []).append(int(i))
    for values in groups.values():
        if shuffle:
            random.shuffle(values)
        for offset in range(0, len(values), size):
            yield values[offset:offset+size]


def metadata(rows):
    return [{k: r[k] for k in ('id', 'label', 'vessel_id', 'port', 'group_id')} for r in rows]


def training_reference_mask(query_rows, reference_rows):
    legal_source = legal_reference_mask(query_rows, reference_rows)
    legal = legal_source & np.array([[q['port'] != r['port'] for r in reference_rows] for q in query_rows])
    fallback = []
    for j, query in enumerate(query_rows):
        same_class = np.array([r['label'] == query['label'] for r in reference_rows])
        use_source = not (legal[j] & same_class).any()
        if use_source:
            legal[j] = legal_source[j]
        fallback.append(use_source)
    return legal, sum(fallback)


def save(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def restore_encoder(checkpoint, device):
    config = dict(checkpoint['encoder_config']); config['weights_path'] = None
    model = Stage1(Stage1Config(**config)).to(device)
    model.load_state_dict(checkpoint['encoder'])
    return model


def load_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['version'] != VERSION:
        raise ValueError('Incompatible representation/checkpoint contract')
    return saved


def prepare(rows, directory, input_config, geometry_config, indices):
    # Only requested split rows are read. No target image contributes training statistics.
    result = {}
    for i in indices:
        try:
            result[int(i)] = prepare_record(rows[i], directory, input_config, geometry_config)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Sample {rows[i]['id']}: {exc}") from exc
    return result


def fit_normalization(samples, indices):
    sums = torch.zeros(2, dtype=torch.float64); squares = sums.clone(); counts = sums.clone()
    for i in indices:
        for p in range(2):
            values = samples[i].images[p][samples[i].valid[p] > 0].double()
            sums[p] += values.sum(); squares[p] += values.square().sum(); counts[p] += len(values)
    mean = sums/counts.clamp_min(1)
    std = (squares/counts.clamp_min(1)-mean.square()).clamp_min(1e-8).sqrt()
    return mean.float(), torch.where(counts > 0, std, torch.ones_like(std)).float()


def normalize_samples(samples, mean, std):
    for s in samples.values():
        s.images = ((s.images-mean[:, None, None, None])/std[:, None, None, None])*s.valid


def scene_statistics(samples, indices):
    values = [[], []]
    for i in indices:
        s = samples[i]
        targets, reliable = background_targets(s.images[None], s.background[None])
        for p in range(2):
            if reliable[0, p]:
                values[p].append(targets[0, p])
    means, stds = [], []
    for part in values:
        x = torch.stack(part) if part else torch.zeros(1, 2)
        means.append(x.mean(0)); stds.append(x.std(0, unbiased=False).clamp_min(1e-3))
    return torch.stack(means), torch.stack(stds)


def eligible_interventions(samples, indices):
    """Qualification before class sampling; includes a distinct compatible Q donor."""
    groups, properties = {}, {}
    for i in indices:
        s = samples[i]
        active = s.valid.flatten(1).sum(1) > 0
        key = (s.geometry.canvas_size, tuple(active.tolist()))
        common = bool(s.background[active].bool().all(0).any())
        query = s.pair_eligible and bool((s.background.flatten(1).sum(1)[active] > 0).all())
        properties[i] = (key, common, query)
        groups.setdefault(key, []).append(i)
    donors = {k: sum(properties[i][1] for i in ids) for k, ids in groups.items()}
    return {i: query and donors[key]-int(common) > 0 for i, (key, common, query) in properties.items()}


@torch.no_grad()
def stage1_accuracy(model, samples, indices, rows, label_map, device, batch_size):
    model.eval(); correct = count = 0
    for ids in batches(samples, indices, batch_size):
        batch = collate([samples[i] for i in ids], device)
        out = model(batch['images'], batch['valid'], hard_gate=True)
        logits = model.class_logits(out, batch['geometries'])
        valid = model.class_valid(batch['geometries']).to(device)
        logits = (logits*valid[..., None]).sum(1)/valid.sum(1).clamp_min(1)[:, None]
        labels = torch.tensor([label_map[rows[i]['label']] for i in ids], device=device)
        correct += int((logits.argmax(1) == labels).sum()); count += len(ids)
    return correct/count


def train_stage1(args, config, rows, split, device):
    geometry_config = GeometryConfig(**config.get('geometry', {}))
    input_kwargs = dict(config.get('input', {})); input_kwargs.pop('mean', None); input_kwargs.pop('std', None)
    input_config = InputConfig(**input_kwargs)
    source_ids, validation_ids = split['train'].tolist(), split['validation'].tolist()
    samples = prepare(rows, args.manifest.parent, input_config, geometry_config, source_ids+validation_ids)
    mean, std = fit_normalization(samples, source_ids)
    normalize_samples(samples, mean, std)
    input_config.mean, input_config.std = tuple(mean.tolist()), tuple(std.tolist())
    scene_mean, scene_std = scene_statistics(samples, source_ids)
    labels = sorted({rows[i]['label'] for i in source_ids}); label_map = {v: i for i, v in enumerate(labels)}
    eligible = eligible_interventions(samples, source_ids)
    controlled = {i: bool(rows[i].get('leakage_controlled', False)) and eligible[i] for i in source_ids}
    trusted = sum(bool(samples[i].background.any()) for i in source_ids)
    complete = not config.get('allow_incomplete_supervision', False)
    if complete:
        if trusted < 2 or not any(controlled.values()):
            raise ValueError('Full Stage1 needs >=2 validated Q observations and audited leakage_controlled views. '
                             'For an explicitly incomplete ablation set allow_incomplete_supervision=true.')
    model = Stage1(Stage1Config(classes=len(labels), **config.get('encoder', {}))).to(device)
    loss_config = LossConfig(**config.get('loss', {}))
    optimizer = stage1_optimizers(model, config.get('learning_rate', 1e-4))
    prior_ids = [i for i in source_ids if controlled[i]] or source_ids
    class_counts = torch.bincount(torch.tensor([label_map[rows[i]['label']] for i in prior_ids]), minlength=len(labels))
    if any(controlled.values()) and (class_counts == 0).any():
        raise ValueError('Controlled views must cover every source class for leakage supervision')
    prior = (class_counts/class_counts.sum()).to(device)
    batch_size, epochs = config.get('batch_size', 8), config.get('stage1_epochs', 20)
    planned_steps = epochs*len(list(batches(samples, source_ids, batch_size)))
    if complete and (batch_size < 2 or planned_steps <= max(loss_config.hard_gate_after,
                                                           loss_config.warmup_steps+loss_config.ramp_steps-1)):
        raise ValueError('Full Stage1 needs batch_size>=2 and training after warmup/ramp/hard-route transition')
    best, history, step = -1., [], 0
    for epoch in range(epochs):
        losses = []
        for ids in batches(samples, source_ids, batch_size, True):
            batch = collate([samples[i] for i in ids], device)
            y = torch.tensor([label_map[rows[i]['label']] for i in ids], device=device)
            legal = torch.tensor([controlled[i] for i in ids], device=device)
            losses.append(stage1_step(model, batch, y, optimizer, prior, scene_mean, scene_std,
                                      legal, loss_config, step)); step += 1
        accuracy = stage1_accuracy(model, samples, validation_ids, rows, label_map, device, batch_size)
        coverage = np.sum([r['controlled_class_counts'] for r in losses], axis=0).tolist()
        if complete and (min(coverage) == 0 or sum(r['scene_targets'] for r in losses) == 0):
            raise ValueError('Full supervision has zero actual controlled-class or scene coverage in this epoch; '
                             'fix eligible donors/batches or explicitly run an incomplete ablation')
        report = dict(epoch=epoch+1, validation_accuracy=accuracy, controlled_class_counts=coverage,
                      losses={k: float(np.mean([r[k] for r in losses])) for k in losses[0]
                              if k != 'controlled_class_counts'})
        history.append(report); print(json.dumps(report), flush=True)
        eligible_checkpoint = not complete or (losses[-1]['hard_gate'] and losses[-1]['constraint_ramp'] == 1)
        if eligible_checkpoint and accuracy > best:
            best = accuracy
            save(args.output, dict(version=VERSION, stage='stage1', encoder=model.state_dict(),
                  encoder_config=asdict(model.config), input_config=asdict(input_config),
                  geometry_config=asdict(geometry_config), scene_mean=scene_mean, scene_std=scene_std,
                  labels=labels, protocol_config=config, manifest_metadata=metadata(rows),
                  source_ids=[rows[i]['id'] for i in source_ids], history=list(history),
                  incomplete_supervision=bool(config.get('allow_incomplete_supervision', False))))
    args.output.with_suffix('.history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')


@torch.no_grad()
def encode_samples(encoder, samples, indices, device, size):
    features = {}
    for ids in batches(samples, indices, size):
        batch = collate([samples[i] for i in ids], device)
        for i, encoded in zip(ids, encoder.encode(batch['images'], batch['valid'], batch['geometries'])):
            features[i] = encoded.to('cpu')
    return features


@torch.no_grad()
def evaluate_h(h, features, indices, rows, label_map, refs, reference_indices, device, match=True):
    h.eval(); predictions = []; correct = 0
    reference_rows = [rows[i] for i in reference_indices]
    ref_labels = torch.tensor([label_map[r['label']] for r in reference_rows], device=device)
    for i in indices:
        q = [features[i].to(device)]
        if match:
            legal = torch.as_tensor(legal_reference_mask([rows[i]], reference_rows), device=device)
            output = h(q, refs, ref_labels, [r['port'] for r in reference_rows], legal)
            logits = output['logits']
        else:
            logits = h.forward_base(q)
        prediction = int(logits.argmax(1)); correct += prediction == label_map[rows[i]['label']]
        predictions.append(dict(id=rows[i]['id'], label=rows[i]['label'], prediction=list(label_map)[prediction]))
    return dict(accuracy=correct/len(indices), count=len(indices), predictions=predictions)


def train_h(args, config, rows, split, device):
    from .bank import select_reference_indices
    saved = load_checkpoint(args.checkpoint)
    if saved['stage'] != 'stage1' or saved['manifest_metadata'] != metadata(rows):
        raise ValueError('H requires this manifest and a Stage1 checkpoint')
    if saved['source_ids'] != [rows[i]['id'] for i in split['train']]:
        raise ValueError('Do not change source split between stages')
    source_ids, val_ids = split['train'].tolist(), split['validation'].tolist()
    samples = prepare(rows, args.manifest.parent, InputConfig(**saved['input_config']),
                      GeometryConfig(**saved['geometry_config']), source_ids+val_ids)
    encoder = restore_encoder(saved, device).freeze_for_h()
    features = encode_samples(encoder, samples, source_ids+val_ids, device, config.get('batch_size', 8))
    label_map = {v: i for i, v in enumerate(saved['labels'])}
    h = HModel(HConfig(classes=len(label_map), **config.get('h', {}))).to(device)
    h.estimate_rms(features[i].to(device) for i in source_ids)
    h.set_stage('base')
    optimizer = torch.optim.AdamW([p for p in h.parameters() if p.requires_grad], lr=config.get('learning_rate', 1e-4))
    best, best_state, history = -1., None, []
    for epoch in range(config.get('base_epochs', 10)):
        order = source_ids.copy(); random.shuffle(order)
        losses = []
        for start in range(0, len(order), config.get('batch_size', 8)):
            ids = order[start:start+config.get('batch_size', 8)]
            y = torch.tensor([label_map[rows[i]['label']] for i in ids], device=device)
            losses.append(base_step(h, [features[i].to(device) for i in ids], y, optimizer))
        score = evaluate_h(h, features, val_ids, rows, label_map, [], [], device, False)['accuracy']
        report = dict(stage='base', epoch=epoch+1, loss=float(np.mean(losses)), validation_accuracy=score)
        history.append(report); print(json.dumps(report), flush=True)
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
    if best_state is None:
        raise ValueError('base_epochs must be positive')
    h.load_state_dict(best_state)
    positions = select_reference_indices([features[i] for i in source_ids],
                    [label_map[rows[i]['label']] for i in source_ids], [rows[i]['port'] for i in source_ids],
                    [rows[i]['vessel_id'] for i in source_ids], [rows[i]['group_id'] for i in source_ids],
                    config.get('prototypes_per_class', 8))
    reference_indices = [source_ids[i] for i in positions]
    reference_rows = [rows[i] for i in reference_indices]
    validate_bank_source(reference_rows, [rows[i] for i in source_ids])
    refs = [features[i].to(device) for i in reference_indices]
    ref_labels = torch.tensor([label_map[r['label']] for r in reference_rows], device=device)
    h.set_stage('match')
    optimizer = torch.optim.AdamW([p for p in h.parameters() if p.requires_grad], lr=config.get('learning_rate', 1e-4))
    best = -1.
    for epoch in range(config.get('h_epochs', 20)):
        order = source_ids.copy(); random.shuffle(order); losses = []; fallback_count = 0
        for start in range(0, len(order), config.get('batch_size', 8)):
            ids = order[start:start+config.get('batch_size', 8)]
            # Training episodes exclude the query port; no query port is used at deployment.
            legal, count = training_reference_mask([rows[i] for i in ids], reference_rows)
            fallback_count += count
            y = torch.tensor([label_map[rows[i]['label']] for i in ids], device=device)
            losses.append(matching_step(h, [features[i].to(device) for i in ids], y, refs, ref_labels,
                         [r['port'] for r in reference_rows], torch.as_tensor(legal, device=device), optimizer))
        if sum(r['match_samples'] for r in losses) == 0:
            raise ValueError('No training query has legal same-class reference support; fix source bank/episodes')
        score = evaluate_h(h, features, val_ids, rows, label_map, refs, reference_indices, device)['accuracy']
        report = dict(stage='match', epoch=epoch+1, losses={k: float(np.mean([r[k] for r in losses])) for k in losses[0]},
                      validation_accuracy=score, source_support_fallback_samples=fallback_count)
        history.append(report); print(json.dumps(report), flush=True)
        if score > best:
            best = score
            output = dict(saved, stage='h', h=h.state_dict(), h_config=asdict(h.config),
                          reference_ids=[rows[i]['id'] for i in reference_indices], h_history=list(history), h_training_config=config)
            save(args.output, output)
    args.output.with_suffix('.history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')


def evaluate(args, rows, device):
    saved = load_checkpoint(args.checkpoint)
    if saved['stage'] != 'h' or saved['manifest_metadata'] != metadata(rows):
        raise ValueError('Evaluation requires the trained H checkpoint and unchanged manifest metadata')
    split = protocol(rows, saved['protocol_config'])
    lookup = {r['id']: i for i, r in enumerate(rows)}
    references = [lookup[v] for v in saved['reference_ids']]
    ids = split[args.split].tolist()
    required = sorted(set(references+ids))
    samples = prepare(rows, args.manifest.parent, InputConfig(**saved['input_config']),
                      GeometryConfig(**saved['geometry_config']), required)
    encoder = restore_encoder(saved, device).freeze_for_h()
    features = encode_samples(encoder, samples, required, device, 8)
    h = HModel(HConfig(**saved['h_config'])).to(device); h.load_state_dict(saved['h']); h.set_stage('match')
    result = evaluate_h(h, features, ids, rows, {v: i for i, v in enumerate(saved['labels'])},
                        [features[i].to(device) for i in references], references, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'predictions'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train-stage1', 'train-h', 'evaluate'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    args = parser.parse_args()
    config = read_json(args.config) if args.config else {}
    seed = config.get('seed', 17); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = load_records(args.manifest)
    if args.command == 'evaluate':
        if not args.checkpoint: parser.error('--checkpoint required')
        evaluate(args, rows, args.device); return
    if not args.config: parser.error('--config required for training')
    if args.command == 'train-h' and not args.checkpoint: parser.error('--checkpoint required')
    if min(config.get(k, 1) for k in ('batch_size', 'stage1_epochs', 'base_epochs', 'h_epochs')) < 1:
        parser.error('batch size and epoch counts must be positive')
    split = protocol(rows, config)
    (train_stage1 if args.command == 'train-stage1' else train_h)(args, config, rows, split, args.device)


if __name__ == '__main__':
    main()
