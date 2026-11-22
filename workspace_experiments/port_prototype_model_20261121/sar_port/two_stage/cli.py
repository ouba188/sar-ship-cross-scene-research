"""Source-only train/evaluate commands for the separate native two-stage path."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from .data import InputConfig, collate, background_targets
from .cache import prepare_record_cached
from .geometry import GeometryConfig
from .stage1 import Stage1, Stage1Config
from .h import HModel, HConfig
from .training import LossConfig, stage1_optimizers, stage1_step, baseline_step, base_step, matching_step
from ..protocol import split_lopo, legal_reference_mask, validate_bank_source, _validate_records

VERSION = 'native3_frozen_g_area_ot_bounded_a_v1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_records(path):
    data = read_json(path)
    rows = data if isinstance(data, list) else data['records']
    return [dict(r, label=str(r['label'])) for r in rows]


def protocol(rows, config):
    mode = config.get('protocol', 'lopo')
    if mode == 'development_split':
        _validate_records(rows)
        mapping = {'train': 'train', 'val': 'validation', 'validation': 'validation',
                   'test': 'test', 'quarantine_historical_selection': 'excluded'}
        split = {k: [] for k in ('train', 'validation', 'test', 'excluded')}
        for i, row in enumerate(rows):
            role = row.get('development_split')
            if role not in mapping:
                raise ValueError(f"Unknown development_split for {row['id']}: {role}")
            split[mapping[role]].append(i)
        for key in ('vessel_id', 'group_id'):
            used = set()
            for name in ('train', 'validation', 'test'):
                values = {rows[i][key] for i in split[name]}
                if used & values:
                    raise ValueError(f'Fixed development split leaks {key} across {name}')
                used.update(values)
        source_classes = {rows[i]['label'] for i in split['train']}
        missing = {name: sorted({rows[i]['label'] for i in split[name]}-source_classes)
                   for name in ('validation', 'test')}
        split = {k: np.asarray(v, dtype=np.int64) for k, v in split.items()}
        split.update(missing_source_classes=sorted(set(missing['validation']+missing['test'])),
                     missing_by_split=missing)
    elif mode == 'lopo':
        split = split_lopo(rows, config['target_ports'], config['validation_ports'])
    else:
        raise ValueError('protocol must be lopo or development_split')
    if split['missing_source_classes'] or any(len(split[k]) == 0 for k in ('train', 'validation', 'test')):
        raise ValueError('Closed-set training requires nonempty splits and source coverage of every class')
    return split


def classification_metrics(labels, predictions, classes):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)), 1)
    support, predicted = matrix.sum(1), matrix.sum(0)
    correct = np.diag(matrix)
    recall = np.divide(correct, support, out=np.zeros(classes, float), where=support > 0)
    f1 = np.divide(2*correct, support+predicted, out=np.zeros(classes, float), where=support+predicted > 0)
    return dict(accuracy=float(correct.sum()/max(matrix.sum(), 1)),
                balanced_accuracy=float(recall[support > 0].mean()) if support.any() else 0.,
                macro_f1=float(f1.mean()), confusion_matrix=matrix.tolist(), count=int(matrix.sum()))


def selection_score(metrics, config):
    metric = config.get('selection_metric', 'accuracy')
    if metric not in ('accuracy', 'balanced_accuracy', 'macro_f1'):
        raise ValueError('selection_metric must be accuracy, balanced_accuracy or macro_f1')
    return metrics[metric]


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
    return [{k: r[k] for k in ('id', 'label', 'vessel_id', 'port', 'group_id', 'development_split')
             if k in r} for r in rows]


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
            result[int(i)] = prepare_record_cached(rows[i], directory, input_config, geometry_config)
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


class SupplementalInterventions:
    """Audited query/donor IDs; deterministic label-blind paired-pixel sampling."""

    def __init__(self, samples, rows, source_ids, config):
        lookup = {rows[i]['id']: i for i in source_ids}
        donor_names = config.get('donor_ids', [])
        planned = config.get('supplemental_batches', [])
        if not donor_names or not planned or len(set(donor_names)) != len(donor_names):
            raise ValueError('Empirical mode needs unique donor_ids and supplemental_batches')
        names = set(donor_names) | {v for part in planned for v in part}
        if not names <= lookup.keys():
            raise ValueError('All supplemental queries and donors must belong to the training split')
        self.donors = [lookup[v] for v in donor_names]
        self.batches = [[lookup[v] for v in part] for part in planned]
        queries = {i for part in self.batches for i in part}
        if set(self.donors) & queries:
            raise ValueError('Dedicated donor_ids must be disjoint from supplemental query IDs')
        for part in self.batches:
            if not part or len(part) > config.get('batch_size', 8):
                raise ValueError('Supplemental batches must be nonempty and fit batch_size')
            if len({samples[i].geometry.canvas_size for i in part}) != 1:
                raise ValueError('Supplemental batch mixes native canvas sizes')
        audited = {i for i in source_ids if rows[i].get('leakage_regularization_eligible', False) is True}
        if audited != queries:
            raise ValueError('Empirical eligible rows must exactly match supplemental query IDs')
        self.sequences = {}
        for i in queries:
            seq = rows[i].get('background_donor_ids', donor_names)
            if not seq or not set(seq) <= set(donor_names):
                raise ValueError('background_donor_ids must be a nonempty subset of dedicated donors')
            self.sequences[i] = [lookup[v] for v in seq]
            q = samples[i]
            active = q.valid.flatten(1).sum(1) > 0
            if not q.pair_eligible or not active.any() or not q.background[active].flatten(1).any(1).all():
                raise ValueError('Supplemental query needs protected target and audited Q in every active polarization')
            for j in self.sequences[i]:
                donor = samples[j]
                donor_active = donor.valid.flatten(1).sum(1) > 0
                if not torch.equal(active, donor_active) or not donor.background[active].bool().all(0).any():
                    raise ValueError('Dedicated donor needs matching polarizations and jointly trusted Q')
        self.counts = {i: 0 for i in queries}
        self.generator = torch.Generator().manual_seed(config.get('seed', 17)+104729)

    def apply(self, batch, ids, samples):
        changed = batch['images'].clone()
        assignments = []
        for position, i in enumerate(ids):
            seq = self.sequences[i]
            j = seq[self.counts[i] % len(seq)]
            self.counts[i] += 1
            donor = samples[j]
            active = torch.where(donor.valid.flatten(1).sum(1) > 0)[0].tolist()
            donor_pixels = torch.where(donor.background[active].bool().all(0).flatten())[0]
            # The same source pixel index field supplies VV and VH, independent of labels.
            field = torch.randint(len(donor_pixels), (changed.shape[-1]**2,), generator=self.generator)
            for p in active:
                values = donor.images[p].flatten()[donor_pixels[field]].reshape_as(changed[position, p])
                changed[position, p] = torch.where(batch['background'][position, p].bool(),
                                                   values.to(changed), changed[position, p])
            assignments.append(j)
        batch['intervention_images'] = changed
        batch['intervention_eligible'] = torch.ones(len(ids), device=changed.device, dtype=torch.bool)
        return assignments


def epoch_batches(samples, source_ids, size, seed, epoch, supplemental=None):
    """Query order is independent of model/intervention RNG and identical across arms."""
    order = list(source_ids)
    random.Random(seed+epoch).shuffle(order)
    yield from ((ids, False) for ids in batches(samples, order, size))
    if supplemental is not None:
        yield from ((ids, True) for ids in supplemental.batches)


@torch.no_grad()
def stage1_metrics(model, samples, indices, rows, label_map, device, batch_size):
    model.eval(); predictions = []; labels = []
    for ids in batches(samples, indices, batch_size):
        batch = collate([samples[i] for i in ids], device)
        out = model(batch['images'], batch['valid'], hard_gate=True)
        logits = model.class_logits(out, batch['geometries'])
        valid = model.class_valid(batch['geometries']).to(device)
        logits = (logits*valid[..., None]).sum(1)/valid.sum(1).clamp_min(1)[:, None]
        predictions.extend(logits.argmax(1).cpu().tolist())
        labels.extend(label_map[rows[i]['label']] for i in ids)
    return classification_metrics(labels, predictions, len(label_map))


def stage1_accuracy(model, samples, indices, rows, label_map, device, batch_size):
    return stage1_metrics(model, samples, indices, rows, label_map, device, batch_size)['accuracy']


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
    leakage_mode = config.get('leakage_mode', 'strict_controlled')
    if leakage_mode not in ('strict_controlled', 'empirical_regularization'):
        raise ValueError('leakage_mode must be strict_controlled or empirical_regularization')
    supplemental = (SupplementalInterventions(samples, rows, source_ids, config)
                    if leakage_mode == 'empirical_regularization' else None)
    if supplemental is None and (config.get('supplemental_batches') or config.get('donor_ids')):
        raise ValueError('Dedicated donor schedules require empirical_regularization mode')
    eligible = eligible_interventions(samples, source_ids)
    controlled = {i: bool(rows[i].get('leakage_controlled', False)) and eligible[i] for i in source_ids}
    trusted = sum(bool(samples[i].background.any()) for i in source_ids)
    baseline = not config.get('encoder', {}).get('gate_enabled', True)
    complete = not baseline and not config.get('allow_incomplete_supervision', False)
    prior_ids = ([i for part in supplemental.batches for i in part] if supplemental else
                 [i for i in source_ids if controlled[i]])
    if complete:
        if trusted < 2 or not prior_ids:
            raise ValueError('Full Stage1 needs >=2 validated Q observations and audited leakage_controlled '
                             'views (strict mode) or eligible supplemental views (empirical mode). '
                             'For an explicitly incomplete ablation set allow_incomplete_supervision=true.')
    model = Stage1(Stage1Config(classes=len(labels), **config.get('encoder', {}))).to(device)
    loss_config = LossConfig(**config.get('loss', {}))
    optimizer = stage1_optimizers(model, config.get('learning_rate', 1e-4))
    prior_ids = prior_ids or source_ids
    class_counts = torch.bincount(torch.tensor([label_map[rows[i]['label']] for i in prior_ids]), minlength=len(labels))
    if not baseline and (supplemental or any(controlled.values())) and (class_counts == 0).any():
        raise ValueError('Qualified leakage views must cover every source class')
    prior = (class_counts/class_counts.sum()).to(device)
    batch_size, epochs = config.get('batch_size', 8), config.get('stage1_epochs', 20)
    planned_steps = epochs*(len(list(batches(samples, source_ids, batch_size)))+
                            (len(supplemental.batches) if supplemental else 0))
    if complete and (batch_size < 2 or planned_steps <= max(loss_config.hard_gate_after,
                                                           loss_config.warmup_steps+loss_config.ramp_steps-1)):
        raise ValueError('Full Stage1 needs batch_size>=2 and training after warmup/ramp/hard-route transition')
    best, history, step = -1., [], 0
    for epoch in range(epochs):
        losses, donor_assignments, query_order = [], [], []
        planned = (epoch_batches(samples, source_ids, batch_size, config.get('seed', 17), epoch, supplemental)
                   if supplemental else ((ids, False) for ids in batches(samples, source_ids, batch_size, True)))
        for ids, extra in planned:
            query_order.extend(rows[i]['id'] for i in ids)
            batch = collate([samples[i] for i in ids], device)
            y = torch.tensor([label_map[rows[i]['label']] for i in ids], device=device)
            legal = torch.tensor([controlled[i] for i in ids], device=device)
            regularization = torch.full((len(ids),), extra, device=device, dtype=torch.bool)
            if supplemental:
                if extra:
                    donors = supplemental.apply(batch, ids, samples)
                    donor_assignments.extend(dict(query_id=rows[i]['id'], donor_id=rows[j]['id'])
                                             for i, j in zip(ids, donors))
                else:
                    batch['intervention_images'] = batch['images']
                    batch['intervention_eligible'] = torch.zeros(len(ids), device=device, dtype=torch.bool)
            losses.append(baseline_step(model, batch, y, optimizer[0]) if baseline else
                          stage1_step(model, batch, y, optimizer, prior, scene_mean, scene_std,
                                      legal, loss_config, step, leakage_mode=leakage_mode,
                                      regularization_views=regularization)); step += 1
        metrics = stage1_metrics(model, samples, validation_ids, rows, label_map, device, batch_size)
        score = selection_score(metrics, config)
        coverage = np.sum([r['controlled_class_counts'] for r in losses], axis=0).tolist()
        empirical_coverage = np.sum([r['regularization_class_counts'] for r in losses], axis=0).tolist()
        active_coverage = empirical_coverage if supplemental else coverage
        if complete and (min(active_coverage) == 0 or sum(r['scene_targets'] for r in losses) == 0):
            raise ValueError('Full supervision has zero actual qualified-class or scene coverage in this epoch; '
                             'fix eligible donors/batches or explicitly run an incomplete ablation')
        report = dict(epoch=epoch+1, validation_accuracy=metrics['accuracy'], validation_metrics=metrics,
                      selection_metric=config.get('selection_metric', 'accuracy'), selection_score=score,
                      controlled_class_counts=coverage,
                      regularization_class_counts=empirical_coverage, leakage_mode=leakage_mode,
                      controlled_samples=sum(r['controlled_samples'] for r in losses),
                      regularization_samples=sum(r['regularization_samples'] for r in losses),
                      paired_samples=sum(r['paired_samples'] for r in losses),
                      losses={k: float(np.mean([r[k] for r in losses])) for k in losses[0]
                              if k not in ('controlled_class_counts', 'regularization_class_counts')})
        if supplemental:
            report.update(query_order=query_order, donor_assignments=donor_assignments,
                          natural_query_count=len(source_ids), supplemental_query_count=sum(map(len, supplemental.batches)))
        history.append(report); print(json.dumps(report), flush=True)
        eligible_checkpoint = baseline or (losses[-1]['hard_gate'] and losses[-1]['constraint_ramp'] == 1)
        if eligible_checkpoint and score > best:
            best = score
            save(args.output, dict(version=VERSION, stage='stage1', encoder=model.state_dict(),
                  encoder_config=asdict(model.config), input_config=asdict(input_config),
                  geometry_config=asdict(geometry_config), scene_mean=scene_mean, scene_std=scene_std,
                  labels=labels, protocol_config=config, manifest_metadata=metadata(rows),
                  source_ids=[rows[i]['id'] for i in source_ids], history=list(history),
                  training_objective='classification_baseline' if baseline else 'disentanglement',
                  leakage_mode=leakage_mode, leakage_prior=prior.detach().cpu(),
                  supervision_metadata=[{k: rows[i][k] for k in ('id', 'leakage_controlled',
                       'leakage_regularization_eligible', 'background_donor_ids') if k in rows[i]} for i in source_ids],
                  incomplete_supervision=not baseline and bool(config.get('allow_incomplete_supervision', False))))
    if best < 0:
        raise ValueError('No eligible checkpoint: run after warmup, ramp and hard-route transition')
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
    h.eval(); predictions = []; targets = []; predicted = []
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
        prediction = int(logits.argmax(1)); targets.append(label_map[rows[i]['label']]); predicted.append(prediction)
        predictions.append(dict(id=rows[i]['id'], label=rows[i]['label'], prediction=list(label_map)[prediction]))
    return dict(classification_metrics(targets, predicted, len(label_map)), predictions=predictions)


def train_base(args, config, rows, split, device):
    return train_h(args, config, rows, split, device, base_only=True)


def train_h(args, config, rows, split, device, base_only=False):
    from .bank import select_reference_indices
    saved = load_checkpoint(args.checkpoint)
    allowed = ('stage1',) if base_only else ('stage1', 'base')
    if saved['stage'] not in allowed or saved['manifest_metadata'] != metadata(rows):
        raise ValueError('Training requires unchanged manifest and a Stage1/base checkpoint')
    if saved.get('incomplete_supervision', False) and not config.get('allow_incomplete_supervision', False):
        raise ValueError('Incomplete-supervision checkpoint cannot enter a complete comparison')
    if saved['source_ids'] != [rows[i]['id'] for i in split['train']]:
        raise ValueError('Do not change source split between stages')
    source_ids, val_ids = split['train'].tolist(), split['validation'].tolist()
    samples = prepare(rows, args.manifest.parent, InputConfig(**saved['input_config']),
                      GeometryConfig(**saved['geometry_config']), source_ids+val_ids)
    encoder = restore_encoder(saved, device).freeze_for_h()
    features = encode_samples(encoder, samples, source_ids+val_ids, device, config.get('batch_size', 8))
    label_map = {v: i for i, v in enumerate(saved['labels'])}
    if saved['stage'] == 'base':
        if any(saved['h_config'].get(k) != v for k, v in config.get('h', {}).items()):
            raise ValueError('Keep H architecture/config unchanged when continuing the saved base')
        h = HModel(HConfig(**saved['h_config'])).to(device)
        h.load_state_dict(saved['h'])
        history = list(saved['h_history'])
        base_path = args.checkpoint
    else:
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
            metrics = evaluate_h(h, features, val_ids, rows, label_map, [], [], device, False)
            score = selection_score(metrics, config)
            report = dict(stage='base', epoch=epoch+1, loss=float(np.mean(losses)),
                          validation_accuracy=metrics['accuracy'],
                          validation_metrics={k: v for k, v in metrics.items() if k != 'predictions'},
                          selection_metric=config.get('selection_metric', 'accuracy'), selection_score=score)
            history.append(report); print(json.dumps(report), flush=True)
            if score > best:
                best, best_state = score, {k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
        if best_state is None:
            raise ValueError('base_epochs must be positive')
        h.load_state_dict(best_state)
        base_path = args.output if base_only else args.output.with_name(args.output.stem+'.base.pt')
        saved = dict(saved, stage='base', h=best_state, h_config=asdict(h.config), reference_ids=[],
                     h_history=list(history), base_training_config=config)
        save(base_path, saved)
        if base_only:
            args.output.with_suffix('.history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            return
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
        metrics = evaluate_h(h, features, val_ids, rows, label_map, refs, reference_indices, device)
        score = selection_score(metrics, config)
        report = dict(stage='match', epoch=epoch+1, losses={k: float(np.mean([r[k] for r in losses])) for k in losses[0]},
                      validation_accuracy=metrics['accuracy'],
                      validation_metrics={k: v for k, v in metrics.items() if k != 'predictions'},
                      selection_metric=config.get('selection_metric', 'accuracy'), selection_score=score,
                      source_support_fallback_samples=fallback_count)
        history.append(report); print(json.dumps(report), flush=True)
        if score > best:
            best = score
            output = dict(saved, stage='h', h=h.state_dict(), h_config=asdict(h.config),
                          reference_ids=[rows[i]['id'] for i in reference_indices], h_history=list(history),
                          h_training_config=config, base_checkpoint=str(base_path))
            save(args.output, output)
    args.output.with_suffix('.history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')


def evaluate(args, rows, device):
    saved = load_checkpoint(args.checkpoint)
    if saved['stage'] not in ('base', 'h') or saved['manifest_metadata'] != metadata(rows):
        raise ValueError('Evaluation requires a trained base/H checkpoint and unchanged manifest metadata')
    match = saved['stage'] == 'h'
    split = protocol(rows, saved['protocol_config'])
    lookup = {r['id']: i for i, r in enumerate(rows)}
    references = [lookup[v] for v in saved['reference_ids']]
    ids = split[args.split].tolist()
    required = sorted(set(references+ids))
    samples = prepare(rows, args.manifest.parent, InputConfig(**saved['input_config']),
                      GeometryConfig(**saved['geometry_config']), required)
    encoder = restore_encoder(saved, device).freeze_for_h()
    features = encode_samples(encoder, samples, required, device, 8)
    h = HModel(HConfig(**saved['h_config'])).to(device); h.load_state_dict(saved['h'])
    h.set_stage('match' if match else 'base')
    result = evaluate_h(h, features, ids, rows, {v: i for i, v in enumerate(saved['labels'])},
                        [features[i].to(device) for i in references], references, device, match=match)
    result.update(stage=saved['stage'], training_objective=saved.get('training_objective', 'disentanglement'),
                  leakage_mode=saved.get('leakage_mode', 'strict_controlled'),
                  incomplete_supervision=saved.get('incomplete_supervision', False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'predictions'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train-stage1', 'train-base', 'train-h', 'evaluate'])
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
    if args.command in ('train-base', 'train-h') and not args.checkpoint: parser.error('--checkpoint required')
    if min(config.get(k, 1) for k in ('batch_size', 'stage1_epochs', 'base_epochs', 'h_epochs')) < 1:
        parser.error('batch size and epoch counts must be positive')
    split = protocol(rows, config)
    commands = {'train-stage1': train_stage1, 'train-base': train_base, 'train-h': train_h}
    commands[args.command](args, config, rows, split, args.device)


if __name__ == '__main__':
    main()
