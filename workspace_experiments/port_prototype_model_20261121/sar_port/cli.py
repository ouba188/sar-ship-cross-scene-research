"""Explicit source-fold training, bank selection, evaluation and deployment.

Run ``python -m sar_port.cli --help``. No action starts training implicitly.
Checkpoint and split metadata record actual source rows, without treating a
historical all-port checkpoint as a clean leave-one-port-out initialization.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_records, prepare_batch
from .protocol import RECORD_FIELDS, legal_reference_mask, split_lopo, validate_bank_source
from .prototypes import select_prototypes


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _metadata(rows):
    return [{key: row[key] for key in RECORD_FIELDS} for row in rows]


def _tree_to(value, device, detach=False):
    if isinstance(value, torch.Tensor):
        return (value.detach() if detach else value).to(device)
    if isinstance(value, dict):
        return {key: _tree_to(item, device, detach) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to(item, device, detach) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to(item, device, detach) for item in value)
    return value


def _load_split(path, rows):
    split = _read_json(path)
    if split["record_ids"] != [row["id"] for row in rows]:
        raise ValueError("Split row order differs from the manifest; rebuild the split for this manifest")
    # Reconstruct the supported protocol rather than trusting hand-edited indices.
    expected = split_lopo(rows, split["target_ports"], split["validation_ports"])
    for name in ("train", "validation", "test", "excluded"):
        if split[name] != expected[name].tolist():
            raise ValueError(f"Split {name} violates the declared port/identity protocol")
    split["missing_source_classes"] = expected["missing_source_classes"]
    return split


def _source_rows(rows, split):
    return [rows[i] for i in split["train"]]


def _check_source_checkpoint(checkpoint, source_rows, split=None):
    """An external all-port state_dict has no source-fold authorization here."""
    if "source_records" not in checkpoint or not checkpoint["source_records"]:
        raise ValueError("Checkpoint has no source-training provenance; use --imagenet for generic pretraining")
    protocol = checkpoint.get("protocol", {})
    selection_ports = protocol.get("selection_ports", protocol.get("validation_ports", []))
    if split is not None and set(selection_ports) & set(split["target_ports"]):
        raise ValueError("Checkpoint selection used a current target port as validation")
    allowed = {row["id"]: row for row in source_rows}
    for row in checkpoint["source_records"]:
        original = allowed.get(row["id"])
        if original is None or any(row[key] != original[key] for key in RECORD_FIELDS):
            raise ValueError("Checkpoint was exposed to rows outside this fold's source-training set")
    validate_bank_source(checkpoint.get("bank_records", []), source_rows)


def _load_bank(path, rows, split):
    bank = _read_json(path)
    indices = bank["indices"]
    if any(not isinstance(i, int) or i < 0 or i >= len(rows) for i in indices):
        raise ValueError("Bank contains an invalid manifest index")
    records = [rows[i] for i in indices]
    if [row["id"] for row in records] != bank["record_ids"]:
        raise ValueError("Bank indices no longer match its observation IDs")
    validate_bank_source(records, _source_rows(rows, split))
    if not records:
        raise ValueError("Prototype bank is empty")
    return records


def _load_model(path, device):
    from .model import ModelConfig, PortPrototypeModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    if "effect_target" not in config:
        config.update(effect_target="frequency", size_prior=False, gate_fusion="learned")
    model = PortPrototypeModel(ModelConfig(**config))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _reference(model, checkpoint, device):
    if "encoded_bank" in checkpoint:
        bank = _tree_to(checkpoint["encoded_bank"], device)
    else:
        bank = model.encode_reference(prepare_batch(checkpoint["bank_records"], device))
    labels = torch.tensor([r["label"] for r in checkpoint["bank_records"]], device=device, dtype=torch.long)
    return bank, labels


def _legal(rows, bank_rows, device, allow_unidentified=False):
    if allow_unidentified:
        # Unknown deployment identities are not fabricated as one shared vessel.
        mask = np.ones((len(rows), len(bank_rows)), dtype=bool)
        for i, row in enumerate(rows):
            for j, ref in enumerate(bank_rows):
                for key in ("vessel_id", "group_id"):
                    if row.get(key) not in (None, "") and row[key] == ref[key]:
                        mask[i, j] = False
    else:
        mask = legal_reference_mask(rows, bank_rows)
    return torch.as_tensor(mask, device=device)


def classification_metrics(labels, predictions, classes):
    """Percent OA, support-present BA, and union-of-true/predicted macro F1."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if not len(labels):
        raise ValueError("Cannot evaluate an empty partition")
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("Evaluation labels are outside the configured class range")
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    true = confusion.sum(1)
    predicted = confusion.sum(0)
    correct = np.diag(confusion)
    present = true > 0
    union = true + predicted > 0
    return {
        "oa": float(100 * correct.sum() / len(labels)),
        "ba": float(100 * (correct[present] / true[present]).mean()),
        "macro_f1": float(100 * (2 * correct[union] / (true[union] + predicted[union])).mean()),
        "count": len(labels),
        "present_classes": np.flatnonzero(present).tolist(),
        "f1_classes": np.flatnonzero(union).tolist(),
        "confusion": confusion.tolist(),
    }


@torch.no_grad()
def _predict(model, rows, bank_rows, reference, reference_labels, batch_size, device,
             allow_unidentified=False):
    was_training = model.training
    model.eval()
    scores = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        data = prepare_batch(batch, device)
        legal = _legal(batch, bank_rows, device, allow_unidentified)
        scores.append(model(data, reference, reference_labels, legal=legal)["logits"].cpu())
    model.train(was_training)
    if not scores:
        raise ValueError("Requested partition is empty")
    return torch.cat(scores)


def split_command(args):
    rows = load_records(args.manifest)
    result = split_lopo(rows, args.target_ports, args.validation_ports)
    payload = {key: value.tolist() if isinstance(value, np.ndarray) else value
               for key, value in result.items()}
    payload.update(manifest=str(Path(args.manifest).resolve()),
                   record_ids=[row["id"] for row in rows],
                   target_ports=args.target_ports, validation_ports=args.validation_ports)
    _write_json(args.output, payload)
    print(json.dumps({name: len(payload[name]) for name in ("train", "validation", "test", "excluded")}))
    if payload["missing_source_classes"]:
        print("Classes absent from source training:", payload["missing_source_classes"])


def select_bank_command(args):
    rows = load_records(args.manifest)
    split = _load_split(args.split, rows)
    source_rows = _source_rows(rows, split)
    if not source_rows:
        raise ValueError("Source training set is empty")
    if args.method == "coverage" and not args.checkpoint:
        raise ValueError("Coverage selection requires a source-trained --checkpoint")
    if args.checkpoint:
        model, checkpoint = _load_model(args.checkpoint, args.device)
        _check_source_checkpoint(checkpoint, source_rows, split)
        with torch.no_grad():
            features = [model.describe(prepare_batch(source_rows[i:i + args.batch], args.device),
                                      descriptor=args.descriptor).cpu().numpy()
                        for i in range(0, len(source_rows), args.batch)]
        descriptors = np.concatenate(features)
    else:
        descriptors = np.zeros((len(source_rows), 1), dtype=np.float32)
    local = select_prototypes(source_rows, descriptors, args.per_class, args.method, args.seed,
                              port_balance=not args.no_port_balance, observation=args.observation)
    indices = [split["train"][i] for i in local]
    chosen = [rows[i] for i in indices]
    validate_bank_source(chosen, source_rows)
    payload = dict(indices=indices, record_ids=[r["id"] for r in chosen],
                   method=args.method, per_class=args.per_class, seed=args.seed,
                   descriptor=args.descriptor if args.checkpoint else None,
                   observation=args.observation, learned_descriptors=bool(args.checkpoint),
                   port_balance=not args.no_port_balance,
                   actual_per_class={str(c): sum(r["label"] == c for r in chosen)
                                     for c in sorted({r["label"] for r in source_rows})},
                   source_checkpoint=str(Path(args.checkpoint).resolve()) if args.checkpoint else None)
    _write_json(args.output, payload)
    print(json.dumps(payload["actual_per_class"]))


def _classification_loss(output, labels, available, aux_weight):
    loss = F.cross_entropy(output["logits"], labels)
    if aux_weight:
        auxiliary = F.cross_entropy(output["auxiliary"].flatten(0, 1),
                                    labels[:, None].expand(-1, 2).flatten(), reduction="none")
        auxiliary = auxiliary.reshape(len(labels), 2)
        loss = loss + aux_weight * ((auxiliary * available).sum(1) / available.sum(1).clamp_min(1)).mean()
    return loss


def _freeze_parameters(model, prefixes):
    counts = {}
    parameters = list(model.named_parameters())
    for prefix in prefixes:
        matched = [parameter for name, parameter in parameters if name.startswith(prefix)]
        if not prefix or not matched:
            raise ValueError(f"Freeze prefix matches no named parameters: {prefix!r}")
        for parameter in matched:
            parameter.requires_grad_(False)
        counts[prefix] = sum(parameter.numel() for parameter in matched)
    return counts


def _checkpoint_payload(model, rows, bank_rows, split, step, metrics, args):
    selection_ports = set(split["validation_ports"])
    for previous in (getattr(args, "initial_protocol", {}), getattr(args, "teacher_protocol", {})):
        selection_ports.update(previous.get("selection_ports", previous.get("validation_ports", [])))
    return dict(
        config=asdict(model.config), model=_tree_to(model.state_dict(), "cpu", detach=True),
        bank_records=bank_rows, source_records=_metadata(rows), step=step,
        validation_metrics=metrics,
        protocol=dict(target_ports=split["target_ports"], validation_ports=split["validation_ports"],
                      selection_ports=sorted(selection_ports),
                      partition_ids={name: [split["record_ids"][i] for i in split[name]]
                                     for name in ("train", "validation", "test")},
                      source_only=True, selection="validation BA then OA",
                      initialization=str(Path(args.initialize).resolve()) if args.initialize else None,
                      imagenet=str(Path(args.imagenet).resolve()) if args.imagenet else None,
                      teacher=str(Path(args.teacher).resolve()) if args.teacher else None,
                      seed=args.seed, frozen_prefixes=args.freeze),
    )


def train_command(args):
    from .model import ModelConfig, PortPrototypeModel
    from .supervision import distillation_losses
    from .disentangle import consistency_scale, paired_order
    rows = load_records(args.manifest)
    split = _load_split(args.split, rows)
    if split["missing_source_classes"]:
        raise ValueError(f"Closed-set training has held-out classes absent from source: {split['missing_source_classes']}")
    source_rows = _source_rows(rows, split)
    validation_rows = [rows[i] for i in split["validation"]]
    if not source_rows or not validation_rows:
        raise ValueError("Train and source-validation partitions must both be nonempty")
    bank_rows = _load_bank(args.bank, rows, split)
    config = ModelConfig(**_read_json(args.config)) if args.config else ModelConfig()
    if (config.prior_weight or config.effect_weight) and not args.teacher:
        raise ValueError("Nonzero prior/effect weights require a source-fold --teacher; bootstrap uses both weights 0")
    if args.steps < 0 or args.batch < 1 or args.eval_every < 1:
        raise ValueError("Use steps >= 0, batch >= 1 and eval-every >= 1")
    if args.batch_sampling == "paired" and args.batch % 2:
        raise ValueError("paired sampling requires an even batch size; use shuffle for odd batches")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = PortPrototypeModel(config).to(args.device)
    if args.initialize:
        initial = torch.load(args.initialize, map_location="cpu", weights_only=True)
        _check_source_checkpoint(initial, source_rows, split)
        model.load_state_dict(initial["model"], strict=True)
        args.initial_protocol = initial.get("protocol", {})
    if args.imagenet:
        model.backbone.load_pretrained(args.imagenet)
    frozen = _freeze_parameters(model, args.freeze)

    reference_labels = torch.tensor([r["label"] for r in bank_rows], dtype=torch.long, device=args.device)
    reference_data = prepare_batch(bank_rows, args.device)
    teacher = teacher_reference = teacher_labels = None
    teacher_rows = []
    if args.teacher:
        from .teacher import measure_teacher, measure_utility_teacher
        teacher, teacher_checkpoint = _load_model(args.teacher, args.device)
        _check_source_checkpoint(teacher_checkpoint, source_rows, split)
        if config.effect_weight and config.effect_target == "utility" and teacher.matcher is None:
            raise ValueError("Coarse/fine utility measurement requires a source teacher with M3 enabled")
        args.teacher_protocol = teacher_checkpoint.get("protocol", {})
        teacher.requires_grad_(False)
        teacher_rows = teacher_checkpoint["bank_records"]
        with torch.no_grad():
            teacher_reference, teacher_labels = _reference(teacher, teacher_checkpoint, args.device)

    backbone = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("backbone.resnet.")]
    other = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("backbone.resnet.")]
    groups = [dict(params=backbone, lr=args.backbone_lr), dict(params=other, lr=args.lr)]
    if not backbone and not other:
        raise ValueError("Freeze prefixes leave no trainable model parameters")
    optimizer = torch.optim.AdamW([group for group in groups if group["params"]], weight_decay=.01)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "config.json", asdict(config))
    _write_json(output_dir / "training.json", dict(steps=args.steps, batch=args.batch, seed=args.seed,
                device=args.device, lr=args.lr, backbone_lr=args.backbone_lr,
                frozen_prefixes=args.freeze, frozen_parameter_counts=frozen, batch_sampling=args.batch_sampling))
    rng = np.random.default_rng(args.seed)
    best_key = (-float("inf"), -float("inf"))
    history = []

    def next_order():
        return paired_order(source_rows, rng) if args.batch_sampling == "paired" else rng.permutation(len(source_rows))

    order, cursor = next_order(), 0

    def evaluate_and_save(step):
        nonlocal best_key
        model.eval()
        with torch.no_grad():
            reference = model.encode_reference(reference_data)
            scores = _predict(model, validation_rows, bank_rows, reference, reference_labels,
                              args.batch, args.device)
        metrics = classification_metrics([r["label"] for r in validation_rows],
                                         scores.argmax(1).numpy(), config.classes)
        payload = _checkpoint_payload(model, source_rows, bank_rows, split, step, metrics, args)
        key = (metrics["ba"], metrics["oa"])
        if key > best_key:
            best_key = key
            torch.save(payload, output_dir / "best.pt")
        torch.save(payload, output_dir / "last.pt")
        print(json.dumps(dict(step=step, validation=metrics)), flush=True)
        return metrics

    evaluate_and_save(0)
    for step in range(1, args.steps + 1):
        if cursor >= len(order):
            order, cursor = next_order(), 0
        indices = order[cursor:cursor + args.batch]
        cursor += len(indices)
        batch_rows = [source_rows[i] for i in indices]
        code = int(rng.integers(8))
        data = prepare_batch(batch_rows, args.device, code=code)
        labels = torch.tensor([r["label"] for r in batch_rows], dtype=torch.long, device=args.device)
        legal = _legal(batch_rows, bank_rows, args.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reference = model.encode_reference(reference_data)
        prediction = model(data, reference, reference_labels, legal=legal)
        loss = _classification_loss(prediction, labels, data["available"], config.aux_weight)
        stats = {"classification": float(loss.detach())}
        if config.budget_weight:
            budget = (prediction['scale_cost'].mean()+reference['scale_cost'].mean())/2
            loss = loss+config.budget_weight*budget
            stats['budget'] = float(budget.detach())
        if teacher is not None and (config.prior_weight or config.effect_weight):
            teacher_legal = _legal(batch_rows, teacher_rows, args.device)
            if config.effect_target == "utility":
                targets = measure_utility_teacher(teacher, data, teacher_reference, teacher_labels,
                                                   legal=teacher_legal, labels=labels, seed=args.seed + step,
                                                   samples=config.utility_samples if config.effect_weight else 0)
                target_key, loss_key = "utility", "utility_huber"
            else:
                targets = measure_teacher(teacher, data, teacher_reference, teacher_labels,
                                          legal=teacher_legal, labels=labels, strength=config.intervention_strength,
                                          measure_effects=bool(config.effect_weight))
                target_key, loss_key = "effects", "effect_huber"
            terms = distillation_losses(prediction, targets["attention"], targets[target_key],
                                        targets["attention_mask"], targets["effect_mask"],
                                        effect_target=config.effect_target)
            loss = loss + config.prior_weight * terms["prior_kl"] + config.effect_weight * terms[loss_key]
            stats.update({key: float(value.detach()) for key, value in terms.items()})
            stats.update(teacher_forwards=targets["teacher_forwards"],
                         supervised_effect_parents=int(targets["effect_mask"].sum()))
        if config.gate and config.consistency_weight:
            weight = config.consistency_weight * consistency_scale(step, config.consistency_warmup_steps,
                                                                   config.consistency_ramp_steps)
            term = dict(loss=prediction["logits"].sum()*0, eligible_donors=0)
            if weight:
                term = model.consistency_loss(data, reference, reference_labels, legal, prediction["logits"],
                                              query_rows=batch_rows, seed=args.seed + step, details=True)
            loss = loss + weight * term["loss"]
            stats.update(consistency=float(term["loss"].detach()), consistency_weight=weight,
                         eligible_donors=term["eligible_donors"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        stats.update(step=step, loss=float(loss.detach()), samples=len(batch_rows))
        history.append(stats)
        print(json.dumps(stats), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate_and_save(step)
            _write_json(output_dir / "history.json", history)
    if not args.steps:
        _write_json(output_dir / "history.json", history)


def evaluate_command(args):
    rows = load_records(args.manifest)
    split = _load_split(args.split, rows)
    if split["missing_source_classes"]:
        raise ValueError(f"Held classes absent from source: {split['missing_source_classes']}")
    model, checkpoint = _load_model(args.checkpoint, args.device)
    _check_source_checkpoint(checkpoint, _source_rows(rows, split), split)
    selected = [rows[i] for i in split[args.partition]]
    with torch.no_grad():
        reference, labels = _reference(model, checkpoint, args.device)
        scores = _predict(model, selected, checkpoint["bank_records"], reference, labels, args.batch, args.device)
    metrics = classification_metrics([r["label"] for r in selected], scores.argmax(1).numpy(), model.config.classes)
    result = dict(partition=args.partition, metrics=metrics,
                  predictions=[dict(id=row["id"], label=row["label"], prediction=int(z.argmax()), logits=z.tolist())
                               for row, z in zip(selected, scores)])
    _write_json(args.output, result)
    print(json.dumps(metrics))


def predict_command(args):
    rows = load_records(args.manifest)
    model, checkpoint = _load_model(args.checkpoint, args.device)
    with torch.no_grad():
        reference, labels = _reference(model, checkpoint, args.device)
        scores = _predict(model, rows, checkpoint["bank_records"], reference, labels, args.batch, args.device,
                          allow_unidentified=True)
    probabilities = scores.softmax(-1)
    _write_json(args.output, [dict(id=row["id"], prediction=int(z.argmax()), probabilities=p.tolist())
                              for row, z, p in zip(rows, scores, probabilities)])


def export_command(args):
    model, checkpoint = _load_model(args.checkpoint, args.device)
    with torch.no_grad():
        reference, _ = _reference(model, checkpoint, args.device)
    exported = {key: checkpoint[key] for key in ("config", "model", "protocol", "source_records", "step")}
    exported.update(encoded_bank=_tree_to(reference, "cpu", detach=True),
                    bank_records=_metadata(checkpoint["bank_records"]), frozen_bank=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)
    print("Exported frozen source bank:", len(exported["bank_records"]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="Create a port/identity-isolated split")
    split.add_argument("--manifest", required=True)
    split.add_argument("--target-ports", nargs="+", required=True)
    split.add_argument("--validation-ports", nargs="+", required=True)
    split.add_argument("--output", required=True)
    split.set_defaults(action=split_command)

    bank = commands.add_parser("select-bank", help="Select real source representatives at fixed class budgets")
    bank.add_argument("--manifest", required=True)
    bank.add_argument("--split", required=True)
    bank.add_argument("--checkpoint")
    bank.add_argument("--per-class", type=int, required=True)
    bank.add_argument("--method", choices=("coverage", "random"), default="coverage")
    bank.add_argument("--descriptor", choices=("body", "global"), default="body")
    bank.add_argument("--observation", choices=("medoid", "first"), default="medoid")
    bank.add_argument("--seed", type=int, default=17)
    bank.add_argument("--no-port-balance", action="store_true")
    bank.add_argument("--batch", type=int, default=8)
    bank.add_argument("--device", default="cpu")
    bank.add_argument("--output", required=True)
    bank.set_defaults(action=select_bank_command)

    train = commands.add_parser("train", help="Explicit source training; never reads target pixels or labels for selection")
    for option in ("manifest", "split", "bank", "output"):
        train.add_argument("--" + option, required=True)
    train.add_argument("--config", help="ModelConfig JSON; loss weights default to source CE only")
    initial = train.add_mutually_exclusive_group()
    initial.add_argument("--initialize", help="Strictly load a compatible source-fold checkpoint")
    initial.add_argument("--imagenet", help="Generic torchvision ResNet-50 pretraining state_dict")
    train.add_argument("--teacher", help="Frozen source-fold teacher with its own reference bank")
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--batch-sampling", choices=("shuffle", "paired"), default="shuffle",
                       help="paired keeps same-class different-vessel/group observations adjacent; use even batches")
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--device", default="cpu")
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--backbone-lr", type=float, default=1e-5)
    train.add_argument("--freeze", action="append", default=[], metavar="PREFIX",
                       help="Freeze matching named-parameter prefixes; repeat for multiple modules")
    train.add_argument("--eval-every", type=int, default=50)
    train.set_defaults(action=train_command)

    evaluate = commands.add_parser("evaluate", help="Evaluate a frozen checkpoint on a declared held partition")
    for option in ("checkpoint", "manifest", "split", "output"):
        evaluate.add_argument("--" + option, required=True)
    evaluate.add_argument("--partition", choices=("validation", "test"), default="test")
    evaluate.add_argument("--batch", type=int, default=8)
    evaluate.add_argument("--device", default="cpu")
    evaluate.set_defaults(action=evaluate_command)

    predict = commands.add_parser("predict", help="Predict unlabelled chips, optionally using an exported frozen bank")
    for option in ("checkpoint", "manifest", "output"):
        predict.add_argument("--" + option, required=True)
    predict.add_argument("--batch", type=int, default=8)
    predict.add_argument("--device", default="cpu")
    predict.set_defaults(action=predict_command)

    export = commands.add_parser("export", help="Freeze encoded source references and remove reference image paths")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--device", default="cpu")
    export.set_defaults(action=export_command)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "batch", 1) < 1:
        raise ValueError("batch must be positive")
    args.action(args)


if __name__ == "__main__":
    main()

