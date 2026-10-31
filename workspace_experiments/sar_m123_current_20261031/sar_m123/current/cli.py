"""Entrypoints: predict, select, fit, adapt. No historical experiment loading."""
import argparse
import json
from pathlib import Path

from .data import read_manifest
from .interventions import select_records
from .model import CLASSES, load_model, predict_records, setup
from .training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('predict', 'select', 'fit', 'adapt'))
    parser.add_argument('--model', required=True, help='Full current checkpoint; init.pt for fit.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--references', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--selection', help='Selection JSON for adapt only.')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--coefficient', type=float, default=.3683066666126251)
    parser.add_argument('--max-steps', type=int, help='Smoke test only, not a full training result.')
    args = parser.parse_args()
    setup(args.device)
    records, references = read_manifest(args.manifest), read_manifest(args.references)
    model = load_model(args.model, args.device)
    if args.command in ('predict', 'select'):
        if args.command == 'predict':
            scores = predict_records(model, records, references, args.device, args.batch_size or 16)
            output = [dict(id=r['id'], prediction=int(s.argmax()),
                           class_name=CLASSES[int(s.argmax())], logits=s.tolist(),
                           probabilities=s.softmax(0).tolist()) for r, s in zip(records, scores)]
        else:
            output = select_records(model, records, references, args.device,
                lambda n, total, r: print(f'SELECT {n}/{total} id={r["id"]} useful={r["useful"]}', flush=True))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        selection = json.loads(Path(args.selection).read_text(encoding='utf-8')) if args.selection else None
        train(model, records, references, args.output, args.device, stage=args.command,
              selection=selection, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
              coefficient=args.coefficient, max_steps=args.max_steps)


if __name__ == '__main__':
    main()
