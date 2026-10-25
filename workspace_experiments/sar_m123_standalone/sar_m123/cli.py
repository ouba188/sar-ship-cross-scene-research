"""Run a JSON chip/OBB request or a list of requests, without query labels."""
import argparse
import json
from pathlib import Path

import torch

from .pipeline import ShipClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='JSON request; image paths are relative to this file')
    parser.add_argument('--assets', type=Path,
                        default=Path(__file__).resolve().parents[1]/'assets')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'),
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seeds', nargs='+', type=int, default=[17], choices=(17, 29, 43),
                        help='Separate frozen configurations, not an ensemble')
    parser.add_argument('--trace', action='store_true', help='Include stage scores and query-reference evidence')
    args = parser.parse_args()
    requests = json.loads(args.input.read_text(encoding='utf-8-sig'))
    single = isinstance(requests, dict)
    if single:
        requests = [requests]
    classifier = ShipClassifier(args.assets, args.device)
    outputs = [classifier.predict(r, args.input.parent, args.seeds, args.trace) for r in requests]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs[0] if single else outputs, indent=2,
                                      ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output.resolve()), samples=len(outputs), seeds=args.seeds)))


if __name__ == '__main__':
    main()
