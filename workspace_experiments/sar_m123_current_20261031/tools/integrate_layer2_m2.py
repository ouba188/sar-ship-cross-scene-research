"""Export the complete spatial model with an M2 at layer2[2], before addition."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sar_m123.current.model import load_model, save_model, setup
from sar_m123.current.spatial_layer2 import from_spatial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--layer2-state', type=Path,
                        help='Retained block2 seed checkpoint; omit for zero-writeback initialization.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path; retain historical checkpoints.')
    setup('cpu')
    source = load_model(args.source, 'cpu')
    state = None if args.layer2_state is None else torch.load(
        args.layer2_state, map_location='cpu', weights_only=True)['state']
    model = from_spatial(source, state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.output, source=str(args.source.resolve()),
               layer2_source=None if args.layer2_state is None else str(args.layer2_state.resolve()),
               training_stage='integrated_unvalidated', cache_boundary='layer1',
               new_m2='zero_initialized' if state is None else 'retained_block2_spatial_transforms',
               evaluation='Integration only; no new OA/BA claim or joint-refit result.')
    print(f'Saved complete integrated checkpoint: {args.output}', flush=True)


if __name__ == '__main__':
    main()
