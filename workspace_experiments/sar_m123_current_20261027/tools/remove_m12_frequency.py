"""Transfer a historical full checkpoint to the spatial-M1/M2 closeout model."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sar_m123.current.model import load_model, save_model, setup
from sar_m123.current.spatial_m12 import from_legacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path; historical weights must remain intact.')
    setup('cpu')
    old = load_model(args.source, 'cpu')
    model = from_legacy(old)
    save_model(model, args.output, source=str(args.source.resolve()),
               training_stage='transferred_unvalidated',
               change='M1 spectral selectors and all M2 band pooling removed; M3 unchanged',
               evaluation='No accuracy claim; refit and paired evaluation required.')
    print(f'Saved unvalidated spatial-M1/M2 checkpoint: {args.output}', flush=True)


if __name__ == '__main__':
    main()
