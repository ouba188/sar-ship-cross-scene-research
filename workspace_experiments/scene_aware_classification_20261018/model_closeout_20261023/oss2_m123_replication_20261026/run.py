"""Reuse the frozen OSS1 four-class algorithms with explicit OSS2 inputs."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'oss1_m123_replication_20261026'
sys.path.insert(0, str(ORIGINAL))
import extract_frontend as extract


def configure():
    extract.HERE = HERE
    extract.DEFAULT_TABLE = HERE / 'stack_table.csv'
    extract.VIEW_TABLE = HERE / 'view_inventory.csv'
    extract.RADAR_TABLE = HERE / 'geometry.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['raw', 'frontend', 'm3', 'analyze'])
    args, extra = parser.parse_known_args()
    configure()
    if args.stage == 'raw':
        sys.argv = [__file__, *extra]
        extract.main()
    elif args.stage == 'frontend':
        import train_frontend as train
        train.HERE = HERE
        sys.argv = [__file__, *extra]
        train.main()
    elif args.stage == 'm3':
        import train_m3 as train
        sys.argv = [__file__, '--table', str(HERE/'stack_table.csv'),
                    '--raw', str(HERE/'frontend_cache/m3_raw.npz'),
                    '--frontend', str(HERE/'frontend'), '--out', str(HERE/'m3_results'), *extra]
        train.main()
    else:
        import analyze_m3 as analysis
        analysis.HERE = HERE
        sys.argv = [__file__, '--out', str(HERE/'m3_results'), *extra]
        analysis.main()


if __name__ == '__main__':
    main()
