"""Run relocated CPU inference while denying file reads from the workspace."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


RUNNER = r'''
import json, os, sys
from pathlib import Path
workspace = os.path.normcase(os.path.abspath(sys.argv[1]))
def forbid_workspace(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if path == workspace or path.startswith(workspace + os.sep):
            raise PermissionError('Portability test forbids original workspace access: ' + path)
sys.addaudithook(forbid_workspace)
from sar_m123.pipeline import ShipClassifier
request = json.loads(Path('input/requests.json').read_text())
model = ShipClassifier('assets', device='cpu')
result = [model.predict(x, Path('input'), seeds=(17,29,43)) for x in request]
Path('result.json').write_text(json.dumps(result, allow_nan=False))
'''


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=root/'validation_local')
    parser.add_argument('--ids', nargs='+', default=['13372', '18767'])
    args = parser.parse_args()
    request = json.loads((args.fixture/'requests.json').read_text())
    selected = [r for r in request if str(r['id']) in args.ids]
    expected = np.load(args.fixture/'expected.npz')
    receipts = []
    with tempfile.TemporaryDirectory(prefix='m123_portable_') as folder:
        target = Path(folder)
        shutil.copytree(root/'sar_m123', target/'sar_m123', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(root/'assets', target/'assets')
        (target/'input/chips').mkdir(parents=True)
        for item in selected:
            for pol in ('vv', 'vh'):
                if item[pol] is not None:
                    shutil.copy2(args.fixture/item[pol]['image'], target/'input'/item[pol]['image'])
        (target/'input/requests.json').write_text(json.dumps(selected))
        subprocess.run([sys.executable, '-X', 'utf8', '-c', RUNNER, str(root.parent)],
                       cwd=target, check=True)
        results = json.loads((target/'result.json').read_text())
        for result in results:
            i = list(expected['obs_ids']).index(int(result['id']))
            for prediction in result['predictions']:
                seed = prediction['seed']
                score = np.array(prediction['class_scores'])
                old = expected[f'final_{seed}'][i]
                error = float(abs(score-old).max())
                assert prediction['class_id'] == int(old.argmax())
                assert error < .002
                receipts.append(dict(id=result['id'], seed=seed, max_score_error=error,
                                     prediction_matches=True))
    report = dict(status='passed', device='cpu', original_workspace_access='denied by Python open audit hook',
                  dependencies='Only relocated package, exported assets, and input chips; environment libraries remain external.',
                  records=receipts)
    (root/'validation_reports').mkdir(exist_ok=True)
    (root/'validation_reports/PORTABILITY.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
