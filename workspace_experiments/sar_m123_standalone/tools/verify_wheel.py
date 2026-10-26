"""Run the packaged CLI from an empty directory against a real chip request."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def check(wheel, assets, parity, report):
    example = json.loads(parity.read_text(encoding='utf-8'))['native_checks'][0]
    with tempfile.TemporaryDirectory(prefix='sar-aligned-package-') as folder:
        root = Path(folder)
        request, output = root/'request.json', root/'prediction.json'
        request.write_text(json.dumps(example['request']), encoding='utf-8')
        script = """
import json, sys
sys.path.insert(0, sys.argv.pop(1))
from sar_m123.cli import main
main()
import sar_m123.aligned as module
print(json.dumps({'loaded_module': module.__file__}))
"""
        command = [sys.executable, '-X', 'utf8', '-c', script, str(wheel.resolve()),
                   '--model', 'aligned', '--assets', str((assets/example['source']).resolve()),
                   '--input', str(request), '--output', str(output), '--seeds', '17', '29', '43',
                   '--device', 'cpu', '--trace']
        done = subprocess.run(command, cwd=root, text=True, encoding='utf-8',
                              capture_output=True, check=True)
        result = json.loads(output.read_text(encoding='utf-8'))
        import numpy as np
        checks = []
        for pred in result['predictions']:
            expected = example['seeds'][str(pred['seed'])]
            score = np.asarray(pred['class_scores'])
            error = float(abs(score-np.asarray(expected['expected'])).max())
            np.testing.assert_allclose(score, expected['expected'], atol=2e-4, rtol=2e-5)
            assert pred['class_id'] == expected['expected_prediction']
            checks.append(dict(seed=pred['seed'], max_abs=error, prediction=pred['class_id']))
        imported = json.loads(done.stdout.strip().splitlines()[-1])['loaded_module']
        if '.whl' not in imported:
            raise RuntimeError(f'Test imported the workspace instead of the wheel: {imported}')
        receipt = dict(status='passed', obs_id=example['obs_id'], source=example['source'],
                       empty_working_directory=True, module_from_wheel=True, device='cpu',
                       query_cache_used=False, checks=checks, stdout=done.stdout)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', required=True, type=Path)
    parser.add_argument('--assets', type=Path, default=Path('assets_aligned'))
    parser.add_argument('--parity', type=Path, default=Path('validation_reports/aligned_parity.json'))
    parser.add_argument('--report', type=Path, default=Path('validation_reports/aligned_wheel.json'))
    args = parser.parse_args()
    check(args.wheel, args.assets, args.parity, args.report)
