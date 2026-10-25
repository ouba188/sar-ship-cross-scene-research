"""One-time conversion of research artifacts into a self-contained asset bundle.

Only this tool reads the legacy workspace; inference never imports it.
"""
import argparse
import json
from pathlib import Path
import shutil


def export_baseline(legacy, assets, backbone):
    target = assets/'upstream/models'
    target.mkdir(parents=True, exist_ok=True)
    names = ['axis_ridge.npz', 'neighborhood_scaler.npz']
    names += [f'{arm}_seed{seed}.pt' for arm in ('five', 'neighborhood', 'calibration')
              for seed in (17, 29, 43)]
    for name in names:
        shutil.copy2(legacy/'regional_utility_pilot/upstream/models'/name, target/name)
    (assets/'backbone').mkdir(parents=True, exist_ok=True)
    shutil.copy2(backbone, assets/'backbone/resnet50.pth')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, required=True,
                        help='model_closeout_20261023 directory')
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=Path(__file__).resolve().parents[1]/'assets')
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    export_baseline(args.legacy_root, args.assets, args.backbone)
    if not args.baseline_only:
        from export_m12 import export as export_m12
        from export_m3 import export as export_m3
        export_m12(args.legacy_root, args.assets)
        export_m3(args.legacy_root, args.assets)
    metadata = dict(format_version=1, seeds=[17, 29, 43], canvas=160,
        input='8-bit grayscale display chips; native crop/pad, no resize',
        class_names=['bulk_carrier', 'fishing_vessel', 'general_cargo',
                     'product_chemical_tanker', 'container_ship', 'crude_oil_tanker',
                     'tug_towing', 'offshore_supply'],
        selected_m3='both + independent VV + null-centered VH + anchored frequency',
        complete=not args.baseline_only,
        provenance='Frozen research checkpoints; no fitting or query labels used by export.')
    (args.assets/'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(dict(assets=str(args.assets.resolve()), complete=metadata['complete'])))


if __name__ == '__main__':
    main()
