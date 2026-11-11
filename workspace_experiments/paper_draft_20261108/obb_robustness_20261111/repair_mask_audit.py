"""Recompute native diagnostic areas; never modify geometry or predictions."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from obb_geometry import conditions, perturb

ROOT = Path(__file__).resolve().parent


def regions(g):
    yy, xx = np.mgrid[:160, :160]
    cx, cy, length, width, angle = g[:5]
    u = (xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle)
    v = -(xx-cx)*np.sin(angle)+(yy-cy)*np.cos(angle)
    def rect(l, w):
        return np.clip(l/2+.5-np.abs(u), 0, 1)*np.clip(w/2+.5-np.abs(v), 0, 1)
    body = rect(length, width)
    return np.stack((body, np.maximum(rect(length, 2*width)-body, 0),
                     np.maximum(rect(1.5*length, width)-body, 0)))


def main():
    audit = pd.read_csv(ROOT/'mask_audit.csv')
    source = json.loads((ROOT/'native_sources.json').read_text(encoding='utf-8'))
    positions = {(name, int(i), pol, r): k for k, (name, i, pol, r) in enumerate(
        audit[['condition', 'index', 'polarization', 'region']].itertuples(index=False, name=None))}
    checks = {k: 0. for k in ('clean_area_px', 'changed_area_px', 'native_l1_px',
                              'original_target_covered', 'soft_iou')}
    outside = np.empty(len(audit))
    for rec in source:
        p = rec['placement']
        footprint = np.zeros((160, 160))
        footprint[p['pad_top']:p['pad_top']+p['crop_height'],
                  p['pad_left']:p['pad_left']+p['crop_width']] = 1
        before = regions(rec['geometry'])*footprint
        for cfg in conditions():
            raw = regions(perturb(rec['geometry'], cfg))
            after = raw*footprint
            for k, name in enumerate(('body', 'side', 'end')):
                ix = positions[cfg['name'], rec['index'], rec['polarization'], name]
                a, b = before[k], after[k]
                intersection = np.minimum(a, b).sum()
                values = dict(clean_area_px=a.sum(), changed_area_px=b.sum(),
                    native_l1_px=np.abs(a-b).sum(),
                    original_target_covered=intersection/max(a.sum(), 1e-12),
                    soft_iou=intersection/max(np.maximum(a,b).sum(), 1e-12))
                for key, value in values.items():
                    checks[key] = max(checks[key], abs(value-float(audit.at[ix,key])))
                outside[ix] = (raw[k]*(1-footprint)).sum()
    assert max(checks.values()) < 1e-6, checks
    receipt = dict(native_diagnostic_max_errors=checks, n=len(audit),
        negative_before=int((audit.outside_footprint_px < -1e-8).sum()),
        negative_after=int((outside < -1e-8).sum()),
        outside_max=float(outside.max()), predictions_untouched=True)
    backup = ROOT/'mask_audit_before_auxiliary_fix.csv'
    if not backup.exists():
        shutil.copy2(ROOT/'mask_audit.csv', backup)
    audit['outside_footprint_px'] = outside
    audit.to_csv(ROOT/'mask_audit.csv', index=False)
    (ROOT/'auxiliary_audit_fix.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
