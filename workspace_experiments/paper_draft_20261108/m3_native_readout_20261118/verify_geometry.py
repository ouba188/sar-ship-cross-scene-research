"""Independent pixel/cell intersection check, including transformed recovered views."""
from pathlib import Path
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('native_geometry_check_model', ROOT/'model.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
np, torch = m.np, m.torch


def check():
    inputs = m.Inputs()
    split = json.loads((ROOT/'split_local.json').read_text(encoding='utf-8'))
    original = next(i for i in split['fit'] if i not in inputs.restored)
    restored = next(iter(inputs.restored))
    selected = [('original', original), ('recovered', restored)]
    for i in split['fit'][:100]:
        views = m.old.old.t.e.task.data.prepare(inputs.rows[i])
        if any(v is not None and v['valid_target'] and v['footprint'].sum() < 160*160
               for v in views):
            selected.append(('partial_footprint', i))
            break
    assert len(selected) == 3
    centers = np.arange(40)*4
    edges = np.r_[-.5, (centers[:-1]+centers[1:])/2, 159.5]
    pixels = np.arange(160)
    overlap = np.maximum(0., np.minimum(edges[1:, None], pixels[None]+.5)-
                         np.maximum(edges[:-1, None], pixels[None]-.5))
    weights = overlap/np.diff(edges)[:, None]
    records = []
    for category, index in selected:
        views = m.old.old.t.e.task.data.prepare(inputs.rows[index])
        maximum = pixel_error = footprint_error = 0.
        for code in range(8):
            data = inputs.data([index], 'small', code)
            for p, view in enumerate(views):
                if view is None or not view['valid_target']:
                    assert not data['read40'][0, p].any()
                    continue
                native = m.raw.tokens.slots(view['geometry'], view['footprint'])
                transformed = np.rot90(native, code % 4, axes=(-2, -1))
                image = np.rot90(view['pixels'], code % 4)
                valid = np.rot90(view['footprint'], code % 4)
                if code // 4:
                    transformed, image, valid = transformed[..., ::-1], image[:, ::-1], valid[:, ::-1]
                expected = np.einsum('ih,shw,jw->sij', weights, transformed, weights, optimize=True)
                maximum = max(maximum, float(abs(expected-data['read40'][0, p].cpu().numpy()).max()))
                pixel_error = max(pixel_error, float(abs(image/255-data['images'][0, p, 0].cpu().numpy()).max()))
                footprint_error = max(footprint_error, float(abs(valid-data['valid'][0, p, 0].cpu().numpy()).max()))
        assert maximum < 2e-6 and pixel_error < 1e-6 and footprint_error == 0
        records.append(dict(category=category, codes=8, mask_max_error=maximum,
                            image_max_error=pixel_error, footprint_max_error=footprint_error))
    (ROOT/'geometry_checks.json').write_text(json.dumps(dict(passed=True, records=records), indent=2), encoding='utf-8')
    print(json.dumps(records), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    check()
