"""Fixed native-PNG Gabor descriptors; no fitting or classifier execution."""
import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import fft
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAIRED = ROOT / 'paired_scale_readout'
WAVELENGTHS = np.array([4., 8., 16.])
ANGLES = np.arange(4) * np.pi / 4
RADII = np.ceil(3 * WAVELENGTHS / 2).astype(int)
QUANTILES = [0, 10, 25, 50, 75, 90, 100]
POL = ('vv', 'vh')
CASES = (16596, 25707)
REGIONS = {'body': slice(0, 8), 'side': slice(8, 16), 'end': slice(16, 20)}
spec = importlib.util.spec_from_file_location('frequency_token_geometry', ROOT / 'extract_tokens.py')
token = importlib.util.module_from_spec(spec)
spec.loader.exec_module(token)
geo = token.geo


def kernel(wavelength, theta):
    sigma = wavelength / 2
    radius = int(np.ceil(3 * sigma))
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    envelope = np.exp(-(xx*xx + yy*yy) / (2*sigma*sigma))
    carrier = np.exp(2j*np.pi*(xx*np.cos(theta) + yy*np.sin(theta))/wavelength)
    dc = (envelope*carrier).sum() / envelope.sum()
    result = envelope * (carrier - dc)
    return result / np.sqrt(np.sum(np.abs(result)**2))


def response(raw, psi):
    radius = psi.shape[0] // 2
    padded = np.pad(raw, radius, mode='reflect')
    filtered = fftconvolve(padded, psi, mode='same')
    return filtered[radius:radius+raw.shape[0], radius:radius+raw.shape[1]]


def mask_coordinates(masks, placement, shape):
    owner, cy, cx = np.nonzero(masks)
    weights = masks[owner, cy, cx]
    yy = cy - placement['pad_top'] + placement['crop_y']
    xx = cx - placement['pad_left'] + placement['crop_x']
    assert ((yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])).all()
    mass = np.bincount(owner, weights=weights, minlength=20)
    return owner, yy, xx, weights, mass


def pool(field, coordinates):
    owner, yy, xx, weights, mass = coordinates
    total = np.bincount(owner, weights=weights*field[yy, xx], minlength=20)
    return np.divide(total, mass, out=np.zeros(20), where=mass > 0)


def coverage_of(shape, coordinates):
    owner, yy, xx, weights, mass = coordinates
    out = np.zeros((20, 3))
    for b, radius in enumerate(RADII):
        interior = ((xx >= radius) & (xx < shape[1]-radius)
                    & (yy >= radius) & (yy < shape[0]-radius))
        total = np.bincount(owner, weights=weights*interior, minlength=20)
        out[:, b] = np.divide(total, mass, out=np.zeros(20), where=mass > 0)
    assert ((out >= -1e-12) & (out <= 1+1e-12)).all()
    return np.clip(out, 0, 1)


def descriptor(energy, active):
    result = np.concatenate((np.log1p(energy/1e-4),
                             energy/(energy.sum(-1, keepdims=True)+1e-8)), axis=-1)
    result[~active] = 0
    return result.astype(np.float32)


def quantiles(values):
    return np.percentile(np.asarray(values), QUANTILES).tolist()


def relative_error(a, b):
    return float(np.max(np.abs(a-b)) / max(float(np.max(np.abs(b))), 1e-12))


def filter_checks(raw, theta):
    # These checks detect DC leakage and the sign error from image y pointing down.
    errors = dict(constant_response_max=0., rotation_energy_relative_max=0.,
                  halfturn_energy_relative_max=0., kernel_sum_abs_max=0.,
                  kernel_l2_error_max=0., image_channel_rotation_relative_max=0.)
    for wavelength in WAVELENGTHS:
        for angle in np.r_[ANGLES, theta+ANGLES]:
            psi = kernel(wavelength, angle)
            original = np.abs(response(raw, psi))**2
            rotated = np.abs(response(np.rot90(raw), kernel(wavelength, angle-np.pi/2)))**2
            halfturn = np.abs(response(raw, kernel(wavelength, angle+np.pi)))**2
            errors['rotation_energy_relative_max'] = max(errors['rotation_energy_relative_max'],
                                                          relative_error(rotated, np.rot90(original)))
            errors['halfturn_energy_relative_max'] = max(errors['halfturn_energy_relative_max'],
                                                          relative_error(halfturn, original))
            errors['constant_response_max'] = max(errors['constant_response_max'],
                float(np.abs(response(np.ones_like(raw), psi)).max()))
            errors['kernel_sum_abs_max'] = max(errors['kernel_sum_abs_max'], float(abs(psi.sum())))
            errors['kernel_l2_error_max'] = max(errors['kernel_l2_error_max'],
                                                float(abs(np.sum(abs(psi)**2)-1)))
        for a, angle in enumerate(ANGLES):
            original = np.abs(response(raw, kernel(wavelength, angle)))**2
            rotated = np.abs(response(np.rot90(raw), kernel(wavelength, ANGLES[(a-2) % 4])))**2
            errors['image_channel_rotation_relative_max'] = max(errors['image_channel_rotation_relative_max'],
                relative_error(rotated, np.rot90(original)))
    assert errors['constant_response_max'] < 1e-11, errors
    assert errors['kernel_sum_abs_max'] < 1e-11, errors
    assert errors['kernel_l2_error_max'] < 1e-12, errors
    assert max(errors[k] for k in errors if 'relative' in k) < 1e-10, errors
    return errors


def mapping_checks(masks, placement, raw, coordinates):
    # Independent dense canvas route must match sparse original-pixel pooling.
    owner, yy, xx, weights, mass = coordinates
    x, y = placement['crop_x'], placement['crop_y']
    w, h = placement['crop_width'], placement['crop_height']
    dx, dy = placement['pad_left'], placement['pad_top']
    canvas = np.zeros((160, 160))
    canvas[dy:dy+h, dx:dx+w] = raw[y:y+h, x:x+w]
    expected = np.divide((masks*canvas).sum((-2, -1)), mass,
                         out=np.zeros(20), where=mass > 0)
    error = float(np.max(abs(pool(raw, coordinates)-expected)))
    assert error < 1e-10, error
    assert abs(weights.sum()-masks.sum()) < 1e-8
    constant = pool(np.ones_like(raw), coordinates)
    np.testing.assert_allclose(constant, mass > 0, atol=1e-12)
    return error


def synthetic_checks():
    rng = np.random.default_rng(1024)
    raw = rng.random((53, 71))
    errors = filter_checks(raw, .371)
    errors['mapping_max_abs'] = 0.
    for shape, placement in [((53, 71), dict(crop_x=0, crop_y=0, crop_width=71,
            crop_height=53, pad_left=45, pad_top=54)),
            ((193, 211), dict(crop_x=29, crop_y=17, crop_width=160,
                             crop_height=160, pad_left=0, pad_top=0))]:
        field = np.arange(np.prod(shape), dtype=float).reshape(shape)/np.prod(shape)
        fp = np.zeros((160, 160))
        dx, dy = placement['pad_left'], placement['pad_top']
        fp[dy:dy+placement['crop_height'], dx:dx+placement['crop_width']] = 1
        masks = token.slots(np.array([79.2, 81.7, 31., 11., .371, 1.]), fp)
        coordinates = mask_coordinates(masks, placement, shape)
        errors['mapping_max_abs'] = max(errors['mapping_max_abs'],
                                        mapping_checks(masks, placement, field, coordinates))
    return errors


def font(size=16):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)


def heatmap(values, vmax, signed=False):
    if signed:
        x = np.clip(values/max(vmax, 1e-12), -1, 1)
        rgb = np.stack((1-np.maximum(-x, 0), 1-abs(x), 1-np.maximum(x, 0)), -1)
    else:
        x = np.clip(values/max(vmax, 1e-12), 0, 1)
        rgb = np.stack((np.minimum(2*x, 1), x*x, np.maximum(1-3*x, 0)*.22), -1)
    return Image.fromarray(np.rint(255*rgb).astype(np.uint8))


def kernel_panel():
    sheet = Image.new('RGB', (1120, 760), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 10), 'Zero-DC complex Gabor | real / imaginary | native support, nearest x2', fill='black', font=font(19))
    for b, wavelength in enumerate(WAVELENGTHS):
        for a, angle in enumerate(ANGLES):
            psi = kernel(wavelength, angle)
            vmax = float(abs(psi).max())
            ox, oy = 12+a*277, 50+b*232
            draw.text((ox, oy), f'lambda={wavelength:g}  phi={a*45} deg  r={RADII[b]}', fill='black', font=font(15))
            # Both components share a signed color scale within each kernel.
            for k, component in enumerate((psi.real, psi.imag)):
                image = heatmap(component, vmax, signed=True)
                image = image.resize((image.width*2, image.height*2), Image.Resampling.NEAREST)
                sheet.paste(image, (ox+k*134, oy+30))
            draw.text((ox, oy+135), f'complex L2=1, max |psi|={vmax:.4f}', fill='black', font=font(13))
    sheet.save(HERE / 'kernels.png')


def case_panel(obs_id, pol, raw, g, placement, energies):
    scale = max(1, min(3, 250//max(raw.shape)))
    pw, ph = raw.shape[1]*scale, raw.shape[0]*scale
    tw, th = max(240, pw+18), ph+65
    sheet = Image.new('RGB', (5*tw, 7*th+55), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f'{obs_id} {pol.upper()} | original PNG / 255 | nearest x{scale} | no resize before filtering',
              fill='black', font=font(18))
    draw.text((12, 31), 'Response display: log1p(E / 1e-4); common vmax per wavelength across both frames and all four angles.',
              fill='black', font=font(15))
    native = Image.fromarray(np.rint(raw*255).astype(np.uint8)).convert('RGB')
    native = native.resize((pw, ph), Image.Resampling.NEAREST)
    sheet.paste(native, (8, 80))
    draw.text((8, 58), f'Original {raw.shape[1]} x {raw.shape[0]}', fill='black', font=font(15))
    overlay = native.copy()
    od = ImageDraw.Draw(overlay)
    center = g[:2] + [placement['crop_x']-placement['pad_left'], placement['crop_y']-placement['pad_top']]
    axis = np.array([np.cos(g[4]), np.sin(g[4])])
    cross = np.array([-axis[1], axis[0]])
    corners = [tuple((center + u*g[2]*axis/2 + v*g[3]*cross/2)*scale)
               for u, v in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    od.line(corners+[corners[0]], fill=(0, 255, 180), width=1)
    # The square is the largest kernel support at one representative body center.
    cx, cy = center*scale
    radius = int(RADII[-1])*scale
    od.rectangle((cx-radius, cy-radius, cx+radius, cy+radius), outline=(255, 90, 50), width=1)
    sheet.paste(overlay, (tw+8, 80))
    draw.text((tw+8, 58), 'OBB + lambda16 support at center', fill='black', font=font(14))
    logs = np.log1p(energies/1e-4)
    for frame in range(2):
        for b, wavelength in enumerate(WAVELENGTHS):
            row = 1+frame*3+b
            oy = 55+row*th
            vmax = max(float(np.percentile(logs[:, b], 99.5)), 1e-12)
            draw.text((8, oy+5), ('ship: g[4]+phi' if frame == 0 else 'image: phi'), fill='black', font=font(16))
            draw.text((8, oy+29), f'lambda={wavelength:g}; r={RADII[b]}', fill='black', font=font(16))
            draw.text((8, oy+53), f'display vmax={vmax:.3f}', fill='black', font=font(15))
            draw.text((8, oy+77), 'cyan: full-support centers', fill='black', font=font(13))
            for a in range(4):
                ox = (a+1)*tw+8
                draw.text((ox, oy+4), f'phi={a*45} deg', fill='black', font=font(15))
                image = heatmap(logs[frame, b, a], vmax).resize((pw, ph), Image.Resampling.NEAREST)
                edge = ImageDraw.Draw(image)
                radius = int(RADII[b])
                if raw.shape[1] > 2*radius and raw.shape[0] > 2*radius:
                    edge.rectangle((radius*scale, radius*scale,
                        (raw.shape[1]-radius)*scale-1, (raw.shape[0]-radius)*scale-1), outline=(0, 255, 200))
                sheet.paste(image, (ox, oy+27))
    sheet.save(HERE / f'responses_{obs_id}_{pol}.png')


def summaries(table, valid, mass, coverage, features):
    coverage_rows, norm_rows = [], []
    for scope in ('all', 'utility_train', 'selection', 'evaluation', 'strategy_fit'):
        rows = np.ones(len(table), bool) if scope == 'all' else table.role.eq(scope).to_numpy()
        for p, pol in enumerate(POL):
            for region, sl in REGIONS.items():
                active = valid[rows, p, None] & (mass[rows, p, sl] > 0)
                mm = mass[rows, p, sl][active].astype(float)
                for b, wavelength in enumerate(WAVELENGTHS):
                    values = coverage[rows, p, sl, b][active]
                    coverage_rows.append(dict(scope=scope, pol=pol, region=region, wavelength=float(wavelength),
                        n_slots=len(values), quantiles=quantiles(values), mean=float(values.mean()),
                        mass_weighted_mean=float(np.sum(values*mm)/mm.sum()),
                        fraction_zero=float((values == 0).mean()), fraction_lt_half=float((values < .5).mean()),
                        fraction_lt_one=float((values < 1-1e-7).mean())))
                for frame, feature in features.items():
                    values = feature[rows, p, sl][active]
                    for block, block_slice in [('all24', slice(None)), ('log12', slice(0, 12)), ('share12', slice(12, 24))]:
                        norms = np.linalg.norm(values[:, block_slice].astype(float), axis=-1)
                        q = quantiles(norms)
                        norm_rows.append(dict(scope=scope, pol=pol, region=region, frame=frame, block=block,
                            n_slots=len(norms), mean=float(norms.mean()),
                            **{f'q{k}': v for k, v in zip(QUANTILES, q)},
                            zero_fraction=float((norms == 0).mean())))
    return coverage_rows, norm_rows


def main():
    started = time.perf_counter()
    token.torch.set_num_threads(2)
    with np.load(PAIRED / 'views.npz') as z:
        ids, valid, mass, bank = [z[k].copy() for k in ('ids', 'valid', 'mass', 'bank')]
    table = pd.read_csv(PAIRED / 'table.csv', dtype={'instance_id': str, 'mmsi': str})
    np.testing.assert_array_equal(ids, table.table_index)
    assert len(ids) == 1805 and len(np.unique(ids)) == 1805
    assert np.isin(bank, ids).all()
    source = table.role.eq('utility_train').to_numpy()
    assert int(valid[source].sum()) == 1205
    wanted = set(table.obs_id)
    snapshots = {int(s['obs_id']): s for s in json.loads(
        (geo.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))
        if int(s['obs_id']) in wanted}
    checks = {'synthetic': synthetic_checks(), 'real_cases': {},
              'mass_relative_max': 0., 'placement_pixel_max_abs': 0., 'mapping_max_abs': 0.,
              'pool_range_violation_max': 0.}
    print('CHECKS synthetic', json.dumps(checks['synthetic']), flush=True)
    features = {frame: np.zeros((len(ids), 2, 20, 24), np.float32) for frame in ('ship', 'image')}
    coverage = np.zeros((len(ids), 2, 20, 3), np.float32)
    fixed_kernels = [[kernel(wavelength, angle) for angle in ANGLES] for wavelength in WAVELENGTHS]
    source_geometry, case_provenance = [], []
    shapes = []
    last = time.perf_counter()
    for j, row in enumerate(table.itertuples()):
        for p, pol in enumerate(POL):
            if not valid[j, p]:
                continue
            pix, fp, g, placement = geo.native(table, snapshots, j, p)
            assert g is not None
            path = getattr(row, 'image_'+pol)
            with Image.open(path) as image:
                raw = np.array(image.convert('L'), dtype=np.float64)/255
            shapes.append(raw.shape)
            masks = token.slots(g, fp)
            coordinates = mask_coordinates(masks, placement, raw.shape)
            actual_mass = coordinates[-1]
            active = mass[j, p] > 0
            np.testing.assert_array_equal(actual_mass > 0, active)
            mass_error = float(np.max(abs(actual_mass-mass[j, p])/np.maximum(actual_mass, 1e-20)))
            assert mass_error < 1e-6, (row.obs_id, pol, mass_error)
            checks['mass_relative_max'] = max(checks['mass_relative_max'], mass_error)
            owner, yy, xx, weights, _ = coordinates
            cy = yy-placement['crop_y']+placement['pad_top']
            cx = xx-placement['crop_x']+placement['pad_left']
            pixel_error = float(np.max(abs(raw[yy, xx]-pix[cy, cx]/255)))
            assert pixel_error == 0
            checks['placement_pixel_max_abs'] = max(checks['placement_pixel_max_abs'], pixel_error)
            checks['mapping_max_abs'] = max(checks['mapping_max_abs'],
                mapping_checks(masks, placement, raw, coordinates))
            coverage[j, p] = coverage_of(raw.shape, coordinates)
            display = np.empty((2, 3, 4, *raw.shape)) if row.obs_id in CASES else None
            for f, frame in enumerate(('ship', 'image')):
                pooled = np.zeros((20, 12))
                for b, wavelength in enumerate(WAVELENGTHS):
                    for a, phi in enumerate(ANGLES):
                        psi = kernel(wavelength, g[4]+phi) if frame == 'ship' else fixed_kernels[b][a]
                        energy = np.abs(response(raw, psi))**2
                        values = pool(energy, coordinates)
                        picked = energy[yy, xx]
                        violation = max(0., float(values[active].max()-picked.max()),
                                        float(picked.min()-values[active].min()))
                        assert violation < 1e-12
                        checks['pool_range_violation_max'] = max(checks['pool_range_violation_max'], violation)
                        pooled[:, 4*b+a] = values
                        if display is not None:
                            display[f, b, a] = energy
                features[frame][j, p] = descriptor(pooled, active)
            if source[j]:
                item = dict(local_index=j, table_index=int(ids[j]), obs_id=int(row.obs_id),
                    instance_id=row.instance_id, pol=pol, image_path=path, native_height=raw.shape[0],
                    native_width=raw.shape[1], length=float(g[2]), width=float(g[3]), theta=float(g[4]),
                    confidence=float(g[5]), center_x=float(g[0]+placement['crop_x']-placement['pad_left']),
                    center_y=float(g[1]+placement['crop_y']-placement['pad_top']),
                    **{k: placement[k] for k in ('crop_x', 'crop_y', 'crop_width', 'crop_height', 'pad_left', 'pad_top')})
                for b, wavelength in enumerate(WAVELENGTHS):
                    suffix = str(int(wavelength))
                    item['wavelength_over_width_'+suffix] = float(wavelength/g[3])
                    item['wavelength_over_length_'+suffix] = float(wavelength/g[2])
                    item['coverage_mass_weighted_'+suffix] = float(np.dot(coverage[j, p, :, b], actual_mass)/actual_mass.sum())
                source_geometry.append(item)
            if display is not None:
                checks['real_cases'][f'{row.obs_id}_{pol}'] = filter_checks(raw, float(g[4]))
                case_panel(int(row.obs_id), pol, raw, g, placement, display)
                case_provenance.append(dict(obs_id=int(row.obs_id), pol=pol, table_index=int(ids[j]),
                    image_path=path, shape=list(raw.shape), geometry=g.tolist(), placement=placement,
                    coverage_mass_weighted=((coverage[j, p]*actual_mass[:, None]).sum(0)/actual_mass.sum()).tolist(),
                    mask_mass=float(actual_mass.sum()), selection='user-specified diagnostic IDs; no label fitting'))
            del masks, coordinates
        if time.perf_counter()-last > 20:
            print('EXTRACT', j+1, '/', len(ids), 'seconds', round(time.perf_counter()-started, 1), flush=True)
            last = time.perf_counter()
    active = valid[..., None] & (mass > 0)
    for frame, values in features.items():
        assert np.isfinite(values).all()
        assert np.count_nonzero(values[~active]) == 0
    assert np.count_nonzero(coverage[~active]) == 0
    assert len(case_provenance) == 4
    checks['invalid_zero'] = True
    checks['invalid_pol_count'] = int((~valid).sum())
    checks['invalid_slot_count'] = int((~active).sum())
    checks['all_finite'] = True
    geometry = pd.DataFrame(source_geometry)
    assert len(geometry) == 1205
    expected = {'length': [9.06, 13.24, 16.84, 24., 30.43, 35.99, 63.16],
                'width': [5.13, 7.20, 8.11, 9.40, 10.62, 12.16, 19.12]}
    geometry_summary = {key: quantiles(geometry[key]) for key in expected}
    for key in expected:
        np.testing.assert_allclose(geometry_summary[key], expected[key], atol=.011, rtol=0)
    ratio_summary = {key: quantiles(geometry[key]) for key in geometry if key.startswith('wavelength_over_')}
    coverage_summary, norms = summaries(table, valid, mass, coverage, features)
    geometry.to_csv(HERE / 'source_geometry.csv', index=False)
    pd.DataFrame(norms).to_csv(HERE / 'frequency_norms.csv', index=False)
    np.savez_compressed(HERE / 'features.npz', ids=ids, e_ship=features['ship'], e_image=features['image'],
                        coverage=coverage, wavelengths=WAVELENGTHS, angles=ANGLES)
    # Reopen only the public output to catch a serialization or axis-order error.
    with np.load(HERE / 'features.npz') as saved:
        np.testing.assert_array_equal(saved['ids'], table.table_index)
        for key, values in [('e_ship', features['ship']), ('e_image', features['image']), ('coverage', coverage)]:
            np.testing.assert_array_equal(saved[key], values)
    checks['saved_array_roundtrip'] = True
    kernel_panel()
    receipt = dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        n=len(ids), valid_pol=int(valid.sum()), utility_valid_pol=len(geometry),
        roles=table.role.value_counts().to_dict(), seconds=time.perf_counter()-started,
        threads=2, device='CPU', new_fits=0, new_classifiers=0, labels_used=False,
        source_views=str(PAIRED/'views.npz'), source_table=str(PAIRED/'table.csv'),
        inherited_arrays=dict(mass='views.npz:mass', valid='views.npz:valid', bank='views.npz:bank',
                              bank_shape=list(bank.shape), ids='original table_index; preserve paired table row order'),
        shapes=dict(e_ship=list(features['ship'].shape), e_image=list(features['image'].shape), coverage=list(coverage.shape)),
        wavelengths=WAVELENGTHS.tolist(), angles=ANGLES.tolist(), sigma=(WAVELENGTHS/2).tolist(),
        radius=RADII.tolist(), kernel_sizes=(2*RADII+1).tolist(), input='PIL convert L / 255, original full PNG',
        filter='numpy reflect padding on the original PNG, scipy.signal.fftconvolve same, crop to original shape',
        kernel='envelope*(complex_carrier - sum(envelope*carrier)/sum(envelope)); complex L2=1',
        feature_order='first 12 log1p(meanenergy/1e-4), next 12 meanenergy/(sum12energy+1e-8); wavelength-major, angle-minor',
        frames=dict(ship='g[4]+phi for all 20 slots', image='phi for all 20 slots'),
        rotation='image x right, y down: np.rot90 -> theta-pi/2; fixed image channels map a -> (a-2)%4',
        coverage_definition='sum(mask*indicator_entire_square_kernel_inside_original_PNG)/sum(mask), each wavelength; invalid=0; no exclusion',
        quantile_percentiles=QUANTILES, geometry_quantiles=geometry_summary, wavelength_ratio_quantiles=ratio_summary,
        native_height_quantiles=quantiles(np.array(shapes)[:, 0]), native_width_quantiles=quantiles(np.array(shapes)[:, 1]),
        boundary_coverage=coverage_summary, checks=checks, cases=case_provenance,
        limitations=['Band shares are not an orthogonal energy decomposition.',
            'Response centers have region assignments; filter support can cross the OBB.',
            'Original PNG boundary reflect extension is an observation-boundary assumption, not 160-canvas padding.',
            'No classifier, predictions, benefit estimate, physical-mechanism naming, or feature standardization fit.'])
    (HERE / 'extraction.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print('COMPLETE', json.dumps({k: receipt[k] for k in ('n', 'valid_pol', 'utility_valid_pol', 'seconds', 'shapes')}), flush=True)


if __name__ == '__main__':
    with fft.set_workers(2):
        main()
