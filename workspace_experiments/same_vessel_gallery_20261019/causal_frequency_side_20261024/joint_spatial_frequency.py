"""Small paired diagnostic; no classifier fits and no physical causal claim."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageDraw, ImageFont
import raw_pair_probe as raw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'round18_joint_spatial_frequency'
SPATIAL = ['log_mean', 'cv', 'rms_radius']
TARGETS = ['fft_ratio', 'fft_direction', 'gradient_ratio']


def measure(im, center):
    yy, xx = np.indices(im.shape)
    dx, dy = xx + .5 - center[0], yy + .5 - center[1]
    r = np.hypot(dx, dy)
    assert min(center[0], center[1], im.shape[1]-center[0], im.shape[0]-center[1]) >= 6
    win = np.where(r < 6, .5*(1+np.cos(np.pi*np.minimum(r/6, 1))), 0)
    mean = (win*im).sum()/win.sum()
    cv = np.sqrt((win*(im-mean)**2).sum()/win.sum())/mean
    radius = np.sqrt((win*im*r*r).sum()/(win*im).sum())*10
    spatial_den = (win*im*r*r).sum()
    patch = win*(im-mean)
    # A common frequency lattice prevents differing chip widths changing band bins.
    # Only zero-valued exterior is replaced; no observed pixel is resized or rotated.
    iy, ix = np.nonzero(win)
    local = patch[iy.min():iy.max()+1, ix.min():ix.max()+1]
    fft = np.fft.fft2(local, s=(64, 64))
    power = np.abs(fft)**2
    assert np.isclose(power.sum()/power.size, (patch*patch).sum(), rtol=1e-10)
    fx, fy = np.meshgrid(np.fft.fftfreq(64, d=10), np.fft.fftfreq(64, d=10))
    freq = np.hypot(fx, fy)
    low = (freq >= 1/120) & (freq < 1/40)
    high = (freq >= 1/40) & (freq <= 1/20)
    band = low | high
    cos2 = (fx*fx-fy*fy)/np.maximum(freq*freq, 1e-20)
    result = dict(log_mean=np.log(mean), cv=cv, rms_radius=radius,
                  spatial_cos2=(win*im*(dx*dx-dy*dy)).sum()/spatial_den,
                  spatial_sin2=(win*im*2*dx*dy).sum()/spatial_den,
                  fft_ratio=np.log(power[high].sum()/power[low].sum()),
                  fft_direction=(power[band]*cos2[band]).sum()/power[band].sum(),
                  gradient_ratio=raw.measure(im, r <= 6)['log_ratio'])
    return result, power, win


def partial(x, y, controls):
    a = np.column_stack([np.ones(len(x))]+[rankdata(c) for c in controls.T])
    rx, ry = rankdata(x), rankdata(y)
    rx -= a @ np.linalg.lstsq(a, rx, rcond=None)[0]
    ry -= a @ np.linalg.lstsq(a, ry, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]


def evaluate(pairs):
    summaries, predictions = [], []
    for (pol, center), g in pairs.groupby(['pol', 'center']):
        g = g.dropna().reset_index(drop=True)
        x = g[SPATIAL+['incidence']].to_numpy()
        angle = g.cos2relative.to_numpy()
        for target in TARGETS:
            y = g[target].to_numpy()
            shape = np.c_[x, g[['spatial_cos2', 'spatial_sin2']].to_numpy()]
            axis = np.c_[x, g[['axis_cos2', 'axis_sin2']].to_numpy()]
            designs = dict(space=x, space_angle=np.c_[x, angle],
                           space_shape=shape, space_shape_angle=np.c_[shape, angle],
                           space_axis=axis)
            pred = {k: [] for k in ['mean']+list(designs)}
            for i in range(len(g)):
                train = np.arange(len(g)) != i
                pred['mean'].append(y[train].mean())
                for name, design in designs.items():
                    scaler = StandardScaler().fit(design[train])
                    model = Ridge(alpha=10).fit(scaler.transform(design[train]), y[train])
                    pred[name].append(model.predict(scaler.transform(design[i:i+1]))[0])
                predictions.append(dict(pol=pol, center=center, target=target, mmsi=g.mmsi[i],
                                        actual=y[i], **{k: v[-1] for k, v in pred.items()}))
            mse = {k: np.mean((np.asarray(v)-y)**2) for k, v in pred.items()}
            summaries.append(dict(pol=pol, center=center, target=target, n=len(g),
                                  rho_angle=spearmanr(angle, y).statistic,
                                  partial_rank=partial(angle, y, x),
                                  partial_rank_shape=partial(angle, y, shape),
                                  rho_with_gradient=spearmanr(y, g.gradient_ratio).statistic,
                                  **{f'mse_{k}': v for k, v in mse.items()},
                                  relative_mse_gain=1-mse['space_angle']/mse['space'],
                                  shape_controlled_gain=1-mse['space_shape_angle']/mse['space_shape']))
    pd.DataFrame(predictions).to_csv(OUT/'oof_predictions.csv', index=False)
    result = pd.DataFrame(summaries)
    result.to_csv(OUT/'summary.csv', index=False)
    print(result.to_string(index=False))


def panels(gallery):
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    for a, b in [(8368, 8369), (25750, 25752), (7324, 7326), (25819, 25820)]:
        canvas = Image.new('RGB', (1000, 760), 'white')
        draw = ImageDraw.Draw(canvas)
        for row, (oid, pol) in enumerate([(a, 'vv'), (b, 'vv'), (a, 'vh'), (b, 'vh')]):
            im = np.load(ROOT/'round16_calibrated_relation/private_windows'/f'{oid}_{pol}.npz')['sigma_noise_amp']
            other = np.load(ROOT/'round16_calibrated_relation/private_windows'/f'{b if oid==a else a}_{pol}.npz')['sigma_noise_amp']
            p = gallery[str(oid)]['images'][pol]['priors']
            z, power, win = measure(im, p['anchor'])
            tile = Image.fromarray(np.uint8(np.clip(im/max(im.max(), other.max()), 0, 1)*255)).convert('RGB')
            d = ImageDraw.Draw(tile); cx, cy = p['anchor']
            d.ellipse((cx-6, cy-6, cx+6, cy+6), outline='orange')
            if p.get('ship_axis'):
                sc = p['ship_axis']['center']; d.ellipse((sc[0]-1, sc[1]-1, sc[0]+1, sc[1]+1), outline='cyan')
            canvas.paste(tile, (10, row*190+30))
            # Spectrum display only: normalized log power, never fed back into measurements.
            sp = np.log1p(np.fft.fftshift(power)/max(power.max(), 1e-20)*1000)/np.log(1001)
            spec = Image.fromarray(np.uint8(sp*255)).convert('RGB')
            canvas.paste(spec, (290, row*190+30))
            draw.text((10, row*190+4), f'{oid} {pol} native amplitude; orange=60m; cyan=OBB center', fill='black', font=font)
            draw.text((510, row*190+40), f'FFT log fine/coarse {z["fft_ratio"]:+.4f}\nFFT directional moment {z["fft_direction"]:+.4f}\nSpatial CV {z["cv"]:.3f}, RMS radius {z["rms_radius"]:.2f}m\nGradient log ratio {z["gradient_ratio"]:+.4f}', fill='black', font=font)
        canvas.save(OUT/f'case_{a}_{b}.png')


def main():
    OUT.mkdir(exist_ok=True)
    gallery = json.loads((ROOT.parent/'gallery_data.js').read_text(encoding='utf8').split('=', 1)[1].strip().removesuffix(';'))['observations']
    cohort = pd.read_csv(ROOT/'round14_raw_pairs/cohort.csv', dtype={'mmsi': str})
    rows = []
    for pair in cohort.itertuples():
        if 26525 in (pair.obs_id_a, pair.obs_id_b):
            continue
        for pol in ('vv', 'vh'):
            for oid in (pair.obs_id_a, pair.obs_id_b):
                p = gallery[str(oid)]['images'][pol]['priors']
                if not p.get('ship_axis'):
                    continue
                im = np.load(ROOT/'round16_calibrated_relation/private_windows'/f'{oid}_{pol}.npz')['sigma_noise_amp']
                for name, c in [('anchor', p['anchor']), ('obb_center', p['ship_axis']['center'])]:
                    z, _, _ = measure(im, c)
                    control, _, _ = measure(im*3, c)
                    assert all(np.isclose(z[k], control[k], atol=1e-10) for k in TARGETS)
                    rows.append(dict(obs_id=oid, mmsi=pair.mmsi, category=pair.class_name_a, pol=pol, center=name,
                                     axis_cos2=np.cos(2*p['ship_axis']['angle']), axis_sin2=np.sin(2*p['ship_axis']['angle']),
                                     incidence=p['radar_incidence'], cos2relative=np.cos(2*(p['ship_axis']['angle']-p['radar_axis'])), **z))
    endpoints = pd.DataFrame(rows)
    endpoints.to_csv(OUT/'endpoints.csv', index=False)
    differences = []
    for pair in cohort.itertuples():
        for (pol, center), g in endpoints[endpoints.mmsi.eq(pair.mmsi)].groupby(['pol', 'center']):
            if len(g) != 2:
                continue
            g = g.set_index('obs_id')
            differences.append(dict(mmsi=pair.mmsi, category=pair.class_name_a, pol=pol, center=center,
                                    **{k: g.loc[pair.obs_id_b, k]-g.loc[pair.obs_id_a, k] for k in SPATIAL+TARGETS+['spatial_cos2', 'spatial_sin2', 'axis_cos2', 'axis_sin2', 'incidence', 'cos2relative']}))
    pairs = pd.DataFrame(differences)
    pairs.to_csv(OUT/'paired_changes.csv', index=False)
    evaluate(pairs)
    panels(gallery)


if __name__ == '__main__':
    main()
