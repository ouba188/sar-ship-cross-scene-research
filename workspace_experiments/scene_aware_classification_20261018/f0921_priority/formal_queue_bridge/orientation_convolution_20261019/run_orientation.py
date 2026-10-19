"""Matched geometry-reference operators on unchanged legal support tasks."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
OLD = BRIDGE / 'learned_geometry_crosspass_20261019'
MAPS = BRIDGE / 'convolution_prepool_20261019'
sys.path.insert(0, str(OLD))
import runner as prior_run
from model import centered, residual_probability
from orientation_model import OrientationReader


def axial_gap(a, b):
    return np.degrees(np.abs((a - b + np.pi / 2) % np.pi - np.pi / 2))


class Study(prior_run.Study):
    def __init__(self):
        self.meta = pd.read_csv(OLD / 'cohort.csv', dtype={'mmsi': str, 'date': str})
        self.tasks = json.loads((OLD / 'tasks.json').read_text())
        self.fit = np.asarray(self.tasks['fit'])
        self.y = self.meta.label.to_numpy(int)
        self.labels = torch.tensor(self.y, device='cuda')
        self.features = dict(np.load(BRIDGE / 'features.npz'))
        self.axes = pd.read_csv(BRIDGE / 'orbit_geometry_readout_20261019/axes.csv')
        self.geo = pd.read_csv(OLD / 'recovery/expanded_geometry.csv')
        np.testing.assert_array_equal(self.axes.row, self.meta.row)
        self.ship_valid = (self.axes.half_width.gt(0) & self.axes.half_length.gt(0)
                           & self.axes.axis_reliability.gt(1e-8)
                           & np.isfinite(self.axes.ship_axis_rad)).to_numpy()
        self.radar_valid = (self.geo.orbit_valid & np.isfinite(self.geo.los_angle_image_rad)).to_numpy()
        self.ship = np.nan_to_num(self.axes.ship_axis_rad.to_numpy())
        self.los = np.nan_to_num(self.geo.los_angle_image_rad.to_numpy())
        along = np.clip(self.axes.half_length.to_numpy() * 32 / 224 * .25, .75, 3)
        across = np.clip(self.axes.half_width.to_numpy() * 32 / 224 * .25, .5, 2)
        along = np.where(self.ship_valid, along, 1.)
        across = np.where(self.ship_valid, across, 1.)
        geometry = np.stack([self.ship, self.los, self.ship_valid, self.radar_valid, along, across], 1)
        self.geometry = geometry.astype(np.float32)
        self.geometry_tensor = torch.tensor(self.geometry, device='cuda')
        # Fixed per-observation random reference, independent of labels and tasks.
        self.wrong_offset = np.random.default_rng(20261019).uniform(np.pi/6, 5*np.pi/6, len(self.meta))
        wrong = self.geometry.copy()
        wrong[:, 1] += self.wrong_offset
        self.wrong_tensor = torch.tensor(wrong, device='cuda')
        maps = torch.tensor(np.load(MAPS / 'maps.npy'), device='cuda')
        fitmaps = maps[self.fit]
        self.mean = fitmaps.mean((0, 3, 4), keepdim=True)
        self.std = fitmaps.std((0, 3, 4), unbiased=False, keepdim=True).clamp_min(.01)
        del fitmaps
        maps = (maps - self.mean) / self.std
        roles = torch.tensor(np.load(BRIDGE / 'masks.npz')['regions'], dtype=torch.float32, device='cuda')
        masks = F.interpolate(roles, (32, 32), mode='area')
        self.valid = (masks.sum((1, 2, 3)) > 1e-8).cpu().numpy()
        self.availability = torch.tensor(self.valid, device='cuda')
        yy, xx = np.mgrid[:8, :8]
        x, y = (xx.flatten()+.5)/4-1, (yy.flatten()+.5)/4-1
        p = np.stack([x, y, x*x, y*y, x*y, np.sqrt(x*x+y*y)], 1)
        p = np.column_stack([np.tile(p, (2, 1)), np.repeat([1., 0.], 64), np.repeat([0., 1.], 64)])
        positions = torch.tensor(np.broadcast_to(p, (len(maps), 128, 8)).copy(), dtype=torch.float32, device='cuda')
        self.inputs = [maps, masks, positions]
        self.rotation_inputs = None
        self.record_coverage(along, across)
        np.savez_compressed(HERE / 'normalization.npz', mean=self.mean.cpu(), std=self.std.cpu(), fit=self.fit)

    def record_coverage(self, along, across):
        report = []
        for name, ids in [('fit', self.fit), ('validation', self.tasks['validation']),
                          ('source', self.tasks['source'])]:
            report.append(dict(split=name, observations=len(ids), image_valid=int(self.valid[ids].sum()),
                               ship_axis=int(self.ship_valid[ids].sum()), radar=int(self.radar_valid[ids].sum()),
                               both=int((self.ship_valid[ids] & self.radar_valid[ids]).sum())))
        pd.DataFrame(report).to_csv(HERE / 'coverage.csv', index=False)
        coverage = self.meta[['row', 'clean_row', 'mmsi', 'label', 'date', 'pass_key']].copy()
        coverage['ship_valid'] = self.ship_valid
        coverage['radar_valid'] = self.radar_valid
        coverage['axis_deg'] = np.degrees(self.ship % np.pi)
        coverage['relative_deg'] = axial_gap(self.ship, self.los)
        coverage['along_radius'] = along
        coverage['across_radius'] = across
        coverage['mismatch_gap_deg'] = axial_gap(self.los, self.los+self.wrong_offset)
        coverage.to_csv(HERE / 'observation_geometry.csv', index=False)
        source = coverage.iloc[self.fit].copy()
        for kind, column in [('image', 'axis_deg'), ('relative', 'relative_deg')]:
            ok = source.ship_valid if kind == 'image' else source.ship_valid & source.radar_valid
            part = source[ok].copy()
            part['angle_bin'] = np.floor(part[column]/30).astype(int)*30
            part.groupby(['label', 'date', 'angle_bin']).mmsi.nunique().rename('ships').to_csv(HERE / f'source_{kind}_coverage.csv')

    def episodes(self):
        records = json.loads((OLD / 'source_episodes.json').read_text())
        (HERE / 'source_episodes.json').write_text(json.dumps(records, indent=2))
        groups = {'fit': [], 'validation': []}
        for r in records:
            groups[r['split']].append(self.context(r['source'], r['support'], r['query']))
        return groups['fit'], groups['validation']

    def embedding(self, reader, ct, rotate=False):
        ids = ct['ids']
        geom = self.wrong_tensor if reader.arm == 'D' else self.geometry_tensor
        maps, masks, position = [x[ids] for x in self.inputs]
        geom = geom[ids]
        if rotate:
            maps = self.rotation_inputs[ids]
            masks = torch.rot90(masks, 1, (-2, -1))
            geom = geom.clone()
            geom[:, :2] -= np.pi/2
        raw, weights = reader(maps, masks, position, geom, self.availability[ids])
        return centered(raw, self.availability[ids], ct['ns']).double(), weights

    def predict(self, reader, ct, geometry=None):
        z, weights = self.embedding(reader, ct, bool(geometry))
        k = ct['base'] + .0625*(z@z.T)
        p, alpha, score = residual_probability(k, ct['prior'], ct['expert'], self.labels[ct['s']], ct['ns'])
        return p, k, alpha, score, weights

    def direction_records(self, reader, ct, task, draw, seed):
        with torch.no_grad():
            z, _ = self.embedding(reader, ct)
            sim = (z[ct['ns']:] @ z[:ct['ns']].T).cpu().numpy()
        old = ct['base'][ct['ns']:, :ct['ns']].cpu().numpy()
        records = []
        for kind, angles, valid in [('image', self.ship, self.ship_valid),
                                    ('relative', (self.ship-self.los) % np.pi, self.ship_valid & self.radar_valid)]:
            for j, q in enumerate(ct['q']):
                if not valid[q]:
                    continue
                s = ct['s']
                delta = axial_gap(angles[q], angles[s])
                same = self.y[s] == self.y[q]
                groups = {'same_near': same & (delta <= 15), 'same_far': same & (delta >= 45),
                          'other_near': ~same & (delta <= 15), 'other_far': ~same & (delta >= 45)}
                rec = dict(arm=reader.arm, seed=seed, task=task['name'], line=task['line'],
                           budget=task['budget'], draw=draw, row=int(q), kind=kind)
                for name, mask in groups.items():
                    mask = mask & valid[s]
                    rec[f'n_{name}'] = int(mask.sum())
                    rec[name] = float(sim[j, mask].mean()) if mask.any() else np.nan
                    rec[f'base_{name}'] = float(old[j, mask].mean()) if mask.any() else np.nan
                rec['hard_relation_gap'] = rec['same_far'] - rec['other_near']
                rec['base_hard_gap'] = rec['base_same_far'] - rec['base_other_near']
                records.append(rec)
        return records


def fit(study, arm, seed, fits, vals):
    torch.manual_seed(seed)
    reader = OrientationReader(arm).cuda()
    optimizer = torch.optim.AdamW(reader.parameters(), lr=.001, weight_decay=.0001)
    schedule = np.random.default_rng(22000+seed).integers(0, len(fits), (200, 2))
    history = []
    start = time.perf_counter()
    for step in range(201):
        if step:
            reader.train()
            optimizer.zero_grad(set_to_none=True)
            loss = 0.
            for index in schedule[step-1]:
                ct = fits[index]
                p = study.predict(reader, ct)[0]
                local = -p[torch.arange(len(ct['q']), device='cuda'), study.labels[ct['q']]].clamp_min(1e-12).log().mean()/2
                local.backward()
                loss += float(local.detach())
            norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
        if step in (0, 50, 100, 200):
            reader.eval()
            with torch.no_grad():
                risks = []
                for ct in vals:
                    p = study.predict(reader, ct)[0]
                    risks.append(float(-p[torch.arange(len(ct['q']), device='cuda'), study.labels[ct['q']]].clamp_min(1e-12).log().mean()))
            rec = dict(arm=arm, seed=seed, step=step, validation_nll=float(np.mean(risks)),
                       training_nll=loss if step else None, gradient_norm=float(norm) if step else None)
            history.append(rec)
            print('STEP', json.dumps(rec), flush=True)
    torch.save(reader.state_dict(), HERE / f'{arm}_seed{seed}.pt')
    pd.DataFrame(history).to_csv(HERE / f'learning_{arm}_{seed}.csv', index=False)
    result = dict(arm=arm, seed=seed, selected_step=200, seconds=time.perf_counter()-start,
                  parameters=sum(p.numel() for p in reader.parameters()), validation_nll=history[-1]['validation_nll'])
    print('FIT', json.dumps(result), flush=True)
    return reader.eval(), result


def main():
    torch.set_num_threads(2)
    # CUDA grid_sample backward does not promise bitwise deterministic gradients.
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.perf_counter()
    study = Study()
    fits, vals = study.episodes()
    contexts = [(t, d, study.context(t['source'], s, t['query'])) for t in study.tasks['tasks'] for d, s in enumerate(t['supports'])]
    training, metrics, predictions, relations = [], [], [], []
    for seed in (0, 1, 2):
        for arm in ('A', 'B', 'C', 'D'):
            reader, timing = fit(study, arm, seed, fits, vals)
            training.append(timing)
            for task, draw, ct in contexts:
                met, records = prior_run.collect(study, reader, arm, seed, task, draw, ct)
                metrics.append(met)
                predictions.extend(records)
                relations.extend(study.direction_records(reader, ct, task, draw, seed))
                if arm == 'A':
                    met, records = prior_run.collect(study, reader, 'base', seed, task, draw, ct)
                    metrics.append(met)
                    predictions.extend(records)
            pd.DataFrame(training).to_csv(HERE / 'training.csv', index=False)
            pd.DataFrame(metrics).to_csv(HERE / 'metrics.csv', index=False)
            pd.DataFrame(predictions).to_csv(HERE / 'predictions.csv', index=False)
            pd.DataFrame(relations).to_csv(HERE / 'direction_relations.csv', index=False)
    met = pd.DataFrame(metrics)
    met.groupby(['line', 'budget', 'arm'])[['oa', 'ba', 'brier', 'nll', 'map_r']].mean().to_csv(HERE / 'summary.csv')
    p = pd.DataFrame(predictions)
    keys = ['task', 'seed', 'draw', 'row']
    old = pd.read_csv(OLD / 'predictions.csv')
    prob = [f'p{i}' for i in range(7)]
    a = p[p.arm.eq('base')].set_index(keys).sort_index()[prob]
    b = old[old.arm.eq('base')].set_index(keys).sort_index()[prob]
    np.testing.assert_allclose(a, b, atol=2e-12)
    receipt = dict(finished=True, seconds=time.perf_counter()-started, fits=len(training), seeds=3,
                   queries=48, passes=2, support_draws=5, selected_step=200, independent_confirmation=False,
                   baseline_replay_max_error=float(abs(a-b).to_numpy().max()),
                   cuda_grid_sample_bitwise_determinism=False, old_classifier_frozen=True)
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
