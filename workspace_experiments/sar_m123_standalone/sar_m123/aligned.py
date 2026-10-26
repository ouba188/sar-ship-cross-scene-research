"""Accepted public four-class raw-summary / aligned-M3 chip inference.

Only exported assets and native chips are read at runtime. The older eight-class
pipeline is independent. Feature orders match the public replication caches.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
import torch
from torch import nn

from . import frequency, regions, tokens
from .backbone import Backbone
from .input import prepare
from .m12 import attention
from .m3 import M3, pair_stats, regional_descriptors, standardize

FEATURES = ('x', 'mean', 'mass', 'joint', 'e_image', 'coverage')
CLASSES = ('bulk', 'general', 'chemical', 'container')


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def project_raw(raw, projection):
    """Project *after* the float32 raw-moment boundary used during training."""
    mass = raw['mass'].astype(np.float32)
    mean = ((raw['layer1']-projection['mean']) @ projection['components'].T).astype(np.float32)
    x = tokens.unit(mean).astype(np.float32)
    x[mass <= 0] = 0
    joint = (raw['joint_raw'] @ projection['components'].T).reshape(*mass.shape, 96)
    return dict(x=x, mean=mean, mass=mass, joint=joint.astype(np.float32),
                e_image=raw['e_image'], coverage=raw['coverage'])


def raw_features(raw, valid):
    """Batch of 1536 regional values (pol/body-side-end/channel), then flags."""
    chunks = []
    for ids in tokens.REGIONS:
        mass = raw['mass'][:, :, ids]
        value = (raw['layer1'][:, :, ids]*mass[..., None]).sum(2)
        chunks.append(value/np.maximum(mass.sum(2)[..., None], 1e-8))
    regional = np.stack(chunks, 2).reshape(len(valid), -1)
    return np.c_[regional, np.asarray(valid, np.float32)].astype(np.float32)


def summary_features(summaries):
    """Batch of 72 body-end-side summaries followed by 32 geometry/quality."""
    fields = []
    for p in range(2):
        g, q = summaries['geometry'][:, p], summaries['quality'][:, p]
        fields.extend((np.log1p(g[:, 0]), np.log1p(g[:, 1]), g[:, 2], g[:, 3]))
        for r in range(3):
            fields.extend((np.log1p(q[:, r, 0]), q[:, r, 1], np.log1p(q[:, r, 2]), q[:, r, 3]))
    return np.c_[summaries['summary1'].reshape(len(fields[0]), -1), np.stack(fields, -1)].astype(np.float32)


def transform(values, scaler):
    return np.where(scaler['active'], (values-scaler['mean'])/scaler['scale'], 0).astype(np.float32)


@torch.inference_mode()
def extract_native(prepared, maps, readers, tau):
    raw = dict(layer1=np.zeros((2, 20, 256), np.float32),
               joint_raw=np.zeros((2, 20, 3, 256), np.float32),
               mass=np.zeros((2, 20), np.float32),
               e_image=np.zeros((2, 20, 24), np.float32),
               coverage=np.zeros((2, 20, 3), np.float32))
    summary = dict(summary1=np.zeros((2, 3, 4, 3)), quality=np.zeros((2, 3, 4)),
                   geometry=np.zeros((2, 4)), valid=np.zeros(2, bool))
    for p, (view, fmap) in enumerate(zip(prepared, maps)):
        if view is None or not view['valid_target'] or view['geometry'] is None:
            continue
        fp, g = view['footprint'], view['geometry']
        summary['valid'][p] = True
        summary['geometry'][p] = np.r_[g[2:4], g[5], 1.]
        masks = regions.soft_regions(g)*fp
        quality = regions.region_quality(regions.soft_regions(g), fp, g)
        flags = (quality['area'] > 0) & (quality['min_neff'] >= 4)
        summary['quality'][p] = np.stack((quality['area'], quality['coverage'], quality['min_neff'], flags), -1)
        slots = tokens.slots(g, fp)
        spectral = frequency.extract(view['raw'], slots, view['placement'])
        raw['mass'][p] = slots.sum((-2, -1))
        raw['e_image'][p], raw['coverage'][p] = spectral['e_image'], spectral['coverage']
        avg, joint = np.zeros((20, 256)), np.zeros((20, 3, 256))
        layer = fmap['layer1']
        reader = torch.as_tensor(readers[p], device=layer.device)
        responses = torch.einsum('cd,kdu->kcu', reader, layer.flatten(2).double())
        for k in range(4):
            rotated = np.rot90(slots, k, (-2, -1))
            w = tokens.cell_mass(rotated).reshape(20, -1)
            energy = np.rot90(spectral['energy_field'], k, (-2, -1))
            ew = tokens.cell_mass(rotated[:, None]*energy[None]).reshape(20, 3, -1)
            z = layer[k].flatten(1).T.cpu().numpy().astype(np.float64)
            jj, mm = tokens.grid_moments(z, w, ew)
            avg += mm/4
            joint += jj/4
            sw = regions.cell_mass(np.rot90(masks, k, (-2, -1))).reshape(1, 3, -1)
            weights = torch.as_tensor(sw, device=layer.device, dtype=torch.float64)
            packed = attention(responses[k:k+1], weights, tau[p])[0].cpu().numpy()
            packed[~flags] = 0
            summary['summary1'][p] += packed/4
        raw['layer1'][p], raw['joint_raw'][p] = avg, joint
    return raw, summary


def local_features(local, base):
    z = base-base.mean(1, keepdims=True)
    rival = np.stack([np.delete(z, c, axis=1).max(1) for c in range(4)], 1)
    return np.concatenate((local.reshape(len(base), 4, 12), np.stack((z, rival, z-rival), -1)), -1)


def gate_features(query, bank, refs, base, dv, dh):
    qvalid, rvalid = query['valid'], bank['valid'][refs]
    valid = qvalid[:, None, None] & rvalid
    has = valid.any(2)
    q, s = query['x'][:, None, None], bank['x'][refs]
    mean, std = pair_stats((q*s).sum(-1), valid)
    agreement = (s[:, :, 0]*s[:, :, 1]).sum(-1)*valid.all(2)
    spatial = np.stack((mean, std, agreement, valid.mean(2),
                        np.broadcast_to(qvalid[:, None], has.shape)), -1).reshape(len(base), 4, 30)
    bv = np.broadcast_to(valid[..., None], valid.shape+(3,))
    cm, cs = pair_stats((query['cov'][:, None, None]*bank['cov'][refs]).sum(-1), bv)
    em, es = pair_stats(bank['energy'][refs], bv)
    ed, _ = pair_stats(abs(query['energy'][:, None, None]-bank['energy'][refs]), bv)
    cover, _ = pair_stats(bank['coverage'][refs], bv)
    qe, qc = (np.broadcast_to(query[k][:, None], em.shape) for k in ('energy', 'coverage'))
    freq = np.stack((qe, em, ed, es, cm, cs, qc, cover), -1)
    freq *= has[..., None, None]
    available = qvalid.any(-1).astype(np.float32)
    values = [base-base.mean(1, keepdims=True), dv, dh]
    scores = (base, base+dv, base+dh)
    for score in scores:
        rival = np.stack([np.delete(score, c, axis=1).max(1) for c in range(4)], 1)
        values.append(score-rival)
    values.extend(softmax(s, axis=1) for s in scores)
    values.extend(np.broadcast_to(available[:, p, None], base.shape) for p in range(2))
    raw = np.concatenate((np.stack(values, -1), spatial, freq.reshape(len(base), 4, 144)), -1)
    return raw.astype(np.float32), available


class AlignedM3(M3):
    """Reuse class-independent matching and checkpoint loaders; four-class IO."""
    def __init__(self, assets, device='cpu'):
        self.root = Path(assets)/'m3'
        self.device = torch.device(device)
        self.projection = load_npz(self.root/'projection.npz')
        self.bank = load_npz(self.root/'bank.npz')
        self.covariance_scaler = load_npz(self.root/'covariance_scaler.npz')
        if self.bank['mass'].shape != (32, 2, 20):
            raise ValueError('Aligned M3 requires 4 classes x 8 source references.')
        self.bank_body = tokens.body_descriptor(self.bank['x'], self.bank['mass']).reshape(4, 8, 64)
        self.bank_regions = regional_descriptors(self.bank, self.covariance_scaler)
        self.models = {}

    def retrieve(self, features, query_mmsi=None):
        body = tokens.body_descriptor(features['x'], features['mass'])
        cosine = np.einsum('nd,ckd->nck', body, self.bank_body)
        if query_mmsi is not None:
            same = np.asarray(query_mmsi, dtype=str).reshape(-1, 1, 1) == self.bank['mmsi'].reshape(1, 4, 8)
            cosine[same] = -np.inf
        selected = np.argsort(-cosine, axis=-1)[..., :2]
        return np.arange(4)[None, :, None]*8+selected

    @torch.inference_mode()
    def forward(self, features, base, seed=17, query_mmsi=None):
        single = features['mass'].ndim == 2
        if single:
            features = {k: v[None] for k, v in features.items()}
        base = np.asarray(base, np.float32).reshape(-1, 4)
        if len(base) != len(features['mass']):
            raise ValueError('Score and local feature batches differ.')
        model = self._load(seed)
        refs = self.retrieve(features, query_mmsi)
        local = self.match(features, refs, model)
        x = self._tensor(standardize(local_features(local, base), model['ck']['readout_scaler']))
        dv, dh = (model[k](x).cpu().numpy() for k in ('vv', 'vh'))
        raw, available = gate_features(regional_descriptors(features, self.covariance_scaler),
                                       self.bank_regions, refs, base, dv, dh)
        gx = self._tensor(standardize(raw, model['ck']['gate_scaler']))
        weights, alpha = model['gate'](gx, self._tensor(available))
        weights, alpha = weights.cpu().numpy(), alpha.cpu().numpy()
        scores = base+weights[:, :1]*dv+weights[:, 1:]*dh
        details = dict(refs=refs, reference_ids=self.bank['source_index'][refs], local=local,
                       delta_vv=dv, delta_vh=dh, gate_raw=raw, available=available,
                       polar_weights=weights, alpha=alpha)
        if single:
            return scores[0], {k: v[0] for k, v in details.items()}
        return scores, details

    __call__ = forward


class AlignedClassifier:
    """assets is one source directory, e.g. assets_aligned/oss1.

    extract returns raw[1538], aux[104], local{six projected arrays}, validity
    and placement. score also accepts batches; predict matches ShipClassifier.
    Optional record.mmsi excludes the same ship during source replay.
    """
    def __init__(self, assets, device='cpu'):
        self.assets = Path(assets)
        self.device = torch.device(device)
        self.manifest = json.loads((self.assets/'manifest.json').read_text(encoding='utf-8'))
        if not self.manifest.get('complete') or self.manifest.get('class_names') != list(CLASSES):
            raise ValueError('Expected a complete accepted four-class asset directory.')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.backbone = Backbone(self.assets/'backbone/resnet50.pth', device)
        self.readers = load_npz(self.assets/'upstream/models/reader_layer1.npz')['W'].reshape(4, 2, 256).transpose(1, 0, 2).copy()
        self.tau = json.loads((self.assets/'upstream/scales.json').read_text(encoding='utf-8'))['tau']['layer1']
        self.raw_scaler = load_npz(self.assets/'readout/raw_scaler.npz')
        self.summary_scaler = load_npz(self.assets/'readout/summary_scaler.npz')
        self.m3 = AlignedM3(self.assets, device)
        self.heads, self.selected_steps = {}, {}

    @torch.inference_mode()
    def extract(self, record, directory=Path('.')):
        prepared = prepare(record, directory)
        maps = self.backbone(prepared)
        raw, summary = extract_native(prepared, maps, self.readers, self.tau)
        batch = {k: v[None] for k, v in raw.items()}
        sums = {k: v[None] for k, v in summary.items()}
        output = dict(raw=raw_features(batch, sums['valid'])[0], aux=summary_features(sums)[0],
                      local=project_raw(raw, self.m3.projection), valid=summary['valid'],
                      placement=[v['placement'] if v else None for v in prepared])
        if record.get('mmsi') is not None:
            output['mmsi'] = str(record['mmsi'])
        return output

    @torch.inference_mode()
    def score(self, features, seed=17):
        if seed not in self.manifest['seeds']:
            raise ValueError(f'Exported seeds are {self.manifest["seeds"]}.')
        if seed not in self.heads:
            ck = torch.load(self.assets/f'readout/raw_summary_seed{seed}.pt', map_location='cpu', weights_only=False)
            head = nn.Sequential(nn.Linear(1642, 128), nn.GELU(), nn.Linear(128, 4))
            head.load_state_dict(ck['state'])
            self.heads[seed] = head.to(self.device).eval().requires_grad_(False)
            self.selected_steps[seed] = ck['step']
        single = np.asarray(features['raw']).ndim == 1
        raw = np.asarray(features['raw'], np.float32).reshape(-1, 1538)
        aux = np.asarray(features['aux'], np.float32).reshape(-1, 104)
        x = np.concatenate((transform(raw, self.raw_scaler), transform(aux, self.summary_scaler)), -1)
        base = self.heads[seed](torch.as_tensor(x, device=self.device)).cpu().numpy()
        scores, matching = self.m3(features['local'], base[0] if single else base, seed, features.get('mmsi'))
        return scores, dict(base=base[0] if single else base, matching=matching,
                            selected_steps=dict(raw_summary=self.selected_steps[seed],
                                                **self.manifest['selections'][str(seed)]))

    def predict(self, record, directory=Path('.'), seeds=(17,), trace=False):
        features = self.extract(record, directory)
        outputs = []
        for seed in seeds:
            scores, details = self.score(features, seed)
            pred = int(scores.argmax())
            item = dict(seed=seed, class_id=pred, class_name=CLASSES[pred],
                        class_scores=scores.tolist(), softmax_scores=softmax(scores).tolist(),
                        polar_weights=details['matching']['polar_weights'].tolist(),
                        selected_steps=details['selected_steps'])
            if trace:
                item['trace'] = to_json(details)
            outputs.append(item)
        return dict(id=record.get('id'), predictions=outputs, class_order=list(CLASSES),
                    target_evidence_available=features['valid'].tolist(), placement=features['placement'],
                    softmax_is_calibrated_probability=False)


def to_json(value):
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
