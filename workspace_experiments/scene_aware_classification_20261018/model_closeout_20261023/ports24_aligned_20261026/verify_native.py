"""Check the eight-class experiment against fresh native image forwards."""
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
from prepare_bundle import HERE, PACKAGE, LOCAL, load, write
from analyze import module
from run import apply_readouts, SEEDS
from sar_m123.backbone import Backbone
from sar_m123.m12 import SummaryReader
from sar_m123.m3 import M3, PairMetric, regional_descriptors
from sar_m123.aligned import raw_features, summary_features
from sar_m123 import tokens, train_ops as op, train_aligned as ta


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    table = pd.read_csv(HERE/'target_truth.csv', dtype={'mmsi': str})
    source = load(HERE/'bundle/features.npz')
    expected = load(HERE/'bundle/external.npz')
    # Class coverage is for checking inference, not for changing or selecting models.
    ids = [int(np.flatnonzero(table.class_id.eq(c))[0]) for c in range(8)]
    missing = np.flatnonzero(expected['mass'][:, 1].sum(1) == 0)
    if len(missing):
        ids.append(int(missing[0]))
    ids = sorted(set(ids))
    ex = module('ports24_native_extract_check', LOCAL/'frozen_joint_coverage/extract.py')
    snapshots = {int(v['obs_id']): v for v in json.loads((ex.joint.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    backbone = Backbone(PACKAGE/'assets/backbone/resnet50.pth', 'cuda')
    reader, m3 = SummaryReader(PACKAGE/'assets'), M3(PACKAGE/'assets', 'cuda')
    raw, aux, local, checks = [], [], [], []
    for i in ids:
        views = []
        for p, pol in enumerate(('vv', 'vh')):
            if expected['mass'][i, p].sum() == 0:
                views.append(None)
                continue
            pix, fp, g, placement = ex.joint.geo.native(table, snapshots, i, p)
            with Image.open(table.iloc[i]['image_'+pol]) as im:
                original = np.asarray(im.convert('L'))
            views.append(dict(pixels=pix, footprint=fp, geometry=g, raw=original,
                              placement=placement, valid_target=True))
        maps = backbone(views)
        summaries = reader.summarize(views, maps)
        lf = m3.extract(views, maps)
        lf['mean'] = (lf['layer1']-m3.projection['mean'])@m3.projection['components'].T
        rr = raw_features(dict(layer1=lf['layer1'][None], mass=lf['mass'][None]), summaries['valid'][None])[0]
        aa = summary_features({k: v[None] for k, v in summaries.items()})[0]
        errors = {}
        for key, fresh in dict(raw=rr, aux=aa, **lf).items():
            if key not in expected:
                continue
            np.testing.assert_allclose(fresh, expected[key][i], atol=2e-4, rtol=2e-5)
            errors[key] = float(abs(fresh-expected[key][i]).max())
        checks.append(dict(obs_id=int(table.iloc[i].obs_id), errors=errors))
        raw.append(rr)
        aux.append(aa)
        local.append(lf)
    raw, aux = np.stack(raw), np.stack(aux)
    bank = source['bank'].ravel()
    features = {k: np.concatenate((source[k][bank], np.stack([f[k] for f in local]))) for k in ta.FEATURES}
    lf = {k: np.stack([f[k] for f in local]) for k in ta.FEATURES}
    refs_query = m3.retrieve(lf)
    np.testing.assert_array_equal(bank[refs_query], expected['refs'][ids])
    refs = np.concatenate((np.zeros((64, 8, 2), int), refs_query))
    covariance = {k: source['covariance_'+k] for k in ('mu', 'scale', 'active')}
    desc = regional_descriptors(features, covariance)
    report = []
    for seed in SEEDS:
        folder = HERE/'training'/f'seed{seed}'
        sc, sa = load(folder/'raw_scaler.npz'), load(folder/'summary_scaler.npz')
        head = nn.Sequential(nn.Linear(1714, 128), nn.GELU(), nn.Linear(128, 8)).eval()
        head.load_state_dict(op.load_checkpoint(folder/'raw_summary.pt')['state'])
        base_query = head(torch.tensor(np.c_[ta.scale_input(raw, sc), ta.scale_input(aux, sa)])).numpy()
        base = np.concatenate((np.zeros((64, 8), np.float32), base_query))
        metric = PairMetric().cuda().eval()
        metric.load_state_dict(op.load_checkpoint(folder/'aligned_metric.pt')['state'])
        ms = op.load_checkpoint(HERE/'initial_assets/warm'/f'seed{seed}/metric_scaler.pt')
        matcher = op.Matcher(features, refs, ms, 'cuda')
        match = np.zeros((len(base), 8, 2, 3, 2), np.float32)
        match[64:] = matcher.predict(np.arange(64, len(base)), metric)
        fresh = apply_readouts(folder, base, match, desc, refs)
        saved = load(HERE/'results'/f'seed{seed}/both.npz')
        index = len(source['obs_id'])+np.asarray(ids)
        errors = {}
        for key in ('bare', 'frequency', 'weight_frequency', 'delta_vv', 'delta_vh'):
            np.testing.assert_allclose(fresh[key][64:], saved[key][index], atol=5e-4, rtol=2e-5)
            errors[key] = float(abs(fresh[key][64:]-saved[key][index]).max())
        np.testing.assert_array_equal(fresh['frequency'][64:].argmax(1), saved['frequency'][index].argmax(1))
        report.append(dict(seed=seed, errors=errors, predictions_identical=True))
    write(HERE/'native_replay.json', dict(chips=checks, configs=report, raw_images_forwarded=True,
        geometry_source='existing verified original detector geometry', resized=False,
        feature_atol=2e-4, score_atol=5e-4))
    print('NATIVE_REPLAY', len(ids), report, flush=True)


if __name__ == '__main__':
    main()
