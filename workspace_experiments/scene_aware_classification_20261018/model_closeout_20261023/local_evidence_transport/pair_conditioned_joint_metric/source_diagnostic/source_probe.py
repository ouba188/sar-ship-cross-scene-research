"""Source-only frozen joint-descriptor diagnostic; no training or GPU use."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BATCH = 16


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_factory(factory):
    def wrapped(*args, **kwargs):
        if str(kwargs.get('device', '')).startswith('cuda'):
            kwargs['device'] = 'cpu'
        return factory(*args, **kwargs)
    return wrapped


def ratio(a, b):
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 1e-8)


def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    train = load('source_joint_train', ROOT/'spatial_frequency_joint_pooling/train.py')
    # The existing Reader hardcodes CUDA. Redirect allocation only, retaining its
    # actual projection, source scaler, reference mapping, and frozen head replay.
    with patch.object(torch, 'tensor', cpu_factory(torch.tensor)), \
         patch.object(torch, 'arange', cpu_factory(torch.arange)), \
         patch.object(torch.nn.Module, 'cuda', lambda self, device=None: self.cpu()):
        reader = train.Reader('joint', 29)
    table = reader.table
    source = np.flatnonzero(table.role.isin(['utility_train', 'selection']))
    f = table.iloc[source].reset_index(drop=True)
    assert f.role.value_counts().to_dict() == {'utility_train': 644, 'selection': 612}
    assert reader.d.shape[-1] == 96 and reader.x.device.type == 'cpu'
    cache = np.load(ROOT/'spatial_frequency_joint_pooling/results/joint_29.npz')
    np.testing.assert_array_equal(cache['ids'], reader.ids)
    old = reader.strong[source]
    new = cache['score'][source]
    y = f.class_id.to_numpy(int)
    rival = np.where(np.arange(8)[None] == y[:, None], -np.inf, old).argmax(1)
    row = np.arange(len(f))
    old_margin = old[row, y] - np.where(np.arange(8)[None] == y[:, None], -np.inf, old).max(1)
    new_margin = new[row, y] - np.where(np.arange(8)[None] == y[:, None], -np.inf, new).max(1)
    change = new_margin - old_margin
    fixed_change = (new[row, y]-new[row, rival]) - (old[row, y]-old[row, rival])
    bc, ac = old.argmax(1) == y, new.argmax(1) == y
    status = np.where(bc & ac, 'CC', np.where(~bc & ~ac, 'WW', np.where(ac, 'rescue', 'harm')))
    chosen = {}
    for kind in ('rescue', 'harm'):
        candidates = np.flatnonzero(f.role.eq('selection').to_numpy() & (status == kind))
        assert len(candidates), kind
        chosen[kind] = int(candidates[np.argmax(np.abs(change[candidates]))])
    # B, true/rival, two actual references, VV/VH, body/side/end.
    numerator = np.zeros((len(f), 2, 2, 2, 3), np.float32)
    mass = np.zeros_like(numerator)
    replay = 0.
    with torch.inference_mode():
        for first in range(0, len(source), BATCH):
            last = min(first+BATCH, len(source))
            ix = torch.as_tensor(source[first:last])
            classes = torch.as_tensor(np.stack([y[first:last], rival[first:last]], 1))
            refs = reader.refs[ix[:, None], classes]
            q, s = reader.x[ix], reader.x[refs]
            d, e = reader.d[ix], reader.d[refs]
            a, b = reader.mass[ix], reader.mass[refs]
            for region, slots in enumerate(train.old.REGIONS):
                sim = torch.einsum('bpid,bckpjd->bckpij', q[:, :, slots], s[..., slots, :])
                qa, sa = a[:, None, None, :, slots], b[..., slots]
                t, _, _, _ = train.old.sinkhorn(sim, qa, sa)
                t = t * ((qa > 0)[..., :, None] & (sa > 0)[..., None, :])
                dq, ds = d[:, :, slots], e[..., slots, :]
                dist = (dq.square().mean(-1)[:, None, None, :, :, None]
                        + ds.square().mean(-1)[..., None, :]
                        - 2*torch.einsum('bpid,bckpjd->bckpij', dq, ds)/96).clamp_min(0)
                numerator[first:last, ..., region] = (t*dist).sum((-1, -2)).numpy()
                mass[first:last, ..., region] = t.sum((-1, -2)).numpy()
                replayed = train.old.pair_score(t, sim, qa, sa).mean(2)
                expected = reader.oldlocal[ix[:, None], classes][:, :, :, region]
                replay = max(replay, float((replayed-expected).abs().max()))
            if first % 256 == 0:
                print('SOURCE', last, '/', len(source), 'CPU', round(time.perf_counter()-start, 1), flush=True)
    # Detect an incorrect old-OT/ref/region reconstruction before interpreting it.
    assert replay < 2e-5, replay
    assert np.isfinite(numerator).all() and np.isfinite(mass).all()
    assert (mass.sum((2, 3, 4)) > 1e-8).all()
    totals = numerator.sum((2, 3, 4))
    masses = mass.sum((2, 3, 4))
    discrepancy = ratio(totals, masses)
    gap = discrepancy[:, 1]-discrepancy[:, 0]
    summaries = []

    def summarize(mask, role, group, region='all', pol='all', num=totals, den=masses):
        ds = ratio(num[mask], den[mask])
        good = np.isfinite(ds).all(1)
        ds = ds[good]
        raw = num[mask][good]
        mg = ds[:, 1]-ds[:, 0]
        summaries.append(dict(role=role, group=group, region=region, polarization=pol,
            n=int(mask.sum()), valid_n=int(good.sum()),
            true_closer_n=int((mg > 0).sum()), true_closer_pct=float(100*(mg > 0).mean()) if len(mg) else None,
            true_mean=float(ds[:, 0].mean()) if len(mg) else None,
            rival_mean=float(ds[:, 1].mean()) if len(mg) else None,
            gap_mean=float(mg.mean()) if len(mg) else None,
            gap_median=float(np.median(mg)) if len(mg) else None,
            raw_true_closer_pct=float(100*(raw[:, 1] > raw[:, 0]).mean()) if len(mg) else None,
            true_mass_mean=float(den[mask][good, 0].mean()) if len(mg) else None,
            rival_mass_mean=float(den[mask][good, 1].mean()) if len(mg) else None))

    for role in ('utility_train', 'selection', 'source_combined'):
        mask = np.ones(len(f), bool) if role == 'source_combined' else f.role.eq(role).to_numpy()
        summarize(mask, role, 'all')
        for kind in ('rescue', 'harm', 'CC', 'WW'):
            summarize(mask & (status == kind), role, kind)
        for cls in range(8):
            summarize(mask & (y == cls), role, 'class_'+str(cls))
        for p, pol in enumerate(('VV', 'VH')):
            for r, name in enumerate(('body', 'side', 'end')):
                summarize(mask, role, 'all', name, pol,
                          numerator[:, :, :, p, r].sum(2), mass[:, :, :, p, r].sum(2))
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE/'source_summary.csv', index=False)

    geo = load('source_native_geometry', ROOT.parents[1]/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    snapshots = {int(s['obs_id']): s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    records, case_rows = [], []
    for kind, j in chosen.items():
        idx = int(source[j])
        rr = reader.refs[idx, [int(y[j]), int(rival[j])]].numpy().ravel().tolist()
        indices = [idx, *rr]
        obs = int(f.iloc[j].obs_id)
        panel = Image.new('RGB', (1120, 610), 'white')
        draw = ImageDraw.Draw(panel)
        draw.text((12, 8), f'{kind} | selection | seed29 | obs {obs} | label {y[j]} | strong {old[j].argmax()} -> jointC {new[j].argmax()}', fill='black', font=font)
        draw.text((12, 30), f'Top1 margin {old_margin[j]:+.5f} -> {new_margin[j]:+.5f}; fixed old rival {rival[j]}; Dtrue {discrepancy[j,0]:.5f}, Drival {discrepancy[j,1]:.5f}', fill='black', font=font)
        provenance = []
        for p, pol in enumerate(('VV', 'VH')):
            for k, ri in enumerate(indices):
                pix, footprint, geometry, place = geo.native(table, snapshots, ri, p)
                x0, y0 = 12+222*k, 65+250*p
                label = 'QUERY' if k == 0 else ('TRUE' if k < 3 else 'OLD RIVAL')
                rec = table.iloc[ri]
                draw.text((x0, y0), f'{pol} {label} {int(rec.obs_id)}', fill='black', font=font)
                draw.text((x0, y0+19), str(rec.class_name), fill='black', font=font)
                draw.text((x0, y0+38), f'{rec.port} | active={int(reader.mass[ri,p].sum()>0)}', fill='black', font=font)
                panel.paste(Image.fromarray(np.asarray(pix, dtype=np.uint8)).convert('RGB'), (x0+20, y0+60))
                provenance.append(dict(obs_id=int(rec.obs_id), table_index=int(reader.ids[ri]), role=str(rec.role),
                    polarization=pol, reference_relation=label, path=str(rec['image_'+pol.lower()]),
                    native_hw=[snapshots[int(rec.obs_id)]['images'][pol.lower()]['h'], snapshots[int(rec.obs_id)]['images'][pol.lower()]['w']],
                    placement=place, active=bool(reader.mass[ri,p].sum()>0)))
        draw.text((12, 577), '160px native crop/pad at 1:1; original PNG gray 0..255; VV/VH separate. Labels do not identify physical causes.', fill='black', font=font)
        filename = f'{kind}_{obs}.png'
        panel.save(HERE/filename)
        record = dict(case=kind, obs_id=obs, table_index=int(reader.ids[idx]), seed=29, role='selection',
            true_class=int(y[j]), old_rival=int(rival[j]), old_pred=int(old[j].argmax()), new_pred=int(new[j].argmax()),
            old_margin=float(old_margin[j]), new_margin=float(new_margin[j]), margin_change=float(change[j]),
            fixed_rival_change=float(fixed_change[j]), true_refs=table.iloc[rr[:2]].obs_id.astype(int).tolist(),
            rival_refs=table.iloc[rr[2:]].obs_id.astype(int).tolist(), true_discrepancy=float(discrepancy[j,0]),
            rival_discrepancy=float(discrepancy[j,1]), panel=filename, provenance=provenance)
        records.append(record)
        for c, relation in enumerate(('true', 'old_rival')):
            for k in range(2):
                for p, pol in enumerate(('VV', 'VH')):
                    for r, region in enumerate(('body', 'side', 'end')):
                        den = float(mass[j,c,k,p,r]); num = float(numerator[j,c,k,p,r])
                        case_rows.append(dict(case=kind, obs_id=obs, relation=relation,
                            reference_obs_id=int(table.iloc[rr[2*c+k]].obs_id), polarization=pol, region=region,
                            old_ot_mass=den, weighted_squared_discrepancy=num,
                            normalized_discrepancy=num/den if den>1e-8 else None))
    pd.DataFrame(case_rows).to_csv(HERE/'case_discrepancy.csv', index=False)
    receipt = dict(status='computed_pending_visual_review', seed=29, device='cpu', batch=BATCH,
        training_steps=0, newly_extracted_features=0, evaluation_queries_analyzed=0,
        source_queries=len(f), role_counts=f.role.value_counts().to_dict(),
        source_ships=int(f.mmsi.nunique()), source_passes=int(f.pass_group.nunique()),
        descriptor='Reader(joint,29).d; utility-standardized 96 coordinates; unmodified cached x and old OT',
        discrepancy='sum_K,P,R,i,j T_old * mean_96((d_q-d_ref)^2) / sum_K,P,R,i,j T_old; zero-mass excluded',
        raw_comparison='same numerator without real-match-mass normalization; mass-entangled sensitivity only',
        case_selection='selection only; maximum absolute change in true-versus-current-max-wrong margin within rescue/harm; lowest table order breaks ties',
        old_local_replay_max_abs=replay, old_head_replay_max_abs=reader.oldhead_replay,
        evaluation_boundary='Existing Reader loads shared caches including evaluation rows, but all diagnostic queries, labels, scores and case choice are subset to source; refs checked source-only.',
        interpretation='old OT frozen; no new conditional metric tested; normalized weighted cosine removes uniform scale, so scalar descriptor closeness is not that model',
        cases=records, summary=summaries, seconds=time.perf_counter()-start)
    assert all(p['role'] != 'evaluation' for c in records for p in c['provenance'])
    used_refs = reader.refs[source].numpy().ravel()
    assert not table.iloc[used_refs].role.eq('evaluation').any()
    receipt['actual_reference_roles'] = table.iloc[np.unique(used_refs)].role.value_counts().to_dict()
    (HERE/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x,np.generic) else str(x)), encoding='utf-8')
    print(summary.query("group in ['all','rescue','harm'] and region=='all'").to_string(index=False), flush=True)
    print(json.dumps([{k:v for k,v in c.items() if k!='provenance'} for c in records], ensure_ascii=False, indent=2), flush=True)
    print('COMPLETE CPU seconds', receipt['seconds'], 'replay', replay, flush=True)


if __name__ == '__main__':
    main()
