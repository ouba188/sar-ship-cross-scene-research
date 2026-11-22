"""Reproducible native OSS2 Q review and diagnostic intervention; never trains.

Review inputs and generated pixels stay local. No inference of valid support
from zero intensity and no automatic controlled-view qualification.
"""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'oss2_q_pilot'
POLS = ('vv', 'vh')


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def build_mask(shape, rectangles, excluded):
    q = np.zeros(shape, bool)
    for x0, y0, x1, y1 in rectangles:
        assert 0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]
        q[y0:y1, x0:x1] = True
    # Pixel centers are integers; exclude even boundary contact of whole pixels.
    for y, x in zip(*np.nonzero(q)):
        if excluded.intersects(box(x-.5, y-.5, x+.5, y+.5)):
            q[y, x] = False
    return q


def contour(ax, geom, color, width=1):
    for g in getattr(geom, 'geoms', [geom]):
        if g.geom_type == 'Polygon':
            ax.plot(*g.exterior.xy, color=color, linewidth=width)


def main():
    OUT.mkdir(exist_ok=True)
    selected = json.loads((ROOT/'q_review_selection_local.json').read_text(encoding='utf8'))
    specs = json.loads((OUT/'review_regions_local.json').read_text(encoding='utf8'))
    evidence = {r['sample_id']: r for r in csv_rows(ROOT/'oss2_alignment/row_evidence_local.csv')}
    coast = {r['sample_id']: r for r in csv_rows(ROOT/'q_review_water_context_local.csv')}
    review_path = OUT/'visual_review_local.json'
    reviewed = json.loads(review_path.read_text(encoding='utf8')) if review_path.exists() else {}
    cache, audit = {}, []
    for r in selected:
        sid = r['id']; spec = specs[sid]; e = evidence[sid]
        assert e['same_pixel_index_derivation_verified'] == 'True'
        arrays = np.stack([np.asarray(Image.open(r[p]['image'])) for p in POLS])
        raw = tifffile.imread(e['raw_path'])
        assert raw.shape[:2] == arrays.shape[1:] and np.isfinite(raw).all()
        target_polygons = [Polygon(r[p]['detections'][r[p]['target_index']]['corners']) for p in POLS]
        target = unary_union(target_polygons)
        # Union of both-polarization detections, plus explicitly reviewed neighbors.
        detections = [Polygon(d['corners']) for p in POLS for d in r[p]['detections']]
        neighbors = [box(*b) for b in spec['neighbor_boxes_xyxy']]
        protection = unary_union([g.buffer(8) for g in detections + neighbors])
        q = build_mask(arrays.shape[1:], spec['q_rectangles_xyxy'], protection)
        assert q.any() and np.isfinite(raw[q]).all() and (raw[q] > 0).all()
        # Recheck the recorded quantizer on selected Q, not just source metadata.
        for p, channel in [('vv', 1), ('vh', 0)]:
            a, b = float(e[f'{p}_raw_to_uint8_slope']), float(e[f'{p}_raw_to_uint8_offset'])
            replay = np.clip(np.floor(a*raw[..., channel].astype(float)+b+.5), 0, 255).astype(np.uint8)
            assert np.array_equal(replay[q], arrays[POLS.index(p)][q])
        visual_ok = reviewed.get(sid, {}).get('q_mask_review_passed', False) is True
        water_ok = coast[sid]['coast_status'] == 'offshore'
        qualified = visual_ok and water_ok
        folder = OUT/sid; folder.mkdir(exist_ok=True)
        Image.fromarray(q.astype(np.uint8)*255).save(folder/'q_mask.png')
        for i, p in enumerate(POLS):
            Image.fromarray(arrays[i]).save(folder/f'original_{p}.png')
        cache[sid] = (r, arrays, q, target, protection, qualified)
        audit.append(dict(sample_id=sid, class_name=r['class_name'], native_hw=list(q.shape),
                          q_pixels=int(q.sum()), q_fraction=float(q.mean()),
                          same_grid_verified=True, raw_q_finite_positive=True,
                          raw_q_quantizer_replay_exact=True, zero_is_nodata_assumed=False,
                          q_visual_review_passed=visual_ok, water_context_passed=water_ok,
                          q_status='reviewed_local_background' if qualified else 'candidate_pending_evidence',
                          background_validated=qualified, valid_mask_status='native_physical_support_unknown',
                          leakage_controlled=False, training_qualified=False,
                          review_scope='selected interior pixels only; no full-chip water/georeference claim',
                          reviewer='codex_visual_review' if visual_ok else None,
                          background_mask=str(folder/'q_mask.png'),
                          reviewer_note=reviewed.get(sid, {}).get('note', 'not reviewed yet')))

    # First render is Q-only; after review, generate cross-ship diagnostic views.
    rng = np.random.default_rng(20261122)
    eligible = sorted(sid for sid, entry in cache.items() if entry[-1])
    interventions = []
    for sid, (r, arrays, q, target, protection, qualified) in cache.items():
        folder = OUT/sid
        altered = arrays.copy(); donor = None; sampled = np.empty(0, dtype=int)
        if qualified:
            candidates = [s for s in eligible if s != sid and cache[s][0]['mmsi'] != r['mmsi']]
            if candidates:
                donor = candidates[int(rng.integers(len(candidates)))]
                donor_arrays, donor_q = cache[donor][1:3]
                source_ids = np.flatnonzero(donor_q)
                sampled = rng.choice(source_ids, size=int(q.sum()), replace=True)
                for i in range(2):
                    altered[i, q] = donor_arrays[i].ravel()[sampled]
                np.savez_compressed(folder/'intervention_indices_local.npz',
                                    recipient_flat_indices=np.flatnonzero(q), donor_flat_indices=sampled)
        assert np.array_equal(altered[:, ~q], arrays[:, ~q])
        if donor is not None:
            for i in range(2):
                assert np.array_equal(altered[i, q], cache[donor][1][i].ravel()[sampled])
        diff = np.abs(altered.astype(np.int16)-arrays.astype(np.int16))
        for i, p in enumerate(POLS):
            Image.fromarray(altered[i]).save(folder/f'intervened_{p}.png')
            Image.fromarray(diff[i].astype(np.uint8)).save(folder/f'absolute_difference_{p}.png')
        interventions.append(dict(sample_id=sid, donor_id=donor, seed=20261122,
                                  changed_pixels_per_pol=[int(np.count_nonzero(x)) for x in diff],
                                  q_mean_abs_delta=[float(x[q].mean()) for x in diff],
                                  outside_q_exactly_unchanged=True, protected_target_exactly_unchanged=True,
                                  paired_donor_indices_verified=True if donor else None,
                                  donor_rule='uniform among other reviewed vessels; no label input',
                                  operation='shared VV/VH donor pixel sampling, no intensity rescaling',
                                  diagnostic_only=True, leakage_controlled=False, training_qualified=False))
        fig, axes = plt.subplots(2, 5, figsize=(16, 6.8), constrained_layout=True)
        for i, p in enumerate(POLS):
            values = [arrays[i], q, arrays[i], altered[i], diff[i]]
            titles = [f'{p.upper()} original [0,255]', 'Q mask (white=selected)',
                      'Q overlay + protection', 'Intervened [0,255]', '|difference| [0,32]']
            for j, (ax, a) in enumerate(zip(axes[i], values)):
                ax.imshow(a, cmap='gray' if j != 4 else 'magma', vmin=0,
                          vmax=1 if j == 1 else (32 if j == 4 else 255), interpolation='nearest')
                if j == 2:
                    overlay = np.zeros((*q.shape, 4)); overlay[q] = [0, 1, 0, .65]
                    ax.imshow(overlay, interpolation='nearest')
                    contour(ax, target, 'cyan'); contour(ax, protection, 'orange', .8)
                ax.set_title(titles[j], fontsize=10); ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlim(-.5, q.shape[1]-.5); ax.set_ylim(q.shape[0]-.5, -.5)
            axes[i, 0].set_ylabel(p.upper())
        state = 'reviewed local Q' if qualified else 'PENDING: not used as intervention donor/recipient'
        fig.suptitle(f"{sid} | {r['class_name']} | native {q.shape[1]} x {q.shape[0]}\n"
                     f"{state}; Q={q.sum()} px; donor={donor or 'none'}; controlled-view eligibility=FALSE", fontsize=12)
        fig.savefig(folder/'comparison_local.png', dpi=135); plt.close(fig)
    save_json(OUT/'q_qualification_local.json', audit)
    save_json(OUT/'interventions_local.json', interventions)
    save_json(OUT/'summary.json', dict(samples=len(audit), reviewed_q=sum(r['background_validated'] for r in audit),
                                     pending_q=sum(not r['background_validated'] for r in audit),
                                     diagnostic_pairs=sum(r['donor_id'] is not None for r in interventions),
                                     controlled_views_qualified=0, training_started=False,
                                     native_valid_masks_recovered=0,
                                     q_pixels=[r['q_pixels'] for r in audit]))
    print(json.dumps(json.loads((OUT/'summary.json').read_text()), indent=2))


if __name__ == '__main__':
    main()
