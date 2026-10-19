"""Test whether the trained directional residual is actually used, without refit."""
import numpy as np
import pandas as pd
import sys
import torch
from threadpoolctl import threadpool_limits
from run_orientation import HERE, Study, OrientationReader, prior_run


def main(arm='C'):
    torch.set_num_threads(2)
    study = Study()
    contexts = [(t, d, study.context(t['source'], s, t['query'])) for t in study.tasks['tasks'] for d, s in enumerate(t['supports'])]
    records, weights, local_stats = [], [], []
    all_queries = sorted(set(i for t in study.tasks['tasks'] for i in t['query']))
    for seed in [0, 1, 2]:
        reader = OrientationReader(arm).cuda().eval()
        reader.load_state_dict(torch.load(HERE / f'{arm}_seed{seed}.pt', weights_only=True))
        spatial = reader.spatial_weight.detach().clone()
        for reference in range(2):
            w = spatial[reference].cpu().numpy()
            weights.append(dict(seed=seed, reference=reference, center_norm=float(np.linalg.norm(w[0])),
                                along_norm=float(np.linalg.norm(w[1])), cross_norm=float(np.linalg.norm(w[2])),
                                anisotropy_norm=float(np.linalg.norm(w[1]-w[2]))))
        with torch.no_grad():
            for split, ids in [('fit', study.fit), ('target_queries', all_queries)]:
                for begin in range(0, len(ids), 8):
                    chunk = ids[begin:begin+8]
                    maps, masks, pos = [x[chunk] for x in study.inputs]
                    geom, available = study.geometry_tensor[chunk], study.availability[chunk]
                    reader.arm = arm
                    raw, _ = reader(maps, masks, pos, geom, available)
                    projected = reader.content(maps.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
                    local = reader.local_response(projected, masks, geom)
                    reader.spatial_weight.zero_()
                    raw0, _ = reader(maps, masks, pos, geom, available)
                    local0 = reader.local_response(projected, masks, geom)
                    reader.spatial_weight.copy_(spatial)
                    reader.arm = 'A'
                    image_raw, _ = reader(maps, masks, pos, geom, available)
                    image_local = reader.local_response(projected, masks, geom)
                    reader.arm = arm
                    for j, i in enumerate(chunk):
                        # Valid center area weights exclude responses never pooled.
                        area = masks[j][None, :, None]
                        relative = float(((local[j]-local0[j]).abs()*area).sum()/(local0[j].abs()*area).sum().clamp_min(1e-8))
                        local_stats.append(dict(seed=seed, split=split, row=int(i),
                            local_residual_relative_abs=relative,
                            reference_local_relative_l2=float(torch.linalg.vector_norm((local[j]-image_local[j])*area.sqrt())/torch.linalg.vector_norm(local[j]*area.sqrt()).clamp_min(1e-8)),
                            reference_raw_relative_l2=float(torch.linalg.vector_norm(raw[j]-image_raw[j])/torch.linalg.vector_norm(raw[j]).clamp_min(1e-8)),
                            raw_cosine_distance_to_zero=float(1-torch.nn.functional.cosine_similarity(raw[j].flatten(), raw0[j].flatten(), dim=0)),
                            raw_cosine_distance_to_image=float(1-torch.nn.functional.cosine_similarity(raw[j].flatten(), image_raw[j].flatten(), dim=0))))
            for task, draw, ct in contexts:
                original, original_k = study.predict(reader, ct)[:2]
                for mode in ['zero_spatial', 'image_reference']:
                    if mode == 'zero_spatial':
                        reader.spatial_weight.zero_()
                    else:
                        reader.arm = 'A'
                    met, recs = prior_run.collect(study, reader, f'{arm}_{mode}', seed, task, draw, ct)
                    newprob = torch.tensor([[r[f'p{k}'] for k in range(7)] for r in recs], device='cuda')
                    met['flips_vs_trained'] = int((newprob.argmax(-1) != original.argmax(-1)).sum())
                    met['mean_abs_probability_delta'] = float((newprob-original).abs().mean())
                    newk = study.predict(reader, ct)[1]
                    met['new_kernel_mean_abs_delta'] = float((newk-original_k).abs().mean()/.0625)
                    records.append(met)
                    reader.spatial_weight.copy_(spatial)
                    reader.arm = arm
    prefix = '' if arm == 'C' else arm+'_'
    pd.DataFrame(records).to_csv(HERE / f'{prefix}operator_interventions.csv', index=False)
    pd.DataFrame(weights).to_csv(HERE / f'{prefix}spatial_weights.csv', index=False)
    pd.DataFrame(local_stats).to_csv(HERE / f'{prefix}local_response_diagnostics.csv', index=False)
    print(pd.DataFrame(records).groupby(['arm', 'line', 'budget'])[['oa', 'ba', 'brier', 'flips_vs_trained', 'mean_abs_probability_delta']].mean().to_string())
    print(pd.DataFrame(local_stats).groupby('split').mean(numeric_only=True).to_string())


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main(sys.argv[1] if len(sys.argv)>1 else 'C')
