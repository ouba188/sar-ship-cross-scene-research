"""Localize original-runtime differences before training expanded tasks."""

import run_coverage as c
import sys
import sklearn
from budget_fusion_source_base import _features
from prepare_learned_frequency import replace_r50

tag = sys.argv[1]
dest = c.OUT/'numeric_probe'
dest.mkdir(exist_ok=True)
source_path = c.OUT/'training/self24ports/Los Angeles/source.npz'
source = c.archive(source_path)
old = c.archive(c.STUDY/'support_budget_fusion_select_20261012/base_cache/Los Angeles/base.npz')
records = {}
for key, ref in [('base_phi', 'phi'), ('mean', 'mean'), ('conditions', 'conditions'), ('base_prior', 'prior_oof')]:
    a, b = source[key], old[ref][:len(source['rows'])]
    records[key] = dict(max_abs=float(abs(a-b).max()),
        relative_frobenius=float(c.np.linalg.norm(a-b)/c.np.linalg.norm(b)),
        argmax_changes=int((a.argmax(1) != b.argmax(1)).sum()))
np = c.np
with c.threadpool_limits(limits=2):
    f = c.Features(c.TASKS/'self24ports/Los Angeles')
    rows = source['rows']
    raw = np.asarray(np.load(c.ROOT/'c4_base_trial_20261009/pooled_c4.npy', mmap_mode='r')[rows], float)
    pca = f.geometry['model']['pca']
    lib = pca.transform(raw)
    centered64 = (raw-np.asarray(pca.mean_, float))@np.asarray(pca.components_, float).T
    offset32 = raw@pca.components_.T-(pca.mean_[None]@pca.components_.T)
    records['pca'] = dict(mean_dtype=str(pca.mean_.dtype), components_dtype=str(pca.components_.dtype),
        centered64_error=float(abs(lib-centered64).max()), offset32_error=float(abs(lib-offset32).max()))
    np.savez_compressed(dest/(tag+'.npz'), library_pca=lib, centered64=centered64, offset32=offset32,
                        **source)
c.write(dest/(tag+'.json'), dict(runtime=sys.executable, numpy=np.__version__, sklearn=sklearn.__version__,
                               comparisons=records))
print(records, flush=True)
