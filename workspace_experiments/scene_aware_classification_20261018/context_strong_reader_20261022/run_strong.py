"""Frozen deeper single-backbone readers, then matched context residuals."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
EXP = HERE.parent/'context_expansion_20261022'
sys.path.insert(0, str(EXP))
from train_expanded import Relation, inputs_for, SEEDS, weights, risk, metrics

KINDS = {'stem': ['stem'], 'layer3': ['layer3'], 'layer4': ['layer4'],
         'stem_layer3': ['stem', 'layer3'], 'stem_layer4': ['stem', 'layer4']}
ARMS = ['whole_unit', 'split_unit', 'split_role']


class View:
    def __init__(self, x, fit, kind):
        self.scale = StandardScaler().fit(x[fit])
        self.pca = PCA(n_components=32 if kind == 'stem' else 128,
                        svd_solver='full' if kind == 'stem' else 'randomized', random_state=17, iterated_power=4)
        self.pca.fit(self.scale.transform(x[fit]))
        z = self.pca.transform(self.scale.transform(x[fit]))
        self.post = StandardScaler().fit(z)

    def transform(self, x):
        return self.post.transform(self.pca.transform(self.scale.transform(x)))


def make_views(raw, fit, names):
    return {name: View(raw[name], fit, name).transform(raw[name]) for name in names}


def combine(z, names):
    if len(names) == 1:
        return z[names[0]]
    return np.concatenate([z[k]/np.sqrt(z[k].shape[1]) for k in names], 1)


def fit_head(z, y, ids, fit, alpha):
    return Ridge(alpha=alpha).fit(z[fit], np.eye(3)[y[fit]], sample_weight=weights(y[fit], ids[fit]))


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    t = pd.read_csv(EXP/'readout/targets.csv', dtype={'mmsi': str})
    raw0 = np.load(EXP/'readout/features.npz')
    deep = np.load(HERE/'deep_features.npz')
    assert np.array_equal(raw0['instance_ids'], deep['instance_ids'])
    raw = {'stem': raw0['target'], 'layer3': deep['layer3'], 'layer4': deep['layer4']}
    y, ids = t.class_id.to_numpy(), t.mmsi.to_numpy()
    alloc = pd.read_csv(EXP/'pass_assignment.csv').set_index('pass_group').group_bin
    bins = t.pass_group.map(alloc).to_numpy()
    candidates, selections, records, predictions, baseline_rows, curves = [], [], [], [], [], []
    (HERE/'models').mkdir(exist_ok=True)
    for fold in range(6):
        d = dict(np.load(EXP/f'fold{fold}_inputs.npz'))
        train, val, test = [d[k] for k in ['train', 'validation', 'test']]
        views = make_views(raw, train, raw.keys())
        choices = []
        for kind, names in KINDS.items():
            z = combine(views, names)
            best = None
            for alpha in [1., 10., 100.]:
                head = fit_head(z, y, ids, train, alpha)
                score = head.predict(z)*3
                vr = risk(y[val], score[val])
                if best is None or vr < best[0]:
                    best = vr, alpha, score, head
            vr, alpha, score, head = best
            choices.append((vr, kind, alpha, score, head))
            candidates.append(dict(fold=fold, kind=kind, alpha=alpha, dimensions=z.shape[1], val_nll=vr,
                                   **metrics(y[test], score[test])))
            for i in test:
                baseline_rows.append(dict(fold=fold, kind=kind, instance_id=t.iloc[i].instance_id, class_id=int(y[i]),
                    prediction=int(score[i].argmax()), **{f's{c}':float(score[i,c]) for c in range(3)}))
            if kind == 'stem':
                assert np.allclose(score, d['base'], atol=3e-4)
        vr, kind, alpha, base, head = min(choices, key=lambda v: v[0])
        selections.append(dict(fold=fold, kind=kind, alpha=alpha, val_nll=vr))
        oof = np.zeros_like(base)
        for b in sorted(set(bins[train])):
            held = train[bins[train] == b]
            fit = train[(bins[train] != b) & ~np.isin(ids[train], ids[held])]
            assert not set(ids[fit]) & set(ids[held])
            z = combine(make_views(raw, fit, KINDS[kind]), KINDS[kind])
            inner = fit_head(z, y, ids, fit, alpha)
            oof[held] = inner.predict(z[held])*3
        np.savez_compressed(HERE/f'fold{fold}_base.npz', base=base, oof=oof, train=train, validation=val, test=test,
                            selected_kind=kind, selected_alpha=alpha)
        weight = torch.tensor(weights(y[train], ids[train]), dtype=torch.float32, device='cuda')
        yy = torch.tensor(y, dtype=torch.long, device='cuda')
        oo = torch.tensor(oof, dtype=torch.float32, device='cuda')
        for seed in SEEDS:
            for arm in ARMS:
                inp = inputs_for(arm, d, 'cuda')
                torch.manual_seed(seed)
                model = Relation('split_unit' if arm == 'split_role' else arm).cuda()
                opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_risk, best_epoch = risk(y[val], base[val]), 0
                for epoch in range(1, 201):
                    opt.zero_grad()
                    s = oo[train]+model(*[x[train] for x in inp])
                    loss = (torch.nn.functional.cross_entropy(s, yy[train], reduction='none')*weight).mean()
                    loss.backward()
                    opt.step()
                    if epoch % 5 == 0:
                        with torch.no_grad():
                            score = base[val]+model(*[x[val] for x in inp]).cpu().numpy()
                        vr = risk(y[val], score)
                        curves.append(dict(fold=fold, arm=arm, seed=seed, epoch=epoch, loss=float(loss.detach()), val_nll=vr))
                        if vr < best_risk-1e-6:
                            best_risk, best_epoch = vr, epoch
                            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(best)
                with torch.no_grad():
                    delta = model(*inp).cpu().numpy()
                assert not delta[~d['m'].any(1)].any()
                score = base+delta
                torch.save(best, HERE/'models'/f'fold{fold}_{arm}_{seed}.pt')
                bp, pp = base[test].argmax(1), score[test].argmax(1)
                records.append(dict(fold=fold, arm=arm, seed=seed, epoch=best_epoch, base_kind=kind, val_nll=best_risk,
                    **metrics(y[test], score[test]), rescued=int(((bp != y[test]) & (pp == y[test])).sum()),
                    harmed=int(((bp == y[test]) & (pp != y[test])).sum())))
                for i in test:
                    predictions.append(dict(fold=fold, seed=seed, arm=arm, instance_id=t.iloc[i].instance_id,
                        class_id=int(y[i]), prediction=int(score[i].argmax()), base_prediction=int(base[i].argmax()),
                        neighbors=int(d['m'][i].sum()), **{f's{c}':float(score[i,c]) for c in range(3)},
                        **{f'b{c}':float(base[i,c]) for c in range(3)}))
                print('fold', fold, 'base', kind, 'seed', seed, arm, 'step', best_epoch, 'net', records[-1]['rescued']-records[-1]['harmed'], flush=True)
        for name, values in [('baseline_candidates', candidates), ('selections', selections), ('metrics', records),
                              ('predictions', predictions), ('baseline_predictions', baseline_rows), ('curves', curves)]:
            pd.DataFrame(values).to_csv(HERE/(name+'.csv'), index=False)
    p = pd.DataFrame(predictions)
    pooled = []
    for (arm, seed), q in p.groupby(['arm', 'seed']):
        pooled.append(dict(arm=arm, seed=seed, **metrics(q.class_id.to_numpy(), q[['s0','s1','s2']].to_numpy()),
                           net=int((q.prediction == q.class_id).sum()-(q.base_prediction == q.class_id).sum())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv', index=False)
    q = p[p.arm.eq(ARMS[0]) & p.seed.eq(SEEDS[0])]
    pd.DataFrame([metrics(q.class_id.to_numpy(), q[['b0','b1','b2']].to_numpy())]).to_csv(HERE/'baseline_pooled.csv', index=False)
    (HERE/'summary.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started, heads=len(records),
        targets=len(t), selection='source-validation balanced NLL only', parameters_context=2659), indent=2), encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
