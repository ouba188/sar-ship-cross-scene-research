"""Source-only pixel fine-tuning pilot with a matched same64 single-R50 readout."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from c4_condition_descriptors import SOURCE, ROOT, load_manifest, read_pair
from unified_head_core import score

OUT = ROOT/'completion_single_finetune_20261010'
CACHE = Path('D:/SAR_completion_cache_20261010')
TASK = ROOT/'learned_frequency_residual_20261009/tasks/oss2'
SEED = 20261010
EPOCHS = 8
BATCH = 64


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    source = load_manifest('self24ports')
    target = load_manifest('oss2')
    with np.load(TASK/'target.npz') as data:
        sr = data['source_rows']
        np.testing.assert_array_equal(data['ids'], target.sample_id.to_numpy(str))
    split = np.load(ROOT/'opensarship2_transfer_20261007/source_split.npz')
    np.testing.assert_array_equal(sr, split['source_rows'])
    assert not np.intersect1d(sr, split['excluded_source_rows']).size
    protocol = dict(dataset='oss2', scope='One external-dataset pilot, not 25-task completion',
        seed=SEED, epochs=EPOCHS, batch=BATCH, source_rows=len(sr), target_rows=len(target),
        source_rows_file=str(TASK/'target.npz'), initialization='ImageNet R50 V2, paired VV/VH',
        training='All R50 parameters and 8-class head, source labels only',
        optimizer='AdamW lr=0.0003 weight_decay=0.0001, cosine schedule',
        loss='Cross entropy with source inverse-square-root class frequency',
        augmentation='Random C4 rotation; no target images in training',
        checkpoint='Fixed final epoch, no target checkpoint or hyperparameter selection',
        evaluation='C4 pooled2048 -> source PCA128/scaling -> balanced ridge prior -> same64 residual ridge',
        residual='phi dot phi + 0.25 normalized-mean dot normalized-mean times original condition RBF; lambda8',
        controls=['matched frozen R50 core', 'fine-tuned R50 core', 'existing full dual and unified reference'],
        limitation='Single model is compact core, not a retraining of every full-model module; one network seed',
        support_draws=5, target_support_labels=64, source_prior_refit=True,
        extra_modules='No SSL, frequency adapter, stable space or incidence in either matched single arm',
        independent_confirmation=False)
    if (OUT/'PROTOCOL.json').exists():
        assert json.loads((OUT/'PROTOCOL.json').read_text()) == protocol
    else:
        write(OUT/'PROTOCOL.json', protocol)
    for name, frame, cols in [('source', source, ['image_relpath_vv', 'image_relpath_vh']),
                              ('oss2', target, ['chip_path_vv', 'chip_path_vh'])]:
        ids = frame.sample_id.to_numpy(str)
        done = CACHE/f'{name}_COMPLETE.json'
        if done.exists():
            np.testing.assert_array_equal(np.load(CACHE/f'{name}_ids.npy'), ids)
            continue
        pixels = np.lib.format.open_memmap(CACHE/f'{name}_inputs128.npy', mode='w+',
                                          dtype=np.uint8, shape=(len(frame), 2, 128, 128))
        paths = list(frame[cols].itertuples(index=False, name=None))
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for lo in range(0, len(frame), 256):
                hi = min(len(frame), lo+256)
                pixels[lo:hi] = np.stack([p[0] for p in pool.map(read_pair, paths[lo:hi])])
                if lo % 4096 == 0:
                    print(f'pixels {name} {hi}/{len(frame)} {time.perf_counter()-start:.1f}s', flush=True)
        pixels.flush()
        np.save(CACHE/f'{name}_ids.npy', ids)
        write(done, dict(rows=len(frame), shape=list(pixels.shape), seconds=time.perf_counter()-start))


def model(pretrained=True):
    if pretrained:
        path = Path(torch.hub.get_dir())/'checkpoints'/Path(ResNet50_Weights.IMAGENET1K_V2.url).name
        if not path.exists():
            raise FileNotFoundError(path)
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    old = net.conv1
    net.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
    if pretrained:
        with torch.no_grad():
            net.conv1.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, 2, 1, 1)*1.5)
    net.fc = nn.Linear(net.fc.in_features, 8)
    return net.cuda()


def tensor(pixels, rows):
    return (torch.as_tensor(np.array(pixels[rows]), device='cuda', dtype=torch.float32)/255-.5)/.25


def train():
    if (OUT/'TRAINING_COMPLETE.json').exists():
        return
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)
    with np.load(TASK/'target.npz') as data:
        rows = data['source_rows']
    labels = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    pixels = np.load(CACHE/'source_inputs128.npy', mmap_mode='r')
    net = model()
    tracked = ('conv1.weight', 'layer2.0.conv1.weight', 'layer4.0.conv1.weight')
    initial = {k: net.state_dict()[k].detach().cpu().clone() for k in tracked}
    counts = np.bincount(labels[rows], minlength=8)
    assert counts.min() > 0
    weights = counts.astype(float)**-.5
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights/weights.mean(), dtype=torch.float32, device='cuda'))
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    logs = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        net.train()
        order = rng.permutation(rows)
        total_loss = 0.
        correct = 0
        epstart = time.perf_counter()
        for step, lo in enumerate(range(0, len(order), BATCH)):
            batch_rows = order[lo:lo+BATCH]
            x = torch.rot90(tensor(pixels, batch_rows), int(rng.integers(4)), (-2, -1))
            y = torch.as_tensor(labels[batch_rows], dtype=torch.long, device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = net(x)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite source loss')
            loss.backward()
            grad = nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach())*len(y)
            correct += int((logits.argmax(1) == y).sum())
            if step % 100 == 0:
                print(f'train epoch{epoch+1}/{EPOCHS} rows{lo+len(y)}/{len(rows)} loss{float(loss.detach()):.4f} grad{float(grad):.3f} elapsed{time.perf_counter()-epstart:.1f}s', flush=True)
        scheduler.step()
        logs.append(dict(epoch=epoch+1, weighted_loss=total_loss/len(rows),
                         augmented_training_oa=100*correct/len(rows), seconds=time.perf_counter()-epstart))
        write(OUT/'training_progress.json', logs)
        print(json.dumps(logs[-1]), flush=True)
    changes = {k: float(torch.linalg.vector_norm(net.state_dict()[k].detach().cpu()-initial[k])) for k in tracked}
    if not all(v > 0 for v in changes.values()):
        raise AssertionError('An expected backbone stage was not updated')
    torch.save(dict(state={k: v.detach().cpu() for k, v in net.state_dict().items()},
                    source_rows=rows, seed=SEED, epochs=EPOCHS), OUT/'r50_final.pt')
    write(OUT/'TRAINING_COMPLETE.json', dict(seconds=time.perf_counter()-started, epochs=logs,
        source_classes=counts.tolist(), parameter_updates_l2=changes, trainable_parameters=sum(p.numel() for p in net.parameters()),
        cuda_peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
        query_labels_used=False, checkpoint_selection='fixed final epoch'))


def extract():
    torch.set_num_threads(2)
    net = model(pretrained=False)
    ckpt = torch.load(OUT/'r50_final.pt', map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt['state'])
    encoder = nn.Sequential(*list(net.children())[:-2], nn.AdaptiveAvgPool2d((2, 2))).eval()
    del ckpt, net
    for name in ('source', 'oss2'):
        if (OUT/f'{name}_FEATURES_COMPLETE.json').exists():
            continue
        pixels = np.load(CACHE/f'{name}_inputs128.npy', mmap_mode='r')
        rows = np.load(TASK/'target.npz')['source_rows'] if name == 'source' else np.arange(len(pixels))
        features = np.lib.format.open_memmap(OUT/f'{name}_c4.npy', mode='w+', dtype=np.float32, shape=(len(rows), 2048))
        started = time.perf_counter()
        with torch.inference_mode():
            for lo in range(0, len(rows), BATCH):
                hi = min(len(rows), lo+BATCH)
                x = tensor(pixels, rows[lo:hi])
                views = []
                for rotation in range(4):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        z = encoder(torch.rot90(x, rotation, (-2, -1)))
                    views.append(z.to(torch.float16).float().mean((-2, -1)).cpu().numpy())
                features[lo:hi] = np.mean(views, axis=0)
                if lo % 4096 == 0:
                    print(f'features {name} {hi}/{len(rows)} {time.perf_counter()-started:.1f}s', flush=True)
        features.flush()
        np.save(OUT/f'{name}_rows.npy', rows)
        write(OUT/f'{name}_FEATURES_COMPLETE.json', dict(rows=len(rows), seconds=time.perf_counter()-started))


def evaluate():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeClassifier
    from frozen_comparison import ba
    with np.load(TASK/'target.npz') as saved:
        data = {k: saved[k] for k in ('source_rows', 'rows', 'ids', 'support', 'support_labels', 'eval_masks', 'conditions')}
    labels = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()[data['source_rows']]
    source_frozen = np.load(ROOT/'c4_base_trial_20261009/pooled_c4.npy', mmap_mode='r')
    arms = ('frozen_single_core', 'finetuned_single_core')
    predictions = np.empty((5, 2, len(data['rows'])), dtype=np.int8)
    source_diagnostics = []
    for ai, arm in enumerate(arms):
        raw_source = (np.asarray(source_frozen[data['source_rows']], dtype=np.float32) if ai == 0
                      else np.load(OUT/'source_c4.npy'))
        raw_target = np.load(ROOT/'cross_dataset_c4_20261009/oss2/pooled_c4.npy' if ai == 0 else OUT/'oss2_c4.npy')
        if ai:
            np.testing.assert_array_equal(np.load(OUT/'source_rows.npy'), data['source_rows'])
            np.testing.assert_array_equal(np.load(OUT/'oss2_rows.npy'), data['rows'])
        pca = PCA(n_components=128, svd_solver='randomized', random_state=SEED)
        xs = pca.fit_transform(raw_source).astype(float)
        xt = pca.transform(raw_target).astype(float)
        center, scale = xs.mean(0), np.maximum(xs.std(0), 1e-6)
        xs, xt = (xs-center)/scale, (xt-center)/scale
        head = RidgeClassifier(alpha=1., class_weight='balanced').fit(xs, labels)
        np.testing.assert_array_equal(head.classes_, np.arange(8))
        prior = head.decision_function(xt)
        phi = np.c_[xt/np.sqrt(128), np.ones(len(xt))]
        mean = xt/np.maximum(np.linalg.norm(xt, axis=1, keepdims=True), 1e-12)
        source_diagnostics.append(dict(arm=arm, source_fit_oa=100*np.mean(head.predict(xs) == labels),
            source_fit_ba=ba(labels, head.predict(xs)), pca_explained_variance=float(pca.explained_variance_ratio_.sum())))
        for draw, support in enumerate(data['support']):
            assert not data['eval_masks'][draw, support].any()
            condition = np.exp(-np.sum((data['conditions'][:, None]-data['conditions'][support][None])**2, axis=2)/.5)
            kernel = phi@phi[support].T+.25*(mean@mean[support].T)*condition
            values, _ = score(kernel, prior, support, data['support_labels'][draw])
            predictions[draw, ai] = values.argmax(1)
        with (OUT/f'{arm}_head.pkl').open('wb') as stream:
            pickle.dump(dict(pca=pca, center=center, scale=scale, head=head), stream)
        del raw_source, raw_target, xs, xt
    np.savez_compressed(OUT/'policies.npz', predictions=predictions, arms=np.array(arms), **data)
    write(OUT/'source_diagnostics.json', source_diagnostics)
    # Target evaluation labels are loaded only after every new prediction is frozen.
    frame = pd.read_csv(ROOT/'opensarship2_transfer_20261007/target_manifest.csv', usecols=['sample_id', 'class_id'])
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str), data['ids'])
    truth = frame.class_id.to_numpy()
    old = np.load(ROOT/'unified_head_20261010/targets/oss2/oss2/policies.npz')
    for key in ('rows', 'ids', 'support', 'eval_masks'):
        np.testing.assert_array_equal(data[key], old[key])
    records, per_class = [], []
    for draw, mask in enumerate(data['eval_masks']):
        modes = {arm: predictions[draw, ai] for ai, arm in enumerate(arms)}
        for name in ('base', 'expert', 'late_softmax_fixed', 'unified_fixed'):
            modes['reference_'+name] = old['predictions'][draw, list(old['modes']).index(name)]
        y = truth[mask]
        for arm, all_pred in modes.items():
            pred = all_pred[mask]
            records.append(dict(draw=draw, arm=arm, n=len(y), oa=100*np.mean(pred == y), ba=ba(y, pred)))
            for c in np.unique(y):
                per_class.append(dict(draw=draw, arm=arm, class_id=int(c), n=int((y == c).sum()), recall=100*np.mean(pred[y == c] == c)))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT/'metrics.csv', index=False)
    metrics.groupby('arm')[['oa', 'ba']].mean().to_csv(OUT/'summary.csv')
    pd.DataFrame(per_class).to_csv(OUT/'per_class.csv', index=False)
    write(OUT/'COMPLETE.json', dict(dataset='oss2', source_only_finetuning=True, source_checkpoint_epochs=EPOCHS,
        support_draws=5, network_seeds=1, query_labels_used_for_fit=False, other_24_tasks_completed=False))
    print(metrics.groupby('arm')[['oa', 'ba']].mean().to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'train', 'extract', 'evaluate'))
    args = parser.parse_args()
    globals()[args.stage]()


if __name__ == '__main__':
    main()
