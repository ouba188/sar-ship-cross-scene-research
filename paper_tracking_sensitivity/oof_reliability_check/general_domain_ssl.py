"""Matched ImageNet versus source-only SAR SimCLR pilot, with shared input access."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

from strong_single_r50_task import SPLIT_SEED, grouped_split, validation_episodes, target_design

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT/'general_domain_ssl_20261011'
CACHE = Path('C:/Users/25791/AppData/Local/Temp/sar_general_domain_20261011')
SOURCE = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
PIXELS = dict(source=Path('D:/SAR_completion_cache_20261010/source_inputs128.npy'),
              oss2=Path('D:/SAR_completion_cache_20261010/oss2_inputs128.npy'),
              f0921=ROOT/'f0921_detector_recrop_20261007/inputs128.npy')
SEED = 20261011
SIZE = 112
PAIRS = 64
EPOCHS = 8
EVAL_EPOCHS = (0, 2, 4, 6, 8)
BACKBONE_LR = 3e-5
TEMP = .2
torch.set_num_threads(2)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def frame():
    return pd.read_csv(SOURCE/'manifest.csv', keep_default_na=False,
        usecols=['instance_id', 'port', 'class_id', 'mmsi', 'product'],
        dtype={'mmsi': str, 'product': str})


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    (CACHE/'features').mkdir(parents=True, exist_ok=True)
    (CACHE/'models').mkdir(exist_ok=True)
    source = frame()
    splits = []
    for dataset, transfer in [('f0921', 'f0921_transfer_20261007'), ('oss2', 'opensarship2_transfer_20261007')]:
        with np.load(ROOT/transfer/'source_split.npz') as z:
            splits.append(z['source_rows'])
            if np.intersect1d(z['source_rows'], z['excluded_source_rows']).size:
                raise ValueError('Historical source exclusion overlap')
        target = target_design(ROOT/'learned_frequency_residual_20261009/tasks'/dataset/'target.npz')
        manifest = ROOT/('f0921_detector_recrop_20261007' if dataset == 'f0921' else transfer)/'target_manifest.csv'
        ids = pd.read_csv(manifest, usecols=['sample_id']).sample_id.to_numpy(str)
        np.testing.assert_array_equal(ids[target['rows']], target['ids'])
        pixels = np.load(PIXELS[dataset], mmap_mode='r')
        if pixels.shape != (len(ids), 2, 128, 128) or pixels.dtype != np.uint8:
            raise ValueError('Unexpected paired target pixel cache')
        np.savez_compressed(OUT/f'{dataset}_design.npz', **target)
    allowed = np.intersect1d(*splits)
    train, val, audit = grouped_split(source, allowed, seed=SPLIT_SEED)
    episodes = validation_episodes(source, val)
    np.testing.assert_array_equal(np.load('D:/SAR_completion_cache_20261010/source_ids.npy'),
                                  source.instance_id.to_numpy(str))
    if np.intersect1d(train, val).size:
        raise ValueError('Source train/validation overlap')
    np.savez(OUT/'source_split.npz', allowed=allowed, inner_train=train, inner_val=val)
    arrays = {'ports': np.array([e['port'] for e in episodes])}
    for i, episode in enumerate(episodes):
        arrays[f'{i}_support'], arrays[f'{i}_query'] = episode['support'], episode['query']
    np.savez(OUT/'validation_episodes.npz', **arrays)
    audit.update(shared_allowed_rows=len(allowed), seed=SEED, source_port_split_seed=SPLIT_SEED,
        source_episode_seed=SPLIT_SEED+1, source_train_rows=len(train),
        source_classes=np.bincount(source.class_id.iloc[train], minlength=8).tolist(),
        source_queries=sum(len(e['query']) for e in episodes), source_episodes=len(episodes),
        external_target_pixels_used_for_training=False, data_cache=str(CACHE))
    write(OUT/'SPLIT.json', audit)
    if not (OUT/'PROTOCOL.json').exists():
        write(OUT/'PROTOCOL.json', dict(date='2026-10-11', stage='Single-neural-seed mechanism pilot',
            seed=SEED, primary_modes=['A_G50', 'B_D50', 'C_G50_G18', 'D_G50_D18', 'E_D50_D18'],
            initialization={'r50': 'ImageNet1K V2', 'r18': 'ImageNet1K V1'},
            domains='G=unchanged ImageNet backbone; D=the same backbone after source SAR SimCLR',
            input='Existing paired128 uint8 -> resize112 -> RGB=[VV,VH,mean] -> ImageNet channel normalization',
            input_access='Both architectures use both polarizations; one view/forward per image at inference',
            ssl='Existing project e206 NT-Xent formula, temperature .2; no F or OSS2 pixels',
            augmentation='Paired-channel synchronized crop scale .8..1 area, independent horizontal/vertical flip; brightness .85..1.15; contrast .85..1.15',
            epochs=EPOCHS, evaluation_epochs=EVAL_EPOCHS, batch_pairs=PAIRS,
            optimizer='AdamW, backbone3e-5, projection1e-3, weight_decay1e-4, cosine schedule',
            projection='GAP -> Linear256 -> ReLU -> Linear128 -> unit normalization',
            trainable='All backbone and projection parameters; BatchNorm training statistics updated',
            precision='CUDA bf16 autocast; float32 similarity/loss; cached raw GAPfloat32',
            selection='Mean BA on eight fixed source episodes; best among positive evaluation epochs, ties earliest. Epoch0 is separate baseline, no forced positive gain',
            source_train='Shared external-allowed intersection, then four held source ports and MMSI/product purge; 21228 rows expected',
            ssl_refit='None in this pilot: chosen inner-trained encoder frozen; final source prior fit on allowed28856',
            readout='Same normalized source-balanced ridge1 prior and balanced support residual ridge8; single sqrt32, pair sqrt16 each, constant coordinate',
            external='F0921 and OSS2 original5x64 support arrays; all predictions saved before query-label readout',
            representation_comparison='Controlled new input/readout, not numerical replacement of historical full model',
            confirmation='One neural seed only. Full23-port LOPO, three seeds and task-trained complete heads are later conditional stages',
            no_target_tuning=True, historical_test_reuse=True, frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    return source, allowed, train, val, episodes


def backbone(arch):
    if arch == 'r50':
        constructor, weights = resnet50, ResNet50_Weights.IMAGENET1K_V2
    elif arch == 'r18':
        constructor, weights = resnet18, ResNet18_Weights.IMAGENET1K_V1
    else:
        raise ValueError(arch)
    cached = Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
    if not cached.exists():
        raise FileNotFoundError(cached)
    model = constructor(weights=weights)
    width = model.fc.in_features
    model.fc = nn.Identity()
    return model, width


def rgb(pair):
    return torch.cat((pair, pair.mean(1, keepdim=True)), dim=1)


def normalize(image):
    mean = image.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = image.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    return (image-mean)/std


def augment(pair):
    n = len(pair)
    scale = torch.empty(n, device=pair.device).uniform_(np.sqrt(.8), 1.)
    flips = torch.where(torch.rand((n, 2), device=pair.device) < .5, -1., 1.)
    affine = torch.zeros((n, 2, 3), device=pair.device)
    affine[:, 0, 0], affine[:, 1, 1] = scale*flips[:, 0], scale*flips[:, 1]
    affine[:, :, 2] = (torch.rand((n, 2), device=pair.device)*2-1)*(1-scale[:, None])
    grid = F.affine_grid(affine, (n, 2, SIZE, SIZE), align_corners=False)
    crop = F.grid_sample(pair, grid, mode='bilinear', padding_mode='border', align_corners=False)
    mean = crop.mean((1, 2, 3), keepdim=True)
    contrast = torch.empty((n, 1, 1, 1), device=pair.device).uniform_(.85, 1.15)
    brightness = torch.empty_like(contrast).uniform_(.85, 1.15)
    crop = ((crop-mean)*contrast+mean)*brightness
    return normalize(rgb(crop.clamp(0., 1.)))


def nt_xent(z, temperature=TEMP):
    # Same normalized NT-Xent and positive ordering as e206_ssl_pretrain.py.
    z = F.normalize(z.float(), dim=1)
    n = len(z)//2
    similarity = z@z.T/temperature
    similarity.fill_diagonal_(-1e4)
    target = torch.cat((torch.arange(n, 2*n), torch.arange(n))).to(z.device)
    return F.cross_entropy(similarity, target)


class SSL(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.backbone, self.width = backbone(arch)
        self.projection = nn.Sequential(nn.Linear(self.width, 256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, x):
        return self.projection(self.backbone(x))


def pixels_batch(pixels, rows, device):
    return torch.from_numpy(np.array(pixels[rows], dtype=np.float32)).to(device)/255.


@torch.inference_mode()
def encode(model, pixels, rows, path, width, batch=64):
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(pixels), width))
    out[:] = np.nan
    device = next(model.parameters()).device
    start = time.perf_counter()
    for lo in range(0, len(rows), batch):
        index = rows[lo:lo+batch]
        pair = pixels_batch(pixels, index, device)
        pair = F.interpolate(pair, size=(SIZE, SIZE), mode='bilinear', align_corners=False, antialias=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            value = model(normalize(rgb(pair))).float()
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite encoded features')
        out[index] = value.cpu().numpy()
        if lo % 16384 == 0:
            print(f'ENCODE {path.stem} {min(lo+batch,len(rows))}/{len(rows)} {time.perf_counter()-start:.1f}s', flush=True)
    out.flush()
    del out
    return time.perf_counter()-start


def cpu_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def validation(feature_path, train, episodes, labels):
    from general_domain_readout import evaluate_source
    return evaluate_source(feature_path, train, episodes, labels)


def train_arch(arch, prepared):
    source, allowed, train, _, episodes = prepared
    dst = OUT/arch
    if (dst/'TRAINING_COMPLETE.json').exists():
        print('REUSE TRAIN', arch, flush=True)
        return
    dst.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = SSL(arch).cuda()
    pixels = np.load(PIXELS['source'], mmap_mode='r')
    labels = source.class_id.to_numpy(int)
    suffix = arch[1:]
    gpath = CACHE/'features'/f'source_G{suffix}.npy'
    current_path = CACHE/'features'/f'source_current_{arch}.npy'
    selected_path = CACHE/'features'/f'source_D{suffix}.npy'
    best_path, resume_path = CACHE/'models'/f'{arch}_best.pt', CACHE/'models'/f'{arch}_resume.pt'
    optimizer = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': BACKBONE_LR},
                                  {'params': model.projection.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
    steps = len(train)//PAIRS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS*steps)
    start_epoch, best_ba, best_epoch, records = 0, -np.inf, None, []
    initial = next(model.backbone.parameters()).detach().cpu().clone()
    if resume_path.exists():
        state = torch.load(resume_path, map_location='cuda', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, best_ba, best_epoch, records = state['epoch'], state['best_ba'], state['best_epoch'], state['records']
        torch.set_rng_state(state['cpu_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        del state
    elif not (dst/'GENERAL.json').exists():
        elapsed = encode(model.backbone, pixels, allowed, gpath, model.width)
        score, details = validation(gpath, train, episodes, labels)
        write(dst/'GENERAL.json', dict(ba=score, episodes=details, feature_seconds=elapsed))
    for epoch in range(start_epoch+1, EPOCHS+1):
        model.train()
        order = np.random.default_rng(SEED+epoch).permutation(train)[:steps*PAIRS]
        losses, norms = [], []
        start = time.perf_counter()
        for step in range(steps):
            rows = order[step*PAIRS:(step+1)*PAIRS]
            pair = pixels_batch(pixels, rows, 'cuda')
            a, b = augment(pair), augment(pair)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model(torch.cat((a, b)))
            loss = nt_xent(z)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite SSL loss')
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            norms.append(float(grad))
            if step % 50 == 0:
                print(f'TRAIN {arch} epoch{epoch}/{EPOCHS} step{step+1}/{steps} loss={losses[-1]:.4f} {time.perf_counter()-start:.1f}s', flush=True)
        record = dict(epoch=epoch, loss=float(np.mean(losses)), gradient_norm=float(np.mean(norms)),
                      seconds=time.perf_counter()-start, backbone_lr=optimizer.param_groups[0]['lr'])
        if epoch in EVAL_EPOCHS:
            encode(model.backbone, pixels, allowed, current_path, model.width)
            score, details = validation(current_path, train, episodes, labels)
            record.update(source_ba=score, validation_episodes=details)
            if score > best_ba+1e-10:
                best_ba, best_epoch = score, epoch
                torch.save(dict(backbone=cpu_state(model.backbone), epoch=epoch, seed=SEED,
                                source_rows=train, input_protocol='paired_RGB112_ImageNet'), best_path)
                shutil.copyfile(current_path, selected_path)
            print(f'VALIDATE {arch} epoch{epoch} BA={score:.4f} selected={best_epoch}/{best_ba:.4f}', flush=True)
        records.append(record)
        write(dst/'CURVE.json', records)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            epoch=epoch, best_ba=best_ba, best_epoch=best_epoch, records=records,
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), resume_path)
    torch.save(dict(backbone=cpu_state(model.backbone), epoch=EPOCHS, seed=SEED, source_rows=train),
               CACHE/'models'/f'{arch}_last.pt')
    change = float((next(model.backbone.parameters()).detach().cpu()-initial).abs().max())
    write(dst/'TRAINING_COMPLETE.json', dict(arch=arch, seed=SEED, epochs=EPOCHS, steps_per_epoch=steps,
        selected_epoch=best_epoch, selected_source_ba=best_ba, selection_at_budget_boundary=best_epoch == EPOCHS,
        first_weight_max_abs_update=change, source_train_rows=len(train), ssl_target_rows=0,
        best_checkpoint=str(best_path), last_checkpoint=str(CACHE/'models'/f'{arch}_last.pt'),
        general_source_features=str(gpath), domain_source_features=str(selected_path),
        max_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20))
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()


def extract_targets():
    if not all((OUT/arch/'TRAINING_COMPLETE.json').exists() for arch in ('r18', 'r50')):
        raise RuntimeError('Freeze both source-selected domain checkpoints first')
    for domain in ('G', 'D'):
        for arch in ('r18', 'r50'):
            model, width = backbone(arch)
            if domain == 'D':
                state = torch.load(CACHE/'models'/f'{arch}_best.pt', map_location='cpu', weights_only=False)
                model.load_state_dict(state['backbone'], strict=True)
                del state
            model = model.cuda().eval()
            for ds in ('oss2', 'f0921'):
                name = f'{ds}_{domain}{arch[1:]}'
                if (OUT/'extraction'/f'{name}.json').exists():
                    continue
                pixels = np.load(PIXELS[ds], mmap_mode='r')
                target = target_design(ROOT/'learned_frequency_residual_20261009/tasks'/ds/'target.npz')
                np.testing.assert_array_equal(target['rows'], np.arange(len(pixels)))
                output = CACHE/'features'/f'{name}.npy'
                elapsed = encode(model, pixels, target['rows'], output, width)
                write(OUT/'extraction'/f'{name}.json', dict(rows=len(pixels), width=width,
                    features=str(output), seconds=elapsed, source_checkpoint_frozen=True,
                    target_query_labels_read=False))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    write(OUT/'ALL_FEATURES_COMPLETE.json', dict(source_encoders=4, external_datasets=2,
        neural_seed=SEED, target_tuning=False, cache=str(CACHE)))


def smoke():
    torch.manual_seed(SEED)
    pixels = np.load(PIXELS['source'], mmap_mode='r')
    rows = np.load(OUT/'source_split.npz')['inner_train'][:PAIRS]
    model = SSL('r50').cuda().train()
    pair = pixels_batch(pixels, rows, 'cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=BACKBONE_LR)
    start = time.perf_counter()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = model(torch.cat((augment(pair), augment(pair))))
        loss = nt_xent(z)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        print(f'SMOKE {step} loss={float(loss):.5f} grad={float(grad):.3f}', flush=True)
    write(OUT/'SMOKE.json', dict(batch_pairs=PAIRS, steps=3, seconds=time.perf_counter()-start,
        peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20, real_source_rows=rows.tolist(),
        finite_loss_and_gradient=True, smoke_weights_discarded=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'smoke', 'train', 'extract', 'all'), default='all')
    parser.add_argument('--arch', choices=('r18', 'r50'))
    args = parser.parse_args()
    prepared = prepare()
    if args.stage == 'smoke':
        smoke()
    if args.stage in ('train', 'all'):
        for arch in ([args.arch] if args.arch else ['r18', 'r50']):
            train_arch(arch, prepared)
    if args.stage in ('extract', 'all'):
        extract_targets()


if __name__ == '__main__':
    main()
