"""Original plus fixed input interventions through unchanged frozen encoders."""

from context import *
import argparse
import time
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torchvision.models import resnet18
from transforms_fixed import transform_pair
from extract_opensarship2_transfer import WS, build_encoder
from c4_condition_descriptors import describe

TF = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])


def encoders():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ck = torch.load(WS / "e206_out/ssl_resnet18.pt", map_location="cpu", weights_only=True)
    net = resnet18(weights=None)
    missing, unexpected = net.load_state_dict(ck["backbone"], strict=False)
    if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
        raise ValueError("Wrong R18 checkpoint")
    return build_encoder(torch.device("cuda")), torch.nn.Sequential(*list(net.children())[:-1]).cuda().eval()


def read(row, condition):
    with Image.open(row.path_vv) as vv, Image.open(row.path_vh) as vh:
        pair, rgb, info = transform_pair(vv, vh, condition)
    info.update(describe(pair))
    return pair, TF(rgb), info


def forward(batch, paired, ssl):
    x = torch.from_numpy(np.stack([b[0] for b in batch])).cuda().float()
    x = (x/255.-.5)/.25
    orbit = []
    for k in range(4):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            spatial = paired(torch.rot90(x, k, dims=(-2, -1))).flatten(2).transpose(1, 2)
        orbit.append(spatial.float().cpu().numpy().astype(np.float16).astype(np.float32).mean(1))
    groups = {}
    for i, b in enumerate(batch):
        groups.setdefault(tuple(b[1].shape), []).append(i)
    extra = np.empty((len(batch), 512), np.float32)
    for indices in groups.values():
        extra[indices] = ssl(torch.stack([batch[i][1] for i in indices]).cuda()).flatten(1).cpu().numpy()
    return dict(raw=orbit[0], c4=np.mean(orbit, axis=0), extra=extra)


def differences(actual, expected):
    rel = np.linalg.norm(actual.astype(float)-expected, axis=1)/np.maximum(np.linalg.norm(expected, axis=1), 1e-12)
    return dict(max_relative_l2=float(rel.max()), mean_relative_l2=float(rel.mean()), max_abs=float(np.max(np.abs(actual-expected))))


def probe(paired, ssl):
    result = []
    for name in DATASETS:
        f = frame(name)
        indices = np.linspace(0, len(f)-1, 16, dtype=int)
        output = forward([read(row, "original") for row in f.iloc[indices].itertuples()], paired, ssl)
        old = archived(name)
        np.testing.assert_array_equal(f.sample_id.to_numpy(str), old["ids"])
        row = dict(dataset=name, rows=indices.tolist())
        for key in ("raw", "c4", "extra"):
            row[key] = differences(output[key], old[key][indices])
            if row[key]["max_relative_l2"] > (.001 if key == "extra" else .02):
                write(OUT / "REPLAY_FAILED.json", row)
                raise ValueError(f"Original feature replay failed: {name} {key} {row[key]}")
        result.append(row)
        print("PROBE", name, json.dumps(row), flush=True)
    write(OUT / "REPLAY.json", dict(datasets=result, original_preprocessing_reproduced=True))


def extract(paired, ssl):
    if not (OUT / "REPLAY.json").exists():
        raise RuntimeError("Pass original-input replay first")
    for name in DATASETS:
        f = frame(name)
        for condition in CONDITIONS:
            dst = OUT / "features" / name / condition
            if (dst / "COMPLETE.json").exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            n = len(f)
            arrays = {key: np.lib.format.open_memmap(dst / (key+".npy"), mode="w+", dtype="float32", shape=(n, dim))
                for key, dim in (("raw", 2048), ("c4", 2048), ("extra", 512))}
            records = []
            started = time.perf_counter()
            for lo in range(0, n, 32):
                batch = [read(row, condition) for row in f.iloc[lo:lo+32].itertuples()]
                output = forward(batch, paired, ssl)
                for key, values in output.items():
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite transformed features")
                    arrays[key][lo:lo+len(batch)] = values
                for j, b in enumerate(batch):
                    records.append(dict(sample_id=str(f.sample_id.iloc[lo+j]), **b[2]))
                if lo % 512 == 0 or lo+32 >= n:
                    print("EXTRACT", name, condition, min(lo+32, n), n, round(time.perf_counter()-started, 1), flush=True)
            for value in arrays.values():
                value.flush()
            pd.DataFrame(records).to_csv(dst / "descriptors.csv", index=False)
            np.save(dst / "ids.npy", f.sample_id.to_numpy(str))
            replay = None
            if condition == "original":
                old = archived(name)
                replay = {key: differences(np.asarray(value), old[key]) for key, value in arrays.items()}
                if any(v["max_relative_l2"] > (.001 if k == "extra" else .02) for k,v in replay.items()):
                    write(dst / "REPLAY_FAILED.json", replay)
                    raise ValueError("Full original-input replay failed")
            write(dst / "COMPLETE.json", dict(rows=n, condition=condition, dataset=name,
                seconds=time.perf_counter()-started, frozen_encoders=True, replay=replay))
            del arrays
    write(OUT / "FEATURES_COMPLETE.json", dict(datasets=DATASETS, conditions=CONDITIONS, no_backbone_training=True))


def contact_sheet():
    canvas = Image.new("RGB", (4*150, len(DATASETS)*160), "white")
    draw = ImageDraw.Draw(canvas)
    for r, name in enumerate(DATASETS):
        row = next(frame(name).iloc[[len(frame(name))//2]].itertuples())
        for c, condition in enumerate(CONDITIONS):
            pair, _, _ = read(row, condition)
            canvas.paste(Image.fromarray(pair[0]).convert("RGB"), (c*150, r*160+28))
            draw.text((c*150+2, r*160+2), f"{name} {condition}", fill="black")
    canvas.save(OUT / "input_contact_sheet.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["probe", "extract"])
    args = parser.parse_args()
    freeze()
    models = encoders()
    with torch.inference_mode():
        if args.stage == "probe":
            probe(*models)
            contact_sheet()
        else:
            extract(*models)
