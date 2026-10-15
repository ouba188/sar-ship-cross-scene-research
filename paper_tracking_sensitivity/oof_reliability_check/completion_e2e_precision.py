"""Small same-pixel control for batch-size and bf16 replay differences."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import pickle
import sys
import numpy as np
import torch
from completion_e2e_pipeline import OUT, WS, Pipeline, relative_error, write


def pooled(encoder, x, bf16, c4=True):
    result = []
    with torch.inference_mode():
        for rotation in range(4 if c4 else 1):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                y = encoder(torch.rot90(x, rotation, dims=(-2, -1)))
            result.append(y.flatten(2).transpose(1, 2).float().cpu().numpy().astype(np.float16).astype(np.float32).mean(1))
    return result[0], np.mean(result, axis=0)


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with (OUT/'self24ports/prepared.pkl').open('rb') as stream:
        state = pickle.load(stream)
    reference = state.pop('reference')
    pipe = Pipeline(state)
    indices = np.arange(64, 80)
    pixels = np.stack([pipe.read(index)[0] for index in indices])
    sys.path.insert(0, str(WS))
    from extract_full_six_spatial_features import build_encoder, read_one
    original = build_encoder(torch.device('cuda'))
    original_pixels = np.stack([np.stack([read_one(path) for path in state['paths'][i]]) for i in indices])
    np.testing.assert_array_equal(pixels, original_pixels)
    split = torch.nn.Sequential(pipe.r50_prefix, pipe.r50_tail)
    for actual, old in zip(split.parameters(), original.parameters()):
        torch.testing.assert_close(actual, old, rtol=0, atol=0)
    records, values = [], {}
    for batch, bf16, allow_tf32 in ((16, True, False), (32, True, False), (64, True, False),
                                    (32, True, True), (64, True, True),
                                    (16, False, False), (32, False, False)):
        torch.backends.cudnn.allow_tf32 = allow_tf32
        x = torch.from_numpy(np.tile(pixels, (batch//16, 1, 1, 1))).cuda().float()
        x = (x/255.-.5)/.25
        raw, c4 = pooled(original, x, bf16, c4=bf16)
        key = f'b{batch}_bf16{bf16}_tf32{allow_tf32}'
        values[key] = (raw[:16], c4[:16])
        records.append(dict(setting=key, original_raw_cache=relative_error(raw[:16], reference['raw'][indices]),
                            original_c4_cache=relative_error(c4[:16], reference['c4'][indices]) if bf16 else None))
        if batch == 16 and bf16:
            sr, sc = pooled(split, x, True)
            records[-1]['split_vs_original_raw'] = relative_error(sr, raw)
            records[-1]['split_vs_original_c4'] = relative_error(sc, c4)
    batch_effects = {}
    for batch in (32, 64):
        batch_effects[f'bf16_b16_vs_b{batch}'] = relative_error(values['b16_bf16True_tf32False'][0], values[f'b{batch}_bf16True_tf32False'][0])
    batch_effects['float32_b16_vs_b32'] = relative_error(values['b16_bf16False_tf32False'][0], values['b32_bf16False_tf32False'][0])
    write(OUT/'PRECISION_PROBE.json', dict(dataset='self24ports', task=state['task'],
        ids=state['ids'][indices].tolist(), same_pixels_exact=True, same_weights_exact=True,
        probe='Original public project extractor versus split deployment encoder,16 predetermined queries; padded identical images for batch32/64; no labels or parameter changes',
        records=records, batch_effects=batch_effects,
        boundary='Float32 outputs also undergo original spatialfloat16 pooling. Only historical task recipe replays; no cache replacement.'))
    print(records, flush=True)
    print(batch_effects, flush=True)
    pipe.close()


if __name__ == '__main__':
    main()
