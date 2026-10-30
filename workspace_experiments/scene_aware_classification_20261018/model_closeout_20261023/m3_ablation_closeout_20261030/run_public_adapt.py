"""Finish the matched OSS2 no-M3 chain with its own source teacher."""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'final_oss2_replication_20261029'
sys.path.insert(0, str(PARENT))
import run


def main():
    run.runtime('cuda')
    rows, cache = run.cohorts(), run.Cache('cuda')
    refs, records = rows['references'], rows['source_adapt']
    run.assert_rows(records, 'source_adapt')
    bank = cache.bank(refs)
    for seed in (17, 29, 43):
        fit = HERE / 'training' / f'no_m3_s{seed}' / 'fit' / 'model.pt'
        output = HERE / 'training' / f'no_m3_s{seed}' / 'adapt' / 'model.pt'
        teacher_file = HERE / 'teachers' / f'no_m3_s{seed}.json'
        if output.exists():
            print('ADAPT_ALREADY_COMPLETE', seed, flush=True)
            continue
        net = run.api.load(fit, 'cuda')
        if teacher_file.exists():
            targets = run.read(teacher_file)
        else:
            def progress(done, total, _):
                if done % 32 == 0 or done == total:
                    print('SELECT_NO_M3', seed, done, total, flush=True)
            targets = run.interventions.select_records(net, records, refs, 'cuda', progress=progress)
            run.write(teacher_file, targets)
        assert [(r['id'], r['label']) for r in targets] == [(r['id'], r['label']) for r in records]
        run.model_api.training_policy(net, 'adapt')
        params = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
        assert {n for n, _ in params} == set(run.model_api.LIVE)
        frozen = {n: value.cpu().clone() for n, value in net.state_dict().items()
                  if n not in run.model_api.LIVE}
        optimizer = run.torch.optim.AdamW([p for _, p in params], lr=3e-5, weight_decay=.01)
        weights = run.np.asarray(run.class_weights4(records, 'adapt'), dtype=run.np.float32)
        rng = run.np.random.default_rng(seed)
        curve, steps = [], 0
        tick = time.perf_counter()
        epochs, batch_size, expected = run.BUDGET['adapt']
        for epoch in range(1, epochs + 1):
            terms_epoch = []
            for ix in run.np.array_split(rng.permutation(len(records)), range(batch_size, len(records), batch_size)):
                group = [records[int(i)] for i in ix]
                labels = run.torch.tensor([r['label'] for r in group], device='cuda')
                weight = run.torch.as_tensor(weights[ix], device='cuda')
                teacher = run.torch.tensor([targets[int(i)]['original'] for i in ix], device='cuda')
                useful = run.torch.tensor([targets[int(i)]['useful'] for i in ix], device='cuda')
                optimizer.zero_grad(set_to_none=True)
                score = cache.score(net, group, refs, bank)
                if steps == 0:
                    assert float((score.detach() - teacher).abs().max()) < 1e-4
                loss, terms = run.training.adaptation_loss(score, labels, teacher, weight, useful,
                                                            run.COEFFICIENT)
                assert run.torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                steps += 1
                terms_epoch.append({k: float(v.detach()) for k, v in terms.items()})
            entry = dict(epoch=epoch, steps=steps,
                         **{k: float(run.np.mean([t[k] for t in terms_epoch])) for k in terms_epoch[0]})
            curve.append(entry)
            print('NO_M3_ADAPT', seed, entry, flush=True)
        assert steps == expected
        assert all(run.torch.equal(net.state_dict()[n].cpu(), old) for n, old in frozen.items())
        run.api.save(net, output, seed=seed, arm='no_m3', stage='adapt')
        restored = run.api.load(output, 'cuda')
        sample = run.sample_rows(rows)
        with run.torch.no_grad():
            replay = max(float((cache.score(net, sample[i:i + 3], refs, bank) -
                                cache.score(restored, sample[i:i + 3], refs, bank)).abs().max())
                         for i in range(0, len(sample), 3))
        assert replay < 1e-5
        native = run.native_error(net, cache, sample, refs, bank)
        run.write(output.parent / 'training.json', dict(seed=seed, arm='no_m3', stage='adapt',
            source_count=len(records), reference_count=len(refs), epochs=epochs, steps=steps,
            coefficient=run.COEFFICIENT, selected_count=sum(r['useful'] for r in targets),
            own_teacher=True, frozen_unchanged=True, save_replay_error=replay,
            native_cache_error=native, curve=curve, seconds=time.perf_counter() - tick))
        print('NO_M3_ADAPT_COMPLETE', seed, flush=True)
        del net, restored
        run.torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
