"""Matched OSS2 no-M3 source fit using the existing fixed public protocol."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'final_oss2_replication_20261029'
sys.path.insert(0, str(PARENT))
import run


def main():
    run.runtime('cuda')
    rows, cache = run.cohorts(), run.Cache('cuda')
    run.HERE = HERE
    run.checkpoint = lambda arm, seed, stage: HERE / 'training' / f'{arm}_s{seed}' / stage / 'model.pt'
    for seed in (17, 29, 43):
        path = run.checkpoint('no_m3', seed, 'fit')
        if not path.exists():
            run.train_one('no_m3', seed, 'fit', rows, cache, 'cuda')
        net = run.api.load(path, 'cuda')
        assert net.arm == 'no_m3' and len(net.m3) == 0
        refs = rows['references']
        bank = cache.bank(refs)
        sample = rows['source_fit'][:4]
        with run.torch.no_grad():
            _, detail = cache.score(net, sample, refs, bank, True)
        assert not detail['relations']
        print('NO_M3_FIT_READY', seed, flush=True)
        del net
        run.torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
