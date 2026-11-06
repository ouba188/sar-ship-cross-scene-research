"""Repeat frozen seed17 development choices, without selecting new epochs."""
from run import *


def main():
    task.current.setup(task.DEVICE)
    selection = json.loads((ROOT/'selection.json').read_text())
    for other in ('mean', 'second'):
        assert selection['class']['metrics']['ba'] > selection[other]['metrics']['ba']
        assert selection['class']['metrics']['oa'] >= selection[other]['metrics']['oa']
    rows, train, hold, rid = task.records()
    a, c = task.arrays(), e.caches()
    c.update({p.stem: np.load(p, mmap_mode='r') for p in (ROOT/'cache').glob('*.npy')})
    task.write(ROOT/'repeat_protocol.json', dict(selection=selection, seeds=[29, 43],
        rule='freeze each method own seed17 position/gain/epoch; no selection on repeat seeds',
        same_carrier=True, independent_pretraining=False, historical_development=True))
    for seed in (29, 43):
        for kind in ('mean', 'second', 'class'):
            sel = selection[kind]
            fit(sel['position'], kind, sel['gain'], seed, 6, rows, train, hold, a, c, sel['epoch'])
            summarize()


if __name__ == '__main__':
    main()
