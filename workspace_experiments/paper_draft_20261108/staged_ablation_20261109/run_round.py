"""Bounded development selection followed by paired repeats; no evaluation selection."""
import json
import gc
import numpy as np
import staged as e
from select_and_evaluate import select


def main():
    e.task.current.setup(e.task.DEVICE)
    protocol=json.loads((e.ROOT/'protocol.json').read_text())
    rows,groups=e.prev.records();a,c=e.prev.arrays()
    a['write']=np.load(e.PREVIOUS/'common_init/write.npy',mmap_mode='r')
    e.resident(a,c)
    for scale in (1.,.3):
        for arm in ('full','plain','se'):
            e.fit(arm,17,scale,protocol,rows,groups,a,c)
    select()
    choices=json.loads((e.ROOT/'selected.json').read_text())['scales']
    for seed in (17,29,43):
        for arm,scale in choices.items():
            e.fit(arm,seed,scale,protocol,rows,groups,a,c)
    del a,c;gc.collect();e.torch.cuda.empty_cache()
    print('ALL_STAGED_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
