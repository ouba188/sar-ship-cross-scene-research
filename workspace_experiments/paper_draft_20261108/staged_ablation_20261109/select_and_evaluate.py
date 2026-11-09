"""Select learning-rate scales on development data; evaluate only frozen choices."""
import json
import argparse
import gc
import numpy as np
import staged as e


def select():
    choices={};decisions=[]
    for arm in ('full','plain','se'):
        options=[]
        for scale in (1.,.3):
            p=e.ROOT/f'{arm}_s17_lr{scale:g}/receipt.json'
            r=json.loads(p.read_text(encoding='utf-8'))
            v=r['metrics']['validation']
            options.append((v['ba'],v['oa'],-v['nll'],scale))
            decisions.append(dict(arm=arm,scale=scale,**v))
        choices[arm]=max(options)[-1]
    for arm in ('no_m1','no_m2','no_m3'):choices[arm]=choices['full']
    e.task.write(e.ROOT/'selected.json',dict(scales=choices,development=decisions,
        rule='seed17 development BA, then OA, then NLL; frozen before evaluation; deletion arms inherit full scale',
        evaluation_used=False))
    print(json.dumps(choices),flush=True)


@e.torch.no_grad()
def evaluate():
    selected=json.loads((e.ROOT/'selected.json').read_text())['scales']
    protocol=json.loads((e.ROOT/'protocol.json').read_text())
    e.task.current.setup(e.task.DEVICE)
    rows,groups=e.prev.records();a,c=e.prev.arrays()
    a['write']=np.load(e.PREVIOUS/'common_init/write.npy',mmap_mode='r')
    e.resident(a,c);records=[]
    for arm,scale in selected.items():
        for seed in (17,29,43):
            folder=e.ROOT/f'{arm}_s{seed}_lr{scale:g}'
            receipt=json.loads((folder/'receipt.json').read_text())
            net=e.Network(arm,seed,protocol['pretrain'])
            net.load_state_dict(e.torch.load(receipt['final'],map_location='cpu',weights_only=True))
            e.configure(net,3);net.requires_grad_(False)
            e.freeze_delta(net,a,rows,groups['references'])
            for split in ('validation','evaluation'):
                ids=groups[split];z=e.predict(net,a,c,rows,ids,groups['references'])
                if split=='validation':
                    cached=np.load(e.Path(receipt['final']).parent/'validation.npz')['score']
                    error=float(abs(z-cached).max());assert error<1e-4,error
                np.savez_compressed(folder/f'{split}.npz',ids=ids,score=z)
                records.append(dict(arm=arm,seed=seed,scale=scale,split=split,
                    **e.task.metrics(np.array([rows[i]['label'] for i in ids]),z)))
            print('EVALUATED',arm,seed,records[-1],flush=True)
            del net;gc.collect();e.torch.cuda.empty_cache()
    e.prev.pd.DataFrame(records).to_csv(e.ROOT/'metrics.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    evaluate() if args.evaluate else select()
