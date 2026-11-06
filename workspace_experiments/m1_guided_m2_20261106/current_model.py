"""Selected joint-head working version; reuse the trained operator unchanged."""
import argparse
import json
import study as s

ROOT=s.ROOT
CONFIG=json.loads((ROOT/'current_version.json').read_text(encoding='utf-8'))


def load_readout(seed=None, device=None):
    seed=CONFIG['default_seed'] if seed is None else seed
    if seed not in CONFIG['seeds']:
        raise ValueError(f'No selected checkpoint for seed {seed}')
    device=s.task.DEVICE if device is None else device
    model=s.RegionalGuidance(CONFIG['arm']).to(device)
    path=ROOT/CONFIG['checkpoint_pattern'].format(seed=seed)
    model.load_state_dict(s.torch.load(path,map_location=device,weights_only=True))
    return model.eval()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('replay','train'),nargs='?',default='replay')
    parser.add_argument('--seed',type=int,choices=CONFIG['seeds'],default=CONFIG['default_seed'])
    args=parser.parse_args()
    s.task.current.setup(s.task.DEVICE)
    rows,groups=s.prev.records();cache=s.load_cache()
    if args.action=='train':
        s.train(CONFIG['arm'],args.seed,cache,rows,groups,freeze_head=CONFIG['freeze_head'])
        return
    model=load_readout(args.seed)
    y=s.np.array([r['label'] for r in rows]);result={}
    for split in ('validation','evaluation'):
        ids=groups[split];score=s.predict(model,cache,ids)
        previous=s.np.load(ROOT/f'class_s{args.seed}'/(split+'.npz'))
        assert s.np.array_equal(ids,previous['ids'])
        error=float(abs(score-previous['score']).max())
        same=bool(s.np.array_equal(score.argmax(1),previous['score'].argmax(1)))
        assert error<1e-4 and same,(split,error,same)
        result[split]=dict(**s.task.metrics(y[ids],score),max_score_error=error,predictions_identical=same)
    record=dict(version=CONFIG['version'],seed=args.seed,head_frozen=False,retrained=False,results=result)
    s.task.write(ROOT/f'current_replay_s{args.seed}.json',record)
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
