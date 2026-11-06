"""Source-only repair for relocating a pretrained class reader to conv2 output."""
import experiment as e
from experiment import *

DEST = ROOT/'source_reader_calibrated'


def prepare(a, c, train):
    model = Network('none')
    features = {'pre': [], 'post': []}
    with torch.no_grad():
        for start in range(0, len(train), 24):
            ids = train[start:start+24]
            x = task.tensor(c['x'], ids)
            occ = task.tensor(c['occupancy'], ids)
            slots = task.tensor(a['slots'], ids)
            v = task.tensor(a['valid'], ids)
            post = []
            for p in range(2):
                reg, en = model.base.region_integrals(slots[:, p], slots.new_zeros(len(ids),20,3,10,10))
                z, _ = model.base.internal_m2[p](x[:, p], reg, v[:, p], en)
                post.append(z)
            block = model.base.backbone.model.layer3[-1]
            z = block.relu(block.bn2(block.conv2(torch.stack(post,1).flatten(0,1)))).reshape_as(x)
            for key, value in [('pre', x), ('post', z)]:
                pooled = (value*occ).sum((-1,-2))/occ.sum((-1,-2)).clamp_min(1e-8)
                features[key].append(pooled)
    old = torch.stack([op.reader.weight.detach().clone() for op in model.m1])
    return {k: torch.cat(v) for k, v in features.items()}, old


def main():
    task.current.setup(task.DEVICE)
    DEST.mkdir(exist_ok=True)
    rows, train, hold, rid = task.records()
    a,c = task.arrays(), caches()
    if not (DEST/'readers.pt').exists():
        features, original = prepare(a,c,train)
        labels = torch.tensor([rows[i]['label'] for i in train], device=task.DEVICE)
        av = task.tensor(a['available'], train)
        frame = pd.DataFrame([rows[i] for i in train])
        hard = set(json.loads((task.ROOT/'source_difficulty.json').read_text())['hard'])
        weight = 1/(frame.groupby('label').mmsi.transform('nunique').to_numpy()*frame.groupby('mmsi').mmsi.transform('size').to_numpy())
        weight *= [2. if i in hard else 1. for i in train]
        weight /= weight.mean()
        weight = torch.tensor(weight,device=task.DEVICE,dtype=torch.float32)
        state, curve = {}, []
        for position in ('pre','post'):
            w = nn.Parameter(original[...,0,0].clone())
            optimizer = torch.optim.AdamW([w],lr=.001,weight_decay=.01)
            for step in range(201):
                scores = torch.einsum('bpc,pkc->bpk',features[position],w)
                ce = F.cross_entropy(scores.flatten(0,1),labels[:,None].expand(-1,2).flatten(),reduction='none').reshape(len(train),2)
                loss = ((ce*av).sum(1)/av.sum(1).clamp_min(1)*weight).mean()
                assert torch.isfinite(loss)
                if step in (0,50,100,200):
                    row = dict(position=position,step=step,weighted_aux_ce=float(loss.detach()))
                    for p in range(2):
                        mask = av[:,p].bool()
                        row[('vv','vh')[p]+'_accuracy'] = float((scores[mask,p].argmax(1)==labels[mask]).float().mean()*100)
                    curve.append(row)
                    print('CALIBRATION',row,flush=True)
                if step == 200:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            state[position] = w.detach().cpu()[...,None,None]
        torch.save(state,DEST/'readers.pt')
        pd.DataFrame(curve).to_csv(DEST/'reader_fit.csv',index=False)
    state = torch.load(DEST/'readers.pt',map_location='cpu',weights_only=True)
    probe = np.r_[train[:3],hold[:3]]
    baseline = predict(Network('none'),a,c,probe)['score']
    arms = ('pre_class_gate','post_class_gate','pre_mean_gate','post_mean_gate')
    errors = {arm:float(abs(predict(Network(arm,state),a,c,probe)['score']-baseline).max()) for arm in arms}
    assert max(errors.values()) < 1e-5, errors
    task.write(DEST/'checks.json',dict(zero_start_errors=errors,source_only_calibration=len(train),calibration_steps=200))
    e.ROOT = DEST
    for arm in arms:
        r = fit(arm,17,rows,train,hold,a,c,state)
        print('CALIBRATED_DONE',arm,r['final'],flush=True)
        e.gc.collect()
        torch.cuda.empty_cache()
    result = []
    for p in DEST.glob('*/receipt.json'):
        r = json.loads(p.read_text())
        result.append(dict(arm=r['arm'],seed=r['seed'],**r['final']))
    pd.DataFrame(result).to_csv(DEST/'summary.csv',index=False)


if __name__ == '__main__':
    main()
