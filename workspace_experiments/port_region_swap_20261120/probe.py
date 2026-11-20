"""Frozen source diagnostic. Private input manifests/checkpoints are not distributed."""
from pathlib import Path
import sys
import csv
import json
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
WORK = OUT.parent
BASE = WORK/'port_coverage_pilot_20261116/pcea_intervention_transfer_20261118'
sys.path.insert(0, str(BASE))
import causal_run as runner
from unified_model import late

def passage(r):
    return '_'.join(r['pass_group'].split('_')[:2])

def dump(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

@torch.inference_mode()
def main():
    torch.manual_seed(20261120)
    a, fit, inner = runner.base.runtime('unified')
    rows = a.rows
    prior = BASE/'causal_work_check_20261119'
    inner = np.setdiff1d(inner, np.load(prior/'selection_local.npz')['validation'])
    score = np.load(prior/'baseline_inner_local.npz')
    expected = dict(zip(score['ids'],score['scores']))
    net = runner.make_model(1.).eval()
    net.load_state_dict(torch.load(prior/'baseline_s17/best.pt',map_location='cuda',weights_only=True))
    with (WORK/'port_class_prototypes_20261120/source_observations_local.csv').open(encoding='utf-8-sig') as f:
        meta = {int(r['index']):r for r in csv.DictReader(f)}
    refs = np.asarray(a.groups['references'])
    slot = next(k for k,i in enumerate(refs) if a.y[i]==3)
    banks=[refs.copy()]
    used={rows[i]['mmsi'] for i in refs}
    seenpass=set()
    candidates=[i for i in fit if a.y[i]==3 and meta[i]['valid']=='True'
                and meta[i]['port']=='Houston' and rows[i]['vv'] and rows[i]['vh']]
    for i in sorted(candidates,key=lambda i:(abs(float(meta[i]['ais_length_m'])-183),i)):
        r=rows[i]
        if r['mmsi'] in used or passage(r) in seenpass: continue
        new=refs.copy(); new[slot]=i
        banks.append(new); used.add(r['mmsi']); seenpass.add(passage(r))
        if len(banks)==3: break
    assert len(banks)==3
    rr=[rows[i] for i in set(np.concatenate(banks))]
    passages={passage(r) for r in rr}; vessels={r['mmsi'] for r in rr}
    refs_encoded=[]
    for bank in banks:
        d=a.data(bank,'small')
        assert d['available'][slot].bool().all()
        refs_encoded.append(net.reference(d))
    queries=[]; selected=set(); rng=np.random.default_rng(20261120)
    for i in rng.permutation(inner):
        r=rows[i]
        if a.y[i] not in (3,4) or r['mmsi'] in vessels|selected or passage(r) in passages: continue
        selected.add(r['mmsi']); queries.append(i)
    paths=lambda r: {p:(r[p]['image'] if r[p] else None) for p in ('vv','vh')}
    def describe(i):
        r=rows[i]
        return dict(id=r['id'],mmsi=r['mmsi'],port=r['port'],passage=passage(r),label=int(a.y[i]),
                    length=meta.get(i,{}).get('ais_length_m'),**paths(r))
    dump('banks_local.json',dict(slot=slot,banks=[[describe(i) for i in b] for b in banks]))
    out=[]; replay=0.; numeric=0.; groups=late.base.selection.GROUPS
    for off in range(0,len(queries),6):
        ids=queries[off:off+6]; qr=[rows[i] for i in ids]; d=a.data(np.asarray(ids),'small')
        def forward(bank, override=None):
            net.scale_override=override
            z,_=net(d,qr,refs_encoded[bank],[rows[i] for i in banks[bank]])
            return z.cpu().numpy(), net.scale_trace['active_tokens'].clone()
        base,_=forward(0)
        replay=max(replay,float(np.max(np.abs(base-np.stack([expected[i] for i in ids])))))
        assert replay < .002, replay
        original=net.scale_trace['choice'].clone(); eligible=net.scale_trace['eligible']
        available=d['available'].bool()
        mass=d['read40'].sum((-1,-2))>0
        modes={}; soft=[base]+[forward(b)[0] for b in (1,2)]
        for u,v in ((0,1),(2,6)):
            # Require both polarizations and equal numbers of populated children in each.
            ok=(available & eligible[:,:,u] & eligible[:,:,v] &
                (mass[:,:,list(groups[u])].sum(-1)==mass[:,:,list(groups[v])].sum(-1))).all(1)
            aa=original.clone(); aa[:,:,u]=1; aa[:,:,v]=0
            bb=original.clone(); bb[:,:,u]=0; bb[:,:,v]=1
            conditions=[]
            for b in range(3):
                za,ta=forward(b,aa); zb,tb=forward(b,bb)
                assert torch.equal(ta[ok],tb[ok])
                conditions.append([za,zb])
            modes[f'{u}:{v}']=(ok.tolist(),conditions)
        repeated,_=forward(0)
        numeric=max(numeric,float(np.abs(base-repeated).max()))
        net.scale_override=None
        for j,i in enumerate(ids):
            out.append(dict(**describe(i),index=int(i),soft=[z[j].tolist() for z in soft],
                pairs={k:[[za[j].tolist(),zb[j].tolist()] for za,zb in cond]
                       for k,(ok,cond) in modes.items() if ok[j]}))
    dump('observations_local.json',out)
    dump('scope.json',dict(query_count=len(out),ports=len({r['port'] for r in out}),
        passages=len({r['passage'] for r in out}),class_counts={str(c):sum(r['label']==c for r in out) for c in (3,4)},
        bank_lengths=[float(meta[b[slot]]['ais_length_m']) for b in banks],
        replay_max=replay,repeat_max=numeric,seed=20261120))
    print('Finished',len(out),'queries, replay',replay,'repeat',numeric)

if __name__=='__main__': main()
