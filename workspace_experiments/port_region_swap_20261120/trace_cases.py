"""Replay selected paired decisions: transport changes are not causal pixel labels."""
from probe import OUT, BASE, runner, late, dump, torch, np, json

@torch.inference_mode()
def main():
    get=lambda n: json.loads((OUT/n).read_text(encoding='utf-8'))
    rows={r['id']:r for r in get('observations_local.json')}
    a,_,_=runner.base.runtime('unified')
    idx={r['id']:i for i,r in enumerate(a.rows)}
    rec=get('banks_local.json')
    banks=[[idx[v['id']] for v in b] for b in rec['banks']]
    net=runner.make_model(1.).eval()
    net.load_state_dict(torch.load(BASE/'causal_work_check_20261119/baseline_s17/best.pt',map_location='cuda',weights_only=True))
    refs=[net.reference(a.data(np.array(b),'small')) for b in banks]
    net.net.matcher.register_forward_pre_hook(lambda mod,args,kw:(args,dict(kw,details=True)),with_kwargs=True)
    cache={}
    def save(mod,args,res):
        cache['T']=[[v['transport'][0].detach().cpu().numpy() for v in tr['regions']] for tr in res[3]]
    net.net.matcher.register_forward_hook(save)
    out=[]
    for e in get('cases_local.json'):
        row=rows[e['id']]; i=row['index']; q=a.data(np.array([i]),'small')
        def run(b,choice=None):
            net.scale_override=choice
            z,_=net(q,[a.rows[i]],refs[b],[a.rows[j] for j in banks[b]])
            return z[0].cpu().numpy(),cache['T']
        pre,t0=run(0)
        if e['axis']=='region':
            choice=net.scale_trace['choice'].clone()
            u,v=map(int,e['pair'].split(':'))
            choice[:,:,u]=0; choice[:,:,v]=1
            pre,t0=run(0,choice)
            choice[:,:,u]=1; choice[:,:,v]=0
            post,t1=run(0,choice)
            expected=row['pairs'][e['pair']][0][::-1]
        else:
            post,t1=run(1); expected=row['soft'][:2]
        err=float(np.abs(np.stack([pre,post])-np.array(expected)).max())
        assert err<.002,err
        # Largest plan-entry changes for the replaced slot and a competing class.
        y=row['label']; rival=int(np.argmax(np.where(np.arange(8)!=y,pre,-np.inf)))
        detail=[]
        for p in (0,1):
            for slot in [rec['slot']]+[k for k,j in enumerate(banks[0]) if a.y[j]==rival and k!=rec['slot']]:
                changes=[(b[slot]-a0[slot]) for a0,b in zip(t0[p],t1[p])]
                reg=int(np.argmax([abs(x).max() for x in changes]))
                x=changes[reg]; j,k=np.unravel_index(abs(x).argmax(),x.shape)
                detail.append(dict(pol=p,refslot=slot,reg=reg,query_slot=int(j),ref_slot=int(k),change=float(x[j,k])))
        out.append(dict(**e,replay_err=err,truth_delta=float(post[y]-pre[y]),rival=rival,
            rival_delta=float(post[rival]-pre[rival]),transport=detail))
    dump('trace_local.json',out)
    print('Traced',len(out),'comparisons, max replay deviation',max(r['replay_err'] for r in out))
if __name__=='__main__': main()
