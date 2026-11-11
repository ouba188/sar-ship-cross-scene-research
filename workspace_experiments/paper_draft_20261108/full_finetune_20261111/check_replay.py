"""Locate live-prefix replay discrepancies before fitting."""
import train as t

inputs=t.Inputs()
for split in ('small','validation'): inputs.prepare_geometry(split)
ids=inputs.groups['validation'][:12]
net=t.LiveFull(17,'local')
old_a,old_c=t.staged.prev.arrays()
old_a['write']=t.np.load(t.PAPER/'experiment_completion_20261109/common_init/write.npy',mmap_mode='r')
report={}
for precision in ('high','highest'):
    t.torch.set_float32_matmul_precision(precision)
    t.torch.backends.cudnn.allow_tf32=precision!='highest'
    with t.torch.no_grad():
        data=inputs.data(ids,'validation')
        a,c=net.extract(data,True)
        errors={k:float((v-t.cost.resident_tensor(old_a[k],ids)).abs().max())
                for k,v in a.items()}
        errors.update({k:float((v-t.cost.resident_tensor(old_c[k],ids)).abs().max()) for k,v in c.items()})
        z=t.predict(net,inputs,ids,'validation')
        old=net.net(old_a,old_c,inputs.rows,ids,inputs.groups['references'])[0].cpu().numpy()
        errors['live_vs_cached_same_precision']=float(abs(z-old).max())
        expected=t.np.load(t.PAPER/'staged_ablation_20261109/full_s17_lr1/validation.npz')['score'][:12]
        errors['cached_vs_record']=float(abs(old-expected).max())
        errors['live_vs_record']=float(abs(z-expected).max())
    report[precision]=errors
    print(precision,errors,flush=True)
t.write(t.ROOT/'replay_diagnosis.json',report)
