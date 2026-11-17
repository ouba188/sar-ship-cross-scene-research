"""Selected-checkpoint PCEA on/off, without optimization or new selection."""
from pathlib import Path
import importlib.util
import sys
import gc

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ports24_sp_diagnose_runner',ROOT/'run.py')
r=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=r
spec.loader.exec_module(r)


if __name__=='__main__':
    m,inputs,_=r.runtime()
    val=inputs.groups['validation']
    results={}
    for strength in (0.,.1):
        dest=r.old.destination('cross_mean',17,strength)
        net=m.Model(17,'cross_mean')
        net.load_state_dict(m.torch.load(dest/'best.pt',map_location='cpu',weights_only=True),strict=True)
        net.accelerate()
        on=m.full.predict(net,inputs,val,'validation')
        replay=r.h.replay(m,val,on,dest/'validation.npz')
        net.exchange.disabled=True
        off=m.full.predict(net,inputs,val,'validation')
        m.np.savez_compressed(dest/'validation_off.npz',ids=val,label=inputs.y[val],score=off)
        results[str(strength)]=dict(replay=replay,on=m.full.metrics(inputs.y[val],on),
            off=m.full.metrics(inputs.y[val],off),max_difference=float(abs(on-off).max()),
            changed=int((on.argmax(1)!=off.argmax(1)).sum()))
        del net
        gc.collect();m.torch.cuda.empty_cache()
    r.write(ROOT/'diagnostics.json',results)
    print(results,flush=True)
