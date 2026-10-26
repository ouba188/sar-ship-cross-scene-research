"""Check zero-step identity and rebuilt score-dependent inputs, no retraining."""
from run import *


def main():
    rows=[]
    for source,target in [('oss1','oss2'),('oss2','oss1')]:
        direction=f'{source}-to-{target}'
        for seed in SEEDS:
            for branch in ('old','strong'):
                out=HERE/direction/f'seed{seed}'/branch
                sel=json.loads((out/'selection.json').read_text())
                for side, repair_name in [('source',f'source_{source}'),('cross',direction)]:
                    z=load(out/f'{side}_predictions.npz')
                    old=load(REPAIR/repair_name/f'seed{seed}/predictions.npz')
                    rebuilt=gate_raw(old['gate_raw'],z['M2'],z['delta_vv'],z['delta_vh'],old['available'])
                    np.testing.assert_array_equal(rebuilt,z['gate_raw'])
                    np.testing.assert_array_equal(z['gate_raw'][...,11:],old['gate_raw'][...,11:])
                    if sel['M2_step']==0:
                        np.testing.assert_array_equal(z['M2'],z['bare'].astype(np.float32))
                    if sel['joint_step']==0:
                        np.testing.assert_array_equal(z['joint'],z['M2'])
                    for name,step in [('vv',sel['vv_step']),('vh',sel['vh_step'])]:
                        if step==0:
                            np.testing.assert_array_equal(z[name],z['M2'])
                    if sel['frequency_step']==0:
                        np.testing.assert_allclose(z['frequency'],z['B_fixed'],atol=1e-6,rtol=0)
                    rows.append(dict(source=source,seed=seed,branch=branch,side=side,
                        gate_rebuild_max_error=float(abs(rebuilt-z['gate_raw']).max()),
                        frozen_descriptors_unchanged=True,selected_zero_outputs_identity=True))
    write(HERE/'verification.json',dict(checks=rows,passed=True))
    print('PASS: 24 source/cross configurations; zero-step identities and base-dependent gate reconstruction.')


if __name__=='__main__':
    main()
