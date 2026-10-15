"""Label-informed amplitude headroom for frozen class-reference metrics."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from sklearn.kernel_ridge import KernelRidge
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
OLD = OUT.parent/'class_reference_metric_pilot_20261014'
spec = importlib.util.spec_from_file_location('frozen_metric_pilot', OLD/'run_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
ALPHAS = [0., .05, .1, .25, .5, 1.]
STAGES = {'initial': OLD, 'matched': OLD/'matched_training'}
VALUES = ['oa', 'ba', 'macro_nll', 'macro_brier']


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def run():
    started = time.perf_counter()
    torch.set_num_threads(2)
    (OUT/'predictions').mkdir(exist_ok=True)
    write('PROTOCOL.json', dict(date='2026-10-14', status='Grid and aggregation fixed before outcomes',
        purpose='Oracle headroom only, labels select alpha on these same queries; not deployable test accuracy',
        alphas=ALPHAS, path='M_alpha=(1-alpha)I+alpha*M_learned; re-solve weighted residual ridge',
        candidates='class_mean/task only, both completed training samplers, all3 training seeds',
        frozen='source geometry2752/prior/reference centers/rho.25/ridge8/full probability fusion.5; no new fits',
        evaluation='All48 existing scenes,5draws,64supports; same/other acquisition separated',
        alpha_zero='Fixed class-reference model, not original visual-kernel model',
        oracle_rules='Global shared alpha and one shared alpha per port; average3seeds5draws, never choose a seed. BA maximization and NLL minimization separately; exact ties choose smaller alpha.',
        primary='Other acquisition, equal scene/draw means within port then equal3ports, mean3seeds',
        secondary='Same-acquisition and per-port curves, seed/draw variation, rescues/harms',
        outside_scope='No selector training, no external data, no main-model promotion, no finer followup grid'))
    d, f, plans, banks, covariance, source = pilot.load_data()
    mapping = {e['episode']: e for e in plans}
    records = json.loads((OLD/'ENROLLMENT.json').read_text(encoding='utf-8'))
    enrollment = [e for e in records if e['role'] == 'held']
    matched = json.loads((STAGES['matched']/'ENROLLMENT.json').read_text(encoding='utf-8'))
    assert enrollment == [e for e in matched if e['role'] == 'held']
    identity = torch.eye(8, dtype=pilot.DTYPE)
    matrices = {}
    for stage, path in STAGES.items():
        for held in sorted({e['held'] for e in enrollment}):
            for seed in pilot.SEEDS:
                name = f"{held.replace(' ', '_')}_class_mean_task_{seed}"
                with np.load(path/'models'/f'{name}.npz') as a:
                    matrices[(stage,held,seed)] = pilot.tensor(a['metric'])
    checks, metrics, events, baselines = [], [], [], []
    number = {}
    solves = 0
    for ei, e in enumerate(enrollment):
        held = e['held']; ti = number.get(held, 0); number[held] = ti+1
        task = pilot.make_task(d,f,mapping[e['episode']],e['query'],e['support'],e['draw'],e['cell'],held,'held')
        key = {k:e[k] for k in ['port','episode','cohort','draw','cell']}
        with torch.no_grad():
            fixed = pilot.forward(task,'class_mean',identity); solves += 1
        fixed_prob = fixed['full'].numpy(); pred0 = fixed_prob.argmax(1)
        for stage,path in STAGES.items():
            cache = path/'predictions'/f"{held.replace(' ', '_')}_{ti:03d}.npz"
            with np.load(cache) as archive:
                np.testing.assert_array_equal(archive['q'],task['q'])
                np.testing.assert_array_equal(archive['s'],task['s'])
                names = archive['arms'].tolist()
                expected_fixed = archive['full'][names.index('class_mean|fixed|-1')]
                np.testing.assert_allclose(fixed_prob,expected_fixed,atol=1e-10,rtol=1e-10)
                checks.append(dict(stage=stage,kind='zero_endpoint',**key,error=float(abs(fixed_prob-expected_fixed).max())))
                original = archive['full'][names.index('original|fixed|-1')]
                expected = {seed:archive['full'][names.index(f'class_mean|task|{seed}')] for seed in pilot.SEEDS}
            baselines.append(dict(stage=stage,**key,**pilot.metrics(task['y'],original)))
            saved = []
            for seed in pilot.SEEDS:
                ps = []
                for alpha in ALPHAS:
                    m = (1-alpha)*identity+alpha*matrices[(stage,held,seed)]
                    with torch.no_grad():
                        result = fixed if alpha == 0 else pilot.forward(task,'class_mean',m)
                    solves += int(alpha != 0)
                    prob = result['full'].numpy(); ps.append(prob)
                    ikey = dict(stage=stage,**key,seed=seed,alpha=alpha)
                    for mode in ['full','expert']:
                        metrics.append(dict(**ikey,mode=mode,**pilot.metrics(task['y'],result[mode].numpy())))
                    pred = prob.argmax(1)
                    for c in np.unique(task['y']):
                        mask = task['y'] == c
                        before,after = pred0[mask] == c,pred[mask] == c
                        events.append(dict(**ikey,class_id=int(c),query_n=int(mask.sum()),support_n=int(task['counts'][c]),
                            rescues=int((~before & after).sum()),harms=int((before & ~after).sum()),
                            recall=100*after.mean(),baseline_recall=100*before.mean()))
                    if alpha == 1:
                        np.testing.assert_allclose(prob,expected[seed],atol=1e-10,rtol=1e-10)
                        checks.append(dict(**ikey,kind='one_endpoint',error=float(abs(prob-expected[seed]).max())))
                    if ti == 0 and seed == pilot.SEEDS[0] and alpha == .25:
                        independent = KernelRidge(alpha=8.,kernel='precomputed').fit(
                            result['ks'].numpy(),task['target'].numpy(),sample_weight=task['root'].numpy()**2)
                        score = task['prior'].numpy()+independent.predict(result['kernel'].numpy())
                        error = float(abs(score-result['scores'].numpy()).max())
                        np.testing.assert_allclose(score,result['scores'].numpy(),atol=1e-10,rtol=1e-10)
                        checks.append(dict(**ikey,kind='interior_independent_krr',error=error))
                saved.append(ps)
            np.savez_compressed(OUT/'predictions'/f"{stage}_{held.replace(' ','_')}_{ti:03d}.npz",
                full=np.array(saved),q=task['q'],s=task['s'],y=task['y'],sy=task['sy'],
                alphas=np.array(ALPHAS),seeds=np.array(pilot.SEEDS))
        if ei % 80 == 0:
            print('PREDICTED',ei,'/',len(enrollment),flush=True)
    for name,rows in [('metrics',metrics),('class_events',events),('checks',checks),('original_baselines',baselines)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    write('COMPUTATION_COMPLETE.json',dict(seconds=time.perf_counter()-started,held_support_cells=len(enrollment),
        expert_solves=solves,training_fits=0,source_fits=0,probability_caches=2*len(enrollment)))
    analyze()


def analyze():
    frame = pd.read_csv(OUT/'metrics.csv')
    full = frame[frame['mode'].eq('full')].copy()
    basekeys = ['stage','cell','port','episode','draw','seed','mode']
    fixed = frame[frame.alpha.eq(0)][basekeys+VALUES]
    paired = frame.merge(fixed,on=basekeys,suffixes=('','_fixed'),validate='many_to_one')
    delta = []
    for v in VALUES:
        paired[v+'_delta'] = paired[v]-paired[v+'_fixed']; delta.append(v+'_delta')
    values = VALUES+delta
    group = ['stage','cell','mode','alpha']
    portseed = paired.groupby(group+['seed','port'])[values].mean().reset_index()
    seedcurve = portseed.groupby(group+['seed'])[values].mean().reset_index()
    portcurve = portseed.groupby(group+['port'])[values].mean().reset_index()
    curve = seedcurve.groupby(group)[values].mean().reset_index()
    sd = seedcurve.groupby(group)[delta].std().reset_index().rename(columns={v:v+'_seed_sd' for v in delta})
    curve = curve.merge(sd,on=group,validate='one_to_one')
    drawcurve = paired.groupby(group+['draw','port'])[values].mean().groupby(group+['draw']).mean().reset_index()
    checkmeans = drawcurve.groupby(group)[values].mean().reset_index().merge(curve,on=group,suffixes=('_draw','_seed'))
    for v in values:
        np.testing.assert_allclose(checkmeans[v+'_draw'],checkmeans[v+'_seed'],atol=1e-12)
    for name,data in [('curve',curve),('curve_by_training_seed',seedcurve),('curve_by_port',portcurve),('curve_by_support_draw',drawcurve)]:
        data.to_csv(OUT/(name+'.csv'),index=False)
    original = pd.read_csv(OUT/'original_baselines.csv').groupby(['stage','cell','port'])[VALUES].mean().reset_index()
    original_mean = original.groupby(['stage','cell'])[VALUES].mean()
    summaries,choices = [],[]
    for (stage,cell),c in curve[curve['mode'].eq('full')].groupby(['stage','cell']):
        ports = portcurve[(portcurve.stage.eq(stage)) & (portcurve.cell.eq(cell)) & portcurve['mode'].eq('full')]
        for criterion in ['ba','macro_nll']:
            ascending = criterion != 'ba'
            best = c.sort_values([criterion,'alpha'],ascending=[ascending,True]).iloc[0]
            for strategy in ['global','per_port']:
                selected=[]; amap={}
                for port,p in ports.groupby('port'):
                    row = p[p.alpha.eq(best.alpha)].iloc[0] if strategy == 'global' else p.sort_values([criterion,'alpha'],ascending=[ascending,True]).iloc[0]
                    selected.append(row);amap[port]=float(row.alpha)
                    choices.append(dict(stage=stage,cell=cell,strategy=strategy,criterion=criterion,port=port,
                                        alpha=float(row.alpha),**{v:row[v] for v in values}))
                avg = pd.DataFrame(selected)[values].mean()
                summaries.append(dict(stage=stage,cell=cell,strategy=strategy,criterion=criterion,
                    alpha_map=json.dumps(amap),**avg.to_dict(),
                    oa_vs_original=avg.oa-original_mean.loc[(stage,cell),'oa'],
                    ba_vs_original=avg.ba-original_mean.loc[(stage,cell),'ba']))
    pd.DataFrame(summaries).to_csv(OUT/'oracle_summary.csv',index=False)
    pd.DataFrame(choices).to_csv(OUT/'oracle_port_choices.csv',index=False)
    events = pd.read_csv(OUT/'class_events.csv')
    np.testing.assert_allclose(events.recall-events.baseline_recall,
                               100*(events.rescues-events.harms)/events.query_n,atol=1e-10)
    keys = ['stage','cell','port','episode','draw','seed','alpha']
    reconstructed = events.groupby(keys).recall.mean().reset_index().merge(full[keys+['ba']],on=keys,validate='one_to_one')
    np.testing.assert_allclose(reconstructed.recall,reconstructed.ba,atol=1e-10)
    write('SUMMARY_CHECKS.json',dict(class_event_rows=len(events),full_metric_reconstructions=len(reconstructed),
        max_ba_error=float(abs(reconstructed.recall-reconstructed.ba).max()),
        seed_draw_aggregation_agrees=True,oracle_not_independent_test=True))
    plot(curve,seedcurve)
    print('GLOBAL CROSS CURVE',flush=True)
    print(curve[curve.cell.eq('other_acquisition') & curve['mode'].eq('full')][
        ['stage','alpha','oa_delta','ba_delta','macro_nll_delta','ba_delta_seed_sd']].to_string(index=False),flush=True)
    print('ORACLE CROSS',flush=True)
    print(pd.DataFrame(summaries).query("cell=='other_acquisition'")[
        ['stage','strategy','criterion','alpha_map','oa_delta','ba_delta','macro_nll_delta','ba_vs_original']].to_string(index=False),flush=True)


def plot(curve,seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs = plt.subplots(2,2,figsize=(10,6.8),sharex=True)
    for col,stage in enumerate(STAGES):
        c=curve[curve.stage.eq(stage)&curve.cell.eq('other_acquisition')&curve['mode'].eq('full')]
        s=seeds[seeds.stage.eq(stage)&seeds.cell.eq('other_acquisition')&seeds['mode'].eq('full')]
        for row,(metric,label) in enumerate([('ba_delta','BA gain (percentage points)'),('macro_nll_delta','Macro NLL change (lower is better)')]):
            ax=axs[row,col]
            for seed,g in s.groupby('seed'):
                ax.plot(g.alpha,g[metric],color='#86a9bb',alpha=.55,linewidth=1)
            ax.plot(c.alpha,c[metric],'-o',color='#176b83',linewidth=2,markersize=4,label='Mean of 3 training seeds')
            ax.axhline(0,color='#555555',linewidth=.8,linestyle='--')
            ax.set_ylabel(label);ax.grid(alpha=.15)
            if row==0:ax.set_title('Initial training' if stage=='initial' else 'Quota-matched training')
            else:ax.set_xlabel('Metric interpolation alpha')
            ax.set_xticks(ALPHAS)
            ax.set_xticklabels(['0','.05','.1','.25','.5','1'],rotation=45,ha='right',fontsize=9)
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Frozen metric amplitude: cross-acquisition diagnostic, 48 scenes / 3 ports')
    fig.text(.5,.012,'Label-informed headroom, not independently evaluated model selection. Thin lines: training seeds.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95))
    fig.savefig(OUT/'alpha_curves.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
