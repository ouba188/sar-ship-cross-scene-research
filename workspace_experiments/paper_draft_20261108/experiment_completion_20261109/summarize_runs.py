"""Report complete matched runs and original-input outcome cases."""
from pathlib import Path
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import raw_current as raw

ROOT = Path(__file__).resolve().parent
NAMES = ['Bulk carrier', 'General cargo', 'Chemical/product', 'Container']
ZH = ['散货船', '普通货船', '化学品/成品油轮', '集装箱船']


def margin(z, y):
    other = z.copy(); other[y] = -np.inf
    return float(z[y]-other.max())


def panel(rows, ids, before, after, stem, text, references):
    row = rows[ids]; label = row['label']; wrong = int(before.argmax())
    choices = [row]
    for cls in (label, wrong if wrong != label else int(np.argsort(before)[-2])):
        choices.append(next((r for r in references if r['label'] == cls and str(r['mmsi']) != str(row['mmsi'])
                             and raw.task.orbit(r) != raw.task.orbit(row)), None))
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), layout='constrained')
    source = []
    for j, rec in enumerate(choices):
        views = raw.task.data.prepare(rec) if rec else [None, None]
        for p, view in enumerate(views):
            ax = axes[p, j]; ax.axis('off')
            if view is not None:
                ax.imshow(view['pixels'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                if view['valid_target']:
                    cx,cy,length,width,angle=view['geometry'][:5]
                    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*np.array([length,width])/2
                    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                    box=corners@rot.T+np.array([cx,cy])
                    ax.plot(box[:,0],box[:,1],color='#31b99a',linewidth=.8)
            else: ax.text(.5, .5, 'Unavailable', ha='center', va='center')
            title = ('Query' if j == 0 else 'Same-class reference' if j == 1 else 'Competing reference')
            status = '' if view is not None and view['valid_target'] else ' [not used]'
            short_id=rec['id'] if rec else 'none'
            if len(short_id)>35:short_id='...'+short_id[-30:]
            ax.set_title(f'{title} / {"VV" if p == 0 else "VH"}{status}\n{short_id}', fontsize=8)
        if rec: source.append(dict(id=rec['id'], vv=rec.get('vv'), vh=rec.get('vh')))
    fig.suptitle(f'{text}\nTrue: {NAMES[label]} | {NAMES[int(before.argmax())]} -> {NAMES[int(after.argmax())]}\n'
                 f'True-class margin {margin(before,label):+.3f} -> {margin(after,label):+.3f}', fontsize=10)
    fig.savefig(stem.with_suffix('.png'), dpi=160); plt.close(fig)
    return dict(id=row['id'], label=label, comparison=text, before=before.tolist(), after=after.tolist(),
                margin_before=margin(before,label), margin_after=margin(after,label), sources=source,
                reference_selection='first lawful actual source reference in each class; illustration, not attention maximum')


def common():
    rows, groups = raw.prev.records(); refs = [rows[i] for i in groups['references']]
    root = ROOT/'common_init'; metrics = []; arrays = {}; records = {}
    for file in sorted(root.glob('*/receipt.json')):
        r = json.loads(file.read_text(encoding='utf-8')); records[r['arm'],r['seed']] = r
        for split, values in r['metrics'].items():
            metrics.append(dict(arm=r['arm'], seed=r['seed'], split=split, **values,
                                trainable=r['trainable'], parameters=r['parameters'], seconds=r['seconds'], peak_gib=r['peak_gib']))
            if split == 'evaluation':
                d = np.load(file.parent/f'{split}.npz'); arrays[r['arm'],r['seed']] = d['score']
                assert np.array_equal(d['ids'], groups[split])
    df = pd.DataFrame(metrics); df.to_csv(root/'metrics.csv',index=False)
    avg = df.groupby(['split','arm'])[['oa','ba','nll','trainable','parameters','seconds','peak_gib']].agg(['mean','std','count'])
    avg.to_csv(root/'summary.csv')
    e = df[df.split=='evaluation']; seeds = sorted(e.seed.unique()); ids = groups['evaluation']
    y = np.array([rows[i]['label'] for i in ids]); cases=[]; changes=[]; class_rows=[]
    for (arm,seed),z in arrays.items():
        for cls in range(4):
            mask=y==cls
            class_rows.append(dict(arm=arm,seed=seed,label=cls,class_name=ZH[cls],n=int(mask.sum()),
                                   recall=float(100*(z.argmax(1)[mask]==cls).mean())))
    pd.DataFrame(class_rows).to_csv(root/'per_class.csv',index=False)
    gallery=root/'cases'; gallery.mkdir(exist_ok=True)
    for arm in ('plain','no_m1','no_m2','no_m3'):
        paired = [seed for seed in seeds if ('full',seed) in arrays and (arm,seed) in arrays]
        if not paired: continue
        for seed in paired:
            full, base = arrays['full',seed], arrays[arm,seed]
            good, old = full.argmax(1)==y, base.argmax(1)==y
            changes.append(dict(arm=arm,seed=seed,rescued=int((good&~old).sum()),harmed=int((old&~good).sum()),
                                delta_oa=float(100*(good.astype(float)-old).mean()),
                                delta_ba=float(100*np.mean([(good[y==c].astype(float)-old[y==c]).mean() for c in range(4)]))))
        # First predefined seed, with both positive and negative cases preserved.
        seed=paired[0]; full,base=arrays['full',seed],arrays[arm,seed]
        good,old=full.argmax(1)==y,base.argmax(1)==y
        masks={'rescue':good&~old,'harm':old&~good,'stable_correct':good&old,'stable_wrong':~good&~old}
        for outcome,mask in masks.items():
            candidates=np.flatnonzero(mask)
            if not len(candidates):continue
            # Deterministic observation order, no best-looking or largest-effect search.
            i=int(candidates[0]);name=f'{arm}_{outcome}_s{seed}_{ids[i]}'
            detail=panel(rows,int(ids[i]),base[i],full[i],gallery/name,f'{arm} -> full / {outcome} / seed {seed}',refs)
            detail.update(file=f'cases/{name}.png',outcome=outcome,arm=arm,seed=int(seed));cases.append(detail)
    pd.DataFrame(changes).to_csv(root/'paired_changes.csv',index=False)
    raw.task.write(root/'case_records.json',cases)
    lines=['# 共同初始化结构实验结果','', '新训练采用同一ImageNet初始化、统一360步联合优化；不是原分阶段工作版的直接复制。',
           '源719、评价378、四类、固定32源参考；目标评价不参与选轮。','',
           '| 配置 | 重复数 | OA均值±标准差 | BA均值±标准差 |','|---|---:|---:|---:|']
    for arm in ('plain','full','no_m1','no_m2','no_m3'):
        z=e[e.arm==arm]
        if len(z):lines.append(f'| {arm} | {len(z)} | {z.oa.mean():.3f} ± {z.oa.std():.3f} | {z.ba.mean():.3f} ± {z.ba.std():.3f} |')
    lines+=['','删除模块会同时去掉其辅助目标/读出接口，具体定义见PROTOCOL_ZH.md。完整模型若不胜过某一删除臂，不能写该模块在本协议下有正增量。',
            '普通骨干若仍在训练不足范围内，后续只能按源端信息给予合理调参机会，不能据本轮固定预算直接宣称超过强单骨干。',
            '三次训练共用ImageNet预训练，但模块初始化、训练排列及共同可训练卷积从共同起点分别更新；不再是只重复末端读出。']
    lines+=['','## 配对差值（Full减去对照，百分点）','',
            '| 对照 | OA差均值 | BA差均值 | 救回事件 | 误伤事件 |','|---|---:|---:|---:|---:|']
    ch=pd.DataFrame(changes)
    for arm,g in ch.groupby('arm'):
        lines.append(f'| {arm} | {g.delta_oa.mean():+.3f} | {g.delta_ba.mean():+.3f} | {g.rescued.sum()} | {g.harmed.sum()} |')
    lines+=['','救回/误伤按种子累计是预测事件，同一船重复出现，不当作独立船只数量。',
            '原图案例按固定seed17和清单顺序，各取一例救回、误伤、持续正确、持续错误；不按图像是否好看挑选。参考是实际源库中首个合法同类/竞争类样本，用于展示，不等同注意力最大的贡献者。']
    (root/'FINDINGS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    body=''.join(f'<section><h2>{html.escape(c["arm"]+" / "+c["outcome"])}</h2><img src="{c["file"]}" style="max-width:100%"></section>' for c in cases)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Common-init cases</title><h1>Common-initialization paired outcomes</h1><p>Actual native160 inputs. References illustrate the legal source bank; not attention maxima.</p>'+body,encoding='utf-8')
    print('\n'.join(lines),flush=True)


def transfer():
    root=ROOT/'oss1_transfer';path=root/'predictions.npz'
    if not path.exists():return
    rows=raw.task.data.read_manifest(root/'target_manifest.json')
    d=np.load(path);assert d['ids'].tolist()==[r['id'] for r in rows]
    y=np.array([r['label'] for r in rows]);metrics=[];parts=[]
    for key in d.files:
        if key=='ids':continue
        metrics.append(dict(model=key,**raw.task.metrics(y,d[key])))
        for c in range(4):parts.append(dict(model=key,label=c,n=int((y==c).sum()),recall=float(100*(d[key].argmax(1)[y==c]==c).mean())))
    pd.DataFrame(metrics).to_csv(root/'summary.csv',index=False)
    pd.DataFrame(parts).to_csv(root/'classes.csv',index=False)
    independent=raw.task.data.read_manifest(root/'target_manifest_no_oss2_eval_overlap.json')
    keep={r['id'] for r in independent}
    cohorts={'historical921':np.ones(len(rows),dtype=bool),
             'no_oss2_evaluation_ship_overlap':np.array([r['id'] in keep for r in rows])}
    report=[]
    for cohort,mask in cohorts.items():
        for arm in ('class','mean','content','no_m3_refit'):
            vals=[raw.task.metrics(y[mask],d[f'{arm}_s{seed}'][mask]) for seed in (17,29,43)]
            report.append(dict(cohort=cohort,arm=arm,n=int(mask.sum()),
                oa=float(np.mean([v['oa'] for v in vals])),oa_std=float(np.std([v['oa'] for v in vals],ddof=1)),
                ba=float(np.mean([v['ba'] for v in vals])),ba_std=float(np.std([v['ba'] for v in vals],ddof=1)),
                nll=float(np.mean([v['nll'] for v in vals]))))
    pd.DataFrame(report).to_csv(root/'cohort_summary.csv',index=False)
    lines=['# 当前工作版OSS2到OSS1冻结迁移','',
           '原图前向；不训练、不更改源参考、不使用目标标签选择检查点。三次结果是末端读出训练重复，共用上游seed17。',
           'historical921沿用已有队列；另一子集排除与OSS2评价集同MMSI的71条。两者嵌套，不是两次独立验证。','',
           '| 队列 | 方法 | 数量 | OA | BA | NLL |','|---|---|---:|---:|---:|---:|']
    for r in report:lines.append(f'| {r["cohort"]} | {r["arm"]} | {r["n"]} | {r["oa"]:.3f} ± {r["oa_std"]:.3f} | {r["ba"]:.3f} ± {r["ba_std"]:.3f} | {r["nll"]:.4f} |')
    upstream=raw.task.metrics(y,d['upstream'])
    lines += ['', f'继承上游单次模型的OA/BA为{upstream["oa"]:.3f}/{upstream["ba"]:.3f}。当前末端组合未超过该上游，不能把相对局部替换的收益写成相对所有旧版本的整体升级。']
    lines+=['','mean/content保留其他模块；no_m3_refit为源端末阶段配套移除对照。它们不是Plain R50，也不是完整流程重新训练消融。',
            '评价标签仅用于本表与后验案例整理；未用于预测。原图与源参考见case_records.json及index.html。']
    (root/'FINDINGS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    source,groups=raw.prev.records();refs=[source[i] for i in groups['references']]
    folder=root/'cases';folder.mkdir(exist_ok=True);cases=[]
    for arm in ('mean','content','no_m3_refit'):
        before,after=d[f'{arm}_s17'],d['class_s17']
        old,good=before.argmax(1)==y,after.argmax(1)==y
        for outcome,mask in dict(rescue=good&~old,harm=old&~good,stable_correct=old&good,stable_wrong=~old&~good).items():
            ids=np.flatnonzero(mask)
            if len(ids):
                i=int(ids[0]);name=f'{arm}_{outcome}_{i}'
                rec=panel(rows,i,before[i],after[i],folder/name,f'{arm} -> full / {outcome} / seed17',refs)
                rec.update(file=f'cases/{name}.png',outcome=outcome,arm=arm);cases.append(rec)
    raw.task.write(root/'case_records.json',cases)
    body=''.join(f'<h2>{html.escape(c["arm"]+" / "+c["outcome"])}</h2><img src="{c["file"]}" style="max-width:100%">' for c in cases)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>OSS1 transfer cases</title><h1>OSS2 to OSS1 frozen transfer</h1>'+body,encoding='utf-8')


if __name__=='__main__':
    common();transfer()
