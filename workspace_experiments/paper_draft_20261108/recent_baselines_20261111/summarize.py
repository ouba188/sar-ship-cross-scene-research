"""Write the completed baseline comparison without touching the manuscript."""
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import ROOT


def main():
    summary=pd.read_csv(ROOT/'summary.csv')
    paired=pd.read_csv(ROOT/'paired_by_seed.csv')
    selection=json.loads((ROOT/'selected.json').read_text(encoding='utf-8'));lr=selection['lr']
    intervals=json.loads((ROOT/'paired_intervals.json').read_text(encoding='utf-8'))
    lines=['# CPINet独立复现：OSS2跨场景结果','',
        '2026-11-11。同一719训练、228验证、378测试；未重跑已有Full/Plain/SE。正式模型与论文不改。',
        'CPINet名称指依据原文独立实现、适配本任务native160和缺极化处理，不是官方权重或原论文分数重现。','',
        '## 主结果','',
        '| 方法 | OA | BA | Macro-F1 | NLL |','|---|---:|---:|---:|---:|']
    for arm in ('CPINet','full','plain','se'):
        r=summary[(summary.arm==arm)&(summary.subset=='all')].iloc[0]
        lines.append(f'| {arm} | {r.oa_mean:.3f} +/- {r.oa_std:.3f} | {r.ba_mean:.3f} +/- {r.ba_std:.3f} | {r.macro_f1_mean:.3f} +/- {r.macro_f1_std:.3f} | {r.nll_mean:.4f} |')
    p=paired[paired.subset=='all']
    lines+=['','均值及样本标准差来自17/29/43三个训练种子，不能把它们当成三个独立目标场景。',
        f'Full减CPINet：OA {p.oa_delta.mean():+.3f}点、BA {p.ba_delta.mean():+.3f}点；OA正向种子{int((p.oa_delta>0).sum())}/3、BA正向{int((p.ba_delta>0).sum())}/3。',
        f'Full相对CPINet救回{int(p.rescued.sum())}、误伤{int(p.harmed.sum())}次，单位为观测×种子事件，不是独立船数。',
        f'船级配对bootstrap 95%区间：OA {intervals["oa"]}；BA {intervals["ba"]}。该区间以现有场景和三种子为条件，不等于对未见港口总体的不确定性。',
        '','## 极化可用性分层','','| 方法 | 双路348 OA/BA | 单路30 OA/BA |','|---|---:|---:|']
    for arm in ('CPINet','full','plain','se'):
        b=summary[(summary.arm==arm)&(summary.subset=='both')].iloc[0]
        s=summary[(summary.arm==arm)&(summary.subset=='single')].iloc[0]
        lines.append(f'| {arm} | {b.oa_mean:.3f} / {b.ba_mean:.3f} | {s.oa_mean:.3f} / {s.ba_mean:.3f} |')
    lines+=['','## 训练与选择','',f'验证集选择学习率{lr:g}；候选为0.01/0.003，seed17各完整300epochs；选中者复用，seed29/43按同一LR各300epochs。未按测试选LR或epoch。',
        '| 种子 | 验证选择epoch | 验证OA/BA | 更新次数 | 样本暴露 | 训练秒数 |','|---|---:|---:|---:|---:|---:|']
    for seed in (17,29,43):
        r=json.loads((ROOT/f'cpinet_s{seed}_lr{lr:g}/complete.json').read_text(encoding='utf-8'))
        b=r['best'];m=b['metrics']
        lines.append(f'| {seed} | {b["epoch"]} | {m["oa"]:.3f} / {m["ba"]:.3f} | {r["steps"]} | {r["exposures"]} | {r["seconds"]:.1f} |')
    lines+=['','CPINet全网从随机初始化训练；Full/Plain/SE为ImageNet起点的分阶段局部更新。数据权限相同，初始化、训练曝光和计算量不同，因此本表是系统比较，不能把差值全归于M1/M2/M3的结构优势。',
        'Full显式使用OBB区域和32个训练域源参考；CPINet复用同一定位裁片后直接处理双极化图像，不使用参考库。训练标签权限相同，额外输入形式不同。',
        '没有把当前程序尚未核实的stem/TL细节称作原论文精确实现。相关选择见CPINET_IMPLEMENTATION_SPEC_ZH.md。',
        '','## 原图与适用范围','',
        '案例入口index.html，分为Full救回、CPINet救回、双方正确和双方错误，按类别保留代表案例；完整逐类与场景表为per_class.csv/per_scene.csv。图像可见事实及解释由主代理另外写入CASE_NOTES_ZH.md。',
        '本轮只完成OSS2固定四分类对照；24港、重做中的OSS1及跨数据集尚未由本次CPINet模型验证，不能据此填数。SCRC-Net未训练。',
        '首次未归一化数值失败及误用C/16的运行只作为实现偏差保留，不进入方法成绩或测试筛选。',
        '代码/协议/汇总结果可归档；原始图像、逐样本预测、权重和本地路径仅留本机。']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for seed in (17,29,43):
        curve=json.loads((ROOT/f'cpinet_s{seed}_lr{lr:g}/curve.json').read_text(encoding='utf-8'))
        axes[0].plot([r['epoch'] for r in curve],[r['train_oa'] for r in curve],label=f'seed {seed}')
        v=[r for r in curve if 'validation' in r]
        axes[1].plot([r['epoch'] for r in v],[r['validation']['ba'] for r in v],label=f'seed {seed}')
    for ax,title in zip(axes,['Balanced augmented training OA','Validation BA (checkpoint selection)']):
        ax.set(xlabel='Epoch',ylabel='Percent',title=title);ax.grid(alpha=.2);ax.legend()
    fig.savefig(ROOT/'training_curves.png',dpi=150);plt.close(fig)
    print('\n'.join(lines[:24]),flush=True)


if __name__=='__main__':main()
