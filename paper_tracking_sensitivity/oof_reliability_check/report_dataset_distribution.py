"""Build a Chinese distribution report and data-derived PNG/PDF figures."""

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from analyze_dataset_distribution import OUT, ORDER, ROOT, SWEEP


DISPLAY = {'self24ports': '自建24港（23个目标港）', 'F0921': 'F_0921', 'OSS1': 'OpenSARShip 1.0',
           'OSS2': 'OpenSARShip 2.0', 'pacs': 'PACS', 'officehome': 'Office-Home'}
EN = {'self24ports': '24-port corpus (23 targets)', 'F0921': 'F_0921', 'OSS1': 'OpenSARShip 1.0',
      'OSS2': 'OpenSARShip 2.0', 'pacs': 'PACS', 'officehome': 'Office-Home'}
SAR_ZH = ['散货船', '渔船', '杂货船', '成品油/化学品船', '集装箱船', '原油轮', '拖轮/拖带船', '海工补给船']
SAR_EN = ['Bulk carrier', 'Fishing vessel', 'General cargo', 'Product/chemical tanker',
          'Container ship', 'Crude oil tanker', 'Tug/towing', 'Offshore supply']
COLORS = ['#247BA0', '#3E9676', '#D99327', '#C65963', '#8071A8', '#57899A', '#7C8740', '#AD705B']


def read(name):
    return pd.read_csv(OUT/f'{name}.csv')


def integer(x):
    return f'{int(x):,}'


def span(a, b):
    return str(int(a)) if a == b else f'{int(a)}–{int(b)}'


class Document:
    def __init__(self):
        self.md, self.web = [], []

    def heading(self, title, level=2):
        self.md.append('#'*level + ' ' + title)
        self.web.append(f'<h{level}>{html.escape(title)}</h{level}>')

    def paragraph(self, value):
        self.md.append(value)
        self.web.append('<p>'+html.escape(value)+'</p>')

    def table(self, headers, rows):
        cells = [[str(v) for v in row] for row in rows]
        self.md.append('\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                                 ['| '+' | '.join(row)+' |' for row in cells]))
        self.web.append('<div class="table-wrap"><table><thead><tr>' +
                        ''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+
                        ''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in cells)+
                        '</tbody></table></div>')

    def figure(self, name, caption):
        path = OUT/'figures'/f'{name}.png'
        self.md.append(f'![{caption}]({path.as_posix()})\n\n{caption}')
        payload = base64.b64encode(path.read_bytes()).decode('ascii')
        self.web.append(f'<figure><img src="data:image/png;base64,{payload}" alt="{html.escape(caption)}">'
                        f'<figcaption>{html.escape(caption)}</figcaption></figure>')

    def link(self, title, path):
        path = Path(path)
        self.md.append(f'[{title}]({path.as_posix()})')
        self.web.append(f'<p><a href="{html.escape(path.as_uri())}">{html.escape(title)}</a></p>')

    def save(self):
        (OUT/'DATASET_DISTRIBUTION_REPORT_ZH.md').write_text('\n\n'.join(self.md)+'\n', encoding='utf-8')
        style = '''body{font-family:"Microsoft YaHei",Arial,sans-serif;color:#242b30;max-width:1180px;margin:36px auto;padding:0 24px;line-height:1.75;background:white}
        h1{font-size:29px;border-bottom:3px solid #247BA0;padding-bottom:12px}h2{font-size:23px;margin-top:38px}h3{font-size:18px;margin-top:26px}
        p{margin:14px 0}table{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0}th,td{border:1px solid #dbe1e4;padding:7px 9px;text-align:right}
        th{background:#edf3f5}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f8fafb}.table-wrap{overflow:auto}
        figure{margin:24px 0}img{max-width:100%;height:auto}figcaption{color:#52616a;font-size:13px}a{color:#176c91}
        @page{size:A4;margin:15mm 12mm} @media print{body{max-width:none;margin:0;padding:0;font-size:10pt}h2,h3{break-after:avoid}tr{break-inside:avoid}thead{display:table-header-group}img{max-height:230mm;object-fit:contain}figure{break-inside:avoid}.table-wrap{overflow:visible}th,td{padding:4px;font-size:8pt;overflow-wrap:anywhere}a{color:inherit;overflow-wrap:anywhere}}'''
        page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>六数据集任务与分布报告</title><style>'+style+'</style></head><body>'+''.join(self.web)+'</body></html>'
        (OUT/'DATASET_DISTRIBUTION_REPORT_ZH.html').write_text(page, encoding='utf-8')


def figures(summary, classes, task_classes, task, draws, allports):
    destination = OUT/'figures'
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    with PdfPages(destination/'DATA_DISTRIBUTION_FIGURES.pdf') as book:
        def save(fig, name):
            fig.savefig(destination/f'{name}.png', dpi=200, bbox_inches='tight')
            fig.savefig(destination/f'{name}.pdf', bbox_inches='tight')
            book.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
        for ax, dataset in zip(axes.flat, ORDER[:4]):
            part = classes.loc[classes.dataset.eq(dataset)].sort_values('class_id')
            y = np.arange(8)
            ax.barh(y, part.share*100, color=COLORS, height=.65)
            ax.set_yticks(y, SAR_EN)
            ax.invert_yaxis()
            ax.set_xlim(0, 80)
            ax.set_xlabel('Target sample share (%)')
            ax.set_title(f'{EN[dataset]} | n={int(part["count"].sum()):,}', loc='left', fontweight='bold')
            for i, row in enumerate(part.itertuples()):
                ax.text(row.share*100+.7, i, f'{row.share*100:.1f}% ({row.count:,})', va='center', fontsize=9)
            ax.grid(axis='x', alpha=.16)
            ax.set_axisbelow(True)
        fig.suptitle('SAR target cohorts: same eight labels, different class proportions', fontsize=15)
        save(fig, '01_sar_class_distribution')

        matrix = allports.pivot(index='port', columns='class_id', values='count').fillna(0).sort_index()
        pct = matrix.div(matrix.sum(axis=1), axis=0)*100
        fig, ax = plt.subplots(figsize=(12, 11), layout='constrained')
        im = ax.imshow(pct, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
        ax.set_yticks(np.arange(len(matrix)), [f'{p}  |  n={int(matrix.loc[p].sum()):,}'+(' *' if p == 'Melbourne' else '') for p in matrix.index])
        ax.set_xticks(np.arange(8), ['Bulk', 'Fishing', 'General\ncargo', 'Product/\nchemical', 'Container', 'Crude oil', 'Tug', 'Offshore'])
        for i in range(len(matrix)):
            for j in range(8):
                v = pct.iloc[i, j]
                ax.text(j, i, f'{v:.1f}' if v else '0', ha='center', va='center', fontsize=8, color='white' if v > 55 else '#222222')
        ax.set_title('All 24 ports: within-port class share (%)\n* Melbourne has 64 samples; no query rows remain after 64-label support.', loc='left', pad=18)
        fig.colorbar(im, ax=ax, shrink=.55, label='Within-port share (%)')
        save(fig, '02_port_class_heatmap')

        p = task_classes.loc[task_classes.dataset.eq('pacs')]
        mat = p.pivot(index='class_id', columns='task', values='target_count')
        class_names = summary['class_names']['pacs']
        fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
        for i, domain in enumerate(mat.columns):
            x = np.arange(7)+(i-1.5)*.2
            bars = ax.bar(x, mat[domain], width=.19, label=domain, color=COLORS[i])
            ax.bar_label(bars, padding=2, fontsize=8)
        ax.set_xticks(np.arange(7), [x.replace('_', ' ') for x in class_names])
        ax.set_ylabel('Target samples')
        ax.set_ylim(0, mat.to_numpy().max()*1.2)
        ax.legend(ncol=4, loc='upper center')
        ax.set_title('PACS: seven labels in every target domain', loc='left', fontweight='bold')
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        save(fig, '03_pacs_domain_classes')

        p = task_classes.loc[task_classes.dataset.eq('officehome')]
        mat = p.pivot(index='class_id', columns='task', values='target_count').sort_index()
        labels = summary['class_names']['officehome']
        fig, axes = plt.subplots(1, 2, figsize=(14, 15), gridspec_kw={'width_ratios': [1.3, 1]}, layout='constrained')
        im = axes[0].imshow(mat, cmap='YlGnBu', aspect='auto', vmin=0, vmax=mat.to_numpy().max())
        axes[0].set_yticks(np.arange(65), [f'{i:02d} {x.replace("_", " ")}' for i, x in enumerate(labels)], fontsize=8)
        axes[0].set_xticks(np.arange(4), mat.columns, fontsize=10)
        for i in range(65):
            for j in range(4):
                v = mat.iloc[i, j]
                axes[0].text(j, i, str(int(v)), ha='center', va='center', fontsize=7, color='white' if v > 68 else '#222222')
        axes[0].set_title('Counts by class and domain', loc='left', pad=15)
        total = mat.sum(axis=1).sort_values()
        axes[1].barh(np.arange(65), total, color='#3E9676', height=.75)
        axes[1].set_yticks(np.arange(65), [labels[i].replace('_', ' ') for i in total.index], fontsize=8)
        axes[1].set_xlabel('Samples across four domains')
        axes[1].set_title('All 65 labels, ordered by count', loc='left', pad=15)
        fig.suptitle('Office-Home: local evaluated cohort, 15,588 samples / 65 classes', fontsize=15)
        save(fig, '04_officehome_class_distribution')

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
        for j, dataset in enumerate(ORDER):
            part = draws.loc[draws.dataset.eq(dataset)]
            coverage = part.support_present_classes / part.target_present_classes*100
            unknown = part.query_share_in_support_unseen_classes*100
            x = j + np.linspace(-.19, .19, len(part))
            for ax, values in zip(axes, [coverage, unknown]):
                ax.scatter(x, values, s=17, alpha=.5, color=COLORS[j])
                ax.plot([j-.24, j+.24], [values.mean()]*2, color='#222222', linewidth=2)
                ax.text(j, min(104, values.max()+4), f'mean {values.mean():.1f}%', ha='center', fontsize=9)
        for ax in axes:
            ax.set_xticks(np.arange(6), ['24 ports', 'F_0921', 'OSS1', 'OSS2', 'PACS', 'Office-\nHome'])
            ax.set_ylim(-3, 110)
            ax.grid(axis='y', alpha=.15)
            ax.set_axisbelow(True)
        axes[0].set_ylabel('Observed support classes / present target classes (%)')
        axes[0].set_title('Class coverage of 64 labels', loc='left', fontweight='bold')
        axes[1].set_ylabel('Query share from classes absent in support (%)')
        axes[1].set_title('Query mass without same-class support', loc='left', fontweight='bold')
        fig.suptitle('Each dot is one task / support draw; these are not independent training runs', fontsize=12)
        save(fig, '05_support_class_coverage')


def build_report(summary, datasets, classes, task_classes, task, draws, support, allports):
    doc = Document()
    ds = datasets.set_index('dataset')
    doc.heading('六数据集分类任务与数据分布详细报告', 1)
    doc.paragraph('日期：2026-10-08。对象：上一轮固定联合模型交接预算实验的实际样本，而非数据集介绍页中的全量规模。统计直接读取已冻结策略、原始样本清单及本地标签元数据，不训练模型、不改变划分、不重选预算。')
    doc.heading('一、先回答：分别是几分类？')
    doc.paragraph('自建24港、F_0921、OpenSARShip 1.0和OpenSARShip 2.0均使用相同的8类船型输出；PACS为7分类；Office-Home为完整65分类，不是历史上另一个8类子集。四个SAR集合汇总均实际含8类，但自建数据的单个目标港只出现3–8类，模型仍保留8个输出。')
    doc.table(['数据集', '模型输出类别', '目标任务', '实际目标样本', '每任务实际类别', '每轮全部任务查询数', '源训练样本/任务'], [
        [DISPLAY[d], int(ds.loc[d].model_classes), int(ds.loc[d].tasks), integer(ds.loc[d].total),
         span(ds.loc[d].target_classes_per_task_min, ds.loc[d].target_classes_per_task_max),
         integer(ds.loc[d].query_rows_per_all_tasks_draw), span(ds.loc[d].source_rows_min, ds.loc[d].source_rows_max)] for d in ORDER])
    doc.paragraph('“目标样本”含64个支持样本和其余查询样本。每个目标任务每轮随机取64个标签，余下用于OA/BA评价；五次抽样分别运行，不能说单次方法使用320标签，也不能把五轮查询相加后当作不同样本总量。自建24港完整清单有38,091条；Melbourne只有64条，抽完支持集后无查询，因此本轮23个目标港共38,027条。Melbourne并非从源域语料中统一删除。')
    doc.paragraph('源训练数是对应目标任务的实际源样本规模，源域之间重复使用不能累加成新样本。SAR外部迁移来自同一自建源语料并按既有身份/产品规则排除重叠；PACS和Office-Home每次使用其余三个域为源域，轮流留出一个目标域。')
    doc.heading('当前实验子集如何得到', 3)
    doc.table(['数据集', '映射/评价前集合', '本轮保留', '口径说明'], [
        ['自建24港', '38,091条、24港', '38,027条、23目标港', 'Melbourne的64条不足以同时留出64支持和非空查询'],
        ['F_0921', '165,880条既有clean候选', '123,489条', '按class_raw细粒度映射；不是使用165,880个目标标签'],
        ['OpenSARShip 1.0', '3,030条本地原生GRDH记录', '1,349条', '剔除粗粒度、非支持及歧义标签；SLC不进入本轮目标评价'],
        ['OpenSARShip 2.0', '9,120条本地原生清单记录', '3,745条', '细粒度8类兼容子集；不是历史5,006片特征缓存'],
        ['PACS', '本地Parquet的9,991条', '全部9,991条', '保留7类、4域'],
        ['Office-Home', '本地Parquet的15,588条', '全部15,588条', '保留65类、4域']])
    doc.paragraph('OSS1的1,681条未纳入GRDH记录包括粗标签1,384、非支持标签229、歧义标签68；OSS2的5,375条未纳入记录包括粗标签3,997、非支持标签1,224、歧义标签154。Cargo不强行当作杂货船，Tanker也不自动当作原油轮。两者最终队列均有8类。F_0921源身份排除使用映射前候选集合，因此源划分文件的165,880与本轮目标123,489不矛盾。详细来源和代码行在同目录标签协议核查中。')
    doc.heading('二、类别空间与评价口径')
    doc.table(['类别ID', '英文标签', '中文含义'], [[i, name, SAR_ZH[i]] for i, name in enumerate(summary['class_names']['sar'])])
    doc.paragraph('SAR输出类别固定为上表8项。目标港中未出现的类不会被当作该港BA的零召回项，但模型若将样本预测到这些类仍算错误。BA是在该轮查询集中实际出现的类别上平均召回率；OA是该轮全部查询样本的正确比例。')
    doc.paragraph('公式：OA = Σc (nc / N) × Recallc；BA = (1 / Kpresent) × Σc Recallc。先对同一任务的五次支持集抽样取均值，再对港口/域等权平均。因此，以下“样本分布”可汇总计数，但不能用汇总样本数加权替代原实验的等任务准确率。')
    doc.paragraph('例如Mombasa目标集合有7类，其中某些类仅1个样本；它被抽入支持集后，该轮查询可能只剩6类。本轮单港查询实际类别范围见逐港表。固定8维输出、目标实际类别数、64标签覆盖类别数是三个不同概念。')
    doc.heading('三、四个SAR目标集合的类别分布')
    doc.table(['类别', '24港完整清单', '本轮23目标港', 'F_0921', 'OpenSARShip 1.0', 'OpenSARShip 2.0'], [
        [SAR_ZH[c], integer(summary['corpus_counts'][c])] +
        [f'{int(classes.loc[(classes.dataset==d)&(classes.class_id==c), "count"].iloc[0]):,} ({100*classes.loc[(classes.dataset==d)&(classes.class_id==c), "share"].iloc[0]:.2f}%)' for d in ORDER[:4]] for c in range(8)])
    doc.figure('01_sar_class_distribution', '图1：四个实际SAR目标集合的类别比例与样本数。23港按样本汇总，仅用于描述数据构成，不是实验准确率的汇总方式。')
    doc.table(['数据集', '最大类比例', '最小非空类样本', '最大/最小类数量比'], [
        [DISPLAY[d], f'{100*ds.loc[d].largest_class_share:.2f}%', integer(ds.loc[d].min_present_class), f'{ds.loc[d].max_min_ratio:.2f}:1'] for d in ORDER])
    doc.paragraph('OpenSARShip 1.0是当前六组中汇总类别失衡最突出的：散货船816/1,349，占60.49%，海工补给船仅12条，最大/最小类比68:1。F_0921数量大，但不等于类别均衡：集装箱船30,638条，海工补给船2,074条，数量比14.77:1。')
    doc.paragraph('自建源语料中渔船占比与F_0921不同，不能把“都为8分类”理解为相同类别先验。下面给出三个外部SAR目标对应的实际源/目标比例；源比例根据各自排除重叠后的源集合计算。差异描述标签构成，不单独证明影像域偏移的成因。')
    external = task_classes.loc[task_classes.dataset.isin(['F0921', 'OSS1', 'OSS2'])]
    doc.table(['类别', 'F_0921 源→目标', 'OSS1 源→目标', 'OSS2 源→目标'], [
        [SAR_ZH[c]] + [f'{100*external.loc[(external.dataset==d)&(external.class_id==c), "source_share"].iloc[0]:.2f}% → {100*external.loc[(external.dataset==d)&(external.class_id==c), "target_share"].iloc[0]:.2f}%' for d in ['F0921', 'OSS1', 'OSS2']] for c in range(8)])
    doc.heading('四、自建24港：逐港规模与类别组成')
    doc.paragraph('23个可评估目标港中，8个港有3类、1个港有6类、2个港有7类、12个港有8类。三类港均只出现散货船、渔船、杂货船；对应任务的源训练集合仍全部含8类。论文宜写“统一8类输出空间下的跨港识别”，而不是“24个港都具有完整8类”。')
    doc.figure('02_port_class_heatmap', '图2：全部24港的港内类别占比；零表示该港清单中没有此类。Melbourne保留展示，但不参与本轮目标查询评价。')
    portcounts = allports.pivot(index='port', columns='class_id', values='count').fillna(0).astype(int)
    porttask = task.loc[task.dataset.eq('self24ports')].set_index('task')
    rows = []
    for port, cc in portcounts.iterrows():
        if port in porttask.index:
            t = porttask.loc[port]
            rows.append([port, integer(cc.sum()), int((cc>0).sum()), integer(t.query_rows),
                         span(t.query_classes_min, t.query_classes_max), f'{100*t.largest_class_share:.2f}%',
                         f'{t.support_classes_mean:.2f}', f'{100*t.support_fraction:.2f}%'])
        else:
            rows.append([port, integer(cc.sum()), int((cc>0).sum()), '0', '不评价', f'{100*cc.max()/cc.sum():.2f}%', '64条全部支持', '100%'])
    doc.table(['港口', '样本数', '实际类别', '每轮查询', '查询类别', '最大类比例', '支持覆盖类均值', '64标签/目标样本'], rows)
    doc.paragraph('港口规模从149条（可评估目标中最小的Mombasa）到5,756条（Ningbo-Zhoushan）不等。固定64标签对应的目标标注比例从约1.11%到42.95%，并非每港相同标注百分比。Port Hedland最大类占91.15%，其港内最大/最小非空类数量比391.5:1；“23港汇总看起来较平衡”不能代表各港都平衡。')
    doc.heading('逐港完整类别计数', 3)
    doc.table(['港口']+SAR_ZH+['合计'], [[p]+[integer(v) for v in row]+[integer(row.sum())] for p, row in portcounts.iterrows()])
    doc.heading('五、PACS与Office-Home的域分布')
    doc.paragraph('这两组是RGB跨域分类验证，两个分支读取同一RGB图像的不同编码器表征，不是VV/VH双极化数据。当前分布统计保持原始类别ID；PACS包含7类，Office-Home保留全部65类。')
    generic = task.loc[task.dataset.isin(['pacs', 'officehome'])]
    doc.table(['数据集', '目标域', '目标样本', '源样本', '每轮查询', '类别数', '最大类比例', '支持覆盖类均值'], [
        [DISPLAY[r.dataset], r.task, integer(r.total), integer(r.source_rows), integer(r.query_rows), r.present_classes,
         f'{100*r.largest_class_share:.2f}%', f'{r.support_classes_mean:.2f}'] for r in generic.itertuples()])
    doc.figure('03_pacs_domain_classes', '图3：PACS四个目标域的7类计数。总体失衡较小，但sketch域中最大/最小类仍达到10.2:1。')
    p = task_classes.loc[task_classes.dataset.eq('pacs')].pivot(index='class_id', columns='task', values='target_count')
    doc.table(['ID', '类别']+list(p.columns)+['合计'], [[i, summary['class_names']['pacs'][i]]+[integer(v) for v in row]+[integer(row.sum())] for i, row in p.iterrows()])
    doc.figure('04_officehome_class_distribution', '图4：Office-Home实际加载的65类在四个域中的数量，以及汇总类别规模。所有65类在每个目标域均存在。')
    doc.paragraph('Office-Home的15,588是本地实际加载并参与本轮实验的条目数，不据此断言所有公开发布版本均为同一规模。完整65类名称及各域计数列于附录；数据相对均衡，不意味着64个随机标签能覆盖65类。')
    doc.heading('六、64个目标标签实际覆盖了什么？')
    doc.table(['数据集', '任务×抽样', '支持覆盖类均值', '覆盖范围', '覆盖全部目标类的抽样', '查询中支持未见类的比例'], [
        [DISPLAY[d], int(ds.loc[d].task_draw_contexts), f'{ds.loc[d].support_classes_mean:.2f}',
         span(ds.loc[d].support_classes_min, ds.loc[d].support_classes_max),
         f'{int(ds.loc[d].support_complete_draws)}/{int(ds.loc[d].task_draw_contexts)}',
         f'{100*ds.loc[d].unseen_query_share_taskmean:.2f}%'] for d in ORDER])
    doc.figure('05_support_class_coverage', '图5：每个任务/抽样的64标签类别覆盖率，以及查询集中支持未见类的样本比例。横线为等任务、等抽样均值；点不是独立网络训练。')
    doc.paragraph('固定64是总标签预算，不是每类64个，也不是按类等额抽样。PACS的20次任务/抽样均覆盖7类；OSS1的5次均未覆盖全部8类；Office-Home每次只覆盖34–46类，平均39.45类，平均约34.66%的查询样本所属类别未在该次支持集中出现。这里的“未见”只指64个目标标签未见，并不表示源训练未见。')
    doc.paragraph('当前门控要求“专家预测类别出现在支持集”才允许交接。因此支持类别缺失会直接限制可接受的专家输出类别。基础模型和专家仍有完整65类源先验输出，不能据此推断未覆盖类别一定预测错误，或把Office-Home退化完全归因于标签覆盖。')
    doc.heading('三个外部SAR任务的逐次支持覆盖', 3)
    dd = draws.loc[draws.dataset.isin(['F0921', 'OSS1', 'OSS2'])]
    doc.table(['数据集', 'draw', '支持覆盖类数', '本次缺失类', '未覆盖类查询样本占比'], [
        [DISPLAY[r.dataset], r.draw, r.support_present_classes,
         r.missing_class_names if isinstance(r.missing_class_names, str) else '无',
         f'{100*r.query_share_in_support_unseen_classes:.2f}%'] for r in dd.itertuples()])
    doc.heading('SAR各类在64标签中的实际数量', 3)
    doc.paragraph('表内为每类支持样本的“均值 [最小, 最大]”，跨本数据集全部目标任务和5次抽样统计；23港包含本来缺少某些类的港口，所以0不必然是抽样漏类。')
    doc.table(['类别']+[DISPLAY[d] for d in ORDER[:4]], [[SAR_ZH[c]]+[
        f'{support.loc[(support.dataset==d)&(support.class_id==c), "support_count"].mean():.2f} '
        f'[{support.loc[(support.dataset==d)&(support.class_id==c), "support_count"].min()}, '
        f'{support.loc[(support.dataset==d)&(support.class_id==c), "support_count"].max()}]'
        for d in ORDER[:4]] for c in range(8)])
    doc.heading('七、采集批次与身份条目')
    meta = read('metadata_summary')
    m = meta.loc[meta.dataset.isin(['F0921', 'OSS1', 'OSS2'])]
    doc.table(['数据集', '目标条目', '采集分组字段', '分组数量', '记录的非空不同身份ID'], [
        [DISPLAY[r.dataset], integer(ds.loc[r.dataset].total), r.group_field, integer(r.distinct_groups), integer(r.distinct_recorded_ids)] for r in m.itertuples()])
    doc.paragraph('F_0921按本地scene字段计5,586组；OSS1和OSS2分别按scene_id计22、38组。它们是清单字段的去重数量，不能不经产品关联核对就称为独立卫星过境次数。不同身份ID仅是已记录非空ID的去重计数，不是重新核验后的真实船舶数。样本数也不是独立船舶数，同一身份可能有多条观测。')
    doc.paragraph('本轮目标支持集按样本抽取，报告不声称支持/查询之间额外实现了船舶身份或采集批次隔离。源/目标的已有身份排除与目标内部支持/查询分组隔离是不同问题；当前报告描述既定协议，不改变实验数据。F_0921的nearest24等字段不是本轮人工核实的真实港口标签，不将其当作已知港口划分。')
    doc.heading('八、这些分布如何影响上一轮预算结果？')
    means = pd.read_csv(SWEEP/'dataset_means.csv')
    rows = []
    for d in ORDER:
        p = means.loc[means.dataset.eq(d)].set_index('policy')
        rows.append([DISPLAY[d], f'{100*ds.loc[d].majority_query_share_taskmean:.2f}%',
                     f'{p.loc["gate_005", "oa_pct"]:.2f}/{p.loc["gate_005", "ba_pct"]:.2f}',
                     f'{p.loc["gate_100", "oa_pct"]:.2f}/{p.loc["gate_100", "ba_pct"]:.2f}'])
    doc.table(['数据集', '查询最大类比例（任务均值）', '5% OA/BA (%)', '不限额门控 OA/BA (%)'], rows)
    doc.paragraph('1. 不能用不同分类数的绝对OA直接给方法排难度顺序。8类SAR、7类PACS、65类Office-Home在标签语义、域划分、表征和64标签覆盖上均不同；高低差异不是单因素实验。')
    doc.paragraph('2. OSS1尤其需要同时看OA和BA。其查询最大类平均占60.53%；本轮5%到不限额门控，OA从43.30升至52.75，BA却从31.76降至29.04。仅展示OA会遗漏类别等权召回退化。最大类比例由查询真值事后统计，只是分布参考，不是一个已合法训练并评测的多数类基线。')
    doc.paragraph('3. 24港必须保留逐港结果。不同港的实际类别数、规模和失衡程度不同。等港平均让小港和大港权重相同，不能拿38,027条的全局类别占比直接解释等港OA。')
    doc.paragraph('4. Office-Home应单列“65类、总计64标签”的设定。支持缺类造成明确的信息覆盖限制，但当前结果不能证明这是模型退化的唯一原因。PACS在支持覆盖完整时，大幅放宽交接仍可能降低准确率，说明类别覆盖与专家可靠性不是同一问题。')
    doc.paragraph('5. 数据分布是观察证据，不是因果归因。当前表格没有证明SAR低分来自某个具体成像因素，也没有证明调整采样就能提点；不在报告中把未做过的干预当作已经验证的模块。')
    doc.heading('九、论文中建议直接写明的实验设定')
    doc.paragraph('“SAR实验共享8类船型标签空间。自建24港语料包含38,091个样本，其中23个满足64样本支持集后仍有查询样本的目标港参与留港评价；三个外部SAR目标集合分别包含123,489、1,349和3,745个映射后样本。通用跨域实验在本地PACS的7类四域和Office-Home的65类四域上开展。每个目标任务随机抽取64个标注样本供两个分支共享，并在其余样本上计算OA和查询实际出现类别的平均召回率BA。报告五次支持集抽样均值，随后对目标港或域等权汇总。”')
    doc.paragraph('“公开SAR规模指当前细粒度标签映射及既定可用性筛选后的评价子集，不将其冒称为完整公开基准。所有集合均已用于开发分析，本报告不将其表述为未接触的独立确认集。5%表示专家预测的决策交接上限，不表示仅使用5%标签，也不表示已实现95%的推理计算节省。”')
    doc.heading('附录A：Office-Home完整65类与各域计数')
    oh = task_classes.loc[task_classes.dataset.eq('officehome')].pivot(index='class_id', columns='task', values='target_count')
    doc.table(['ID', '类别']+list(oh.columns)+['合计'], [[i, summary['class_names']['officehome'][i]]+[integer(v) for v in row]+[integer(row.sum())] for i, row in oh.iterrows()])
    doc.heading('附录B：文件与复现入口')
    doc.paragraph('主统计输入为预算扫描保存的34项任务清单、各任务冻结策略中的support_local_rows/eval_mask/row_ids、目标manifest、各外部任务source_split，以及本地PACS/Office-Home parquet的label/domain字段。仅读取标签和元数据，不解码图像、不加载特征或训练权重。')
    doc.link('类别映射与筛选协议核查', OUT/'LABEL_PROTOCOL_AUDIT_ZH.md')
    for title, path in [
        ('六数据集概要表', OUT/'dataset_summary.csv'), ('逐港/域规模、类别及支持覆盖', OUT/'task_summary.csv'),
        ('全部类别计数与比例', OUT/'dataset_class_counts.csv'), ('逐任务源/目标类别计数', OUT/'task_class_counts.csv'),
        ('170个任务/抽样的支持覆盖', OUT/'support_draws.csv'), ('逐类逐次支持与查询计数', OUT/'support_class_counts.csv'),
        ('全部24港类别计数', OUT/'all24ports_class_counts.csv'), ('采集分组完整计数', OUT/'acquisition_group_counts.csv'),
        ('五张矢量图合订PDF', OUT/'figures/DATA_DISTRIBUTION_FIGURES.pdf'),
        ('既有完整预算实验报告', SWEEP/'BUDGET_FINDINGS_ZH.md')]:
        doc.link(title, path)
    doc.paragraph('复现脚本：E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/analyze_dataset_distribution.py 和 report_dataset_distribution.py。使用ClearSAR现有环境，先统计、后制图与报告。已检查34任务/170抽样齐全、支持和查询互斥、各类计数之和与样本规模一致。')
    doc.save()


def main():
    summary = json.loads((OUT/'distribution_summary.json').read_text(encoding='utf-8'))
    datasets, classes, tc = read('dataset_summary'), read('dataset_class_counts'), read('task_class_counts')
    task, draws, support, ports = read('task_summary'), read('support_draws'), read('support_class_counts'), read('all24ports_class_counts')
    figures(summary, classes, tc, task, draws, ports)
    build_report(summary, datasets, classes, tc, task, draws, support, ports)
    print(f'Saved Chinese Markdown/standalone HTML report and five PNG/PDF figures: {OUT}')


if __name__ == '__main__':
    main()
