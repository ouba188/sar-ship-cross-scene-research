"""Finalize reports from the completed bounded CSVs; no raw scans or extra joins."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def stats(v):
    a = pd.to_numeric(pd.Series(v), errors='coerce').dropna().to_numpy(float)
    a = a[np.isfinite(a)]
    return dict(n=len(a), min=float(a.min()), median=float(np.median(a)),
                p95=float(np.quantile(a,.95)), max=float(a.max())) if len(a) else dict(n=0)


def md(frame):
    cols = list(frame.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---']*len(cols)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    return '\n'.join(lines)


def main():
    c = pd.read_csv(OUT/'cohort_attributes.csv', dtype={'mmsi':str})
    t = pd.read_csv(OUT/'track_summary.csv', dtype={'mmsi':str})
    m = pd.read_csv(OUT/'track_messages.csv', dtype={'mmsi':str})
    s = pd.read_csv(OUT/'track_segments.csv', dtype={'mmsi':str})
    p = pd.read_csv(OUT/'selected32_condition_changes.csv', dtype={'mmsi':str})
    pp = pd.read_csv(OUT/'same_vessel_conditions.csv', dtype={'mmsi':str})
    summary = json.loads((OUT/'summary.json').read_text(encoding='utf8'))
    zero_moving = s.displacement_m.eq(0) & s.reported_endpoint_mean_sog_knots.ge(1)
    zero = s.displacement_m.eq(0)
    s.loc[zero, ['displacement_bearing_deg','abs_bearing_cog_delta_deg']] = np.nan
    s.loc[zero, 'moving_cog_eligible'] = False
    for idx, row in t.iterrows():
        seg = s[s.global_row.eq(row.global_row)]
        msgs = m[m.global_row.eq(row.global_row)]
        aa = stats(seg.abs_bearing_cog_delta_deg)
        t.at[idx,'abs_bearing_cog_delta_deg_median'] = aa.get('median',np.nan)
        t.at[idx,'abs_bearing_cog_delta_deg_n'] = aa['n']
        t.at[idx,'draught_positive_raw_variants'] = msgs.loc[msgs.draught_raw.gt(0),'draught_raw'].nunique()
        t.at[idx,'draught_zero_records'] = msgs.draught_raw.eq(0).sum()
        t.at[idx,'draught_positive_records'] = msgs.draught_raw.gt(0).sum()
    t.to_csv(OUT/'track_summary.csv',index=False)
    s.to_csv(OUT/'track_segments.csv',index=False)
    summary['track_stats']['abs_bearing_cog_delta_deg_median'] = stats(t.abs_bearing_cog_delta_deg_median)
    summary['segment_stats']['abs_bearing_cog_delta_deg'] = stats(s.abs_bearing_cog_delta_deg)
    summary['zero_displacement_segments'] = int(zero.sum())
    summary['zero_displacement_reported_moving_segments'] = int(zero_moving.sum())
    summary['draught_within_window'] = dict(raw_variation_tracks=int(t.draught_raw_variants.gt(1).sum()),
        positive_variation_tracks=int(t.draught_positive_raw_variants.gt(1).sum()),
        zero_and_positive_tracks=int((t.draught_zero_records.gt(0)&t.draught_positive_records.gt(0)).sum()))

    residual_groups = []
    for name,mask in [('all',np.ones(len(s),bool)), ('gap_ge10s',s.gap_s.ge(10)), ('gap_ge30s',s.gap_s.ge(30)),
                      ('positive_displacement_moving_cog',s.moving_cog_eligible)]:
        q=s.loc[mask]
        residual_groups.append(dict(subset=name,segments=len(q),
            speed_abs_residual_median_knots=stats(q.speed_residual_knots.abs()).get('median',np.nan),
            speed_abs_residual_p95_knots=stats(q.speed_residual_knots.abs()).get('p95',np.nan),
            bearing_cog_valid=int(q.abs_bearing_cog_delta_deg.notna().sum()),
            bearing_cog_abs_median_deg=stats(q.abs_bearing_cog_delta_deg).get('median',np.nan),
            bearing_cog_abs_p95_deg=stats(q.abs_bearing_cog_delta_deg).get('p95',np.nan)))
    pd.DataFrame(residual_groups).to_csv(OUT/'motion_consistency_by_gap.csv',index=False)
    jumps=s[s.speed_residual_knots.abs().gt(10)].copy()
    jumps.to_csv(OUT/'large_position_speed_residual_segments.csv',index=False)
    summary['large_position_speed_residual'] = dict(threshold_knots=10,segments=len(jumps),
        tracks=int(jumps.global_row.nunique()), ships=int(jumps.mmsi.nunique()))
    summary['low_to_moving_pairs'] = int((p.first_sog_knots.le(.5)&p.second_sog_knots.ge(2)).sum())
    summary['moving_to_low_pairs'] = int((p.first_sog_knots.ge(2)&p.second_sog_knots.le(.5)).sum())
    unique_pairs=pp.drop_duplicates(['mmsi','first','second'])
    complete=unique_pairs[unique_pairs.both_motion_exact]
    summary['unique_endpoint_pairs']=len(unique_pairs)
    summary['both_exact_unique_endpoint_pairs']=len(complete)
    summary['all_recovered_pair_changes']={f:stats(complete[f].abs()) for f in
        ['sog_change_knots','cog_change_moving_both_deg','true_heading_change_deg','draught_positive_both_change_raw']}
    summary['status']='completed_bounded_32_ships_no_background'
    summary['completed_utc']=datetime.now(timezone.utc).isoformat()
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

    # Check actual output joins and old-cache agreement, not a new exhaustive audit.
    old=pd.read_csv(OUT.parents[1]/'same_vessel_condition_study_20261010/motion_knowledge_20261011/motion_knowledge.csv',dtype={'mmsi':str})
    old=old[old.match_status.eq('matched')]
    check=c.merge(old,on=['instance_id','mmsi','product'],suffixes=('_new','_old'),validate='one_to_one')
    check=check[check.ais_epoch_new.notna()]
    for field in ['ais_epoch','sog_knots','cog_deg','true_heading_deg']:
        assert np.allclose(check[field+'_new'],check[field+'_old'],equal_nan=True),field
    assert len(c)==2713 and c.global_row.is_unique
    assert len(p)==32 and p.mmsi.nunique()==32 and len(t)==64
    assert len(m)==890 and not m.duplicated(['mmsi','epoch','lon','lat','raw_sog','raw_cog','raw_true_heading']).any()
    assert len(s)==int((t.distinct_timestamps-1).sum())
    assert m.groupby('global_row').size().sort_index().tolist()==t.set_index('global_row').distinct_dynamic_records.sort_index().tolist()
    assert c[c.raw_length_m.notna()].raw_length_m.eq(c[c.raw_length_m.notna()].object_length_m).all()
    assert c[c.raw_beam_m.notna()].raw_beam_m.eq(c[c.raw_beam_m.notna()].object_beam_m).all()
    assert s.loc[zero,'abs_bearing_cog_delta_deg'].isna().all()
    receipt=dict(status='passed',focused_unit_tests=5,prior_exact_rows_compared=len(check),
                 compared_fields=['ais_epoch','sog_knots','cog_deg','true_heading_deg'],
                 raw_object_dimension_agreement_rows=int(c.raw_length_m.notna().sum()),
                 current_rows=2713,selected_ships=32,tracks=64,distinct_dynamic_records=890,
                 validation='exact keys, ambiguity, dedup, sentinel/circular/knots units, zero-displacement bearing, saved-row alignment',
                 no_training=True,no_download=True,no_runtime_upgrade=True)
    (OUT/'VALIDATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')

    cov=pd.read_csv(OUT/'coverage_by_group.csv')
    cl=cov[cov.grouping.eq('class_name')][['group','rows','valid_sog','valid_cog','valid_true_heading','raw_size_pair','positive_raw_draught']]
    cl.columns=['类别','队列行','SOG','COG','真艏向','精确长宽','非零吃水原码']
    ports=t.port.nunique()
    selected_classes=p.class_id.value_counts().sort_index()
    class_map=c.drop_duplicates('class_id').set_index('class_id').class_name.to_dict()
    class_text='、'.join(f'{class_map[k]} {v}艘' for k,v in selected_classes.items())
    fields=[('精确动态记录匹配',1322),('有效SOG（节）',int(c.sog_knots.notna().sum())),
        ('有效COG（度）',int(c.cog_deg.notna().sum())),('有效真艏向（度）',int(c.true_heading_deg.notna().sum())),
        ('双极化UID一致的AIS时间',int(c.ais_epoch.notna().sum())),('精确记录长宽',int(c.raw_length_m.notna().sum())),
        ('精确记录吃水原码（含0）',int(c.draught_raw.notna().sum())),('非零吃水原码',int(c.draught_raw.gt(0).sum())),
        ('对象JSON长宽',int(c.object_size_pair_available.sum())),('船级众数长宽',int(c.identity_mode_size_pair_available.sum()))]
    cv=pd.DataFrame([dict(字段=n,有效行=v,分母=2713,覆盖率=f'{v/2713:.2%}') for n,v in fields])
    bstat=summary['segment_stats']['abs_bearing_cog_delta_deg']
    trstat=summary['track_stats']['abs_bearing_cog_delta_deg_median']
    gap=summary['track_stats']['gap_max_s']
    dv=summary['draught_within_window']
    lg=summary['large_position_speed_residual']
    report=f'''# 24港运动与属性有界结果

2026-10-17，已完成。产出只在本目录；无训练、下载、环境升级或后台任务。全队列仅做缓存/对象回接，短轨迹在32艘船收口。

## 1. 可直接交给主线的结论

- 当前2713行覆盖2517个MMSI、448产品、24港。精确动态记录匹配1322行，其中本轮原始记录1287行、仅复用旧缓存35行；有效SOG1320、COG1275、真艏向1137行。不是全队列全字段已恢复。
- 32艘预选首末船、64段短轨迹共890条不同动态记录，890个船/采集内时刻，无重复拷贝膨胀。每段2–51条，中位7条；观测跨度6–381秒，中位339.5秒。
- 首末配对32艘的SOG绝对变化中位1.65节，16/32变化至少2节；真艏向可用27艘，中位绝对变化110度、22/27至少30度。长宽32/32两端一致。两端非零吃水原码可比较16艘，其中14艘变化，但单位及属性更新时间未知。
- 稳定低速窗口与跨采集运动变化同时存在，可用于主线图像响应的分层和案例选择；这些结果不证明运动造成图像变化或带来分类收益。

## 2. 覆盖和来源

{md(cv)}

当前2713行由returned 1899、legacy 814组成。双极化JSON时间一致1868；22行只有一个可用极化对象，另9条returned两端JSON均不可用，814条legacy没有这套UID/时间链。这里的22行不是已证实的时间矛盾。845行不具备严格的对象时间锚点，546行有时间但未在当前读取预算中取得精确运动。不能用最近AIS补成精确匹配。

匹配链：`global_row -> manifest instance/product/MMSI -> VV/VH UID -> JSON AIS UTC -> 同产品、同MMSI、同epoch原记录`。双极化不算两条AIS证据。坐标检查仅使用对象明确存储的经纬度；本轮没有把SAR框中心当AIS经纬度。

复用 `recover_v6_relation_observations.py::parse_export` 和 `build_returned_local_knowledge_v1.py::load_file_maps`；时间、缺失码和精确歧义处理复用 `extract_motion_knowledge.py`。原始字段补充用标准CSV读取器，仅提取来源行号及既定列，不另写TXT解析器。874个本地JSON，54个相关AIS文件，33,707,421字节（32.15 MiB）、341,924条源文件行；源文件中非队列船行不作为本研究样本。

旧缓存直接复用107行，其中72行在已需读取的产品中重新匹配，另外35行仍保留旧来源。107行的时间/SOG/COG/真艏向与旧结果一致；1287行原始长宽与对应对象JSON一致。船级众数长宽2609行另列，绝不回填为同期测量。

{md(cl)}

覆盖明显偏向returned和常见产品；例如渔船SOG仅10/252，而油化船340/457。主线不能将完整条件子集的响应关系无条件推广到全队列，缺失值也不应当零。

## 3. 32船选择和窗口

沿用输入309条极化配对（VV161、VH148），共164艘、169个不同端点组合。按MMSI排序，每艘取原配对中按first/second/pol排序的首个组合；要求两端均有双极化一致时间、精确原始记录，且各有至少两个不同观测时刻。检查74艘后得到32艘：26艘缺少精确时间锚点，16艘至少一端不足两个时刻。其余90艘未继续检查；没有为了扩类别或寻找更好结果继续扫描。

这32艘覆盖{ports}港、7类：{class_text}。无bulk_carrier；不是随机或类别均衡代表样本，结论严格限定在实际子集。`ship_selection.csv` 保留每一步分母和跳过原因。

窗口固定为产品起始前180秒至结束后180秒，因此25秒产品的名义窗口可到385秒；不是一律中点±180秒。本次最大实际跨度381秒。每段最大消息间隔中位{gap['median']:.1f}秒、最大{gap['max']:.0f}秒；39/64段存在>60秒空档，32/64存在>120秒空档。60/64段在产品中点前后均有消息。跨采集的几天/几月间隔不桥接成连续轨迹。

34/64段所有有效SOG都<=0.5节，只称“观测记录全低速”。另外提供每段SOG均值、标准差、范围和低速记录比例；没有输出停泊时长，没有在空档内假定持续静止。全32船首末中低速<=0.5节转为>=2节有{summary['low_to_moving_pairs']}艘，反向有{summary['moving_to_low_pairs']}艘；阈值仅作事后描述，不训练或调参。

这里“不同消息”精确定义为导出动态列0–9不同的记录，去除重复副本，并另报不同时刻。没有无线电报文序号，因此不宣称恢复了所有真实发射报文；丢包与未导出消息无法由这批数据计数。此次890条记录在MMSI/epoch/位置/运动组合上也全不同。

## 4. 局部运动一致性和实际异常

826个相邻时刻段用WGS84测地距离/时间计算位移速度，与两端SOG均值比较；这是描述性一致性，转向、报告时延和位置误差都会影响它，不能直接当加速度真值。段级绝对速度残差中位0.300节，P95=4.121节、最大86.300节；先对每条轨迹取中位再跨轨迹汇总，中位0.148节，避免高频消息船主导全部解释。

92段位移为0，其中47段报告运动。零位移无定义方向，已将位移方向和COG方向残差记缺失。只在非零位移且两端SOG>=1节时比较COG，实际{bstat['n']}段，中位绝对差{bstat['median']:.3f}度，P95={bstat['p95']:.3f}度；按轨迹取中位后有{trstat['n']}段轨迹可用，中位{trstat['median']:.3f}度。COG保持对地运动方向语义，不当作船体朝向；低速COG即使编码有效也不用于这项方向一致性。

原始跳变确实存在：global_row=29284，MMSI477166600，在epoch1754963700→1754963704的4秒内坐标移动197.135米，位移速度95.800节、两端报告9.5节；随后6秒又反向移动147.503米。这是已观察到的位置/时间/SOG不一致，原因未确认，不直接解释为船只加速。绝对速度残差>10节共{lg['segments']}段、{lg['tracks']}条轨迹、{lg['ships']}艘船，详见 `large_position_speed_residual_segments.csv`。保留这些负结果，未将异常过滤后再宣称全体稳定。

`motion_consistency_by_gap.csv` 提供原全部段及gap>=10秒、>=30秒的描述性敏感性分层。不是用当前表现反向挑出的“最佳时间阈值”，也不把密集相邻段视为独立船舶证据。

## 5. 属性与同船条件变化

字段零起算语义：4=SOG/10（节），8=COG/10（度），9=真艏向（度），14=长、15=宽，16=吃水原码；5是定位精度，不是导航状态。长宽单位沿用导出消费者与JSON中的length_m/beam_m，未由登记机构独立认证；14和15不相加。SOG1023/COG3600/heading511等缺失码不置零，SOG1022单独按下界编码标记，本队列有效匹配中没有1022。

AIS时间差以动态消息UTC减产品起止中点计算，不是精确到船的成像时间；1868行范围-180.5至178.5秒。有效SOG行绝对时间差中位20.5秒、P95=137.55秒、最大180.5秒。原JSON整数秒时间差与精确中点可能相差0.5秒，不能用整数180秒直接判错。

吃水1287行中337为0，950为非零；原码单位未验证，不转换成米，不反推载荷/干舷。64条轨迹中{dv['raw_variation_tracks']}条出现原码变化，其中{dv['zero_and_positive_tracks']}条为零/正值并存，{dv['positive_variation_tracks']}条有多个不同正值。动态时间戳不提供静态/航次吃水的更新时间；窗口内变化不证明几分钟内实际吃水改变。

预选全部配对中，有208/309极化行、109/164艘船两端恢复精确动态记录；去除极化重复后114个端点组合。完整字段可用分母逐列不同。主统计32船每艘只取一个组合，详见 `selected32_condition_changes.csv`：SOG变化32艘，真艏向27艘，两端运动且COG有效9艘（绝对方向变化中位65.3度），两端正吃水16艘（原码绝对变化中位4、最大35）；原码变化没有升级为物理因果结论。

## 6. 主线接口与产物

- `cohort_attributes.csv`：2713行，主键global_row；精确运动、时间差、对象/原始/船级众数尺寸分列，source_txt+source_line和source_json_paths可追溯。
- `same_vessel_conditions.csv`：保持原309极化配对行，first/second接global_row；含条件、差值、可用性及原图像响应变化字段。169端点组合、164船不能混同309独立样本。
- `selected32_condition_changes.csv`：32艘的单一预选首末组合，适合主线图像案例回接。
- `track_summary.csv`：64条船/采集窗口，跨度、间隔、消息数、速度与一致性摘要。
- `track_messages.csv` / `track_segments.csv`：890条去重消息与826相邻段；消息保留原始来源文件和行号。
- `coverage_by_group.csv` / `ship_selection.csv` / `read_sources.csv`：真实类/港/来源分母、选择过程、读取成本。
- `summary.json` / `INITIAL_COVERAGE.json` / `VALIDATION.json`：最终统计、早期覆盖和验证收据。

实际来源：当前inventory、既定same_vessel_pairs、eight_class_adaptive manifest；同步目录 `E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata/local_knowledge_sync_v1/`；旧motion_knowledge_20261011缓存；`E:/Hermes/paper_tracking/port_ais_2024_2025_from_zip/dedup_tables/vessels_to_lookup_dedup.csv`。完整54个已读TXT路径在read_sources.csv，不需要主线重新搜索目录。

本轮未继续全量原始回接：已记录路径显示全部448产品单副本约719 MB，但按最新收口要求停在32船和32.15 MiB。剩余546条有一致对象时间的行属于未读覆盖缺口，不是已证明原始AIS不存在。其余845行首先缺精确对象时间链。本次统计为因素分析输入，不改变主模型、标签、主线图像/分类结果或brain。

验证：5项有针对性的测试通过（精确身份/时刻及原码列、歧义拒绝、重复消息、缺失码/角度、测地速度单位/零位移方向）；107行旧缓存一致，1287行尺寸与对象一致，2713行/32船/64窗口/890消息/826段保存对齐通过。无后台任务。
'''
    (OUT/'FINDINGS_ZH.md').write_text(report,encoding='utf8')
    handoff=f'''# 24港运动/属性交接：已完成

状态：当前有界集合已收口，无后台进程，无后续扫描、训练或下载。

主报告：`E:/临时会话/all_factor_probe_20261017/ports_motion/FINDINGS_ZH.md`。

## 可立即使用

- `cohort_attributes.csv` 按global_row连接当前inventory：2713行中精确运动记录1322；有效SOG1320、COG1275、真艏向1137。精确长宽/吃水原码1287；非零吃水950。对象长宽1890、船级众数2609独立列出。
- `same_vessel_conditions.csv` 保留原309极化配对，两端精确覆盖208行/109艘；去重为114端点组合。请勿把极化重复当独立船。
- `selected32_condition_changes.csv` 是32船单一首末组合；`track_summary.csv` 是64短窗口，`track_messages.csv` 为890条不同动态记录，`track_segments.csv` 为826相邻段。

## 数字结论与限制

32船/64窗口的记录数中位7（2–51），实际跨度中位339.5秒（6–381），最大间隔中位113.5秒、最大360秒；34/64窗口全记录SOG<=0.5节，只称低速代理，不报停泊时长。

32船SOG绝对变化中位1.65节、16艘>=2节；27艘有效真艏向中位变化110度。32船长宽均不变；16艘两端正吃水原码中14艘变化，单位和更新时间未验证。index5是定位精度；动态UTC不是吃水更新时间。

筛选为MMSI排序的前32合格船，覆盖7类/{ports}港，无bulk_carrier，不能代表全体。少数位置往返跳变导致速度残差；零位移段方向已置缺失。所有异常保留，见报告和large_position_speed_residual_segments.csv。

本轮只读54个TXT、32.15 MiB；不继续约719 MB的448产品全读。未恢复546行是有对象时间但本轮未读，845行缺严格时间锚点。5项定向测试及实际输出对齐通过，详见VALIDATION.json。

如主线需要展示速度向量，可由有效SOG/COG导出east=SOG*sin(COG)、north=SOG*cos(COG)，单位仍为节；不是对水速度，也未投影到SAR有向视线。现有产出不包含分类效用断言。
'''
    (OUT/'HANDOFF.md').write_text(handoff,encoding='utf8')
    print(json.dumps(dict(status=summary['status'],handoff=str(OUT/'HANDOFF.md'),
        raw_bytes=summary['read_raw_bytes'],zero_displacement=int(zero.sum()),
        bearing_stats=bstat,track_bearing_stats=trstat,draught=dv,jumps=lg,
        transition_counts=[summary['low_to_moving_pairs'],summary['moving_to_low_pairs']],validation=receipt),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
