# Reference Pool Audit

日期：2026-10-24。范围：只读追溯参考池、数据角色、覆盖与原始 chip；不训练、不改 run.py/PROTOCOL_ZH.md、不操作 Git。未读取本轮分类结果来选择原图或候选参考。以下脚本仅生成 pool_inventory.csv 及两幅原图对照。

## 已确认的候选规则

1. 与原库范围一致的主候选池是 `indices.npz['fit_train']` 的 1205 条观测；按原有顺序、在 body mass 有效后 `drop_duplicates('mmsi')`，得到 1004 艘船，各类 306/232/162/141/62/55/29/17。所有观测均有有效 body mass。旧库每类 8 艘，共 64 艘。
2. 更宽的外层合法池是 `role == 'strategy_fit'` 的 1646 条、1290 MMSI、112 pass。它与 utility_train / selection / evaluation 的 MMSI 和 pass 集合交集均为 0。三个查询角色的标签不得用于建立、筛选或排序参考库；可按既定用途用于效用学习、选点、最终报告。
3. 内部 `fit_validation` 296 条可以作为外层固定查询的参考，但必须标为扩大参考池实验。其标签已经用于上游 reader alpha 和各级头选点，不是未接触数据；纳入参考后，不再用该内部 validation 声称对参考机制独立验证。主线程本轮只取 fit_train，因此没有这项范围变化。
4. `fit_excluded_mmsi` 145 条仅被内部划分排除，仍属于 strategy_fit；它们与外部三个角色隔离，外层参考使用本身不造成外部标签泄漏。但其 111 艘船全部出现在内部 validation，不能当新独立船，不能重新并入 reader 训练同时维持原内部验证独立性。主对照不纳入；扩池时单列身份重复。
5. 固定查询公平换库：保持每类 8 个不同 MMSI、逐槽四组有效性模式、相同冻结 token/PCA/模型/原始 RMS/查询集合；预定随机种子，尽量排除原库 MMSI，记录确实不可避免的复用。同一库必须同时计算 learned 和 identity，差值为 Z_W(bank)-Z_I(bank)。不按 evaluation 结果挑库。

## 追溯与训练接触

`conditional_relation_operator/learned/run.py` -> 父目录 `run.py` -> `edge_difference/run.py` -> `support_transfer_learning/run.py::load()`。load 读取 `regional_utility_pilot/upstream/table.csv`、`local_evidence_transport/tokens.npz` 和 `projection_layer1.npz`，按冻结 mean/components 得到归一化 PCA32 token，bank 是 projection 文件中保存的 8x8 表位置索引，不是 obs_id 或原始 row_index。

原库来自 `match_readout.py::make_bank()`：fit_train -> body mass>0 -> 保留每 MMSI 第一条 -> 每类 body descriptor KMeans(8, random_state=17,n_init=10) -> 距中心最近且不重复的 8 条。PCA32 仅从 fit_train 的有效 layer1 token 随机抽至多 20000 个拟合。token 提取使用冻结 ResNet50，不是本轮拟合的监督辅助 reader；因此“已有 reader 训练接触”与“token backbone 本轮训练”不能混写。

已读取 layer1/layer3 reader 保存的 `fit_indices`、`validation_indices`，分别逐项匹配 table 中 fit_train 的1205个原 row_index、fit_validation 的296个原 row_index。监督 reader 的 scaler/PCA128/postscale/Ridge 使用 fit_train，alpha 用 fit_validation；上游 full 基线也在 strategy_fit 内拟合/选点。fit_train 对这些监督 reader 是训练内数据，fit_validation 是选点接触；均不能称整栈 OOF。conditional learned 的 M_b/M_n/a 后续受 utility644 监督、selection612 选点，原参考64也通过固定参考张量参与这些梯度，但这不等于用查询标签给新参考指定类别。

内部 fit_train/fit_validation 的 MMSI/pass 交集均为0；fit_excluded_mmsi 与 validation 的 MMSI/pass 交集为111/0。pass 按 `platform + product字段6 + product字段7` 定义（例如 S1A_057818_072156），不是逐 PNG 或逐裁片编号。全局不同 MMSI/pass 是独立单位的计数，不代表各单位统计上完全独立。

## 覆盖统计

表中 `N/M/P` 分别为观测数、不同 MMSI、不同 pass；不同类别的 pass 可以重叠，不能把各类 P 相加当总数。VH 不可用指当前 token/关系输入不可用，不是原图文件缺失。

|类别|strategy_fit N/M/P|fit_train N/M/P|内部 validation N/M/P|内部排除 N/M/P|strategy_fit / fit_train VH不可用观测|
|---|---:|---:|---:|---:|---:|
|0 bulk|471/382/95|367/306/75|79/76/19|25/20/19|10 / 3|
|1 fishing|291/236/21|285/232/16|4/4/4|2/1/2|137 / 135|
|2 general cargo|246/199/69|193/162/57|38/37/11|15/11/10|13 / 9|
|3 chemical/product|334/237/50|170/141/33|98/96/13|66/50/14|18 / 9|
|4 container|115/89/35|69/62/26|28/27/8|18/12/6|2 / 0|
|5 crude oil|102/81/36|66/55/23|28/26/10|8/7/7|2 / 1|
|6 tug|50/38/29|34/29/22|10/9/6|6/5/4|5 / 2|
|7 offshore|37/28/19|21/17/11|11/11/6|5/5/5|13 / 10|
|总计|1646/1290/112|1205/1004/88|296/286/23|145/111/36|200 / 169|

1646 条全部 VV token 可用，全部 VV/VH 原图路径存在。`upstream/valid.npz` 与 `tokens.npz['valid']` 在整个3387行表上完全一致。四组合法边按 body 与对应 side/end 的 mass 同时大于0计算，条件算子要求至少2条；本表没有“极化 token 标为有效但边数不足2”的额外失效行。原库8条VH不可用为 obs **16657、15607、15202、5626、26391、28628、8183、22924**。

### 原库及同范围新增覆盖

|类别|实际 canonical池 N=M / P / VH有效数|原库 N=M / P / VH有效数|排除原库全部MMSI后的 fit_train N/M/P|原库未覆盖的 fit_train class-pass数|canonical中新MMSI且VH有效数|
|---|---:|---:|---:|---:|---:|
|0|306/73/304|8/7/8|358/298/75|68|296|
|1|232/16/121|8/5/5|274/224/16|11|116|
|2|162/48/154|8/8/7|181/154/53|49|147|
|3|141/32/134|8/6/7|160/133/32|27|127|
|4|62/23/62|8/7/8|60/54/24|19|54|
|5|55/20/54|8/6/8|54/47/21|17|46|
|6|29/18/27|8/7/7|23/21/15|15|20|
|7|17/10/10|8/7/6|10/9/7|4|4|
|总计|1004/85/866|64/35/56|1120/940/87|不相加|810|

注意两个层次：fit_train 除原64**观测**还有1141条；若排除原64**船只**的所有观测，则为1120条、940艘。strategy_fit 除原64观测还有1582条，尚未入原库的不同船为1226艘。不要用观测数冒充新增独立船数。

原始去重规则锁定每船第一条观测，会丢弃部分同船更可用的 VH 观测：例如渔船 fit_train 任一观测 VH 有效的不同船为132，但 canonical 仅121；化学品135->134、拖船28->27。当前五库沿用原规则，属于已匹配的抽样定义，不是本轮实现错误。若改为每船挑 VH 可用观测，应单列为“观测选择/覆盖改变”，不能混称只换船。海工任一观测可用也仅10艘，原库已占6艘，新的可用船确实仅4艘，不是第一条规则制造的短缺。

### 五个实际 bank

仅读取 `banks.csv` 和源元数据核查，不读取本轮性能表选图。实际六库均为64条/64 MMSI、每类8条，全部 fit_train、各槽四组可用模式与原库完全一致，且索引/obs_id/MMSI/pass/class 与 table 逐项吻合。五个新库各有且仅有2个旧MMSI/旧观测，全部为海工VH有效槽位。VV side/end 各64，VH side/end 各56；每类VH有效数固定为8/5/7/7/8/8/7/6。

|bank|总pass数|各类pass数（0至7）|旧MMSI复用|新MMSI|
|---|---:|---|---:|---:|
|original|35|7,5,8,6,7,6,7,7|64|0|
|101|32|8,3,8,5,6,4,7,7|2|62|
|307|30|8,3,8,7,5,7,6,7|2|62|
|509|30|7,6,6,7,5,7,6,7|2|62|
|711|35|6,7,8,7,5,8,7,6|2|62|
|907|35|7,6,7,8,7,6,7,7|2|62|

五新库合计320个槽位仅覆盖239个不同观测/船。海工新的双极化可用船固定为 **374023000(obs13000)、577065000(31851)、376145000(13276)、563612000(30692)**，五库全部包含这4艘。它们是重复暴露，不是5组独立的参考证据。新库渔船pass为3/3/6/7/6，与原5不同，故“模式匹配”不等于“场景组成相同”。

## 两组原图事实

抽样规则：在源库覆盖而非性能上选两个有明确约束的类别。每图展示一个原库VH无效样本、一个原库双极化有效样本、一个实际新库中新增class-pass的双极化样本。选择过程无随机抽样；新库编号101来自预定seed101，海工新例出现在全部5个种子。不是依据救回/误伤选例，不能据此推断本轮预测翻转原因或全体比例。

两图已实际渲染查看，6个观测、12个极化图均读取 table 的 `image_vv/image_vh` 原PNG，64x64、固定灰度0..255、nearest展示、不裁剪、不旋转、不逐图拉伸。PNG已是历史8bit产品，不能把灰度差当校准后后向散射或物理成因；模型实际输入另经既有160画布crop/pad、冻结骨干/C4和几何slot提取，此图展示的是进入该流程之前的原chip。

|图|table_index / obs_id|MMSI|pass|类别/库角色|VV/VH合法边 side,end|
|---|---|---|---|---|---|
|fishing|1312 / 16657|412421791|S1A_057818_072156|原库、VH无效|8,4 / 0,0|
|fishing|1251 / 15874|412413902|S1A_058693_0744B4|原库、双极化|8,4 / 8,4|
|fishing|1401 / 17937|412431825|S1A_059043_0752E1|101库、新船及该类新pass|8,4 / 8,4|
|offshore|585 / 8183|352001054|S1A_058338_073664|原库、VH无效|8,4 / 0,0|
|offshore|1107 / 13344|377172000|S1A_059038_0752B2|原库，被101/509/711复用|8,4 / 8,4|
|offshore|1082 / 13000|374023000|S1A_061132_079EF0|五新库共同包含、新船及该类新pass|8,4 / 8,4|

每个案例的精确原始路径已记录于 `pool_inventory.csv` 对应obs行的 `image_vv/image_vh` 字段，可直接定位；本报告脚本还逐极化打印完整路径与快照target_index。原图来源分别是 `E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_merged_20260915/images/`（渔船）和 `.../v5_context_dynamic_64_128_return_20260915/images/`（海工），快照来自 `E:/临时会话/same_vessel_gallery_20261019/local_condition_module/view_snapshot.json`。

**渔船图 `raw_fishing_coverage.png`：**

- 16657：VV中央近竖直亮条明显，VH相应中央仍可见较弱的断续灰亮条；VH最大灰度132、无>=200像素，快照target_index=-1，当前VH输入无效。不能称VH文件缺失，也不能从图像断言不存在船体回波。
- 15874：两极化中央均是细长近竖直响应，亮点位置和强弱不同；VV>=200像素4个，VH13个。两极化均有接受的目标和8/4条边。
- 17937：两极化都可见多个分离亮团，从左上到右下分布，较前两例并非单个孤立中央亮条；快照两极化target_index=1，不能把全图亮点都当目标或把它们命名为某物理结构。该例来自原渔船bank未覆盖的pass，却和15874一样拥有完整四组有效模式。说明有效性匹配保留不了周围响应构型，不证明该差异导致分类下降。

**海工图 `raw_offshore_coverage.png`：**

- 8183：VV/VH中央均有面积较大的白亮团，右上边缘另有显著亮团。VH有215个>=200像素（VV189个），但VH target_index=-1，当前边为0/0。这直接排除了“VH不可用必然是原图黑/无亮目标”的解释；可确认的是目标接入未成立，不能仅靠外观断定检测错误或更换标签。
- 13344：两极化中央亮响应相对紧凑、近竖直，背景较暗，均为有效参考；它同时出现在三个新库，所以新库不是完全不同的一组图像条件。
- 13000：VV中央较紧凑亮团，VH中央亮团有更明显的上下延展；全图颗粒纹理较亮。VV/VH全图平均灰度87.31/80.77，高于13344的40.73/51.36，这只是8bit显示事实。它为原海工bank新增了船只与pass的图像条件；Port Said港口本身已被原库obs905覆盖，不能称新增港口。作为仅4个新双极化可用候选之一，该例被五库反复使用。

本审计没有计算查询预测或因果贡献。可保留的结论是：类内候选、过境和极化有效性存在具体覆盖缺口，且四组模式相同的真实图像仍有明显构型差异；性能变化是否由这些差异还是原库依赖造成，要结合主线程固定模型干预，不能由本图单独决定。

## 实际风险与设计边界

- **当前未发现外部角色标签/身份泄漏。** 六个实际库只用fit_train，外层strategy_fit与三查询角色MMSI/pass均零交叉；不用再为此扩大审计或改协议。
- **原库是KMeans代表点，新库是模式分层随机替代，且class-pass数变化。** 这是合法的参考敏感性干预，能够说明冻结关系参数对这组替代参考的依赖；它同时改变代表点选择方式和场景组成，不能独自证明“过拟合参考身份”是唯一原因。当前不要求追加训练；报告保留这一解释边界即可。
- **海工可替代性不足是实际数据约束。** 六个VH槽仅4个新船可填，至少2旧船复用合理；五库共享4新船，不能以5次一致变化当作5份独立海工证据。
- **不把所有1646条默认当本轮的1004船候选。** validation已经参与选点，内部排除项重复validation身份，若扩池必须另标范围。训练池/选点池的内部独立性与外层查询隔离是两个不同问题。
- **逐update换库时缓存绑定必须同步。** 已读的 `learned.Reader.__init__` 将bank绑定至 `sb/sn/ms/rv`，`reference/ready` 也随库缓存。仅改 `.bank` 无效；应该同步更新实际参考张量和同库identity。原RMS若冻结就保持原值，不能一支重新估计而另一支沿用旧值。本审计只告知接口约束，不修改主实现。
- **训练参考与多库选点的接触应如实记录。** 新训练从同1004船池反复重采样，可能接触五诊断库的船，这是合法source训练；这五库之后不能再叫未见参考库。旧checkpoint与新模型按同一个selection612、多库集合和汇总准则选点，才能对齐选择口径；evaluation485只作冻结后报告，不据其结果挑库/种子。这里是新实验的边界提示，不是声称尚未审阅的新代码存在泄漏。

完成项：逐观测CSV共1646行，包含三种ID、原图路径、内外角色、reader接触、canonical候选标记、原库/五新库成员、极化有效性及四组边数。六库身份/索引/模式核查通过；两幅PNG已查看。仅修改本报告及其三个生成产物。统计脚本首次在打印numpy整数的JSON时中断，修为显式整数序列化后完整成功；未发生训练或数据变更。

## 可复现脚本

脚本以 `apply_patch` 写在本报告中，执行时从此代码块编译，避免额外脚本文件。仅写同目录 `pool_inventory.csv`、`raw_fishing_coverage.png`、`raw_offshore_coverage.png`。审计检查针对实际风险：索引错位、查询身份交叉、与原库有效模式不一致、CSV元数据与真实索引不一致；发现后将对应库标为不可用于当前公平比较，不启动修复或训练。

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path('E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/conditional_relation_operator/reference_resampling')
ROOT = HERE.parents[1]
UP = ROOT.parent / 'regional_utility_pilot/upstream'
t = pd.read_csv(UP / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
z = np.load(ROOT / 'tokens.npz')
mass = z['mass']
valid = z['valid']
upvalid = np.load(UP / 'valid.npz')['valid']
old = np.load(ROOT / 'projection_layer1.npz')['bank'].astype(int)
ix = np.load(UP / 'indices.npz')
groups = ['VV_side', 'VV_end', 'VH_side', 'VH_end']
counts = np.stack([
    ((mass[:, p, body] > 0) & (mass[:, p, near] > 0)).sum(-1)
    for p in range(2)
    for body, near in [(np.arange(8), np.arange(8, 16)),
                       (np.array([0, 1, 6, 7]), np.arange(16, 20))]
], axis=1)
available = counts >= 2
assert np.array_equal(z['instance_ids'].astype(str), t.instance_id.to_numpy())
assert np.array_equal(upvalid, valid)
assert np.array_equal(ix['fit_train'], np.flatnonzero(t.upstream_role.eq('fit_train')))
fit = ix['fit_train']
eligible = fit[mass[fit, :, :8].sum((1, 2)) > 0]
canonical = t.iloc[eligible].drop_duplicates('mmsi').index
q = t[t.role.isin(['utility_train', 'selection', 'evaluation'])]
src = t[t.role.eq('strategy_fit')].copy()
for key in ['mmsi', 'pass_group']:
    assert not set(src[key]) & set(q[key])
for layer in ['layer1', 'layer3']:
    ck = np.load(UP / 'models' / ('reader_' + layer + '.npz'))
    assert np.array_equal(ck['fit_indices'], t.iloc[ix['fit_train']].row_index)
    assert np.array_equal(ck['validation_indices'], t.iloc[ix['fit_validation']].row_index)

inv = src.copy()
inv.insert(0, 'table_index', inv.index)
inv['in_original_bank'] = inv.index.isin(old.ravel())
inv['original_mmsi'] = inv.mmsi.isin(t.iloc[old.ravel()].mmsi)
inv['fit_train_candidate'] = inv.index.isin(eligible)
inv['canonical_fit_train'] = inv.index.isin(canonical)
inv['reader_supervised_fit_row'] = inv.index.isin(ix['fit_train'])
inv['reader_selection_row'] = inv.index.isin(ix['fit_validation'])
inv['reader_fit_mmsi_seen'] = inv.mmsi.isin(t.iloc[ix['fit_train']].mmsi)
inv['reader_validation_mmsi_seen'] = inv.mmsi.isin(t.iloc[ix['fit_validation']].mmsi)
inv['pca32_fit_scope'] = inv.index.isin(ix['fit_train'])
inv['old_class_pass'] = [t.loc[i, 'pass_group'] in set(t.iloc[old[int(t.loc[i, 'class_id'])]].pass_group) for i in inv.index]
for p, pol in enumerate(['vv', 'vh']):
    inv[pol + '_token_valid'] = valid[inv.index, p]
    inv[pol + '_raw_exists'] = inv['image_' + pol].map(lambda s: Path(s).is_file())
for g, group in enumerate(groups):
    inv[group + '_edges'] = counts[inv.index, g]
    inv[group + '_usable'] = available[inv.index, g]

banks = pd.read_csv(HERE / 'banks.csv', dtype={'bank': str, 'mmsi': str})
bank_stats = []
for name, block in banks.groupby('bank', sort=False):
    block = block.sort_values(['class_id', 'slot'])
    positions = block['index'].to_numpy(int)
    actual = t.iloc[positions]
    assert len(positions) == 64 and actual.mmsi.nunique() == 64
    assert actual.upstream_role.eq('fit_train').all()
    assert np.array_equal(actual.obs_id.to_numpy(), block.obs_id.to_numpy())
    assert np.array_equal(actual.mmsi.to_numpy(), block.mmsi.to_numpy())
    assert np.array_equal(actual.pass_group.to_numpy(), block.pass_group.to_numpy())
    assert np.array_equal(actual.class_id.to_numpy(), block.class_id.to_numpy())
    assert np.array_equal(available[positions].reshape(8, 8, 4), available[old])
    assert np.array_equal(available[positions], block[groups].to_numpy(bool))
    inv['bank_' + name] = inv.index.isin(positions)
    bank_stats.append(dict(bank=name, n=len(block), mmsi=actual.mmsi.nunique(),
        passes=actual.pass_group.nunique(), old_mmsi=actual.mmsi.isin(t.iloc[old.ravel()].mmsi).sum(),
        old_observations=np.isin(positions, old).sum(),
        class_passes=actual.groupby('class_id').pass_group.nunique().tolist()))
inv.to_csv(HERE / 'pool_inventory.csv', index=False, encoding='utf-8-sig')

def summary(label, frame):
    out = frame.groupby('class_id').agg(
        observations=('mmsi', 'size'), mmsi=('mmsi', 'nunique'),
        passes=('pass_group', 'nunique'), vh_valid=('vh_token_valid', 'sum'))
    print(label, '\n' + out.to_string())
    print('total', len(frame), frame.mmsi.nunique(), frame.pass_group.nunique())

for label, frame in [
    ('strategy_fit', inv),
    ('fit_train', inv[inv.fit_train_candidate]),
    ('canonical_fit_train', inv[inv.canonical_fit_train]),
    ('fit_validation', inv[inv.reader_selection_row]),
    ('fit_excluded_mmsi', inv[inv.upstream_role.eq('fit_excluded_mmsi')]),
    ('new_train_mmsi', inv[inv.fit_train_candidate & ~inv.original_mmsi]),
    ('original_bank', inv[inv.in_original_bank]),
    ('canonical_new_dual', inv[inv.canonical_fit_train & ~inv.original_mmsi & inv.vh_token_valid])]:
    summary(label, frame)
print('BANKS', json.dumps(bank_stats, default=int))
print('raw_missing', inv[['vv_raw_exists', 'vh_raw_exists']].eq(False).sum().to_dict())
print('newbanks_union', banks[banks.bank.ne('original')].mmsi.nunique(),
      banks[banks.bank.ne('original')]['index'].nunique())

snapshots = {int(s['obs_id']): s for s in json.loads(Path(
    'E:/临时会话/same_vessel_gallery_20261019/local_condition_module/view_snapshot.json'
).read_text(encoding='utf-8'))}
panels = [
    ('raw_fishing_coverage.png', 'Fishing reference coverage',
     [(16657, 'Original: VH tokens unavailable'),
      (15874, 'Original: both polarizations usable'),
      (17937, 'Bank 101: new vessel and class-pass')]),
    ('raw_offshore_coverage.png', 'Offshore reference coverage',
     [(8183, 'Original: VH tokens unavailable'),
      (13344, 'Original: reused by banks 101/509/711'),
      (13000, 'New vessel: shared by all five banks')])]
for filename, title, cases in panels:
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), layout='constrained')
    fig.suptitle(title + '\nNative PNGs; nearest display; fixed 0-255 gray; no geometric transform', fontsize=14)
    for rownum, (obs, label) in enumerate(cases):
        row = t[t.obs_id.eq(obs)].iloc[0]
        i = int(row.name)
        for p, pol in enumerate(['vv', 'vh']):
            with Image.open(row['image_' + pol]) as im:
                pixels = np.array(im.convert('L'))
            ax = axes[rownum, p]
            ax.imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_title(label + '\nobs ' + str(obs) + ' | ' + pol.upper() +
                ' | ' + str(pixels.shape[1]) + 'x' + str(pixels.shape[0]) +
                ' | edges ' + str(counts[i, p*2]) + '/' + str(counts[i, p*2+1]), fontsize=10)
            ax.set_xlabel(str(row.mmsi) + ' | ' + row.pass_group + '\n' + row.port, fontsize=9)
            ax.set_ylabel('native pixel y', fontsize=9)
            priors = snapshots[obs]['images'][pol]['priors']
            target = priors.get('target_index')
            det = priors['detections'][target] if target is not None and target >= 0 else None
            print('RAW', json.dumps(dict(obs_id=obs, table_index=i, pol=pol,
                path=row['image_' + pol], shape=list(pixels.shape),
                minimum=int(pixels.min()), maximum=int(pixels.max()),
                mean=float(pixels.mean()), bright_ge200=int((pixels >= 200).sum()),
                token_valid=bool(valid[i, p]), target_index=target,
                confidence=det.get('confidence') if det else None,
                shape_usable=det.get('shape_usable') if det else None)))
    fig.savefig(HERE / filename, dpi=140)
    plt.close(fig)
print('AUDIT_AND_PANELS_COMPLETE')
```
