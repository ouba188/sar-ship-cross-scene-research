# 冻结后的确认范围

seed17的10个fit全部完成，尚未加载本轮378评价图像。原10个fit记录完整保留。

四方案最佳验证228的OA/BA均为68.859649/63.042496，top1与旧coordinated全部相同，均只救回000414。head_only选head_small/epoch3；broadcast_head与affine_head选head_small/epoch1；affine_frozen选base/epoch4。

按事先BA/OA/-NLL顺序，在两个affine架构中选择affine_frozen（NLL1.162696，affine_head为1.163140）；head_only是配套追加训练对照（NLL1.161771）。选择依据不是378结果。

仅给affine_frozen与head_only补seed29/43，再对三种子的固定检查点评价378。旧冻结头广播coordinated的三种子结果来自上一轮缓存，同划分/起点/配对预算，不重训。broadcast_head与affine_head保留seed17验证结果，不声称它们有三种子扩围。

本轮头部LR1e-5的负结果不能被head_small掩盖；较小LR避免了部分退化，但既有+1并非模块独立收益。若确认仍未超过简单对照，不自动追加regionmean/独立极化训练或更多调参。
