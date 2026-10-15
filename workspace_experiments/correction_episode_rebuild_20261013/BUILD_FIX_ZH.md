# build边界修正

初次build退出于`joint_core.make_parts -> StratifiedGroupKFold(4)`：
`ValueError: n_splits=4 cannot be greater than the number of members in each class.`

新B16支持可能各类最多3例，船身份仍16个不同，故四折船隔离可行，只是原分层实现拒绝。新run.py仅在max(class_count)<4时采用固定seed的KFold四分，16例一船一行保证船级隔离，paired_oof仍逐折断言。其他任务不变；不筛掉任务、不降低折数或预算。未训练任何新选择器、未查看新目标结果时修复。
