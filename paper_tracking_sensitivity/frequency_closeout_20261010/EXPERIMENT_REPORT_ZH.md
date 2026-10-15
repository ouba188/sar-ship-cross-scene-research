# 最终校准对照

| dataset | stage | mode | oa | ba | macro_f1_8 | delta_oa_vs_learned | delta_ba_vs_learned | delta_oa_vs_standalone_local | delta_ba_vs_standalone_local | delta_oa_vs_margin_selected | delta_ba_vs_margin_selected | delta_oa_vs_pair_local | delta_ba_vs_pair_local | positive_draws_ba_vs_learned | positive_draws_ba_vs_standalone_local | positive_draws_ba_vs_margin_selected | positive_draws_ba_vs_pair_local |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f0921 | full | calibrated | 31.0316 | 26.4011 | 23.7185 | -0.0755 | 0.0417 | -0.0783 | 0.0362 | -0.1530 | -0.0876 | -0.1496 | -0.0858 | 3 | 2 | 0 | 0 |
| f0921 | full | learned | 31.1072 | 26.3593 | 23.7858 | 0.0000 | 0.0000 | -0.0028 | -0.0056 | -0.0775 | -0.1294 | -0.0741 | -0.1275 | 0 | 2 | 2 | 2 |
| f0921 | full | source_selected | 31.0316 | 26.4011 | 23.7185 | -0.0755 | 0.0417 | -0.0783 | 0.0362 | -0.1530 | -0.0876 | -0.1496 | -0.0858 | 3 | 2 | 0 | 0 |
| oss2 | full | calibrated | 35.1155 | 32.1760 | 27.0976 | -0.2391 | -0.1965 | -0.2391 | -0.2841 | -0.2554 | -0.2935 | -0.2445 | -0.3037 | 1 | 1 | 1 | 0 |
| oss2 | full | learned | 35.3545 | 32.3724 | 27.2773 | 0.0000 | 0.0000 | 0.0000 | -0.0876 | -0.0163 | -0.0971 | -0.0054 | -0.1072 | 0 | 1 | 2 | 3 |
| oss2 | full | source_selected | 35.3545 | 32.3724 | 27.2773 | 0.0000 | 0.0000 | 0.0000 | -0.0876 | -0.0163 | -0.0971 | -0.0054 | -0.1072 | 0 | 1 | 2 | 3 |
| self24ports | full | calibrated | 63.2828 | 54.2098 | 32.1798 | 0.0744 | -0.0284 | -0.0375 | -0.0955 | -0.0315 | -0.0162 | -0.0330 | -0.0376 | 2 | 2 | 2 | 1 |
| self24ports | full | learned | 63.2084 | 54.2383 | 32.1738 | 0.0000 | 0.0000 | -0.1119 | -0.0670 | -0.1059 | 0.0122 | -0.1074 | -0.0092 | 0 | 1 | 2 | 3 |
| self24ports | full | source_selected | 63.3049 | 54.3111 | 32.2198 | 0.0965 | 0.0728 | -0.0154 | 0.0058 | -0.0094 | 0.0850 | -0.0109 | 0.0637 | 4 | 2 | 2 | 3 |

OA/BA为百分数，增量为百分点；先平均五组支持抽样，再按任务等权。固定交接使用原learned位置。只有一个新增校准器，未搜索阈值；source_selected只从原主模型learned和calibrated中选择。其他历史结果仅作参考，不参与本轮选择。目标真值仅用于全部预测冻结后的评估。weighted correctness/Brier/ECE以真实评估类平衡权重作事后诊断，不能当作理论校准保证。收口决策与文献见CLOSEOUT_ZH.md和RELATED_WORK_ZH.md。
