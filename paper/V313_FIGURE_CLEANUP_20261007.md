# v3.13 图像清理记录（2026-10-07）

依据用户“先把非313版本的图删掉”的要求，按来源逐项确认后删除图片文件。此次未改实验数据、配置、脚本、checkpoint、预测文件或论文正文。

共清理 28 个不同路径：当前主线 v3.13 工作区删除 21 个已跟踪图片；E:\DCT-Reg 同步删除这 21 个图片，并清理另外 7 个仅存在于旧本地工作区的图片。两个工作区中保留的图像资源与来源记录共 125 个文件，清理前后 SHA-256 校验一致。

## 已删除：旧版性能与审计图（13 个）

以下 10 个 paper_outputs 图片来自 v3.10。来源为 scripts/run_paper_completion.py、scripts/generate_figure3_all_cancers.py、scripts/kaplan_meier_analysis.py、scripts/kp_multi_cancer.py；输入指向 dct_v3.10 实验目录和 v310 模型。

- paper_outputs/figure3_dose_response_blca_5fold.png
- paper_outputs/figure3_dose_response_partial.png
- paper_outputs/figure_statistical_comparison.png
- paper_outputs/km_curves/km_curves_full_model.png
- paper_outputs/km_curves_all_cancers_grid.png
- paper_outputs/km_curves_multi_cancer/km_curves_blca.png
- paper_outputs/km_curves_multi_cancer/km_curves_hnsc.png
- paper_outputs/km_curves_multi_cancer/km_curves_kirc.png
- paper_outputs/km_curves_multi_cancer/km_curves_lusc.png
- paper_outputs/km_curves_multi_cancer/km_curves_skcm.png

另有 3 个 v3.10 干预审计图片。audit_results/blca_fold0_audit_summary.json 中记录了 v310 checkpoint；audit_results/AUDIT_REPORT.md 标明 DCTV310DirectionalRegularizedTransport。原始审计记录和报告保留。

- audit_results/visualizations/alpha_vs_risk_aggregate.png
- audit_results/visualizations/intervention_curves.png
- audit_results/visualizations/risk_change_distribution.png

## 已删除：v3.11 槽风险解释图（6 个）

scripts/proof_D_visualization.py 的 DATA_DIR 指向 results/dct_v311_blca_uni_fixed/per_slot_export，故这些图不属于 v3.13。

- results/proof_D_visualization/proof_D_all_folds_combined.png
- results/proof_D_visualization/proof_D_fold0.png
- results/proof_D_visualization/proof_D_fold1.png
- results/proof_D_visualization/proof_D_fold2.png
- results/proof_D_visualization/proof_D_fold3.png
- results/proof_D_visualization/proof_D_fold4.png

## 已删除：DCT_V32 图（2 个）

图片标题标为 DCT_V32，scripts/run_paper_completion.py 的复制来源为 results/dct_v32_calibration_curve.png 和 results/dct_v32_decision_curve.png。DCT_V32 是原有模型标签，不据此推断为 v3.2。

- paper_outputs/figure_calibration.png
- paper_outputs/figure_dca.png

## 已删除：旧本地工作区独有图片（7 个）

这些文件已不在当前主线工作区。旧架构图按文件名和图内版本确认；旧论文图片按 paper/README.md、paper/V313_README.md、scripts/recover_paper_sources.py 的来源说明与实际图内容确认。原始旧论文 DOCX 和历史归档保留。

- paper/jpg/dct_v311_architecture.png
- paper_outputs/dct_v310_complete_module_diagram.png
- paper/jpg/DCT architecture.png
- paper/jpg/Passenger or Driver.png
- results/fig1_audit_placeholder.png
- results/fig2_performance.png
- results/fig3_ablation.png

## 保留范围

- paper/figures 中的 v3.13 架构、Exp0–Exp6 消融、Full/Direct/Independent 对照、transport sweep、KM、病例图、队列通路图及其数据记录。
- v3.13 论文中的 SlotSPE 基线对照；它属于本论文实验，不按非 v3.13 模型清除。
- paper/history 中 v3.13 撤回对照图及其说明；版本仍是 v3.13，撤回原因不属于此次版本清理范围。
- 所有实验原始数据、报告、代码、旧论文文档、历史压缩归档和 third_party 参考图片。

保留不代表已达到论文使用标准。现有病例、通路和 KM 图仍按 V313_SLOTSPE_FIGURE_MAP.md 的计划完善，不能把旧版图替代为 v3.13 证据。

完整的本地文件路径与清理前 SHA-256 保存在 E:\DCT-Reg\tmp\v313_figure_cleanup_20261007.json（verification=passed）。清理完成时仅修改本地；后续应用户“重新写，然后重新push”的要求，与更新后的 v3.13 作图计划和服务器提示词一并交付。服务器执行新计划时仍须保留旧实验目录与已有产物，不按此记录删除服务器数据。
