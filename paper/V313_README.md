# DCT v3.13 论文

当前中文正文为 DCT_v313_初稿.md，Word 为 DCT_v313_初稿.docx。2026-10-08 接入已有实验图，不运行模型，不改实验原始数据。

图 1–7 均已接入正文：架构、七组损失消融、训练机制对照、固定模型运输计划替换、BLCA/KIRC 总体 KM、BLCA 病例非组织子图及队列通路绝对注意力差值。图 6 仅包含 TCGA-2F-A9KP 的通路—槽矩阵和实际运输关联；组织空间热图、Top-5 组织块和 KIRC 病例仍未完成。

图 7 按每折验证患者风险 midrank 百分位划分 Q1–Q4。主图使用各通路组均值减去四组均值等权平均的差值（保留正负号与原始权重单位），保留真实量级，不做逐行 z-score。BLCA 四组各 95 人；KIRC 为 120/123/122/123 人。图注与正文已统一。原始平均权重放在附录 D 的补充图 S3，使用从零开始的绝对色标。

图 5 采用不同分组口径：每折训练患者风险中位数固定高低阈值，再汇总验证患者。BLCA 380 人（low=172，high=208），KIRC 488 人（low=225，high=263）；数值按患者 JSON 核对，未报告 log-rank p 值。旧逐折 KM 和旧病例风险方向错误的总览图不接入正文。

图 2、3 的排版已重绘。图 2 包含五折散点与样本标准差；图 3 使用事实 sweep 的精确 Full 折值和 20 项重跑控制臂审计折值，差值从未四舍五入数据计算。实验成绩不变，原稿先对均值取四位再相减的末位差异已同步修正。

作图脚本 scripts/prepare_v313_manuscript_panels.py 只读取已保存记录，生成 PDF/SVG/PNG。输出目录 figures/v313_manuscript_integrated/ 包含来源 hash、原始数值快照和差值矩阵；原始图与 JSON 保留。scripts/build_v313_paper.py 仅从 Markdown 构建 Word，保留可编辑公式，不生成新实验结果。

十队列原始溯源、SlotSPE native/matched 完整配方与逐折结果、Full/control 最终患者和配置配对仍待补。补充图 S1 槽诊断与 S2 实测效率未生成。新增服务器实验仅同步了进度报告，原图与逐折数据需下载并核验后再接入，不从进度摘要直接填写结果。

旧 DCT_唯一初稿.* 属于 v3.10，不作为本稿依据。真实组织病例图资源未齐的状态保留在 V313_MANUSCRIPT_STATUS.json 中。

复现现有图与 Word（只作图和排版，不执行模型）：

```powershell
python scripts/prepare_v313_manuscript_panels.py
python scripts/build_v313_paper.py
powershell -NoProfile -File scripts/export_v313_paper_pdf.ps1
```

最后一步使用本机 Word 导出内部排版检查 PDF 到 tmp/v313_paper_qa/。本次成稿已完成全部 23 页视觉检查，含 12 个图片面板、9 张表和 15 个原生公式对象。

图 5 的服务器作图入口仍为 scripts/plot_fig5_km_curves.py；仅消费既有 export --km 产物，缺少训练风险中位数或完整五折时停止。新增实验仍由作者执行：

```bash
python scripts/plot_fig5_km_curves.py \
  --exports results/v313_interpretability_v1/exports \
  --arm exp6 --cancer blca --cancer kirc --check-only

python scripts/plot_fig5_km_curves.py \
  --exports results/v313_interpretability_v1/exports \
  --arm exp6 --cancer blca --cancer kirc \
  --output paper/figures/fig5_oof
```

框架图与连线说明见 V313_ARCHITECTURE_REDESIGN.md 和 figures/v313_architecture_redesign/；Word 已采用该图。解释设计与剩余资源条件见 V313_INTERPRETABILITY_IMPLEMENTATION_PLAN.md。
