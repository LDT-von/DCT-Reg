# DCT v3.13 论文

当前中文正文为 DCT_v313_初稿.md，Word 为 DCT_v313_初稿.docx。2026-10-07 版本继续沿用原方法结构和可编辑公式，更新十队列结果、七组训练目标、重跑运输机制对照及解释分析方法。

图 1 已接入依据实际代码重绘的框架图，提供中英文 PDF/SVG/PNG，M1-M4 标出相对 SlotSPE 的改动；图 4 已插入实际十折运输计划 sweep；图 2–3、5–7 继续留空。图号依次为架构、损失消融、逐折机制对照、固定模型运输干预、KM、病例解释、队列通路热图。源文件 fig3_sweep_* 与 fig3_transport_sweep.* 属于服务器旧编号；论文版 fig4_transport_sweep_manuscript.* 由相同 JSON 重绘，仅改版式、图号和样本标准差口径。尚未收到图片的结果段不预写趋势或生物学结论。

scripts/plot_v313_manuscript_sweep.py 只读取已保存的十折 JSON 生成论文版图，无训练或 checkpoint 推理。正文 §4.8、讨论和结论已纳入固定模型低敏感性观察；它不替代重训控制臂。逐患者最佳预测对齐与边缘残差仍未在本次 JSON 中交付。LLM_LEDGER.md 末尾保留针对 24f3f10 的来源核对补记。

控制臂表由 results/v313_evidence_v2/controls_audit.json 汇总，20/20 complete，服务器审计记录 passed=True。历史异常批次不引用。十队列值来自作者提供的逐折汇总，统一重算样本标准差；完整原始来源、SlotSPE 四组完整数值及公平比较条件仍列在文稿附录 B。

scripts/build_v313_paper.py 从 Markdown 生成 Word；本次构建不重画或覆盖已有架构图，不运行模型。scripts/export_v313_paper_pdf.ps1 用 Word 生成内部排版检查 PDF。V313_MANUSCRIPT_STATUS.json 保存本次采用的数值来源与待补清单。

旧 DCT_唯一初稿.* 属于 v3.10，不作为本稿数值依据；history/retracted_transport_controls_20261005/ 中的旧机制图也不恢复引用。解释代码的独立计划见 V313_INTERPRETABILITY_IMPLEMENTATION_PLAN.md，新设计接口尚不因此变成可运行实现。


图 5 的作者最新要求：每个癌种一张总体图，仅 Full 高/低风险两条曲线。
先按各折训练风险中位数将验证患者分组，再汇总五折组别和结局；每名患者
只出现一次。蓝色低风险、红色高风险，附删失标记、名义 95% CI 和在险人数。
旧 km_*_per_fold.png 保留为历史产物，其风险标签和阈值存在问题，不接入正文。

新脚本仅读取 export --km 的产物，没有训练或推理功能。已有患者预测和
训练阈值必须齐备，缺项时报错。默认 exp6；--arm direct 或 independent
可分别生成控制臂图。任意癌种可重复指定 --cancer，只要具有全五折导出。
本次只做合成数据校验，尚未生成新的真实总体曲线。

服务器运行（由作者执行）：

```bash
python scripts/plot_fig5_km_curves.py \
  --exports results/v313_interpretability_v1/exports \
  --arm exp6 --cancer blca --cancer kirc --check-only

python scripts/plot_fig5_km_curves.py \
  --exports results/v313_interpretability_v1/exports \
  --arm exp6 --cancer blca --cancer kirc \
  --output paper/figures/fig5_oof
```

每癌种输出 PNG、可编辑文字 PDF、患者分组 CSV 和来源/阈值 JSON。
患者 CSV 中保留原始 risk 仅供核对，不计算跨折 pooled C-index。
--show-logrank 是可选的探索性统计，不作为独立外部验证显著性。
若 check-only 报缺折或缺 km_train_median，先按经过核验的 Full manifest
补对应 export --km；不要用控制臂清单代替 Full 清单，也不回退到验证中位数。


2026-10-08 框架图修订：见 `V313_ARCHITECTURE_REDESIGN.md` 与 `figures/v313_architecture_redesign/`。Markdown 正文和图状态已接入英文新图；已有 Word 文件保留，下一次运行构建脚本会采用新图。中文图用于说明，各版本共享同一计算拓扑。

框架图连线修订：预测主干横向排列，cross/self 重建分为两条水平支路。取消跨面板长折线，同名 S/T/g 标记说明张量复用；模型计算拓扑保持一致。中英文 SVG/PDF/PNG 与正文图注同步更新。
