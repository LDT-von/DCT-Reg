# DCT v3.13 OT 与 UNI2 h 论文参照

本次按作者要求，将正文比较对象改为 OT 生存预测方法和采用 UNI2-h 的生存预测方法。公开分数已从原文汇总，不需要服务器重新搜集；服务器后续负责同条件复现原始记录。当前表格包含 9 个参照方法与 DCT，保留全部十癌种列。SlotSPE 不再作为本次主动比较对象。

## 优先阅读与比较的方法

| 方法 | 来源与定位 | 已核实的输入及终点 | 对 DCT 的意义 |
|---|---|---|---|
| MOTCat | ICCV 2023，[原文](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Multimodal_Optimal_Transport-based_Co-Attention_Transformer_with_Global_Structure_Consistency_for_ICCV_2023_paper.pdf) | 原论文病理与组学，ResNet50，OS；新表采用 TTA 作者用 UNI 在 DSS 上复现的数值 | 直接检验多模态 OT 耦合的价值；原论文 OS 值不混入 DSS 排名 |
| TTA | [Together Then Apart 原文](https://arxiv.org/html/2511.18089)，[代码](https://github.com/Y-Research-SBU/TTA) | 病理与组学，UNI，DSS，五折 site-stratified | 原型与联合 UOT 对齐，兼顾模态的独特信息；优先多模态参照 |
| OTSurv | MICCAI 2025，[原文](https://papers.miccai.org/miccai-2025/paper/1359_paper.pdf)，[代码](https://github.com/Y-Research-SBU/OTSurv) | 仅病理，UNI 1024 维，DSS | 检验带全局质量与局部不确定性约束的 OT 聚合；输入少于 DCT |
| STEPH | CVPR 2026，[原文](https://arxiv.org/html/2603.10526)，[官方补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Liu_Sparse_Task_Vector_CVPR_2026_supplemental.pdf)，[代码](https://github.com/liupei101/STEPH) | 仅病理，补充材料明确 UNI2-h 1536 维，DSS | 相同病理编码器的参考；利用跨癌种知识，输入和训练策略仍不同 |
| ProtoPathway | [预印本](https://arxiv.org/html/2605.21454)，[作者代码及 UNI2-h 说明](https://github.com/AmayaGS/ProtoPathway) | 病理与转录组，UNI2-h，OS | 原型病理与通路组学结构最接近；适合方法讨论，须重新评估 DSS 后才能进入同终点主表 |
| ME-Mamba | [原文](https://arxiv.org/html/2509.16900)，[期刊页面](https://www.sciencedirect.com/science/article/pii/S0895611126000364) | 病理与组学，ResNet50 / UNI / CONCH 三组；不是 UNI2-h；当前读取文本未明确 OS/DSS | OT 局部对齐与多专家融合候选；暂不加入 DSS 排名 |
| MCSP-OTMR | [出版社摘要](https://www.sciencedirect.com/science/article/pii/S0957417426027314)，DOI 10.1016/j.eswa.2026.133823 | 病理与组学，OT 重建与 Fisher 模态重加权；完整配置、数值表未取得 | 与 OT 重建主张直接相近，必须讨论；不得把 OT 重建本身当作尚无人提出的新意 |
| CROPKT / ROUPKT | [原文](https://arxiv.org/html/2508.13482)，[代码](https://github.com/liupei101/CROPKT) | 仅病理，UNI2-h，DSS，跨癌种迁移 | 补充同编码器与跨癌种参照；当前先采用 STEPH 官方表中的已核验 MIL 结果 |

## 已汇总的公开数值

主动比较表：paper/tables/v313_ot_uni2h_comparison/table_v313_ot_uni2h_reference.{docx,pdf,png,svg,html,json}。

数值输入与来源 SHA256：paper/V313_OT_UNI2H_COMPARISON_INPUT.json。共 49 个可按相同癌种名称对齐的公开均值，另有本研究真实 50 个折值。输入同时保留论文全部原始队列名称、均值和离散度，未把均值虚构成五折结果。

表中 MOTCat 来自 TTA 表 1 的作者复现，而不是 MOTCat 原论文，也不是 SlotSPE 表中的转录。OTSurv 来自原论文表 1。STEPH、ABMIL、TransMIL、ILRA、R2T-MIL、Patch-GCN 来自 STEPH 官方补充表 S2；后五项是 STEPH 论文中的复现值，不能写成这些模型原始论文的分数。

原文没有单独报告的队列记为 NR。STEPH 的 KIPAN、LUNG、STES 分别为合并队列，不能替换 KIRC、LUAD/LUSC、STAD。TTA/OTSurv 的 CRC 暂不映射到 COADREAD，必须核对患者定义后才可映射。每列已有外部数值数量：BRCA 9、COADREAD 6、KIRC 3、UCEC 6、LUAD 1、LUSC 0、HNSC 6、SKCM 6、BLCA 9、STAD 3。LUSC 只有 DCT，因此不标第一名。

红色加粗和下划线仅表示该列可用 DSS 报告值中的第一与第二。不同论文的模态、患者纳入、划分和 checkpoint 选择没有统一，本研究 DCT 仍是 best-validation / legacy_val 开发汇总；这些名次不能作为统一独立测试下的 SOTA 结论。不同队列集合的 Overall 不计算混合排名。

ProtoPathway 表 2 的 OS 均值单独记录：BRCA 0.649，BLCA 0.646，COADREAD 0.740，HNSC 0.642，STAD 0.674，原文 Overall 0.670。它们不进入上面的 DSS 排名。其表注将 ± 标为 standard error，不能擅自称作五折标准差。

ME-Mamba 表 1 中 UNI 版本：BLCA 0.6583，BRCA 0.7313，UCEC 0.7508，GBMLGG 0.8624，LUAD 0.6695。这里只记录公开数值，待终点核验，不替代 DSS 主表。

## 当前方法叙事需要收紧的地方

已有论文覆盖多模态 OT 对齐、原型表示与 OT 重建。DCT 应结合实际代码说明：模态内跨患者复用的原型坐标；阶段条件、多几何运输与事件风险读取；重建复用预测的运输计划，以停止梯度的预测阶段门控聚合，再恢复编码通路 tokens。与 MCSP-OTMR 的差异只能依据已经读到的摘要作初步定位；取得全文后还须核对是否存在同样的运输复用和监督设计。

优先同条件复现对象为 MOTCat、TTA、OTSurv、STEPH；ProtoPathway 在适配 DSS 后作为结构相近的多模态候选。公开表已经有数值，同条件表仍需真实训练记录，不能以文献排名代替消融、独立评估和机制对照。

## 第 6 节汇总图如何阅读

用户指定的 d2d4877 只更新进度记录。第 6 节记录五类图，每类 BLCA/KIRC 各一张，共 10 张，PNG 与 PDF 共 20 文件。服务器目录为 /data1/DCT-Reg/results/v313_additional_20261008/figures_summary/。本机没有该目录，下面解释依据已推送的汇总脚本，未声称已经看到服务器图像。

| 图 | 想验证什么 | 如何读 |
|---|---|---|
| reconstruction | 正确运输/配对是否有助于恢复组学通路表示 | 重建误差越低越好；患者检索越高越好。低误差不自动等于患者特异性，检索需同时超过随机基准 |
| pathway_advantage | 哪些通路从正确运输中获益 | 横向差值是对照误差减 native 误差；正值支持 native 更准，负值表示对照更准。不是通路重要性或风险相关性 |
| pairing | 病理与组学配对打乱后预测是否变化 | 同时看 C-index 与风险变化。风险变了但 C-index 没有改善，不能证明预测收益；重复打乱不是新增独立样本 |
| patch_deletion | 模型选出的组织块是否比随机块更影响预测 | 比较删除高分、低分、随机块后的 C-index 和风险变化。需要对照和折间稳定性，不只看一条曲线下降 |
| patch_budget | 少量组织块能否保留排序能力 | 横轴是保留比例，比较策略的 C-index。当前脚本从删除比例换算横轴，需确认与原始 budget 记录一致 |

这些图只覆盖两个癌种、Full 的现有记录；它们不等于十癌种 KM，也不是外部方法比较。对照中的 Direct 仅去掉 cross 重建的运输使用，不能简称整个模型无 OT，因为预测主路仍使用 OT。Independent 去掉槽对联合耦合，不能泛称网络没有任何 attention。

导读暂不把没有本机核验的服务器统计或图片结论写成正文已验证结果。
