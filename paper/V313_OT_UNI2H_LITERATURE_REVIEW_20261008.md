# DCT v3.13 OT 与 UNI2 h 论文参照

本次按作者进一步要求，沿用 SlotSPE 表 1 中其他方法的数据，只排除 SlotSPE 自身全部三行。扩展表包含 21 个参照方法与 DCT，所有方法标注年份，保留十癌种和相同十队列的 Overall。公开分数已直接汇总，不需要服务器重搜；服务器后续负责同条件复现原始记录。

## 优先阅读与比较的方法

| 方法 | 来源与定位 | 已核实的输入及终点 | 对 DCT 的意义 |
|---|---|---|---|
| MOTCat (2023) | ICCV 2023，[原文](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Multimodal_Optimal_Transport-based_Co-Attention_Transformer_with_Global_Structure_Consistency_for_ICCV_2023_paper.pdf) | 原论文病理与组学，ResNet50，OS；扩展表采用 SlotSPE 作者用 UNI 在 DSS 上重评估的完整十队列值 | 直接检验多模态 OT 耦合的价值；原论文 OS 值不混入 DSS 排名 |
| TTA (2025) | [Together Then Apart 原文](https://arxiv.org/html/2511.18089)，[代码](https://github.com/Y-Research-SBU/TTA) | 病理与组学，UNI，DSS，五折 site-stratified | 原型与联合 UOT 对齐，兼顾模态的独特信息；优先多模态参照 |
| OTSurv (2025) | MICCAI 2025，[原文](https://papers.miccai.org/miccai-2025/paper/1359_paper.pdf)，[代码](https://github.com/Y-Research-SBU/OTSurv) | 仅病理，UNI 1024 维，DSS | 检验带全局质量与局部不确定性约束的 OT 聚合；输入少于 DCT |
| STEPH (2026) | CVPR 2026，[原文](https://arxiv.org/html/2603.10526)，[官方补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Liu_Sparse_Task_Vector_CVPR_2026_supplemental.pdf)，[代码](https://github.com/liupei101/STEPH) | 仅病理，补充材料明确 UNI2-h 1536 维，DSS | 相同病理编码器的参考；利用跨癌种知识，输入和训练策略仍不同 |
| ProtoPathway (2026) | [预印本](https://arxiv.org/html/2605.21454)，[作者代码及 UNI2-h 说明](https://github.com/AmayaGS/ProtoPathway) | 病理与转录组，UNI2-h，OS | 原型病理与通路组学结构最接近；适合方法讨论，须重新评估 DSS 后才能进入同终点主表 |
| ME-Mamba (2026) | [原文](https://arxiv.org/html/2509.16900)，[期刊页面](https://www.sciencedirect.com/science/article/pii/S0895611126000364) | 病理与组学，ResNet50 / UNI / CONCH 三组；不是 UNI2-h；当前读取文本未明确 OS/DSS | OT 局部对齐与多专家融合候选；暂不加入 DSS 排名 |
| MCSP-OTMR (2026 在线) | [出版社摘要](https://www.sciencedirect.com/science/article/pii/S0957417426027314)，DOI 10.1016/j.eswa.2026.133823 | 病理与组学，OT 重建与 Fisher 模态重加权；完整配置、数值表未取得 | 与 OT 重建主张直接相近，必须讨论；不得把 OT 重建本身当作尚无人提出的新意 |
| CROPKT / ROUPKT (2025) | [原文](https://arxiv.org/html/2508.13482)，[代码](https://github.com/liupei101/CROPKT) | 仅病理，UNI2-h，DSS，跨癌种迁移 | 补充同编码器与跨癌种参照；当前先采用 STEPH 官方表中的已核验 MIL 结果 |

## 已汇总的公开数值

主动比较表：paper/tables/v313_ot_uni2h_comparison/table_v313_ot_uni2h_reference.{docx,pdf,png,svg,html,json}。

数值输入与来源 SHA256：paper/V313_OT_UNI2H_COMPARISON_INPUT.json。共 183 个可按相同癌种名称对齐的公开均值，另有本研究真实 50 个折值。原表 15 个非 SlotSPE 方法共 150 个十队列均值、150 个离散度、15 个原文 Overall 原样保留；这些数据是 SlotSPE 作者的重评估，并非各方法原论文的分数。

原表方法为 MLP、SNN、SNNTrans、ABMIL、TransMIL、CLAM-SB、CLAM-MB、Porpoise、MCAT、MOTCat、CMTA、SurvPath、PIBD、MMP、LD-CVAE。额外六项为 ILRA、R2T-MIL、Patch-GCN、OTSurv、TTA、STEPH，贡献 33 个对齐的公开均值。MOTCat、ABMIL、TransMIL 在本表统一采用完整十队列来源；未从 TTA 或 STEPH 的复现分数中逐格取更高值。

每列已有外部数值数量：BRCA 21、COADREAD 19、KIRC 17、UCEC 19、LUAD 16、LUSC 15、HNSC 19、SKCM 19、BLCA 21、STAD 17。所有癌种可排名。额外方法没单独报告的队列保留 NR，原始合并队列不改名。Overall 仅对相同完整十队列行排名；STEPH 十三队列、TTA 五队列、OTSurv 六队列的总体值不混入这一列。

年份以原文对应的方法发表年为主，全部记录 year、year_kind 和 year_source。MLP 1998 来自原文引用的 Haykin 教材，不是 MLP 首次提出年；SNNTrans 2021 取其组件引用中的后一年，组合为 SNN 2017 与 TransMIL 2021，不是独立发表年份。两者表内加单星。TTA 标初次 arXiv 年份 2025，但保存的分数取 2026 年 v2。DCT 2026 是当前研究年份，以双星标识，不表示已经发表。

| 年份 | 方法 |
|---|---|
| 1998 引用教材 | MLP |
| 2017 | SNN |
| 2018 | ABMIL |
| 2021 | SNNTrans（组件引用）、TransMIL、CLAM-SB、CLAM-MB、MCAT、Patch-GCN |
| 2022 | Porpoise |
| 2023 | MOTCat、CMTA、ILRA |
| 2024 | SurvPath、PIBD、MMP、R2T-MIL |
| 2025 | LD-CVAE、OTSurv、TTA（初次预印本） |
| 2026 | STEPH、DCT（当前工作） |

原表年份引用见来源 PDF 第 7 页 4.2；新增来源核验包括 [ILRA ICLR 2023](https://openreview.net/pdf?id=01KmhBsEPFO)、[RRT-MIL CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_Feature_Re-Embedding_Towards_Foundation_Model-Level_Performance_in_Computational_Pathology_CVPR_2024_paper.html)、[TTA arXiv 提交记录](https://arxiv.org/abs/2511.18089)和 [STEPH CVPR 2026 说明](https://arxiv.org/abs/2603.10526)。R2T-MIL 是 STEPH 表 S2 使用的名称，原 CVPR 论文称为 RRT-MIL；本表保留数值来源的行名，不混为新方法。

红色加粗和下划线表示可用 DSS 公开值中的第一、第二。编码器、模态、患者纳入、划分和评分口径尚未统一，不能当作同条件独立测试下的 SOTA。相同完整癌种集合的 Overall 仍有这些协议差异。

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
| patch_budget | 少量组织块能否保留排序能力 | 横轴是保留比例，比较策略的 C-index。修正版横轴直接取 budget.retained_fraction，并检查跨折设置一致 |

这些图只覆盖两个癌种、Full 的现有记录；它们不等于十癌种 KM，也不是外部方法比较。对照中的 Direct 仅去掉 cross 重建的运输使用，不能简称整个模型无 OT，因为预测主路仍使用 OT。Independent 去掉槽对联合耦合，不能泛称网络没有任何 attention。

导读暂不把没有本机核验的服务器统计或图片结论写成正文已验证结果。


诊断审计更新：旧 CSV 的 reconstruction/native 与 shuffled 是重建误差，不是 C-index；旧患者检索使用重复归一化的训练均值，已改为 v2。上文导读的检索结论需基于修正版重新导出，不以旧 Top-1 判断患者特异性。具体影响和服务器步骤见进度第 8 节及 V313_DIAGNOSTIC_BUGFIX_SERVER_PROMPT_20261008.txt。
