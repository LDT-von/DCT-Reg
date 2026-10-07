# DCT v3.13 新增图和机制验证（2026-10-08）

本轮范围只针对 **DCT v3.13**。在远端 `d7a16d44e663281fa1a45e80d7c171b36c75a3d3` 的证据工具基础上增加代码，不改模型、训练目标或已有论文图。开发时只做合成测试，没有启动真实数据处理、推理或训练。

## 1. 论文检索得到的启发

检索日期：2026-10-08。下面是论文实际分析与本次实验设计的对应关系；本次新增对照是针对 DCT 的推导，并非声称原论文实施过这些相同对照。

| 原始论文 | 读到的分析 | 对 v3.13 的启发 |
|---|---|---|
| [SlotSPE，ICLR 2026 / arXiv v2](https://arxiv.org/html/2512.01116v2)，§4.3、4.7、H、I | 缺失组学、槽位/通路/WSI解释、效率、临床协变量与校准 | 已有病例热图之外，需要量化高亮区域是否影响预测。原始 SlotSPE 已有重建，不能把“有重建”当作 v3.13 独占创新。 |
| [MOTCat，ICCV 2023](https://arxiv.org/html/2306.08330v2)，§4.3、4.4、附录 B/C | OT/微批消融、微批大小、共注意力图、风险分层 | 检验耦合组织和采样预算。已有 Direct/Independent 训练对照保留；本次冻结 checkpoint 的干预是补充机制诊断。 |
| [SurvPath，CVPR 2024，官方代码](https://github.com/mahmoodlab/SurvPath) 与 [官方补充材料](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Jaume_Modeling_Dense_Multimodal_CVPR_2024_supplemental.pdf) | patch/通路/基因多级归因、跨模态对应、临床协变量比较 | 在热图旁加 top/bottom/random 删除曲线，检查关注程度与预测影响是否相符。 |
| [PathGen，Nature Communications 2026](https://www.nature.com/articles/s41467-025-66961-9)，Results: similarity evaluation / transparency evaluation | 合成与真实转录组相似度、预测增益、共注意力一致性 | DCT 输出是潜在通路 token：使用同目标空间的误差与患者检索，不能照搬原始基因表达的 Spearman 解释。 |
| [BMLSurv，Pattern Recognition 2026](https://www.sciencedirect.com/science/article/pii/S0031320325010593) | 论文摘要提出病理/组学学习能力差异及梯度传播失衡 | 配对破坏对照可以检查预测对病理患者身份的依赖；它不是梯度失衡的直接证明。 |
| [MCSP-OTMR，ESWA 2026](https://www.sciencedirect.com/science/article/pii/S0957417426027314) | 摘要提出 OT 跨模态重建、缺失组学和模态重加权 | 这是非常接近的相关工作。论述应落实到 v3.13 的具体耦合与 token 重建路径，避免笼统声称 OT+重建本身首次出现。这里只阅读到摘要/预览。 |

## 2. 已有证据与新增证据

远端已有 Exp0–6、Full/Direct/Independent、SlotSPE 对比、KM、耦合混合 sweep、病例热图、队列通路图、效率及槽位诊断。请继续使用 `paper/V313_EVIDENCE_RUNBOOK.md` 的 manifest 审计流程；本轮不重复训练这些项目。

本轮新增三类实验，使用已有最佳 checkpoint：

| 优先级 | 实验 | 核心问题 | 新图 |
|---|---|---|---|
| 1 | 患者特异重建 + 患者配对破坏 | 解码器是否只记住通路均值？接收患者的真实 WSI 是否带来额外信息？ | 重建误差/患者检索图、全通路重建优势热图、错配患者的预测变化图 |
| 2 | top/bottom/random patch 删除 | 热图关注的 patch 是否比同数量随机 patch 更影响预测？ | 删除比例—ΔC-index 与删除比例—绝对风险变化曲线 |
| 3 | 冻结模型的 patch 预算 | 只保留部分评估 patch，预测是否稳定？ | 保留比例—C-index 与保留比例—绝对风险变化曲线 |

每个 fold 生成五类图，均保存 PDF 和 PNG。逐患者原始数组在 `additional.npz`，实验参数/身份/哈希在 `additional.json`，图索引在 `figure_index.json`，逐折指标在 `fold_metrics.csv`。

### A. 重建/患者配对

固定 checkpoint 与接收患者组学，在同一个解码器内计算六个条件：

1. `native`：该 checkpoint 训练时的真实跨模态路径；Full 为 transport，Direct 为 direct。
2. `product`：用 factual 边缘分布的乘积替代每个 stage/geometry 的耦合，保持 factual event gate，隔离耦合组织的作用。
3. `direct`：原始 WSI slots 直接进入同一个解码器。
4. `shuffled`：同一 fold 的另一名验证患者提供 WSI slots；保留接收患者组学 slots 与目标，重新计算 costs、OT 和 event gate。每次是一一错配，无自己配自己，不跨 fold。
5. `train_mean`：仅训练患者提供的、逐通路归一化 token 均值。验证患者不参与均值估计。
6. `self`：omics slots 自重建，作为条件内参照；必须查看有效 self 系数，未训练的 self 分支不能据此评价自重建质量。

误差严格使用模型的 `0.5*(cosine distance + Smooth-L1)`，在 LayerNorm 后的通路 token 上计算。检索先减去训练均值，再展平并计算 cosine，判断重建能否在当前 fold 的候选患者中找回正确目标。报告 Top-1、MRR、1/N 机会水平、零向量数量和目标患者间方差；并列按最差名次处理，恒定解码不会凭行顺序获得虚假成功。

所有错配重复的 donor ID、重建误差、检索名次、预测风险及 C-index 均保留，不挑最有利的一次。热图包含所有通路，不按生存结果挑通路；名字在数组 `pathway_names` 内。

**怎样解读：** native 比 train_mean/shuffled 重建更好且 centered retrieval 有辨识力，支持“存在患者特异信息”；native 比 product 更好，支持耦合组织有信息。错配后的 C-index 下降说明预测依赖正确的病理/组学配对。若没有差异，就应如实报告，不能称其有效。

**边界：** recipient omics 仍参与 OT 和 gate，这不是 WSI-only imputation；decoder token 不是原始基因表达。冻结 checkpoint 的 direct/product 路径可能偏离训练分布，不能替代训练匹配的 Full/Direct/Independent 比较。不同 checkpoint 的目标编码器可能不同，不应直接把潜在误差大小当作跨模型质量排名。

### B. patch 删除

使用既有 exporter 的 final slot pooling + prototype rollout。将 WSI slots 的注意力按 slot 等权平均为一个预先固定的 patch 分数；比较最高分、最低分和随机 patch 删除。排名只用完整 bag 的注意力，不用验证结局或删除后的预测选排名。

每个删除后的 bag 均重新编码 slots、重算成本/所有 OT plans 和 survival head。默认删除真实 patch 的 `0,10%,25%,50%,75%`，向下取整，至少保留一个。random 默认 5 个重复；top/bottom 不重复相同推理。记录真实 patch 数和实际删掉的数量，不能把不足一枚 patch 的取整变化称为有意义扰动。

输出 `ΔC-index` 与 `|Δrisk|`。risk 可能增大也可能减小，因此不预设单向风险变化。随机重复范围仅表示采样变动，**不是置信区间**。注意力趋于均匀时，代码通过固定随机 tie-break 处理并保存分数范围；此时 top/bottom 曲线缺乏可区分排序，不能挑它来支持解释性。

padding 用已验证 feature row 身份排除。先单独建立 **unpadded reference**，所有删减/预算曲线相对此基线；原预测和去 padding 预测的差异另行保存，不混作 patch 删除效应。

**怎样解读：** top 相比 random 更大 `|Δrisk|` 表示注意力排序与模型敏感性有关；若同时 C-index 损失更大，进一步支持其关注的是预后相关信息。这是模型内扰动证据，不能代替病理学家组织标注或生物学验证。

### C. patch 预算

随机顺序由 `diagnostic seed + patient ID + repeat` 确定，不依赖模型 arm、执行顺序或设备。每个重复使用嵌套子集，默认保留 `100%,90%,75%,50%,25%` 的**已有评估 bag 的真实 patch**；不能把它解释成从整张 WSI 重新提取了相应百分比。

每个重复分别计算患者 C-index，再报告均值和范围；不先平均风险后制造更好分数。只比较相同患者、特征、评估 bag、split/seed 和有效训练配方；这检验冻结 checkpoint 的输入预算敏感性，不是“不同 patch 预算重训”的结果。

## 3. 服务器命令

使用已经审计过的 `core.json`（例如含 Full、Direct、Independent），路径定义与既有 runbook 一致。

```bash
cd /data1/DCT-Reg
git pull --ff-only origin main
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
TOOL=scripts/run_v313_additional_evidence.py
MANIFEST="$PWD/results/v313_evidence_v2/core.json"
OUT="$PWD/results/v313_additional_20261008"
```

先检查选了哪些任务。默认只是列计划，不加载 checkpoint、不读取真实特征、不创建输出：

```bash
$PY "$TOOL" run --manifest "$MANIFEST" --arm exp6 \
  --cancer blca --fold 0 --output "$OUT/check"
```

由你在服务器执行一折检查（不训练）：

```bash
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL" run --manifest "$MANIFEST" \
  --arm exp6 --cancer blca --fold 0 --output "$OUT/check" \
  --device cuda:0 --execute
$PY "$TOOL" plot --exports "$OUT/check" --output "$OUT/figures_check"
```

一折核验后，Full 全癌种/全折，另用新目录：

```bash
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL" run --manifest "$MANIFEST" \
  --arm exp6 --output "$OUT/full" --device cuda:0 --execute
$PY "$TOOL" plot --exports "$OUT/full" --output "$OUT/figures_full"
```

`--arm exp6` 只选清单中 Full，实际清单不一定覆盖十癌。各癌先看独立 fold；不要将不同 checkpoint 的原始风险混合计算 C-index。

随后，复用完全相同的诊断 seed/比例/重复数，在训练匹配的控制 checkpoint 上运行 patch 诊断：

```bash
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL" run --manifest "$MANIFEST" \
  --arm direct --arm independent --experiment patches \
  --output "$OUT/patch_controls" --device cuda:0 --execute
$PY "$TOOL" plot --exports "$OUT/patch_controls" --output "$OUT/figures_patch_controls"
```

只跑第一类用 `--experiment reconstruction`，只跑后两类用 `--experiment patches`。默认 5 次随机重复，可用 `--repeats` 修改，但所有比较条件要一致。运行每个 checkpoint 只载入一次；训练均值只在重建实验计算。默认每位患者的 patch 诊断约 48 次子集前向，耗时取决于服务器和特征读取，不提供未经测量的时间估计。

## 4. 证据检查与论文使用

- manifest 先整体审计；任何缺文件、划分错误、旧预测重放不一致均停止。严格 checkpoint 加载、最佳 epoch 温度与训练折 reference buffer 沿用原 exporter。
- 保存训练 commit 与导出 commit、有效重建系数、config/checkpoint/split/预测哈希，每位患者特征文件哈希及采样行校验值。
- 原始标准 forward 与手动 replay 必须相符。eval 结束逐项验证 buffer 未改变，不做梯度更新。
- 图入口验证数组哈希，拒绝重复实验身份、跨折患者重叠和非空输出目录。没有完成导出的实验不会进入图。
- 默认 `legacy_val` 使用验证集选择 checkpoint：所有新增曲线都是**最佳验证模型上的机制诊断**。没有独立外测数据时，不能写成 held-out test superiority。
- 不做基于结果选比例/随机种子/通路/病例，不把 random repeats 当独立样本，不自动输出显著性星号。需要真实多 seed 或外部队列时先预设 protocol，再实际运行。

优先放正文：**患者特异重建/检索图 + 患者错配图**。patch 删除及预算图适合正文机制小节或补充材料。已有定性热图可与 patch 删除曲线并排；先确认数据支持假设，再形成论文结论。
