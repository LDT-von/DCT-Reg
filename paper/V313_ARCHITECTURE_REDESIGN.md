# v3.13 框架图与消融结论说明

## 直接结论

当前结果支持 v3.13 完整训练方案有用。UNI2-h 五折最佳验证记录中，Full 相对同架构、仅 NLL 的 Exp0：BLCA 从 0.7001 到 0.7238，提升 2.37 个 C-index 百分点；KIRC 从 0.8120 到 0.8224，提升 1.04 个百分点。这是对完整辅助目标组合的支持证据。

这组损失消融保持预测架构不变，因此没有分别消融原型坐标、多几何或阶段事件读出，不能单凭它把收益拆给每个网络模块。五折最佳验证均值不等于多种子显著性或独立测试结论。

Full 对 Direct 的当前均值差为 BLCA +2.09、KIRC +1.94 个百分点；对 Independent 为 +1.45、+1.70 个百分点。它们提供机制训练对照的支持信号，但 Full 与控制臂完整患者、结局、配置和评分口径的匹配核验仍须完成。Direct 只改变 cross 重建记忆；Independent 同时改变预测和重建的联合耦合。

冻结 Full 的权重后替换耦合，是另一个问题：检查训练完成的预测器是否敏感于替换。该干预中的 C-index 接近不变，不能抵消重新训练的对照结果，也不能自动说明传输无用；它限制的是“预测强烈依赖精细匹配”的解释。目前应同时报告两类结果。

## 相对 SlotSPE 的实际模块改动

| 图中标记 | SlotSPE 基础 | v3.13 实际实现 | 合理的贡献表述 |
|---|---|---|---|
| A，沿用基础 | 病理特征、通路编码、两路 Slot Attention | 仍使用两模态 token 编码与局部槽注意力 | 槽压缩基础继承，不能单独作为新贡献 |
| M1，原型坐标重分配 | 原始槽由 Slot Attention 得到 | 局部槽输出再竞争分配到跨患者复用的原型坐标；病理和组学使用两个独立原型库 | 提供可复用的槽坐标，为后续传输构造结构化输入；生物学语义仍需验证 |
| M2，阶段多几何传输 | 原方法主要使用模态内注意力和迭代跨模态注意力进行交互 | 槽对的余弦、欧氏、正点积代价，加上学习的阶段代价、证据代价及证据边际，再通过 Sinkhorn 构造各阶段、各几何的计划 | 将跨模态关系表示为有边际约束的联合传输计划；4 个潜在阶段与 3 种几何是当前默认实例 |
| M3，传输事件预测 | 选择槽、自注意力/跨注意力分支融合后预测 | 传输计划与槽对内容联合构造事件 tokens，经过阶段嵌入和事件 Transformer；事件 logits 与阶段门控并行生成，再加权得到患者 logits | 将预测组织为传输事件的组合；预测不只读取 OT 矩阵 |
| M4，复用预测计划的重建 | 已有模态内与跨模态重建；原跨重建用组学槽对病理图块进行注意力，再解码通路特征 | 病理槽经预测使用的同一批计划重心传输到组学槽坐标；几何均值、列归一化、停止梯度的阶段门控聚合；与组学槽 self 分支共用通路查询解码器，重建停止梯度的编码通路 tokens | 独特性在于让同一传输计划同时服务预测和重建，以及具体的槽坐标搬运关系；不能把“有跨模态重建”本身称为首创 |

M1-M3 是 v3.13 完整框架相对 SlotSPE 的结构差异，部分来自 DCT 之前的版本；M4 是 v3.13 的版本特定改动。相对自身前一版的新增点与相对外部基线的改动点应分开表述。

IPCW 排序、逐槽 NLL、多样性约束是训练设计，放在图底部。NLL、IPCW 和重建等已有损失形式不能逐一包装成首创方法。

## 新图

文件目录：`paper/figures/v313_architecture_redesign/`。

- `fig1_v313_framework_en.svg` / `.pdf` / `.png`：英文论文版。
- `fig1_v313_framework_zh.svg` / `.pdf` / `.png`：中文解释版。
- `manifest.json`：模型源文件哈希、源提交、参考布局和示意标识。

沿用 SlotSPE Figure 1 的双路输入、编码器、槽压缩视觉顺序，原生重绘全部矢量元素，并按实际 DCT 拓扑替换中央交互与预测模块。紫色连接标出“预测计划复用于重建”；橙色虚线明确重建使用的门控停止梯度。原型库按模态分开，不暗示同号病理槽与组学槽有已证实的相同生物学含义。

底部重建只在训练时运行。目标是编码后的通路 tokens，不是原始基因表达；当前跨重建依赖接收者组学参与计划与门控，不能由此声称 WSI-only 缺失组学补全。图中阶段是学习的潜在阶段，不是已标注的疾病分期。所有微型矩阵、图块和 token 都是结构示意，不能当成患者实验结果。

英文图注建议：

> Overview of DCT v3.13. Histology and pathway tokens are compressed by local slot attention and reassigned to modality-specific prototype coordinates reused across patients (M1). Stage-conditioned multi-geometry costs and evidence-conditioned marginals define Sinkhorn transport plans (M2). The plans and slot-pair content jointly form event representations for gated survival prediction (M3). During training, the same prediction plans transport histology slots to omics-slot coordinates for pathway-token reconstruction, alongside an omics self-reconstruction branch with a shared decoder (M4). Reconstruction uses stop-gradient encoded pathway targets and a stop-gradient prediction gate. Solid arrows denote prediction flow and dashed arrows denote training-only branches. All illustrated tensors are schematic.

## 代码对应关系

- M1：`DistributionalCounterfactualTransport._semantic_slots` / `_encode_transport_slots`。
- M2：`DistributionalCounterfactualTransport._cost_tensor` / `_plans_from_cost_tensor`。
- M3：`_selected_stage_events` / `MultiScaleOTFusion.forward` / `_encode_logits_from_plans`。
- M4：`DCTV313TransportReconstruction._transport_wsi_to_omic` / `reconstruction_losses` / `OmicsTokenReconstructionDecoder`。
- 仅训练时重建：v3.13 `forward` 的 `if not self.training: return super().forward(...)`。

运行绘图命令（只生成图，不运行模型）：

```powershell
python scripts/draw_v313_architecture_slotspe_layout.py
```

依赖：`reportlab`、`pypdfium2`、`Pillow`；默认字体目录为 Windows `C:/Windows/Fonts`，可用 `--font-dir` 指定包含 Arial 和 SimHei 字体的目录。用 `--source-root` 核对其他 v3.13 checkout；用 `--output-dir` 指定输出目录。生成器在导出前核对源文件，并记录源提交与 SHA-256，无模型推理或患者数据读取。

参考：[SlotSPE 论文 Figure 1 与方法章节](https://arxiv.org/html/2512.01116v2#S3)。当前证据数值来自本仓库 `paper/DCT_v313_初稿.md` 的损失消融与训练控制臂记录，保留其最佳验证和匹配状态限制。
