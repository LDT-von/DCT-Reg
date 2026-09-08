# 官方 SlotSPE 外部参考基线

## 状态与边界

本文件是 DCT-Reg 与官方 SlotSPE 的唯一比较记录。

- **SlotSPE 数值**：用户提供的官方 `g.+h.`（基因组 + 组织病理学）结果；这些是外部参考值，不是本仓库的复现结果。
- **红色、最高、排名等标记**：未记录，也不用于任何结论。
- **DCT 数值**：仅为本仓库 `paper/DCT_唯一初稿.md` 中的历史本地记录；它们不是 DCT v3.10 正式结果。`experiments/REGISTRY.csv` 中 E1--E5 均未完成，正式结果状态为 `pending`。
- **禁止的表述**：在没有同一协议的官方 SlotSPE 本地运行前，不得声称 DCT 胜过、追平或落后 SlotSPE。

## 官方源码来源

- 上游仓库：<https://github.com/zylvemvet/SlotSPE>
- 固定提交：`02051a2083add7b427727e0200e4263516903561`
- 本地只读副本：`third_party/SlotSPE/`
- 下载方式：该固定提交的 GitHub 源码包；不改写、不注册为本仓库模型。

官方入口是 `third_party/SlotSPE/survival.py`。它读取打包的 `splits/5fold/<study>/fold_<k>.csv`，在每个验证 fold 内按最大验证 C-index 选择 checkpoint，并把五折均值写入 `summary.csv`。因此其原始流程是五折 train/validation 汇总，不是 DCT 正式协议要求的嵌套外层持出评估。

## SlotSPE `g.+h.` 外部参考值

| 癌种 | C-index |
|---|---:|
| BRCA | 0.779 |
| COADREAD | 0.773 |
| KIRC | 0.815 |
| UCEC | 0.813 |
| LUAD | 0.683 |
| LUSC | 0.634 |
| HNSC | 0.642 |
| SKCM | 0.688 |
| BLCA | 0.708 |
| STAD | 0.671 |
| Overall（十癌种算术平均） | 0.721 |

十个癌种数值的未四舍五入算术平均为 `0.7206`，显示为 `0.721`。

## 与 DCT 历史本地记录的限定比较

下表仅列出双方都有数值的五个癌种。DCT 的 KIRC 项只有三个已列 fold，不能称为完整五折。所有差值仅作待验证的数值差，不能作为性能结论。

| 癌种 | SlotSPE 外部参考 (`g.+h.`) | DCT 历史本地记录 | DCT − SlotSPE |
|---|---:|---:|---:|
| BLCA | 0.7080 | 0.7208 | +0.0128 |
| HNSC | 0.6420 | 0.6471 | +0.0051 |
| KIRC | 0.8150 | 0.8579（仅 3 fold） | +0.0429 |
| LUSC | 0.6340 | 0.6313 | −0.0027 |
| SKCM | 0.6880 | 0.6556 | −0.0324 |
| 共同五癌种算术平均 | 0.6974 | 0.7025 | +0.0051 |

这些差值不具有比较效力，原因包括：SlotSPE 是官方多模态 `g.+h.` 流程；DCT 历史记录没有完整的正式证据包；checkpoint 选择、外层评估和数据工件尚未对齐。

## 两个方法实际解决的问题

### 官方 SlotSPE

SlotSPE 是一个**多模态生存预测基线**。它从 WSI patch 和组学通路特征分别产生 slots，以迭代跨模态注意力融合；再使用可微 top-k slot 选择、模态内 Transformer 和重建辅助损失来产生离散时间生存风险。官方启动脚本的标准设置是：`Pathways`、`combine` signature、DSS 终点、4 个时间 bin、WSI/组学各 8 个 slots、10 次 slot attention 迭代、4096 个 patch、30 epochs。

### DCT-Reg

DCT-Reg 也使用病理和组学 token，但研究目标不是单纯提高 C-index。它对由训练折生存信息得到的低风险/高风险运输代价锚点进行干预，重新求解 Sinkhorn coupling，再读取冻结风险头的响应。其目标是检验：跨模态 OT 结构是否真的承载可复现且方向正确的预后信息。冻结 v3.10 的训练目标是 NLL、IPCW 排序和方向运输响应；正式协议还要求 fixed-coupling、随机/置乱锚点和连续剂量对照。

因此，SlotSPE 回答“多模态 slots 能否预测生存”，DCT-Reg 试图额外回答“模型中的 OT 结构是否参与并以可审计方式影响风险”。后一个问题不能由 C-index 或注意力图单独证明。

## “0.721 算不算基线？”

**算外部文献基线，不算本地实验基线。**

它可用于：说明任务难度、设定重跑目标、在明确标注“官方报告值”的相关工作表中呈现。它不能用于：计算 DCT 的正式胜负、选择 DCT 超参数、或替代同一数据/特征/划分下的 SlotSPE 运行。

要把它升级为本地强基线，必须使用 `third_party/SlotSPE/` 的未修改官方代码，锁定同一 WSI 特征、RNA 表、DSS 终点、患者级 splits、随机种子集合、训练预算和 checkpoint 规则，保存每 fold 的预测、配置、日志和版本哈希。若同时要满足 DCT 的正式证据标准，则还需要预先定义一个双方都能执行的外层持出 checkpoint-selection 协议；官方 SlotSPE 当前脚本本身不提供该嵌套评估。

## 本机复现预检（2026-09-08）

预检未启动训练。官方 SlotSPE 的命令行和源码可导入，但当前数据不能直接用于其未修改的训练入口。

- 十癌种的 RNA 表、官方临床表和官方五折 split 文件均已具备；十癌种病例集合与本地临床表完全重合。
- 本机 UNI2-h 目录包含对应 slide 的 1536 维 `float32` HDF5 `features` 数据集，文件名与官方临床 `wsi` 字段匹配。
- 官方 `dataset/dataset_survival.py` 只查找同名 `.pt`，并以 `torch.load` 读取；它找不到当前 `.h5` 时会打印 `missing file`，随后用零张量替代。这种运行无效，不能产生 SlotSPE 基线分数。
- HDF5 `features` 可无损转换为形状 `(patches, 1536)` 的 PyTorch tensor；转换产物必须保留原 slide stem、写入独立目录，并记录每个输入/输出的 SHA-256。不能将 `.h5` 仅改扩展名。
- 官方脚本的默认 `encoding_dim=1024`，而本机特征维度为 1536。因此本地官方运行必须显式传入 `--encoding_dim 1536`；这属于输入兼容性配置，不是模型架构改写。
- 患者五折划分只有 BLCA 的五个 fold 与官方包逐字节一致。BRCA、COADREAD、HNSC、KIRC、LUAD、LUSC、SKCM、STAD、UCEC 的五个 fold 均不一致，不能把已有 DCT 结果与官方 SlotSPE 数值或新 SlotSPE 运行直接比较。

因此，下一步的最小有效工作是：先把 BLCA 的 HDF5 特征转换到独立 `.pt` 目录，使用官方 BLCA split、官方临床/RNA 元数据和 `--encoding_dim 1536` 跑未修改 SlotSPE；仅在逐 slide 加载覆盖率为 100%、输出工件完整且 checkpoint 规则预先锁定后，该结果才能称为本地 SlotSPE 强基线。其余九癌种需要先选择并固定“官方 split”或“本地 split”的共同协议，再开始任何横向比较。
