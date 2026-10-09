# DCT v3.13 新增图和机制验证：实时进度（2026-10-08）

这是用户发令"你可以跑起来，再怎么跑 blca 和 kirc 的五折吧"之后的**实时进度**。
说明文档：远端已落地 `paper/V313_ADDITIONAL_EVIDENCE.md` 与 5 个新文件 + 65 个合成测试（`35be29d`）。
本机起点：`bae1c80` paper/document/csv-all。

## 1. 已完成

- **BLCA exp6 五折**（`results/v313_additional_20261008/blca_full/`，exp6 = Full transport + cross reconstruction），
  5 折 × 1 run = 5 run，耗时约 99 min（commit_hash: `bae1c80db204dd46726539f1452de72e3b632b08`，source_commit = `bae1c80`）。
  逐患者 `additional.npz`（每个约 1 MB）+ `additional.json`。
  Plot：25 PDF + 25 PNG = 50 张，位置 `figures_blca_full/`。
  `fold_metrics.csv` + `figure_index.json`（source/array SHA-256 齐全）。

- **KIRC exp6 五折**（`results/v313_additional_20261008/kirc_full/`），5 折 × 1 run = 5 run。
  Plot：25 PDF + 25 PNG = 50 张，位置 `figures_kirc_full/`。

- **merged manifest**：`results/v313_evidence_v2/core.json`（30 run，audit 通过率 30/30；合并 `full_manifest.json` + `controls.json`，仅记录，不入 git）。

## 2. 进行中

- **patch controls**（`results/v313_additional_20261008/patch_controls/`，`--experiment patches`），
  arms `direct` + `independent`，cancers BLCA + KIRC，五折共 20 run。

  当前进度（截至 2026-10-08 01:58 UTC+8）：
  - `direct_blca_f0_s3` ✅
  - `direct_blca_f1_s3` ✅
  - `direct_blca_f2_s3` ✅
  - `direct_blca_f3_s3` ✅
  - `direct_blca_f4_s3` ✅
  - `direct_kirc_f0_s3` ⏳ factual 66/98（进行中）
  - 剩余 14 折（`direct_kirc_f1..4`, `independent` × BLCA/KIRC × f0-f4）

  节奏：~22 min/折（patch-only，无 reconstruction）；剩余 14 折 ≈ 5 h，预计 ~07:00 UTC+8 完成。

  Plot：`figures_patch_controls/` 待 20 折全部完成后再 plot。
  每折 2 类图（patch_deletion、patch_budget，各含 PDF+PNG = 4 文件），20 折共 40 文件（20 张图）。
  patches experiment 没有 patient_pairing、pathway_reconstruction、topk_patch、random_deletion（代码里不存在这些输出）。

## 3. 已知结论（不预设未来结果）

仅基于 **已完成的 BLCA + KIRC exp6 五折**：

### A. 重建

`fold_metrics.csv` 中 native vs shuffled 的 **C-index 几乎相同**（典型 4 位小数前三位一致，例如 f0 native 0.8649 vs shuffled 0.8648）。
原因：cindex 是验证集 **原始** forward + 原始 outcomes 上的统计量；`shuffled` 路径只计算 **shuffled forward** 路径的 C-index，与 native 同列对比时不衡量 native vs shuffled 风险差。
- plan §A 的"patient_pairing"图真正显示的是 native vs shuffled 的**逐患者预测差异**（BLCA f0 mean |Δrisk| ≈ 1.39，KIRC f0 mean |Δrisk| ≈ 3.94）。
- retrieval_top1 ~ 0.013（=1/76 ~ 1/98 chance），**未显示出 native 重建明显优于随机 chance**——plan §A 的"如实报告无差异"情况。

### B. patch 删除

`fold_metrics.csv` 的 `top@0.5 vs random@0.5 cindex` 差异极小（<0.005）。但 `deletion_risk` 的 `|Δrisk|` 曲线（fig `patch_deletion.png`）top 删除曲线在 75% 删除时 BLCA f0/4 上比 random 略高（0.65 vs 0.62）。
- 含义：删除 patch 后**单点**风险变化明显，但 **C-index**（76-98 患者群体）对个体风险变动不敏感。

### C. patch 预算

保留比例 100% → 25% 时 C-index 与 risk 变化可见。fig `patch_budget.png` 已绘。

### 跨癌种观察

| pattern | BLCA | KIRC |
|---|---|---|
| native 重建误差（通路均值）| ~0.66 | 高 |
| train_mean vs native 差异 | 中 | 中 |
| native vs shuffled C-index | ~相同 | ~相同 |
| native vs shuffled mean |Δrisk| | ~1.39 | ~3.94 |
| top vs random patch deletion | 75% 删除时略大 | 75% 删除时略大 |

KIRC 上 shuffled 重建误差**与 native 差距更大**（KIRC f0 ~0.87 vs BLCA ~0.66），提示 KIRC 上 WSI 信息对通路重建贡献**强于** BLCA。

## 4. 复核清单

- [x] merged manifest 30/30 audit 通过
- [x] BLCA exp6 f0 单张端复现图已检查（5 类图 × 5 折）
- [x] KIRC exp6 f0 pathway_reconstruction + patient_pairing 已检查
- [ ] patch controls 完成后 plot
- [ ] 论文写作中遵循 "如实报告没有差异" 原则，不声称训练匹配之外的能力
- [ ] 不混 checkpoint、不跨 fold 混 raw risk 计算 cindex（plan 4 节）
- [ ] 未自动添加显著性星号、未挑 splice 用实际比较

## 5. 后续

剩余 15 折 patch controls 由后台进程持续运行；用户中断会话也不影响。
完成后执行：

```bash
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
"$PY" /data1/DCT-Reg/scripts/run_v313_additional_evidence.py plot \
  --exports /data1/DCT-Reg/results/v313_additional_20261008/patch_controls \
  --output /data1/DCT-Reg/results/v313_additional_20261008/figures_patch_controls
```

并把 20 折 patch controls 的 2 类图与 exp6 五折对比分析（patches experiment 只有 patch_deletion / patch_budget，无 patient_pairing / pathway_reconstruction）：
- Direct（无 OT）vs exp6（Full OT）：看 OT + cross reconstruction 是否带来 patch 删除/预算上的差异
- Independent（no co-attention）vs exp6：看耦合组织是否带来 patch 删除/预算上的差异

## 6. 汇总图（per-cancer summary figures）

用户要求"一个实验一个癌症一张图，不要分折"。**已生成 10 张汇总图**（5 类 × 2 cancer，每类 PDF + PNG = 20 文件），
位置 `results/v313_additional_20261008/figures_summary/`。
原 100 张分折图（`figures_blca_full/`, `figures_kirc_full/`, `figures_check/`）已删除。

汇总脚本：`scripts/summarize_v313_exp6.py`（`6791e1e`）。

| 类 | BLCA | KIRC |
|---|---|---|
| reconstruction | `blca_exp6_summary_reconstruction.{pdf,png}` | `kirc_exp6_summary_reconstruction.{pdf,png}` |
| pathway_advantage | `blca_exp6_summary_pathway_advantage.{pdf,png}` | `kirc_exp6_summary_pathway_advantage.{pdf,png}` |
| pairing | `blca_exp6_summary_pairing.{pdf,png}` | `kirc_exp6_summary_pairing.{pdf,png}` |
| patch_deletion | `blca_exp6_summary_patch_deletion.{pdf,png}` | `kirc_exp6_summary_patch_deletion.{pdf,png}` |
| patch_budget | `blca_exp6_summary_patch_budget.{pdf,png}` | `kirc_exp6_summary_patch_budget.{pdf,png}` |

**用户反馈**（2026-10-08 11:46 UTC+8）："看不懂，但还是记录一下吧"——本节用于记住用户已翻看但未理解的口语反馈。

### 第 6 节本机阅读说明

2026-10-08 本机已同步包含 d2d4877 的远端版本。该 commit 只更新进度说明；上述服务器 figures_summary 目录没有同步到本机，不能将读取脚本称为已查看实际图片。五类图的横轴、误差方向和解释边界见 paper/V313_OT_UNI2H_LITERATURE_REVIEW_20261008.md。它们覆盖 BLCA/KIRC 的机制诊断，不等于十癌种比较或十癌种 KM。

更正第 5 节对照简称：Direct 不在 cross 重建中使用运输，但预测主路仍有 OT，不能写成整个模型“无 OT”；Independent 去掉槽对联合耦合，不能泛称网络完全没有 attention。此说明不改变服务器已有分数或图像。

## 7. 外部比较方向已调整

作者要求不与 SlotSPE 比较，改找 OT 生存预测或使用 UNI2-h 的相关论文。新表包含 MOTCat [TTA]、TTA、OTSurv、STEPH 及其 UNI2-h MIL 参照，已汇总 49 个同名称队列公开均值。ProtoPathway 使用 UNI2-h 但评价 OS，单独记录，不进入 DSS 排名。MCSP-OTMR 与 OT 重建叙事接近，列为优先全文复核对象。没有发起新模型运行，也没有声称公开报告值属于同条件复现。


### 比较表扩展及年份

作者进一步要求保留 SlotSPE 表中其他方法的数据，仅去掉 SlotSPE 自身。当前表已扩展为 21 个参照方法加 DCT，每个方法标注年份；原表其他 15 行的完整十队列数值、离散度和 Overall 原样保留，额外六项贡献 33 个对齐均值，共 183 个公开均值。十癌种均有比较对象。Overall 仅排名完整相同十队列行，额外方法不同队列集合不混入。MLP 与 SNNTrans 的引用年份及 DCT 当前工作年份加注说明。没有新增模型运行。


## 8. 本机诊断代码审计与勘误（2026-10-08）

本节覆盖上文的解读，保留旧进度作为历史记录。只做代码审计、合成回归测试及合成图排版检查，没有启动真实患者导出、推理或训练。服务器原始 additional.json/npz 和十张实际图片仍未同步本机。

### 已确认的问题及影响

1. **CSV 指标误读。** 旧 additional_plots.py 在 experiment=reconstruction、condition=native/shuffled 的 value 保存 mean_error，不是 C-index。上文第 3 节引用 fold_metrics.csv 得出 native/shuffled C-index 近似相同的结论不能成立；0.8649/0.8648 的示例须回查原始字段。真正的配对 C-index 在 additional.json 的 pairing.native_cindex / shuffled_cindex，并由原始风险计算。新版 CSV 增加 metric/direction，并独立输出 pairing 与 budget 行。此前依据这段进度作出的“错配后排序几乎不变”判断撤回，等待核验 JSON/NPZ。
2. **检索中心重复归一化。** training_mean 已是训练患者逐个 LayerNorm 后的 token 算术均值，旧 exporter 又对均值 LayerNorm，用改变后的向量中心化。训练均值对照也被重复归一化。这偏离减去训练均值的检索定义，会改变 Top-1、MRR 和 centered energy。两项新增合成测试在旧代码失败，修复后通过。新版保留精确算术均值并标 retrieval_metric_version=2；旧 Top-1/MRR 不能充当修正版结果。旧数组未保存完整解码表征，无法仅凭误差和秩恢复正确检索，须用同一 checkpoint、患者和诊断设置重新导出 reconstruction，不需要重训。修复不改变训练模型、重建距离、配对风险或 patch 诊断。修复后指标是否提高尚未知。
3. **汇总入口缺少校验且图注写死。** 旧 summarize_v313_exp6.py 绕过 array SHA256 与患者身份检查；没有核对跨折通路顺序；把打乱次数固定标为 16×5；随机检索基线只取 fold0 患者数。预算横轴还从 deletion 记录推导。新版复用严格导出校验，要求完整五折、无重复患者和一致通路顺序，按真实 repeats/N 与 budget.retained_fraction 作图，记录来源和图像哈希。配对 C-index 逐折连线，风险变化按折展示；不混合不同 checkpoint 原始风险排名。重建误差棒明确为五折样本 SD。

### 检查中未发现对应实现错误的部分

C-index 与训练用 sksurv 指标在含删失/风险并列的合成样例一致；错配风险确实重新求解接收患者组学与供体病理的成本、计划和风险头，不是重复 native 分数。两侧槽编码各自独立，没有把供体组学缓存混入接收患者。patch 子集重新编码并重算运输；padding 排除，随机 tie-break 和嵌套预算保留。native 重建距离与模型训练距离一致。此处是代码与合成验证，不能替代服务器实际数组审计。

### 接下来怎样看图

先从旧的两癌种 Full 导出重画 pathway_advantage、pairing、patch_deletion、patch_budget，共八张；新版入口会重算配对统计并检查与正文 Full 折值是否一致。重建检索两张需重新导出 reconstruction 后画 v2。旧记录里根据不同癌种绝对误差推断病理贡献更强的结论继续撤回；应比较各癌种内控制误差减 native 误差。保留原始输出，不删除旧图，不用合成预览充当真实实验图。

服务器完整核验、重导出和出图提示词：paper/V313_DIAGNOSTIC_BUGFIX_SERVER_PROMPT_20261008.txt。

验证记录：additional evidence、summary 和 core evidence 三组共 38 项测试通过；五张合成汇总 PNG 已逐张检查排版。合成 QA 不属于模型真实实验结果，本轮未启动服务器作业。
