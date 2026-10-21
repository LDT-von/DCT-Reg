# SlotSPE 问题清单与 DCT v3.13 诚实评估（第 2 版）

**日期** 2026-10-21（第 2 版，修正第 1 版的数学错误与叙事错误）
**主证据** BLCA / uni2-h / seed 3 / fold 0 / 30 epoch（v3.13）

---

## 0. 第 1 版勘误

| # | 旧说法 | 错误类型 | 修正后说法 |
|---|---|---|---|
| E1 | "v3.13 cos > 0.95 比例为 1.0（56/56 对均超标）" | 推导错误：等模长假设未经验证；min_pair_distance 不能代表所有对 | v3.13 cosine **未直接测量**。仅 Euclidean distance 有实测值 |
| E2 | "WSI pair_term 在 epoch 29 反弹到 0.509" | 字段混淆：0.509 是 `min_pair_distance`，不是 pair_term。实际 WSI pair_term = **0.024** | 区分 `pair_term`、`min_pair_distance`、`mean_pair_distance`、`violating_frac` 各字段 |
| E3 | "Omics 单调递减" | 与图不符（实际有明显波动） | 按实测序列描述 |
| E4 | "主损失覆盖了分离约束" | 无梯度/消融证据的因果断言 | 标为"待查"，不写因果 |
| E5 | "v3.13 无 top-k → 不存在置换敏感性" | 过度推断：只说明无相同 top-k 路径，完整模型稳定性需对应测试 | 删除该断言 |
| E6 | "两者相同（cos 均为 1.0）" | 基于未测数据得出比较结论 | 仅描述各自实测值，明确未做直接比较 |
| E7 | "9 个癌种 + BLCA" 列表漏写 BRCA | 打字错误 | 已更正 |

---

## 1. 数据来源与审计

### v3.13
- **路径**：`results/dct_v313_full_wsi_three_decoders/blca_seed3/.../epoch_curve_fold0.csv`
- **实验**：BLCA，fold 0，seed 3，30 epoch，legacy_val 协议
- **列名来源**：`train_v313_diversity_{wsi,omic}_{pair_term,mean_pair_distance,min_pair_distance,violating_pair_fraction,duplicate_pair_fraction,full_collapse_patient_fraction}`
- **模态**：WSI、Omics 各 8 槽，56 有向对
- **Cosine**：未直接测量；下界估算值 0.9993 来自距离推导，**不作为已验证数据**

### SlotSPE（诊断性证据，非五折均值）
- **路径**：`results/slotspe_proof/p1246_diagnostic_blca_fold0_20261020/diagnostic.json`
- **实验**：BLCA，fold 0，seed 3，epoch 3，77 验证患者
- **Cosine**：实测值（per-patient per-pair cosine 直接计算），77/77 患者所有 56 对均 > 0.95
- **⚠ 重要限制**：单 checkpoint，非五折均值；初始化 cos（0.5605）来源 FINAL_RECEIPT.md §3，未独立核验

### 十癌种泛化图
- **9 癌种来源**：`results/dct_v313_uni2h/<study>/...`（brca/coadread/hnsc/kirc/luad/lusc/skcm/stad/ucec）
- **BLCA 来源（暂定）**：`results/dct_v313_blca_uni2h/blca/...`（5 折 × 30 epoch，与九癌种**不同实验批次**）
- **⚠ BLCA 跨批次警告**：图中 BLCA 曲线来自不同批次，跨批次比较仅供示意

---

## 2. 修正后的实测数据表（v3.13 BLCA fold0 seed3）

### WSI 分支

| 字段（CSV 列名） | epoch 0 | epoch 1 | epoch 29 | 说明 |
|---|---|---|---|---|
| `train_v313_diversity_wsi_mean_pair_distance` | 0.407 | 1.531 | **6.883** | 批内均值距离 |
| `train_v313_diversity_wsi_min_pair_distance` | 0.137 | 0.474 | **0.509** | 所有患者对的最小均值距离 |
| `train_v313_diversity_wsi_pair_term` | 0.780 | 0.058 | **0.024** | 多样性惩罚项（越小越好） |
| `train_v313_diversity_wsi_violating_pair_fraction` | 0.950 | 0.175 | **0.060** | 违反分离约束的对占比 |
| `train_v313_diversity_wsi_duplicate_pair_fraction` | 0.000 | 0.000 | **0.000** | 精确零距离对占比 |
| `train_v313_diversity_wsi_full_collapse_patient_fraction` | 0.000 | 0.000 | **0.000** | 所有 8 槽均相等的患者占比 |

### OMIC 分支

| 字段（CSV 列名） | epoch 0 | epoch 1 | epoch 29 | 说明 |
|---|---|---|---|---|
| `train_v313_diversity_omic_mean_pair_distance` | 0.265 | 2.427 | **6.987** | |
| `train_v313_diversity_omic_min_pair_distance` | 0.085 | 0.340 | **0.327** | |
| `train_v313_diversity_omic_pair_term` | 0.912 | 0.180 | **0.064** | ⚠ 与旧版不符：非单调递减 |
| `train_v313_diversity_omic_violating_pair_fraction` | 0.999 | 0.307 | **0.124** | |
| `train_v313_diversity_omic_duplicate_pair_fraction` | 0.000 | 0.000 | **0.000** | |
| `train_v313_diversity_omic_full_collapse_patient_fraction` | 0.000 | 0.000 | **0.000** | |

---

## 3. SlotSPE 实测数据

| 字段 | 值 | 来源 |
|---|---|---|
| 训练后最大余弦（77 患者均值） | **0.9996** | diagnostic.json p2.mean_max_offdiag_cos |
| 训练后最大余弦（77 患者最小） | **0.9992** | diagnostic.json p2.min_max_offdiag_cos |
| 所有 56 对均 > 0.95 的患者占比 | **77/77 = 1.0** | diagnostic.json p2.fraction_patients_with_offdiag_high_cos |
| 初始化最大余弦 | 0.5605 | FINAL_RECEIPT.md §3（⚠ 未独立核验）|

---

## 4. 诚实结论

### 4.1 v3.13 Euclidean distance（实测）

- **mean_pair_distance**：WSI 从 0.41 → 6.88，OMIC 从 0.27 → 6.99，距离增大
- **min_pair_distance**：WSI 从 0.14 → 0.51，OMIC 从 0.09 → 0.33；两组均有提升但量级不大
- **duplicate_pair_fraction**：WSI/OMIC 全程均为 **0.0** → 无精确零距离对
- **full_collapse_patient_fraction**：WSI/OMIC 全程均为 **0.0** → 无患者全部 8 槽完全相同

### 4.2 v3.13 Cosine similarity（未实测）

- v3.13 cosine **未直接测量**。用 `min_dist=0.509`、`r≈21.87` 代入 `1−d²/(2r²)` 得到的下界 ≈ 0.9993 **不能作为已验证数据**。
- 等模长假设（所有 8 个槽向量长度相同）未经核实。
- 即使该估算成立，`min_pair_distance` 也只描述最近一对，不能推出全部 56 对。
- **不得声称"v3.13 所有 56 对均超过 cos > 0.95"。**

### 4.3 SlotSPE Cosine（实测）

- 77 名患者 × 56 对，实测最大余弦均 > 0.95（mean 0.9996，min 0.9992）
- 初始化 cos ≈ 0.41，训练后升至 0.9996；sigma ≈ 1，未收缩
- **⚠ 仅单 checkpoint（fold0/seed3/epoch3），非五折均值**

### 4.4 v3.13 vs SlotSPE 比较

- 两种方法使用**不同的测量协议**：
  - v3.13：每 epoch 批内汇总统计（distance 均值/最小值）
  - SlotSPE：逐患者逐对 cosine 直接计算
- 不能将 v3.13 的 Euclidean distance 与 SlotSPE 的 cosine 放在同一坐标轴进行比较
- 不能声称"v3.13 在 cosine 上与/优于/劣于 SlotSPE"

### 4.5 v3.13 在 Euclidean 层面的贡献

基于 Euclidean 距离（**实测数据**）：
- duplicate_pair_fraction = 0.0（无精确塌缩）
- full_collapse_patient_fraction = 0.0（无患者全塌缩）
- mean_dist 从 0.41 → 6.88（增大）
- 与 SlotSPE 不同：无 top-k，不存在相同的推理期随机性

---

## 5. 已知缺口（诚实声明）

| 项 | 状态 |
|---|---|
| v3.13 cosine similarity | **未直接测量**。当前所有 cos 数值均为未验证估算 |
| v3.13 十癌种泛化图的 pair 级诊断 | uni2h CSV 仅 22 列，无 per-patient per-pair cosine；泛化到其他癌种时 pair 级表现未知 |
| WSI pair_term 非单调反弹 | epoch 1→29，WSI pair_term 0.78→0.06→0.02；OMIC 0.91→0.18→0.06，原因待查 |
| 目标带阈值 | `[0.001, 0.050]` 来自 v311 model.py，未对 v313 校准 |
| v3.13 置换敏感性 | 未测试；无 top-k 不等于完整模型对置换稳定 |
| SlotSPE 初始化 cos | 0.5605 来自 FINAL_RECEIPT.md，未独立加载 checkpoint 核实 |

---

## 6. 产物

```
docs/research/dct_v313_slot_diversity_figure/
  v313_vs_slotspe_p2_comparison_v2.png / .pdf   # 修正版 P2 对照图
  v313_vs_slotspe_p2_comparison_v2.csv           # 逐字段实测值 + 估算值对照
  v313_slot_diversity_panel_blca_fold0.png / .pdf  # v3.13 Euclidean 面板
  v313_tencancer_mean_curves.png / .pdf          # 十癌种均值曲线
  v313_tencancer_mean_curves.csv                 # 十癌种均值数据
  CONCLUSION.md                                  # 本修正版
scripts/plot_v313_vs_slotspe_p2_v2.py           # 修正版脚本
scripts/plot_v313_tencancer_mean_curves.py       # 十癌种均值脚本
```

数据源：
- v3.13 BLCA：`results/dct_v313_full_wsi_three_decoders/blca_seed3/.../epoch_curve_fold0.csv`
- v3.13 9-cancer：`results/dct_v313_uni2h/<study>/.../epoch_curve_fold{0..4}.csv`
- v3.13 BLCA（暂定泛化）：`results/dct_v313_blca_uni2h/blca/...`（不同批次）
- SlotSPE：`results/slotspe_proof/p1246_diagnostic_blca_fold0_20261020/diagnostic.json`
- SlotSPE init cos：FINAL_RECEIPT.md §3
