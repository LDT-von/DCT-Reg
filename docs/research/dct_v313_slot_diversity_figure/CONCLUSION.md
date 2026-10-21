# SlotSPE 问题清单与 DCT v3.13 诚实评估（第 3 版）

**日期** 2026-10-21（第 3 版，加入 v3.13 cosine 实测）
**主证据** BLCA / uni2-h / seed 3 / fold 0 / 30 epoch（v3.13）

---

## 0. 版本历史

| 版本 | 日期 | 关键变化 |
|---|---|---|
| 第 1 版 | 2026-10-20 | 初始版本，含多个数学错误 |
| 第 2 版 | 2026-10-21 | 修正 E1–E7，标记 v3.13 cosine 未实测 |
| **第 3 版** | **2026-10-21** | **实测 v3.13 cosine（76 患者 best checkpoint forward），发现与 SlotSPE 同量级冗余** |

---

## 1. 数据来源与审计

### v3.13
- **路径**：`results/dct_v313_full_wsi_three_decoders/blca_seed3/.../epoch_curve_fold0.csv`
- **Cosine 实测**：`scripts/measure_v313_slot_cosine.py` → `v313_slot_cosine_blca_fold0_measured.json/.csv`
  - 加载 `model_best_s0.pth`，对 76 名验证患者逐一 forward（`replay()`）
  - 提取 `slots_wsi` [1,8,D] 和 `slots_omic` [1,8,D]
  - 计算每患者 28 个上三角对（8 选 2）的 cosine similarity
  - 阈值 cos > 0.95，汇报各指标
- **模态**：WSI、Omics 各 8 槽，28 对（无向）
- **Cosine 实测值**：见下表

### SlotSPE（诊断性证据，非五折均值）
- **路径**：`results/slotspe_proof/p1246_diagnostic_blca_fold0_20261020/diagnostic.json`
- **实验**：BLCA，fold 0，seed 3，epoch 3，77 验证患者
- **Cosine**：实测值，77/77 患者所有 28 对均 > 0.95

### 十癌种泛化图
- **9 癌种来源**：`results/dct_v313_uni2h/<study>/...`（brca/coadread/hnsc/kirc/luad/lusc/skcm/stad/ucec）
- **BLCA 来源**：`results/dct_v313_blca_uni2h/blca/...`（5 折 × 30 epoch，**不同实验批次**）
- **⚠ BLCA 跨批次警告**：图中 BLCA 曲线来自不同批次，跨批次比较仅供示意

---

## 2. v3.13 Cosine 实测（第 3 版新增）

**来源**：`docs/research/dct_v313_slot_diversity_figure/v313_slot_cosine_blca_fold0_measured.json`
**实验条件**：BLCA fold0 seed3 best checkpoint（model_best_s0.pth），76 验证患者，逐一 forward

| 指标 | WSI | OMIC | 说明 |
|---|---|---|---|
| **frac_above (mean)** | **1.0000** | **0.9732** | 超过阈值的对占比 |
| **frac_above (median)** | **1.0000** | **1.0000** | |
| **frac_above (min)** | **1.0000** | **0.7143** | 最差患者 |
| **frac_above (max)** | **1.0000** | **1.0000** | |
| **patients ALL 28 pairs > 0.95** | **76/76** | **59/76** | 全冗余患者数 |
| **max_cosine (mean)** | **0.9981** | **0.9997** | |
| **max_cosine (min)** | **0.9981** | **0.9976** | 最差患者 |
| **max_cosine (max)** | **0.9981** | **0.9999** | |
| **mean_cosine (mean)** | **0.9917** | **0.9908** | |
| **min_cosine (mean)** | **0.9810** | **0.9729** | |

### 与 SlotSPE 的比较（第 3 版更新）

| 指标 | SlotSPE WSI | v3.13 WSI | v3.13 OMIC |
|---|---|---|---|
| frac_above (mean) | 1.0000 | **1.0000** | 0.9732 |
| max_cosine (mean) | 0.9996 | **0.9981** | 0.9997 |
| patients ALL pairs > thr | 77/77 | **76/76** | 59/76 |

**关键发现（第 3 版）**：
- v3.13 WSI 槽的 cosine 冗余**与 SlotSPE 同量级**：76/76 患者 frac > 0.95，max cosine ≈ 0.998
- v3.13 OMIC 略好但仍有 59/76 患者全冗余
- v3.13 虽然有 Euclidean pair distance 惩罚项（mean_dist → 6.88），但 cosine 仍接近 1.0

---

## 3. 修正后的实测数据表（v3.13 BLCA fold0 seed3，Euclidean）

### WSI 分支

| 字段（CSV 列名） | epoch 0 | epoch 1 | epoch 29 | 说明 |
|---|---|---|---|---|
| `train_v313_diversity_wsi_mean_pair_distance` | 0.407 | 1.531 | **6.883** | 批内均值距离 |
| `train_v313_diversity_wsi_min_pair_distance` | 0.137 | 0.474 | **0.509** | 所有患者对的最小均值距离 |
| `train_v313_diversity_wsi_pair_term` | 0.780 | 0.058 | **0.024** | 多样性惩罚项 |
| `train_v313_diversity_wsi_violating_pair_fraction` | 0.950 | 0.175 | **0.060** | 违反分离约束的对占比 |
| `train_v313_diversity_wsi_duplicate_pair_fraction` | 0.000 | 0.000 | **0.000** | 精确零距离对占比 |

### OMIC 分支

| 字段（CSV 列名） | epoch 0 | epoch 1 | epoch 29 | 说明 |
|---|---|---|---|---|
| `train_v313_diversity_omic_mean_pair_distance` | 0.265 | 2.427 | **6.987** | |
| `train_v313_diversity_omic_min_pair_distance` | 0.085 | 0.340 | **0.327** | |
| `train_v313_diversity_omic_pair_term` | 0.912 | 0.180 | **0.064** | |
| `train_v313_diversity_omic_violating_pair_fraction` | 0.999 | 0.307 | **0.124** | |
| `train_v313_diversity_omic_duplicate_pair_fraction` | 0.000 | 0.000 | **0.000** | |

---

## 4. 诚实结论

### 4.1 v3.13 Euclidean distance（实测）

- **mean_pair_distance**：WSI 从 0.41 → 6.88，OMIC 从 0.27 → 6.99，距离显著增大
- **min_pair_distance**：WSI 从 0.14 → 0.51，OMIC 从 0.09 → 0.33
- **duplicate_pair_fraction**：全程均为 **0.0** → 无精确零距离对
- **violating_pair_fraction**：WSI 从 0.95 → 0.06，OMIC 从 0.999 → 0.12，大幅改善

### 4.2 v3.13 Cosine similarity（实测，第 3 版）

- **WSI**：frac_above = 1.0000，max_cosine mean = 0.9981，76/76 患者所有 28 对全部 > 0.95
- **OMIC**：frac_above = 0.9732，max_cosine mean = 0.9997，59/76 患者所有对 > 0.95
- **与 SlotSPE 对比**：v3.13 WSI 的 cosine 冗余**与 SlotSPE 同量级**（SlotSPE 77/77，v3.13 76/76）
- **与 Euclidean distance 对比**：虽然 Euclidean mean_dist 增大了 16 倍，但 cosine 仍接近 1.0——这说明欧氏距离改善不等于余弦相似度改善（向量"同方向但短"的情况）
- **不得声称 v3.13 解决了槽冗余问题**

### 4.3 SlotSPE Cosine（实测）

- 77 名患者 × 28 对，**所有 77 患者**最大余弦均 > 0.95（mean 0.9996，min 0.9992）
- **⚠ 仅单 checkpoint（fold0/seed3/epoch3），非五折均值**

### 4.4 v3.13 vs SlotSPE 比较

| 维度 | v3.13 | SlotSPE | 比较结论 |
|---|---|---|---|
| Euclidean mean_dist | 6.88（WSI） | 未测 | v3.13 Euclidean 有改善 |
| Cosine frac_above | 1.0000（WSI） | 1.0000 | **同量级冗余** |
| max_cosine | 0.9981（WSI） | 0.9996 | v3.13 略低但同量级 |
| 精确重复对（d=0） | 0.0（CSV） | 28/28（硬编码） | v3.13 无精确重复 |
| top-k 随机性 | 无 | 有 | SlotSPE 多一噪声源 |

---

## 5. 已知缺口（诚实声明）

| 项 | 状态 |
|---|---|
| v3.13 cosine similarity | ✅ **实测（2026-10-21）**：WSI frac>0.95=1.0000，OMIC=0.9732 |
| v3.13 cosine 其他折/种子 | ❌ 未测（当前仅 fold0 seed3） |
| v3.13 十癌种泛化图的 pair 级诊断 | uni2h CSV 无 per-patient cosine；泛化未知 |
| WSI pair_term 非单调反弹 | 待查 |
| 目标带阈值 | `[0.001, 0.050]` 来自 v311 model.py，未对 v313 校准 |
| SlotSPE 初始化 cos | 0.5605 来自 FINAL_RECEIPT.md，未独立加载 checkpoint 核实 |

---

## 6. 产物

```
docs/research/dct_v313_slot_diversity_figure/
  v313_slot_cosine_blca_fold0_measured.json   # v3.13 cosine 实测（JSON）
  v313_slot_cosine_blca_fold0_measured.csv    # v3.13 cosine 实测（CSV）
  v313_vs_slotspe_p2_comparison_v2.png / .pdf
  v313_vs_slotspe_p2_comparison_v2.csv
  v313_slot_diversity_panel_blca_fold0.png / .pdf
  v313_tencancer_mean_curves.png / .pdf
  v313_tencancer_mean_curves.csv
  CONCLUSION.md                              # 本第 3 版
scripts/
  measure_v313_slot_cosine.py               # v3.13 cosine 实测脚本
  plot_v313_vs_slotspe_p2_v2.py
  plot_v313_tencancer_mean_curves.py
```

数据源：
- v3.13 BLCA Euclidean：`results/dct_v313_full_wsi_three_decoders/blca_seed3/.../epoch_curve_fold0.csv`
- v3.13 BLCA cosine（**实测**）：`v313_slot_cosine_blca_fold0_measured.json`（commit edc44c5）
- v3.13 9-cancer：`results/dct_v313_uni2h/<study>/...`
- SlotSPE cosine：`results/slotspe_proof/p1246_diagnostic_blca_fold0_20261020/diagnostic.json`
