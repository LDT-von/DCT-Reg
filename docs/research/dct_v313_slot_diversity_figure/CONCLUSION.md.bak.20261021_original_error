# SlotSPE 问题清单与 DCT v3.13 诚实评估

**日期** 2026-10-21（修正版）| **P2 主证据** BLCA / uni2-h / seed 3 / fold 0 / 30 epoch
**泛化证据** 9 study × 5 fold = **45 runs**（`results/dct_v313_uni2h`）

---

## 0. 修正声明（2026-10-21）

初版 CONCLUSION 声称"v3.13 无槽冗余、cos > 0.95 比例为 0.0"——这是**反向错误**。

**修正**：v3.13 最近槽对距离 0.509 → cos ≥ 0.9996，**所有 56 对均超过 0.95 阈值**，比例为 **1.0**，与 SlotSPE 相同。

初版错误版本保留备份路径：`docs/research/dct_v313_slot_diversity_figure/CONCLUSION.md.bak.20261021_original_error`（由文件系统保留）。

---

## 1. P2 三图诚实对照

| # | 子图 | SlotSPE | DCT v3.13 | 判定 |
|---|---|---|---|---|
| (a) | cos > 0.95 的槽对比例 | **1.0**（28/28 对全部超标） | **1.0**（56/56 对全部超标） | ⚠️ **两者相同**，v3.13 在 cos 阈值层面没有更好 |
| (b) | 最大 inter-slot cosine | 初始化 0.5605 → 训练后 **0.9996** | 最终 **≥ 0.9996** | ⚠️ 收敛终点数值相同 |
| (c) | pair 分离惩罚项 | — | WSI 非单调（0.78→0.03→0.51）；OMIC 单调递减 | ⚠️ WSI 有反弹 |

![P2 三图修正版](/data1/DCT-Reg/docs/research/dct_v313_slot_diversity_figure/v313_vs_slotspe_p2_comparison.png)

**图 (a) 的修正**：初版错误地将 0.9996 理解为"不超标"（写 0.0），实则 0.9996 >> 0.95，两者的 cos 阈值比例均为 **1.0**。

**图 (b) 的修正**：v3.13 的下界 ≥ 0.9996 与 SlotSPE 训练后的 0.9996 数值相近，但机制不同：
- SlotSPE：从初始化 cos ≈ 0.41 被训练**压塌**到 0.9996（sigma ≈ 1，未收缩）
- v3.13：从相同初始化出发，pair_term 在 epoch 1 就降到 0.03（迅速满足约束），但后续反弹；几何距离保持在 0.509 使 cos ≥ 0.9996

**图 (c) 的修正**：WSI pair_term 非单调，epoch 1 时约束被满足（0.03），epoch 29 反弹到 0.51——说明 WSI 分支的分离约束在后期被主损失覆盖。

---

## 2. v3.13 真正优于 SlotSPE 的地方（不是 P2 cos）

| 指标 | SlotSPE | v3.13 | 说明 |
|---|---|---|---|
| 精确重复对比例 | 未报告 | **0.0**（全程） | 图 (d)：无零距离槽对，避免完全重合 |
| 患者级全塌缩比例 | 未报告 | **0.0**（全程） | 图 (e)：无患者的所有槽完全相同 |
| 推理期随机性 | **有**（每次 forward 重采样） | **无**（deterministic init） | v3.13 的 top-k 机制不生效 |
| 置换敏感性 | **有**（top-k 仲裁） | **无**（无 top-k） | v3.13 结构上不存在 |

![v3.13 多样性面板（Euclidean 距离，非 cosine）](/data1/DCT-Reg/docs/research/dct_v313_slot_diversity_figure/v313_slot_diversity_panel_blca_fold0.png)

---

## 3. 诚实结论

**v3.13 在 Euclidean 距离层面**（mean_dist 0.41→6.88，min_dist 0.14→0.51）**比 SlotSPE 好**：几何距离增大了，duplicate_pair_fraction = 0，full_collapse = 0。

**但在 cosine similarity 层面**，v3.13 和 SlotSPE 一样糟糕：cos 都在 0.9996 左右，所有槽对的几何方向几乎相同。

**机制差异**：
- SlotSPE 塌缩是"训练驱动"（sigma 未收缩，梯度把它推到一起）
- v3.13 高 cos 是"高维几何必然"（Euclidean 距离 0.51 在 256 维空间里对应 cos ≈ 0.9996）；多样性损失有效但 Euclidean 距离的绝对量级（0.51）相对于向量长度（18.8）太小，cos 依然极高

**因此：v3.13 的贡献不是"解决 P2 cos 塌缩"，而是避免了 SlotSPE 的推理期随机性和置换敏感性，同时用 Euclidean 距离（而非 cosine）来度量多样性。**

---

## 4. 已知缺口（诚实声明）

| 项 | 状态 |
|---|---|
| 45 run（brca 以外的 9 癌种）无 pair 级诊断 | CSV 仅 22 列，无 `duplicate_pair_fraction` / `mean_pair_distance`；泛化到其他癌种时 pair 级表现**未知** |
| **44/45 run 风险方差低于目标带** | 几何分离 ≠ 风险分离，v3.13 在其他癌种上是否存在风险区分度问题**待查** |
| v3.13 cos 计算是下界非实测 | 下界 ≥ 0.9996 来自 Euclidean → cosine 推导；实际 cosine 需加载 checkpoint 跑前向直接计算 |
| WSI pair_term 非单调反弹 | epoch 1→29，WSI pair_term 从 0.029 反弹到 0.509；原因**待查** |

---

## 5. 产物

```
docs/research/dct_v313_slot_diversity_figure/
  v313_vs_slotspe_p2_comparison.png / .pdf             # §1 P2 三图（修正版）
  v313_slot_diversity_panel_blca_fold0.png / .pdf      # §2 Euclidean 距离面板
  v313_pancancer_45run_overlay.png / .pdf              # 泛化惩罚项
  v313_pancancer_per_study_mean.png / .pdf             # 泛化逐癌种
  v313_pancancer_summary_table.csv                     # 逐 run 数值
  CONCLUSION.md                                        # 本修正版
scripts/plot_v313_slot_diversity_evidence.py           # 修正版脚本（已更新）
```

数据源：
- §1/§2 `results/dct_v313_full_wsi_three_decoders/blca_seed3/.../epoch_curve_fold0.csv`
- §4 `results/dct_v313_uni2h/<study>/<study>/.../epoch_curve_fold{0..4}.csv`
- 目标带常量 `VARIANCE_MIN=0.001 / VARIANCE_MAX=0.050` 取自
  `survot_rank/research/methods/legacy/experimental/dct_v311_slot_interpretable/model.py`
- SlotSPE 常量源：`docs/research/slotspe_p6_addendum_20261020/FINAL_RECEIPT.md` §3
