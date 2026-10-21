# SlotSPE 问题清单与 DCT v3.13 结论

**日期** 2026-10-21 | **P2 主证据** BLCA / uni2-h / seed 3 / fold 0 / 30 epoch
**泛化证据** 9 study × 5 fold = **45 runs**（`results/dct_v313_uni2h`）
**结论** v3.13 **不存在精确槽冗余**（无重复对、无全塌缩），且结构上不存在置换敏感性的放大器。
⚠️ 但泛化 run 显示 **per-slot 风险区分度未达 v311 目标带**（44/45 runs 偏低）——见 §5(b)。

---

## 1. 三条结论速览

| # | 问题 | SlotSPE | **DCT v3.13** | 证据 |
|---|---|---|---|---|
| P2 | 8 个 slot 塌缩成近重复 | 🔴 塌缩 | 🟢 **0 精确塌缩**（BLCA 实测）| 图1(d,e) |
| P2 泛化 | 惩罚项收敛 ≠ 槽有区分度 | — | ⚠️ **44/45 run 风险方差低于目标带** | §5(b) |
| P4 | 置换改变输出 | 🔴 2.55e-02 | 🟢 **无 top-k，不存在** | §4 |
| P6 | 推理期随机 | 🔴 风险抖动 | 🟢 **确定性 init** | §4 |

---

## 2. P2 槽冗余：SlotSPE 的核心缺陷

**现象**：训练后 slot 几乎全等，模型实际只用 1 个 slot。

| 指标 | 初始化 | 训练后 |
|---|---|---|
| inter-slot cosine (mean) | 0.4053 | — |
| inter-slot cosine (**max**) | 0.5605 | **0.9996** |
| `slots_logsigma.exp()` | — | 1.0045 (wsi) / 1.0001 (omic) |

**sigma 未收缩 → 不是初始化问题，是训练压塌的。**

![SlotSPE 塌缩 vs v3.13](/data1/DCT-Reg/docs/research/dct_v313_slot_diversity_figure/v313_vs_slotspe_p2_comparison.png)

---

## 3. v3.13 证明：无槽冗余

![v3.13 多样性面板](/data1/DCT-Reg/docs/research/dct_v313_slot_diversity_figure/v313_slot_diversity_panel_blca_fold0.png)

| 指标 | epoch 0 | epoch 29 | 判定 |
|---|---|---|---|
| 精确重复对 `duplicate_pair_fraction` | 0.0 | **0.0** | 🟢 30 epoch 全程为 0 |
| 患者全塌缩 `full_collapse_patient_fraction` | 0.0 | **0.0** | 🟢 全程为 0 |
| 平均 slot 距离 | 0.407 | **6.88** | 🟢 变远离 |
| 最近 slot 对距离 | 0.137 | **0.509** | 🟢 远离 0 三个数量级 |
| 违反分离约束的对占比 | 0.950 | **0.060** | 🟢 94% 达标 |

**决定性证据是面板 (b)**：每患者**最相似的那一对** slot 的距离。塌缩必然贴近 0，实测 0.51/0.33。

**根因**：`diversity.py` 对全部 56 个有向对逐个计算，先罚后平均——单个塌缩对无法被远点掩盖。

---

## 4. P4 / P6：为什么 v3.13 结构上免疫

| 机制 | SlotSPE | v3.13 |
|---|---|---|
| readout 选择 | Gumbel **top-k** (k=2/8) | 无 top-k，梯度回到全部 slot |
| 塌缩后果 | k-th 与 k+1-th 分差 min **2.1e-04** → 数值仲裁 → 29 个 mask 元素改变 | 不存在仲裁路径 |
| 推理期 slot 初始化 | **每次 forward 重采样** → 种子变动 logits 达 7.39e-02 | `dct_slot_init_mode: deterministic` |

> v3.13 的 `topk_ratio / temperature / top_k_method` 是**失效残留参数**，不生效。

---

## 5. 泛化证据：9 cancer × 5 fold = 45 runs

![45 run 叠加](/data1/DCT-Reg/docs/research/dct_v313_slot_diversity_figure/v313_pancancer_45run_overlay.png)

![9 study 均值](/data1/DCT-Reg/docs/research/dct_v313_slot_diversity_figure/v313_pancancer_per_study_mean.png)

数据源 `results/dct_v313_uni2h`：blca 外的 **brca / coadread / hnsc / kirc / luad / lusc / skcm / stad / ucec**，各 5 fold，共 45 run。

| study | div ep0 | div ep29 | 降幅 | var ep29 | 落入目标带 |
|---|---|---|---|---|---|
| brca | 0.01852 | 0.00133 | 14.4× | 0.00004 | 0/5 |
| coadread | 0.03187 | 0.00143 | 23.4× | 0.00002 | 0/5 |
| hnsc | 0.02933 | 0.00144 | 23.0× | 0.00011 | 0/5 |
| kirc | 0.02969 | 0.00138 | 22.1× | 0.00012 | 0/5 |
| luad | 0.02899 | 0.00130 | 22.2× | 0.00010 | 0/5 |
| lusc | 0.02919 | 0.00171 | 17.5× | 0.00007 | 0/5 |
| skcm | 0.03159 | 0.00175 | 20.6× | 0.00053 | **1/5** |
| stad | 0.04598 | 0.00182 | 26.9× | 0.00004 | 0/5 |
| ucec | 0.03979 | 0.00192 | 26.5× | 0.00011 | 0/5 |

**两条读数必须合看：**

- **(a) 多样性惩罚项 45/45 run 全部下降 14–27×，终点 < 2e-3。** 这是**惩罚函数**（`slot_diversity_loss` 返回值，越小越好），不是距离。下降 = 约束被满足，**不可**读成"多样性变差"。
- **(b) 但 per-slot hazard 方差 44/45 run 终点低于 v311 目标带 [0.001, 0.050]**（ep29 均值 2e-5 ~ 5e-4）。仅 skcm/fold2 达标。

**因此：惩罚项收敛不足以单独证明槽在承担不同的风险角色。** §3 的 BLCA 结论建立在 pair 距离与重复对上，那套诊断在 uni2h CSV 中不存在（见 §5 缺口）。

---

## 6. 已知缺口（诚实声明）

| 项 | 状态 |
|---|---|
| v3.13 **inter-slot cosine** 未实测 | CSV 只存 distance。补测需加载 checkpoint 跑一次前向 |
| 完全塌缩时 pair 梯度为零 | 代码已标注；实测 `full_collapse = 0.0`，未触发 |
| 确定性初始化 + 完全塌缩 = 零梯度死锁 | 理论可达，当前 epoch0 距离 0.41 远离该状态 |
| **45 run 无 pair 级诊断** | uni2h CSV 仅 22 列，无 `duplicate_pair_fraction` / `mean_pair_distance`。"45 run 无塌缩"**未实测** |
| **44/45 run 风险方差低于目标带** | §5(b)。槽的几何分离与风险分离**解耦**，前者满足不代表后者。BLCA 的 pair 级结论**不可**外推到这 45 run（其 CSV 无该诊断）|
| uni2h 是否走 repaired 分支 | `experiment_settings.txt` 未记录 revision；但惩罚项 14–27× 的收敛形态与 BLCA repaired 一致 |

---

## 7. 产物

```
docs/research/dct_v313_slot_diversity_figure/
  v313_slot_diversity_panel_blca_fold0.png / .pdf      # §3 BLCA fold0 pair 级证据
  v313_vs_slotspe_p2_comparison.png / .pdf             # §2 与 SlotSPE 对照
  v313_pancancer_45run_overlay.png / .pdf              # §5 45 run 叠加
  v313_pancancer_per_study_mean.png / .pdf             # §5 9 study 均值 ± std
  v313_pancancer_summary_table.csv                     # §5 逐 run 数值
scripts/plot_v313_slot_diversity_evidence.py   # §2/§3，直接读 CSV，无随机性
scripts/plot_v313_pancancer_evidence.py        # §5，直接读 CSV，无随机性
```

数据源：
- §2/§3 `results/dct_v313_full_wsi_three_decoders/blca_seed3/.../epoch_curve_fold0.csv`
- §5 `results/dct_v313_uni2h/<study>/<study>/.../epoch_curve_fold{0..4}.csv`
- 目标带常量 `VARIANCE_MIN=0.001 / VARIANCE_MAX=0.050` 取自
  `survot_rank/research/methods/legacy/experimental/dct_v311_slot_interpretable/model.py`
  （**注意**：`scripts/_archive_v310_v311/analyze_v311_slot_interp.py` 里的 0.005 是过期值，勿用）
- SlotSPE 常量源：`docs/research/slotspe_p6_addendum_20261020/FINAL_RECEIPT.md` §3