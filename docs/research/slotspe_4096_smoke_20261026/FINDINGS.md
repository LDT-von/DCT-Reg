# SlotSPE 原版 @ 4096 patch — 端到端单折实测（BLCA fold 0, seed 3）

日期：2026-10-26
目的：在 **4096 patch** 预算下验证 SlotSPE 原版单臂端到端可跑通，并复检此前在 256 patch 下记录的 4 项机制问题。

范围：**单折单 seed 的机制复检 + patch/epoch 因子分解**。不是泛化实验，不是消融，不是治疗性能结论。

---

## 1. 运行身份

| 项 | 值 |
|---|---|
| 入口 | `scripts/run_slotspe_proof.py --config configs/slotspe_clean_original_server.json --patches {256,4096} --folds 0 --epochs {1,30} --batch-size 8 --device cuda:1 --run` |
| 输出根 | `/tmp/slotspe_4096_e2e`（1 ep）、`/tmp/slotspe_256_e2e`（1 ep）、`/tmp/slotspe_4096_30ep`（30 ep） |
| run_identity (4096/1ep) | `320418eef6cb4d8357ad70ccf1fa9df80929a14370551d46f104afd9775ef120` |
| arm | `original`（唯一臂；未加运输、未加 ABC、未替换 decoder） |
| 源码根 | `/data1/DCT-Reg/third_party/SlotSPE`（11 个文件哈希已记录于 `prepared/manifest.json`） |
| checkpoint sha256 (4096/1ep) | `8e2d592fca0d6889e6432470230c44758cb6151999e3e33d131ffd94ee1c7c67` |
| same_initial_state sha256 | `109d31ccac93ec39e1e5ffdaaadccaa4a1cb0be051361f227cf52e1095b2ca76` |
| 参数量 | 33,093,390（与 256 patch 基线**完全一致**） |

### 队列口径变化（重要）

4096 patch 使「真实 patch 数不足」成为主要排除原因：

| | 256 patch | 4096 patch |
|---|---|---|
| train | 302 | **270** |
| val | 77 | **69** |
| excluded | 2 | **42** |
| complete-case 患者 | 379 | 339 |

`native_model_audit.json` → `padding.source_loader_opportunity_counts_from_exclusions`：
- `fewer_real_patches_than_configured_no_padding`: **41**
- `missing_clinical_endpoint_or_wsi_string`: 1

**4096 patch 下 42 例排除中 41 例是"该患者真实 patch 不够 4096"。**
被排除者真实 patch 数介于 256 与 4096 之间——这是 patch 预算提升的直接代价，不是数据缺陷。
因此 4096 与 256 的 val 队列**不是同一批患者**（69 ⊂ 77，少 8 人），两者 val C-index 不可直接比较。

---

## 2. 关键结论：padding 量级差异源于 **epoch**，不是 patch 数

初看 4096/1ep 的 padding 敏感度比 256/30ep 基线小三个数量级
（max risk 差 5.29e-4 vs 0.783），但这同时变了两个因子。
因此补跑 **256 patch / 1 epoch** 作同 epoch 对照，形成 2×2：

| 运行 | patch | epoch | 患者队列 | max risk 差 | p90 | median | mean |
|---|---|---|---|---|---|---|---|
| 256/1ep | 256 | 1 | 77 | 0.002880 | 0.001617 | 5.03e-04 | 7.35e-04 |
| 4096/1ep | 4096 | 1 | 69 | **0.000529** | 0.000384 | 1.26e-04 | 1.84e-04 |
| 256/30ep | 256 | 30 | 77 | **0.782642** | 0.245810 | 1.35e-02 | 7.86e-02 |
| 4096/30ep | 4096 | 30 | 69 | 见 §6 | | | |

上表三行的 max/p90/median/mean 均在**同一 138 个 (patient, added_rows) 配对单元**上计算
（256/1ep 与 4096/1ep 的 val 患者交集，patch 数不影响患者身份），
probe seed 相同，因此是严格的配对比较。

**读法：**
- **同一 epoch 下**，4096 patch 的 padding 敏感度**反而更小**（max 0.000529 vs 0.002880，约 1/5）。
  4096 patch 下补零行相对 token 数占比更小（16/4096 vs 16/256），这与机制一致。
- **同一 patch 下**，1 epoch → 30 epoch 使 padding 敏感度**放大 ~270 倍**（max 0.0029 → 0.783）。

**结论：基线的 0.783 数量级主要由训练轮数（模型被训练的程度）驱动，不是 patch 数驱动。
先前基于跨包比较的怀疑（"4096 下 padding 影响更小"）不成立，且方向被反转。**

同一批 5 名最敏感患者在三次运行中的追踪（added_rows=64）：

| 患者 | 256/30ep | 256/1ep | 4096/1ep |
|---|---|---|---|
| TCGA-H4-A2HO | 0.7826 | 0.000624 | 0.000300 |
| TCGA-GV-A3QI | 0.7056 | 0.000730 | 0.000202 |
| TCGA-BT-A20O | 0.6711 | 0.000473 | 0.000354 |
| TCGA-CF-A27C | 0.6020 | 0.000507 | 0.000422 |
| TCGA-FD-A3SP | 0.5857 | 0.000954 | 0.000224 |

这 5 人在 4096/1ep 队列内，且 4096 下全部**更小**——患者身份不是混淆项。

---

## 3. 关键结论：自然 gate near_zero 同为 epoch 驱动

| 运行 | 观测数 | near_zero |
|---|---|---|
| 256/1ep | 3080 | **0** |
| 4096/1ep | 2760 | **0** |
| 256/30ep | 3080 | **161** |

与 padding 同一模式：1 epoch → 30 epoch 使 near_zero 从 0 变为 161。
两种解释仍需 4096/30ep 区分：
- 1 epoch 的 `pred_keep_slot` 尚未学到会触发坍缩的分数分布
  （4096/1ep 的 aux 侧仍有 1 个参数为 `zero`）；
- 队列/观测数略少（2760 vs 3080）。

**在 4096/30ep 出来前，不能记作"4096 下没有 gate 坍缩"。**
这与 `docs/research/slotspe_gate_collapse_20261022/FINDINGS.md` 的既有结论一致：
坍缩需要模型进入特定分数区域，而该区域是训练塑造出来的。

---

## 4. 四项机制复检（4/4 复现，4096/1ep）

| 检查项 | 状态 | 关键数字 |
|---|---|---|
| `original_disconnection` | **observed** | 8/8 `pred_keep_slot` 参数 `main.kind=None`；aux 侧 7 nonzero + 1 zero |
| `evaluation_randomness` | **observed** | 20 rep，val C-index ∈ [0.7001, 0.7167]，std=**0.00481**；单患者 risk range max=**0.0268** |
| `padding_sensitivity` | **observed** | 69 患者 × {16, 64} 补零行 → max_logit_diff=**1.69e-3**，max_risk_abs_diff=**5.29e-4**；自然零行 0 |
| `self_interference` | **observed** | `max_self_pool_change`=**2.618** |
| `natural_gate_near_zero` | **not_observed** | 69 患者 × 40 slot = 2760 观测，near_zero=**0**，nonfinite=0 |

RNG 重放：`rng_replay.max_logit_difference = 0.0`（同 seed 完全一致）。

### 各次运行对照

| 指标 | 256/1ep | 4096/1ep | 256/30ep（基线） |
|---|---|---|---|
| 参数量 | 33,093,390 | 33,093,390 | 33,093,390 |
| selector 梯度断路 | 8/8 `main.kind=None` | 8/8 `main.kind=None` | 8/8 `main.kind=None` |
| val C-index mean | 0.7338 | 0.7057 | 0.6959 |
| val C-index std | 0.00762 | 0.00481 | 0.00776 |
| patient risk std mean | 0.00831 | 0.00465 | 0.1230 |
| patient risk range max | 0.0467 | 0.0268 | 1.408 |
| padding max_logit_diff | 1.44e-2 | 1.69e-3 | 3.12e-3 |
| padding max_risk_abs_diff | 2.88e-3 | 5.29e-4 | 7.83e-1 |
| 自然零行 | 0 | 0 | 0 |
| max_self_pool_change | 2.506 | 2.618 | 2.700 |
| natural_gate near_zero | 0/3080 | 0/2760 | 161/3080 |
| 峰值显存 | 0.99 GiB | 4.71 GiB | 0.99 GiB |
| 训练耗时 | 118 s | 170 s | 678 s |

**val C-index 三者不可比**：队列不同（77 vs 69）、epoch 不同、patch 不同。
patient risk std 的巨大差异（0.0047–0.0083 vs 0.1230）同样由 epoch 主导——30 epoch 后模型对 RNG 更敏感。

---

## 5. 资源实测（支撑后续预算规划）

| 配置 | 峰值显存 | mean step | epoch 训练时长 |
|---|---|---|---|
| 256 patch / bs8 | 0.99 GiB | — | 26.9 s |
| 4096 patch / bs8 | 4.71 GiB | 0.339 s | 33.6 s |
| 4096 patch / bs32 | 15.98 GiB | 0.450 s | — |
| 256 patch / bs8（30 ep 基线） | 0.99 GiB | — | ~20 s |

独立 smoke（`scripts/slotspe_4096_smoke.py`，不改源码、不写患者结果）：
`smoke_256.json` 峰值 0.389 GiB / `smoke_4096_bs8.json` 4.06 GiB / `smoke_4096_bs32.json` 15.98 GiB。

**4096 patch 的边际训练成本很小**（epoch 33.6 s vs 256 patch 的 20–27 s），
主要成本在**显存**（4.71 GiB vs 0.99 GiB）。batch=8 是 4096 patch 的合理选择；
batch=32 需 15.98 GiB 且无提速收益（step 0.450 vs 0.339 s）。

---

## 6. 4096 / 30 epoch（补全 2×2 的第四格）

**状态：运行中**（`/tmp/slotspe_4096_30ep`）。
这一格决定：

1. **padding 敏感度在 4096 下的真实量级**——若与 256/30ep 同量级（~0.78），
   则确认"敏感度由 epoch 而非 patch 数驱动"在充分训练下同样成立；
   若显著更小，则说明存在 patch 数与训练程度的交互。
2. **自然 gate 坍缩率在 4096 下的真实值**——确认 0/2760 是欠训假象还是真实差异。
3. **充分训练下的 patient risk 随机性**（256/30ep 为 std 0.1230 / range 1.408）。

**必须强调**：这一格出来之前，§2 的结论只在 **1 epoch** 下成立。
**充分训练下 padding 敏感度是否仍随 patch 数下降，目前未验证。**

---

## 7. 本次确立的（可引用）

- 4096 patch 端到端**可跑通**：4 stage 全部 completed，1 epoch / 34 steps / 170 s / 峰值 4.71 GiB
  （RTX 5090, cuda:1, torch 2.10.0+cu128）。
- 4096 patch 的**队列代价已量化**：42 例排除中 41 例因真实 patch 不足 4096；val 队列 77 → 69。
- **padding 敏感度的量级由 epoch 主导**：同 epoch 下 4096 反而比 256 更小（1/5）；
  同 patch 下 1→30 epoch 放大 ~270 倍。基线 0.783 不应被归因于 patch 数。
- **自然 gate near_zero 同为 epoch 主导**：1 epoch 下 0/3080，30 epoch 下 161/3080。
- **4 项机制问题在 4096 patch 下全部复现**（含梯度断路 8/8），且与 patch 数、epoch 无关。
- 参数量 33,093,390 与 patch 数无关（三次运行一致）。
- 显存/时间预算表（§5），batch=8 为 4096 patch 的合理配置。

## 8. 本次未确立的（不可引用）

- 任何患者生存预测性能结论。val C-index 0.7057（4096/1ep）是 **1 epoch、69 人、单折**的探索性数值，
  既非泛化数字，也不可与 256/30ep 的 0.6959 或 256/1ep 的 0.7338 比。
- **充分训练下 4096 patch 的 padding 敏感度与 gate 坍缩率**——§6 第四格未完成。
- "4096 下无 gate 坍缩"——与 1 epoch / 队列大小混淆。
- OT 必要性、修正归因、临床因果、问题必然发生——`proof_summary.json` 的 `claims_not_supported` 已列出，本轮无新增。

---

## 9. 后续建议（待用户拍板）

1. **等 4096/30ep 出来**，补全 2×2 并复核 §2/§3 的结论是否在充分训练下成立。
2. 若成立，再考虑 4096 + 30 epoch × 5 折（预估 ~5.5 h/折 × 5）。
3. **保持 256 patch 作为固定参照**：所有机制数字仍以 256/30ep 为准，4096 作为新预算的独立支线，
   不混用两者的 val 数字。

---

## 10. 产物清单

`docs/research/slotspe_4096_smoke_20261026/`

| 文件 | 内容 |
|---|---|
| `smoke_256.json` | 256 patch / bs8 独立 smoke：峰值 0.389 GiB，mean step 0.66 s |
| `smoke_4096_bs8.json` | 4096 / bs8：峰值 4.06 GiB，mean step 0.339 s |
| `smoke_4096_bs32.json` | 4096 / bs32：峰值 15.98 GiB，mean step 0.450 s |
| `resolved_config_4096.json` | 4096 配置快照 |
| `e2e_fold0_1ep/` | 4096 / 1 epoch 端到端产物（SHA256 已与 `/tmp/slotspe_4096_e2e` 逐文件校验一致） |
| `e2e_fold0_1ep_256control/` | 256 / 1 epoch 对照产物（§2 因子分解用） |

`e2e_fold0_1ep/` 内含：`curve.json` / `eval_repeats.json` / `native_model_audit.json` /
`natural_gate_observations.json` / `patient_slot_intervention.json` / `summary.json` /
`trained_selector_gradients.json` / `validation_predictions.csv` / `proof_summary.json` /
`proof_summary.txt` / `queue_status.json` / `resolved_config.json`

**未纳入版本库**：`last.pt`（132.7 MB）、`shared_initial_state.pt`、prepared cache（8.3 GB）。
checkpoint 哈希已记录在 `summary.json` 与 receipt 中，需要时可按哈希定位复核。
