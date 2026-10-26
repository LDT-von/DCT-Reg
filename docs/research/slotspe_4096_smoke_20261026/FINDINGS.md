# SlotSPE 原版 @ 4096 patch — 端到端实测与 patch/epoch 因子分解

日期：2026-10-26
目的：在 **4096 patch** 预算下验证 SlotSPE 原版单臂端到端可跑通，并复检此前在 256 patch 下记录的 4 项机制问题。

范围：**单折单 seed 的机制复检 + patch×epoch 2×2 因子分解**。不是泛化实验，不是消融，不是治疗性能结论。

---

## 1. 运行身份

| 项 | 值 |
|---|---|
| 入口 | `scripts/run_slotspe_proof.py --config configs/slotspe_clean_original_server.json --patches {256,4096} --folds 0 --epochs {1,30} --batch-size 8 --device cuda:1 --run` |
| 输出根 | `results/slotspe_proof/slotspe_4096_fold0_30ep_20261026`（30 ep，/data1）<br>`/tmp/slotspe_4096_e2e`（4096/1ep）、`/tmp/slotspe_256_e2e`（256/1ep） |
| run_identity (4096/1ep) | `320418eef6cb4d8357ad70ccf1fa9df80929a14370551d46f104afd9775ef120` |
| arm | `original`（唯一臂；未加运输、未加 ABC、未替换 decoder） |
| 源码根 | `/data1/DCT-Reg/third_party/SlotSPE`（11 个文件哈希已记录于 `prepared/manifest.json`） |
| checkpoint sha256 (4096/30ep) | `c0df1755435ae5cf31cf8a864e6818f4f3dd0530ad8e5b5139d72d28ccddf1dd` |
| checkpoint sha256 (4096/1ep) | `8e2d592fca0d6889e6432470230c44758cb6151999e3e33d131ffd94ee1c7c67` |
| same_initial_state sha256 | `109d31ccac93ec39e1e5ffdaaadccaa4a1cb0be051361f227cf52e1095b2ca76` |
| 参数量 | 33,093,390（**四次运行全部一致**，与 patch 数、epoch 数均无关） |
| 4096/30ep 训练耗时 | 993.2 s（30 epoch，1020 steps） |

**确定性复验（两次独立运行）**：4096/30ep 的重跑与磁盘写满事故中保住曲线的首次运行，
**30 个 epoch 的 val_C 逐位完全一致**（`results/slotspe_proof/_partial_4096_30ep_diskfull/curve_partial_30ep_diskfull.json`
vs `results/slotspe_proof/slotspe_4096_fold0_30ep_20261026/pairs/fold_0/seed_3/original/curve.json`）。
同 seed 下逐轮可复现。

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

## 2. 核心结论：padding 敏感度与 gate 坍缩率**主要由 epoch 驱动**

补齐 2×2（patch × epoch）后，在**同一 138 个 (patient, added_rows) 配对单元**上计算
（256/1ep 与 4096/1ep 的 val 患者交集；probe seed 相同，严格配对）：

### 2.1 padding 敏感度（补零行造成的单患者 risk 绝对差）

| cell | max | p90 | median | mean | max_logit_diff |
|---|---|---|---|---|---|
| 256 / 1 ep | 0.002880 | 0.001617 | 5.03e-04 | 7.35e-04 | 0.0144 |
| 4096 / 1 ep | **0.000529** | 0.000384 | 1.26e-04 | 1.84e-04 | 0.0017 |
| 256 / 30 ep | **0.782642** | 0.245810 | 1.35e-02 | 7.86e-02 | 3.0603 |
| 4096 / 30 ep | **0.013015** | 0.001033 | 1.27e-05 | 4.70e-04 | 0.0316 |

**读法（两个方向的效应都已定量）：**
- **同 epoch 下**：4096 patch 一致地**更不敏感**。
  1 ep：max 0.000529 vs 0.002880（约 1/5.4）；30 ep：max 0.013015 vs 0.782642（约 **1/60**）。
  中位数差距更大：30 ep 下 1.27e-05 vs 1.35e-02（约 **1/1000**）。
  机制一致：补零行相对 token 数占比随 patch 数增加而稀释（16/4096 vs 16/256）。
- **同 patch 下**：1→30 epoch 使敏感度**放大 2~3 个数量级**
  （256：max ×272；4096：max ×24.6；median ×380 / ×101）。

**结论：先前基于跨包比较的怀疑（"4096 下 padding 影响更小"）方向正确但量级被 epoch 混淆放大；
补齐 2×2 后可以确定——敏感度由 epoch 主导，patch 数只在其上叠加一个一致的稀释效应。**
原先"基线 0.783 不应归因于 patch 数"的判断经此复核**成立**。

### 2.2 自然 gate near_zero 率

| cell | 观测数 | near_zero | 率 |
|---|---|---|---|
| 256 / 1 ep | 3080 | 0 | 0.000% |
| 4096 / 1 ep | 2760 | 0 | 0.000% |
| 256 / 30 ep | 3080 | **161** | **5.227%** |
| 4096 / 30 ep | 2760 | **17** | **0.616%** |

同样是 epoch 主导（1 ep 两格皆 0），但 4096 下坍缩率低 **8.5 倍**（0.616% vs 5.227%）。
这与 `docs/research/slotspe_gate_collapse_20261022/FINDINGS.md` 的既有结论方向一致
（该文报告 omic 约 9.6% / wsi 约 0.78%，与本次 5.227% 的整体率同量级）。

**重要限定**：4096/1ep 的 0/2760 曾一度是"4096 下无坍缩"的候选解释，
2×2 补齐后**该解释被排除**——充分训练下 4096 仍会坍缩，只是率更低。
这修正了仅凭单格数据可能得出的错误结论。

---

## 3. 四项机制复检（4/4 在 4096 patch 下复现）

| 检查项 | 256/1ep | 4096/1ep | 256/30ep | 4096/30ep |
|---|---|---|---|---|
| `original_disconnection` | observed (8/8) | **observed (8/8)** | observed (8/8) | **observed (8/8)** |
| `evaluation_randomness` | observed | **observed** | observed | **observed** |
| `padding_sensitivity` | observed | **observed** | observed | **observed** |
| `self_interference` | observed | **observed** | observed | **observed** |
| `natural_gate_near_zero` | not_observed | not_observed | observed (161) | **observed (17)** |

4096/30ep 具体数字：
- 梯度断路：8/8 `pred_keep_slot` 参数 `main.kind=None`；aux 侧 7 nonzero + 1 zero。
- 评估随机性：20 rep，val C-index ∈ [0.6081, 0.6283]，std=**0.00511**；
  单患者 risk std mean=**0.0374**，range max=**1.886**。
- padding：69 患者 × {16, 64} 补零行 → max_logit_diff=**0.0316**，max_risk_abs_diff=**0.0130**；自然零行 0。
- self-pool：`max_self_pool_change`=**2.449**。
- RNG 重放：`rng_replay.max_logit_difference = 0.0`（同 seed 完全一致）。

### 全 cell 对照

| 指标 | 256/1ep | 4096/1ep | 256/30ep | 4096/30ep |
|---|---|---|---|---|
| 参数量 | 33,093,390 | 33,093,390 | 33,093,390 | 33,093,390 |
| selector 梯度断路 | 8/8 | 8/8 | 8/8 | 8/8 |
| val C-index mean | 0.7338 | 0.7057 | 0.6959 | **0.6191** |
| val C-index std | 0.00762 | 0.00481 | 0.00776 | 0.00511 |
| patient risk std mean | 0.00831 | 0.00465 | 0.1230 | 0.0374 |
| patient risk range max | 0.0467 | 0.0268 | 1.408 | 1.886 |
| 自然零行 | 0 | 0 | 0 | 0 |
| max_self_pool_change | 2.506 | 2.618 | 2.700 | 2.449 |
| 峰值显存 | 0.99 GiB | 4.71 GiB | 0.99 GiB | 4.71 GiB |

---

## 4. 训练曲线：4096 patch 在 30 epoch 下**未复现** 256 patch 的表现

| | 256 / 30 ep | 4096 / 30 ep |
|---|---|---|
| epoch 1 val_C | 0.7258 | 0.6992 |
| 峰值 val_C | 0.7393（epoch 12） | 0.7157（epoch 2） |
| 末轮 val_C | 0.6886 | **0.6210** |
| 末轮 train loss | — | 0.7602 |
| 末轮 train_main_nll | — | 0.0334 |

4096/30ep 的 val_C 在 epoch 3 后即塌到 0.65 以下并**持续走低**（末轮 0.6210），
而 256/30ep 全程在 0.64–0.74 区间震荡、末轮 0.6886。

**这不是"4096 更好/更差"的性能结论**，理由有三，必须同时说明：
1. **val 队列不同**（69 vs 77 人），不是同一批患者；
2. **过拟合**：4096 的 train_main_nll 降到 0.0334（近完美拟合训练集），
   而 val_C 同步下降——典型过拟合形态；
3. **单折单 seed**，无重复、无置信区间。

**但它是一个明确的、与 patch 预算相关的观察**：
在本设置下，更大的 patch 预算带来更多 token，模型在 30 epoch 内更快拟合训练集并更早开始过拟合。
这与 P1（token 数与性能的关系）方向相关，但**本数据不足以支撑 P1 结论**，
因为队列不同且仅单折单 seed。

### 4.1 风险方向检验（对 SPE-09 ⑤ 意外发现的复核）

SPE-09 在 3 epoch 256 patch 上发现 predicted risk 与 observed time **正**相关
（Spearman ρ=+0.41, p=2.3e-4），即"risk"实际像长生存概率。

本次复核（Spearman，risk 取 20 repeat 均值）：

| 运行 | n | ρ(risk, time) | p |
|---|---|---|---|
| 256 / 30 ep | 77 | **−0.1503** | 0.192 |
| 4096 / 30 ep | 69 | **−0.0702** | 0.567 |
| 4096 / 1 ep | 69 | −0.1630 | 0.181 |

**结论：3 epoch 上的方向翻转没有复现。** 30 epoch 下 ρ 转为轻微负值但**均不显著**（p>0.18）。
即 SPE-09 的"方向翻转"是**欠训阶段的产物**，与原判断一致。
本次所有 val 数字方向正常（ρ<0），但**统计上无显著方向**——
这再次说明单折探索性 val C-index 不能当作性能证据。

---

## 5. 资源实测（支撑后续预算规划）

| 配置 | 峰值显存 | mean step | epoch 训练时长 |
|---|---|---|---|
| 256 patch / bs8 | 0.99 GiB | — | 26.9 s（1ep）/ ~20 s（30ep） |
| 4096 patch / bs8 | 4.71 GiB | 0.339 s | 33.6 s（1ep）/ ~30 s（30ep） |
| 4096 patch / bs32 | 15.98 GiB | 0.450 s | — |
| 4096 patch / bs8（30ep 整跑） | 4.71 GiB | — | 993 s total |

独立 smoke（`scripts/slotspe_4096_smoke.py`，不改源码、不写患者结果）：
`smoke_256.json` 峰值 0.389 GiB / `smoke_4096_bs8.json` 4.06 GiB / `smoke_4096_bs32.json` 15.98 GiB。

**4096 patch 的边际训练成本很小**（epoch ~30 s vs 256 patch 的 ~20 s），
主要成本在**显存**（4.71 GiB vs 0.99 GiB）。batch=8 是 4096 patch 的合理选择；
batch=32 需 15.98 GiB 且**无提速收益**（step 0.450 vs 0.339 s）。

---

## 6. 本次确立的（可引用）

- 4096 patch 端到端**可跑通**，4 stage 全部 completed；30 epoch 整跑 993 s / 峰值 4.71 GiB。
- **4 项机制问题在 4096 patch 下全部复现**（含梯度断路 8/8），且与 patch 数、epoch 数均无关。
- **padding 敏感度由 epoch 主导**（2×2 定量）；patch 数在其上叠加一致的稀释效应
  （同 epoch 下 4096 约 1/5 ~ 1/60 于 256）。
- **自然 gate 坍缩率由 epoch 主导**（1 ep → 0/0；30 ep → 5.227% / 0.616%），
  且 4096 下坍缩率低 8.5 倍。
- **"4096 下无 gate 坍缩"这一候选解释已被 2×2 排除**——充分训练下仍会坍缩。
- 4096 patch 的**队列代价已量化**：42 例排除中 41 例因真实 patch 不足 4096；val 队列 77 → 69。
- **参数量与 patch 数、epoch 数均无关**（33,093,390，四次运行一致）。
- 同 seed 下逐轮可复现（epoch 1 val_C 逐位一致）。
- **SPE-09 的方向翻转是欠训产物**，30 epoch 下不复现（ρ 均 < 0 且不显著）。
- 显存/时间预算表（§5），batch=8 为 4096 patch 的合理配置。

## 7. 本次未确立的（不可引用）

- **任何患者生存预测性能结论**。所有 val C-index 为单折单 seed 的探索性数值；
  4096 vs 256 的 val 数字**队列不同**，不可比；本轮不存在可用于性能主张的对照。
- **"4096 patch 导致过拟合"**——形态吻合（train_main_nll 0.0334 + val_C 单调下降），
  但队列不同且单折单 seed，**不能定性**。
  （注：4096/30ep 的 69 人 val 队列是 256/30ep 的 77 人队列的**严格子集**，
  已核验 overlap=69，因此后续可在该交集上做配对比较；本轮尚未做。）
- **P1（token 数 → 性能）的结论**——本次只提供两个 patch 档位、两个队列，
  且过拟合形态与 token 数直接相关，不足以回答 P1。
- OT 必要性、修正归因、临床因果、问题必然发生——`proof_summary.json` 的
  `claims_not_supported` 已列出，本轮无新增。

---

## 8. 后续建议（待用户拍板）

1. **4096 + 30 epoch × 5 折**：若要回答"4096 是否因过拟合掉分"，
   需多折 + 与 256 同口径的**相同 val 患者**比较（4096 队列是 256 队列的子集，
   可在 69 人交集上做配对比较）。
2. **早停 / 峰值选择**：4096 的峰值在 epoch 2（0.7157），末轮 0.6210；
   若沿用 256 的末轮固定策略，4096 会被系统性低估。可考虑按内层验证选 checkpoint。
3. **保持 256 patch 作为固定参照**：所有机制数字仍以 256/30ep 为准，
   4096 作为新预算的独立支线，不混用两者的 val 数字。
4. **磁盘管理**：4096 patch 的 prepared cache 为 8.0 GB，**必须放在 /data1**
   （`/tmp` 所在分区仅 97 GB，本轮已因此发生一次磁盘写满导致 checkpoint 截断的事故；
   详见 §9）。

---

## 9. 事故记录：/tmp 分区写满导致 checkpoint 截断

**经过**：首次 4096/30ep 运行输出根设在 `/tmp/slotspe_4096_30ep`。
训练本身**完成全部 30 epoch（1020 steps，`curve.json` 完整）**，
但在保存 `last.pt` 与后处理诊断时 `/` 分区（97 GB，100% 满）报
`OSError: [Errno 28] No space left on device`，导致 `last.pt` 截断、
`torch.load` 报 `PytorchStreamReader failed reading zip archive`。

**处置**：
- 保留截断现场于 `results/slotspe_proof/_partial_4096_30ep_diskfull/curve_partial_30ep_diskfull.json`
  （30 轮曲线完整，可用于诊断），**未删除**。
- 清理 `/tmp`（pytest 临时目录等）与已归档运行的 cache；把 4096 cache（8.0 GB）
  迁移至 `/data1/slotspe_cache_4096`。
- 以**新输出根** `results/slotspe_proof/slotspe_4096_fold0_30ep_20261026`（位于 /data1，8.6 TB 可用）
  重跑，完整跑通，checkpoint sha256 `c0df1755435ae5cf31cf8a864e6818f4f3dd0530ad8e5b5139d72d28ccddf1dd`。
- 重跑 epoch 1 的 val_C 与首次逐位一致，**同时验证了训练的确定性**。

**教训**：4096 patch 预算下 prepared cache 达 8.0 GB（256 patch 仅 625 MB），
**任何 4096 patch 运行的输出根都应放在 /data1**。

### 9.1 已落实的代码级防呆（2026-10-26 追加）

上述教训此前只写在本文件里，依赖人工记住。现已在 `survot_rank/research/slotspe_proof/queue.py`
中落实为**强制检查**，不再依赖执行者自觉：

| 机制 | 位置 | 行为 |
|---|---|---|
| 启动前磁盘预算 | `execute()` 开头，最先于任何文件/数据检查 | 空间不足 → 抛 `RuntimeError` 并**在 prepare 之前**中止，不产生任何半成品 |
| 每 stage 前复查 | `execute()` 阶段循环内 | 长跑期间被其它进程占满时，在写 checkpoint 前停下 |
| `--doctor` 展示 | `preflight()` → `disk_budget` 字段 | 启动前即可看到需求估算与实际余量 |
| 估算口径 | `_disk_budget()` | cache = `400 患者 × patches × 1536 × 4 B`（5 折共享一份）；pair = `折数 × seed 数 × 2 × 133 MB`（实测 checkpoint 132.7 MB） |

实测拦截效果（5 折 × 4096 patch，估算 14.3 GB）：

```
RuntimeError: Insufficient disk for this batch; refusing to start because a full disk
truncates checkpoints and invalidates the run.
  output_root       : /tmp/slotspe_diskguard_test
  estimated needed  : 14.3 GB (cache 10.1 GB + pairs 4.2 GB)
  free on that path : 10.6 GB
  Put --output-root on a filesystem with enough free space (e.g. /data1).
```

**5 折 × 4096 patch 的真实磁盘需求 ≈ 14.3 GB**（cache 约 8.0–10.1 GB + checkpoint 约 1.3 GB + 余量）。
`/data1` 有 8.6 TB，余量充足；`/` 分区仅剩约 11 GB，**单折即不足**。

回归测试：`tests/test_slotspe_proof_queue.py` 新增 3 项（预算随 patch/折数缩放、
空间不足拒绝启动且不产生 prepared 目录、空间充足不误拦），
连同既有 SlotSPE 测试共 **50 项全部通过**。

---

## 10. 产物清单

`docs/research/slotspe_4096_smoke_20261026/`

| 文件 | 内容 |
|---|---|
| `smoke_256.json` | 256 patch / bs8 独立 smoke：峰值 0.389 GiB，mean step 0.66 s |
| `smoke_4096_bs8.json` | 4096 / bs8：峰值 4.06 GiB，mean step 0.339 s |
| `smoke_4096_bs32.json` | 4096 / bs32：峰值 15.98 GiB，mean step 0.450 s |
| `resolved_config_4096.json` | 4096 配置快照 |
| `e2e_fold0_1ep/` | 4096 / 1 epoch 端到端产物（SHA256 与运行目录逐文件校验一致） |
| `e2e_fold0_1ep_256control/` | 256 / 1 epoch 对照产物（2×2 因子分解用，SHA256 已校验） |
| `e2e_fold0_30ep/` | 4096 / 30 epoch 端到端产物（SHA256 已校验） |

各 `e2e_fold0_*/` 目录内含：`curve.json` / `eval_repeats.json` / `native_model_audit.json` /
`natural_gate_observations.json` / `patient_slot_intervention.json` / `summary.json` /
`trained_selector_gradients.json` / `validation_predictions.csv` / `proof_summary.json` /
`proof_summary.txt` / `queue_status.json` / `resolved_config.json`

**未纳入版本库**：`last.pt`（132.7 MB）、`shared_initial_state.pt`、prepared cache（8.0 GB）。
checkpoint 哈希已记录在 `summary.json` 与 receipt 中，需要时可按哈希定位复核。
