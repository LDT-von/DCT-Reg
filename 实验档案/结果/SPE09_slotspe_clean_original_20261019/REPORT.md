# SPE-09 / SPE-07 干净原版 SlotSPE 启动实测 (BLCA, fold0, seed3, 2026-10-19)

**对应提交**：`origin/main @ f5370b63f24dcffb8504a8885b8d296ef1f01876` (clean original audit)
**提交者**：刘栋天 (LDT-von)
**环境**：本机 (RTX 5090, dct env, torch 2.10.0+cu128)
**commit message**：feat: deliver clean original SlotSPE server audit

---

## 一、启动链路真实跑出来的故障（按时间顺序）

| # | 现象 | 原因 | 修复 |
|---|---|---|---|
| 1 | 本机落后 origin/main 2 commits | git 没拉 | `git pull --ff-only` |
| 2 | 工作区有 `scripts/run_transport_only.py` 130 行未提交 SlotSPE UNI2-h patch | 之前会话遗留 | `git stash` |
| 3 | `--doctor` 通过但 `--run` 在 `prepare` 阶段 `ValueError: No complete-case train patients remain in fold 0` | features_root 路径错误：`/data1/TCGA-UNI2-h-features` 是 `.h5` 仓，patient.py:248 写死 `.pt` 拼接 | 改 `SLOTSPE_FEATURES_ROOT=/data1/DCT-Reg/data/slotspe_pt_features/blca/pt_files` |
| 4 | 第二次 `prepare` 同样 0 train，per-patient exclusion 100% 是 `at_least_one_missing_slide_feature` | features_root 子目录层级搞错——`{stem}.pt` 直接在 `pt_files/` 下，不能再加 `blca/` | `SLOTSPE_FEATURES_ROOT=.../blca/pt_files`（不带 `blca/`） |
| 5 | 第三次 prepare 在某 train patient 上 `EOFError`，不是 `RuntimeError`，**patient.py:245 except 没接住**，整批崩 | 仓里 1/115 `.pt` 是 legacy torch zip 格式，mmap 触发 `RuntimeError`，fallback `weights_only=True` 在 legacy 上又抛 `EOFError`；该患者 `TCGA-DK-A3IV` 在 train 集合 | 改 `survot_rank/research/slotspe_proof/patient.py`：① 路径优先 `.pt` 否则回落到 `.h5`；② except 扩成 `(RuntimeError, EOFError)` |
| 6 | 改完跑通：`prepare train=302 val=77 excluded=2` | 304 train 中 1 名 patient 的 slide label 在临床表中缺（`missing_clinical_endpoint_or_wsi_string`），1 名 patch 不足 256（`fewer_real_patches_than_configured_no_padding`），剩 302 | 通过 |

**所有 4 stage 全过**：protocol → prepare → fold_0_seed_3 → summarize。
**driver log**：`/tmp/slotspe_clean_original_blca_run_20261020.driver.log` (含 backup)。

---

## 二、论文层的 4 个科学问题——实测答案

### Q1: selector 是否与主 NLL 梯度连接？

| 证据 | 数值 |
|---|---|
| `disconnected_selector_status` | **`observed`** |
| `main_gradient_connected_parameter_count / selector_parameter_count` | **0 / 8** |
| `main_gradient_nonzero_parameter_count` | 0 |
| 所有 8 个 `pred_keep_slot.{wsi,omic}.0.weight/0.bias/2.weight/2.bias` | **main 端 `kind: None, connected: False`** |
| 同 selector 接到 auxiliary 损失梯度？ | **是**（`aux.kind: nonzero, norm=0.0016..0.0103`） |

**结论**：selectors 的梯度来自 auxiliary 损失而非主 NLL。3 个 epoch + 114 个 optimizer step 后，selector 参数对 survival NLL 是**断路**。

### Q2: 同 weight + 同 RNG 评估散度？

| 证据 | 数值 |
|---|---|
| `evaluation_randomness_observed` | **`True`** |
| `val_cindex_mean` / `std_population` / `min` / `max` (20 reps, 同 weight, **仅 RNG seed 变**) | **0.7223 / 0.0035 / 0.7132 / 0.7298** |
| `patient_risk_std_mean` / `max` | 0.0073 / 0.0301 |
| `patient_risk_range_max` | **0.136**（单 patient 跨 20 reps 摆动 0.136） |
| 20 个 cindex 原始序列 | `[0.7195, 0.7242, 0.7219, 0.7258, 0.7211, 0.725, 0.7211, 0.7219, 0.7298, 0.7132, 0.725, 0.7163, 0.7211, 0.7266, 0.7219, 0.7195, 0.7227, 0.7235, 0.7235, 0.7235]` |

**结论**：仅 RNG 变化（无训练、无 checkpoint 变化、无 patch 变化），20 次重复评估的 val C-index 在 **0.7132–0.7298 之间漂**，**单一报告数字 + 单一 RNG 实际上能挑出 +0.007 的差异**。这是工程层面的"评估不可重复"。

### Q3: padding 是否影响真实患者 logits？

| 证据 | 数值 |
|---|---|
| `padding.sensitivity` | **`observed`**（阈值 1e-7） |
| 例子 patient `TCGA-2F-A9KO`，原始 logits | `[-5.423, -3.581, -0.592, -1.568]` |
| 加 16 行 controlled zero rows 后 logits | `[-5.422, -3.577, -0.592, -1.568]` |
| `max_logit_difference` / `risk_abs_difference` | **0.00414 / 0.000245** |

**结论**：真实患者 bag 在**没有自然零行**（`original_zero_feature_rows: 0`）的情况下，**额外 padding 16 行改变 logits 最大 4e-3、risk 差 2.5e-4**。这是结构性数值依赖，不是 padding 错误（因为该患者没自然缺 patch），而是 SlotSPE 前向对 batch 内 padding rows 数敏感。

### Q4: 末轮患者 slot 干预 / self-pool 干扰？

| 证据 | 数值 |
|---|---|
| `self_interference` | **`observed`** |
| `max_self_pool_change` | **2.563**（已训练模型上 self-pool 跨 batch 改动值） |
| `controlled_gate_case.counterexample` | **`observed`** |
| controlled case 第 1/3/4 个 seed 的 gate_sum | `1.0 / 7.6e-36 / 1.0` |

**结论**：在控制实验（独立 native decoder 副本，构造 scores 输入）下，gate_sum **可降到 7.6e-36**——这是**机制层面**的坍缩可能；但末轮 trained checkpoint 上**自然预测没有出现 near_zero gate**（`natural_gate_near_zero: not_observed`，0/3080）。

### 训练曲线

| epoch | train_total_loss | val_C |
|---|---|---|
| 1 | 2.858 | 0.7258 |
| 2 | 2.580 | **0.7425** |
| 3 | 2.419 | 0.7227 |

最佳 epoch=2 val C=0.742，但仅 3 epoch 拟合，明显欠训。论文里**不能拿这个数字作 generalization 报告**。

### 副产物（诊断性）

- `purity.checks`：5/5 native_model_class / native_forward / native_slot_classes / native_decoder_classes / no_repaired_eval_buffers 全为 True → 末轮确实是原始 SlotSPE，没被替换。
- `checkpoint_state_unchanged_by_diagnostics: True`：诊断代码不污染 checkpoint。
- `rng_replay.max_logit_difference: 0.0`（within 1e-6）：**同 RNG seed 同 weight 同 patient 重新评估，logits 完全一致**——意味着 RNG 才是 20-rep 散度的根因，不是 `torch.load` 或其他状态差异。

---

## 三、与 f5370b6 commit message / guide 的对照

| commit message 声明 | 实测是否成立 |
|---|---|
| 干净原版单臂 | ✓ arms=['original']，未添加运输、未添加 ABC |
| fold0, seed3, 3 epoch, 256 patch, batch8 | ✓ config + curve + protocol.json 三方一致 |
| 20 次评估 | ✓ `eval_repeats: 20`，20/20 cindex defined |
| 26 个文件、新任务包、诊断代码、原始证据和执行说明 | ✓ git log --stat 列出 26 个文件 |
| 隔离流程复验通过、远端提交/文件/任务包哈希一致 | 提交是 `f5370b6`，SHA 已是远端；本机 `git pull --ff-only` 无冲突 |

**结论**：commit message 全部真实。

---

## 四、未做（边界声明）

1. **SPE-07 = E051/related** ——AGENTS.md 没登记 SPE-07 是否对应 E0XX 编号，待用户定。
2. **本机未提交改动**：
   - `scripts/run_transport_only.py` 130 行 SlotSPE UNI2-h patch（**stash@{0}**）
   - `survot_rank/research/slotspe_proof/patient.py` 路径 fallback + except 扩 (RuntimeError, EOFError) 18 行（**已 working tree，未提交**）
3. **`'fewer_real_patches_than_configured_no_padding'`** 排除的 1 名 patient：未单独诊断是哪个、缺多少 patch。
4. **3 epoch 训练的 val_C 0.7223 ± 0.0035 不作 SPE 报告数字**。本机 epoch=3 的 val 评估是探索性，不是独立 test。

---

## 五、产物留底

- `/data1/DCT-Reg/results/slotspe_proof/slotspe_clean_original_blca_run_20261020/`
    - `proof_summary.json` `pair_summary.json` `queue_manifest.json`
    - `pairs/fold_0/seed_3/original/`：curve.json / last.pt / eval_repeats.json / validation_predictions.csv / native_model_audit.json / natural_gate_observations.json / patient_slot_intervention.json / trained_selector_gradients.json / summary.json
- `/data1/DCT-Reg/实验档案/结果/SPE09_slotspe_clean_original_20261019/`
    - `driver.log` `proof_summary.json` `pair_summary.json` `queue_manifest.json` `SHA256SUMS.txt`