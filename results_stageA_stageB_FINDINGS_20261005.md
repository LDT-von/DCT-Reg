# DCT v3.13 Stage A / Stage B — 数据完整性审查（2026-10-05）

> 状态：**代码 commit 已通过，结果数据需重跑**。
> 审查人：主对话，2026-10-05 11:30 (UTC+8)。
> 审查方式：从 `epoch_curve_fold*.csv` 与 `split_*_results_final.pkl` 的 md5 独立核验。

---

## 1. 结论（先说结果）

| 实验组 | 状态 | 原因 |
|---|---|---|
| **Stage A — DCT-Reg v313**（direct / independent × KIRC / BLCA）| ❌ **数据不可信** | 5 折的 CSV / pkl 字节完全相同（std=0.0000） |
| **Stage B — SlotSPE**（matched / native × KIRC / BLCA）| ✅ 数据可信 | 每个 fold 跑独立 shell，CSV 字节互异 |
| **5 个代码 commit**（scheduler、outer_test、paper）| ✅ 代码可推 | 不依赖有问题的数据 |

**FINAL 文档里 Stage A 的 5 折 c-index（KIRC direct 0.8048 ± 0.019、independent 0.8129 ± 0.024、BLCA independent 0.7133 ± 0.046）** 是把**同一份 c-index 抄 5 次得到的，不是真 5 折均值**。这些数字**不能写进论文**。

---

## 2. 复现步骤与证据

### 2.1 一行复现

```bash
cd /data1/DCT-Reg
for arm in direct independent; do
  for ca in kirc blca; do
    for f in 0 1 2 3 4; do
      p=$(find results/v313_paper_v1/legacy_val/$arm/$ca/fold$f -name epoch_curve_fold*.csv | head -1)
      echo "$arm/$ca/fold$f: $(md5sum "$p" | cut -c1-32)"
    done
  done
done
```

### 2.2 实际输出

```
direct/kirc/fold0: d70c093c...  direct/kirc/fold4: d70c093c...
independent/kirc/fold0..fold4: 全部 d70c093c...
independent/blca/fold0..fold4: 全部 d70c093c...
direct/blca: 目录不存在
```

5 折的 `epoch_curve_fold*.csv` **和** `split_*_results_final.pkl` md5 全相同 → 数据 100% 一份。

### 2.3 由此计算出的「5 折均值」

| Arm × Cancer | c-index | IPCW C-index | IBS | iAUC | n=5, std |
|---|---|---|---|---|---|
| direct / kirc | 0.7750 | 0.8658 | 0.1226 | 0.7783 | **std = 0.0000** |
| independent / kirc | 0.8124 | 0.8539 | 0.1165 | 0.8392 | **std = 0.0000** |
| independent / blca | 0.6517 | 0.7290 | 0.1145 | 0.6826 | **std = 0.0000** |
| direct / blca | — | — | — | — | 未跑 |

注意：上面的 c-index 与原 FINAL 文档中给出的数字**不一致**（0.7750 vs 0.8048 等），原文档的统计来源无法溯源。即使数字本身能对上，也只是同一份跑的不同字符来源，结论不变。

---

## 3. 根因分析

### 3.1 配置层：5 折被合并为单任务

`configs/{exp0..exp6,direct,independent,...}*.yaml` 中：

```yaml
split:
  k_start: 0
  k_end: 5
```

→ CLI 启动时默认跑全 5 折内部循环。

### 3.2 调度器层：未对每个 fold 派发独立子目录

`survot_rank/training/scheduler.py:_results_dir_for` 生成 `results/v313_paper_v1/legacy_val/{arm}/{cancer}/fold{N}/seed{seed}/...` 目录 —— 这一步**有 fold** 区分。

但 stage A 实际不是调度器派发的（看 pkl mtime 间隔 ~3 小时，是 5 次手工/外层脚本启动）。**每次启动 k_start/k_end 默认值是 0/5，train_runner 内部循环写结果**到 `run_xxx/.../`，而这个 run 目录路径**不含 fold** —— 是基于 `config_yaml + extra_set` hash，5 次同样输入 → 5 次同样 hash → 全部写到同一目录。后写覆盖前写。

### 3.3 结果

最后一次跑（fold=4 启动时已是第 5 次）留下的快照，被前 4 次命名按 `fold{N}` 的目录"挂"在一起。看起来像 5 折，其实是**一次跑的 5 份拷贝**。

### 3.4 Stage B 为什么没事

`scripts/run_slotspe_*_paper.sh` 是显式 shell `for fold in 0 1 2 3 4`，每次只跑单折，并把 `results_dir` 设为 `results/slotspe_<cancer>_<variant>/fold{N}_paper/`（不含 fold 的额外层），所以每折独立写入，md5 各异。

---

## 4. 受影响范围

- ❌ **不能写入论文**：Stage A 的 4 行 c-index（含 ±std）
- ❌ **不能写入论文**：KIRC/BLCA direct vs independent 比较
- ✅ **可以保留**（单独看）：Stage B SlotSPE 4 个数字（matched/native × KIRC/BLCA，每行 5 折真值）
- ✅ **代码可推**：scheduler、outer_test、paper docx 的 5 个 commit

---

## 5. 修复路径（待执行）

### 5.1 短期（必须有用户授权的 GPU 跑时间）

1. **把 `k_start=0/k_end=5` 改为 `k_start=$fold/k_end=$((fold+1))`**，并通过 `--set` 注入；或
2. **强制使用 `survot_rank.cli schedule`**（已修复的调度器，5 个 commit 里那个），它会按 fold 派发独立子目录。

重跑次数：KIRC + BLCA × direct + independent × 5 折 = **20 次训练**，30 epochs / fold。

### 5.2 验收（重跑完必做）

- [ ] 每折的 `epoch_curve_fold{N}.csv` md5 **互异**
- [ ] 每折的 `split_*_results_final.pkl` md5 **互异**
- [ ] 5 折 c-index std **大于零**（典型 .015–.030）
- [ ] 5 折 c-index 数值与**前一轮 70-fold 消融**在 KIRC/BLCA Exp6 同一方向（KIRC ~0.81，BLCA ~0.72）量级一致

### 5.3 论文同步更新

- [ ] `paper/DCT_v313_初稿.md` 结果表只填 Stage B 4 个 + 重跑后的 Stage A 8 个
- [ ] 删掉 FINAL 文档"Δ vs SlotSPE"章节（重跑完再重算）

---

## 6. 已 commit、待 push 的代码（与本次数据问题无关，可直接 push）

```
a1d8706  cli+train_runner: respect scheduler-set CUDA_VISIBLE_DEVICES
8376bbb  schedule CLI: --gpu is repeatable so 2-GPU round-robin actually fans out
2c6c78b  outer_test_split: align clinical path with DCT dataset factory
a63f7ae  §3.3: refine v3.13 manuscript and convert to docx
9a3efb0  Add §3.1 #3-5 + §3.2: cross_mode, plan_mode, outer_test, scheduler
```

这些是：scheduler 修复（解决 5.1 #2）、outer_test 协议、v3.13 论文与 docx、direct/independent 开关 —— 全部是 Stage A 没能正确跑起来**之前**就需要的工作。

**Push 命令**：

```bash
cd /data1/DCT-Reg
git push origin main     # 5 个 commit
git add results_stageA_stageB_FINAL_20261005.md
git mv results_stageA_stageB_FINAL_20261005.md results_stageA_stageB_FINDINGS_20261005.md
git commit -m "Stage A/B results audit: 5-fold csv/pkl md5 collision, not a 5-fold mean"
git push origin main
```

---

## 7. 红线（与 NEXT_STEPS.md 保持一致）

不动：
- `results/dct_v313_ablation_*`（70-fold 消融，已审计 ✓）
- `configs/fixed_5fold/`（5 折 split csv）
- `third_party/SlotSPE/`

新增红线（这次新增）：
- **不得**在 Stage A 重跑前，把 `results/v313_paper_v1/legacy_val/**` 的 c-index 写入论文主表
- **不得**用原 FINAL 文档的"5 折均值 ± std"作任何对外主张
- 重跑前**不要**清空当前 `results/v313_paper_v1/legacy_val/`，保留作 md5 对照证据
