# DCT v3.13 Transport-Aware Omics Reconstruction

**模型代号**：`dct_v313_transport_reconstruction`（alias: `dct_v313`）
**状态**：candidate / experimental
**目录**：`methods/legacy/experimental/dct_v313_transport_reconstruction.py`
**类**：`DCTV313TransportReconstruction`
**配置**：`configs/dct_v313_blca_uni.yaml`
**实验任务**：BLCA UNI 5-fold，patch=4096，max_epochs=30

---

## 1. 一句话定位

在 v3.11 fix-v2（per-slot NLL + 多样性）的基础上，新增**运输感知的 omics 重建正则**：用 WSI→Omics 的多几何 Sinkhorn 计划，把 WSI 表征运输回 omics 坐标，监督 omics 语义槽；并让 omics 槽自重建 omics 自身作为锚点。

---

## 2. 损失函数（完整配方）

v3.13 总损失 = **5 项之和**，其中第 5 项是 v3.13 新增的消融对象：

```
L_total
 = 1.00 * L_surv                                       ← 主任务
 + 0.10 * L_rank_ipcw                                  ← 排序一致
 + 0.05 * L_per_slot_nll                               ← v3.11 冻结
 + 0.10 * L_per_modality_diversity                     ← v3.11 冻结
 + ramp(t) * λ * ( 0.5 * L_self_recon                  ← 消融
                  + 0.5 * L_cross_recon )              ← 消融
```

| 损失 | λ | 含义 | 来源 |
|------|---|------|------|
| **L_surv** | 1.00 | 离散时间生存 NLL | `bag_loss=nll_surv`（YAML） |
| **L_rank_ipcw** | 0.10 | Uno 风格 IPCW 成对排序 | `dct_lambda_ipcw_rank=0.10`（YAML） |
| **L_per_slot_nll** | 0.05 | 每个槽的生存 NLL（v3.11 fix-v2） | `FROZEN_ARGUMENTS` |
| **L_per_modality_diversity** | 0.10 | 槽跨模态不坍缩（v3.11 fix-v2） | `FROZEN_ARGUMENTS` |
| **L_self_recon** | λ × 0.5 × ramp | omics 槽自重建 omics 自身 | v3.13 新增 |
| **L_cross_recon** | λ × 0.5 × ramp | 跨模态 Sinkhorn 路径重建 | v3.13 新增 |

**ramp(epoch)**：第 2 epoch 开始线性爬升，5 epoch 拉到满 → 早期训练不干扰主任务。
**λ_recon 默认 0.10**：其中 self=0.05、cross=0.05。

### 历史残留但已关闭（=0.0）

下面 5 个 v3.8 时代的辅助损失，**不会进入总损失**，仅作为参数残留在 YAML 里：

- `dct_v38_lambda_direction`、`dct_v38_lambda_dose`、`dct_v38_lambda_reconfiguration`：方向族（DCR≈0.526≈随机，v3.11/13 关掉）
- `dct_lambda_listwise`、`dct_lambda_etar`：列表排序族（被 IPCW-rank 替代）

---

## 3. 消融实验（BLCA UNI 5-fold）

### 3.1 设计

| 标签 | 改动 | 含义 |
|------|------|------|
| baseline | 全开, λ_recon=0.10 | v3.13 完整配方 |
| **A no_cross** | 关 cross 重建 | 仅 self 重建生效 |
| **B no_self** | 关 self 重建 | 仅 cross 重建生效 |
| **C double_w** | λ_recon=0.20 | self/cross 权重 ×2 |

调度：`scripts/run_dct_v313_ablations_5fold.sh`
日志：`logs/v313_abl_<tag>/fold<k>/fold<k>.log`
结果：`results/dct_v313_ablation_<tag>/blca/`

### 3.2 结果（2026-09-18 更新）

| 实验 | fold0 | fold1 | fold2 | fold3 | fold4 | **mean ± std** |
|------|-------|-------|-------|-------|-------|---------------|
| baseline | 0.7266 | 0.7259 | 0.6573 | 0.6623 | 0.7304 | **0.7005 ± 0.033** |
| **A no_cross** | 0.7266 | 0.7250 | 0.6733 | 0.7060 | 0.6806 | **0.7023 ± 0.022** |
| **B no_self** | 0.7227 | 0.7267 | 0.6789 | 0.6787 | 0.7171 | **0.7048 ± 0.022** |
| **C double_w** | 0.7274 | 0.7250 | 跑中 | 跑中 | 跑中 | (2/5) |

C 的 fold2/3/4 由 rerun 脚本顺序跑完（fold 偶数→GPU0，奇数→GPU1），预计 ~1.5 h 完成。

### 3.3 解读

| 消融 | 与 baseline 相比 | 解读 |
|------|-----------------|------|
| **A 关 cross** | +0.0018 均值, std↓0.011 | 跨模态 Sinkhorn 重建几乎**无贡献**，且关掉后训练更稳 |
| **B 关 self** | +0.0043 均值, std↓0.011 | omics 自重建**也无正向贡献**，甚至轻微拖累 |
| **C λ×2** | fold0/1 与 baseline 持平 | 重建项放大无帮助 |

**结论雏形**：v3.13 新增的重建损失（self + cross）对 BLCA UNI 5-fold **贡献有限甚至为负**。从可解释性角度仍可保留，但作为预测性能正则可考虑降权或移除。

---

## 4. 运行方式

### 4.1 完整 5-fold

```bash
PYTHONPATH=/data1/DCT-Reg CUDA_VISIBLE_DEVICES=0 \
  /home/ubuntu/.conda/envs/trisurv/bin/python -m survot_rank.cli train \
  --config configs/dct_v313_blca_uni.yaml
```

### 4.2 消融矩阵

```bash
bash scripts/run_dct_v313_ablations_5fold.sh
```

3 个变体 × 5 fold，顺序启动，奇偶 fold 分流到两张 GPU（fold 0/2/4 → GPU 0，fold 1/3 → GPU 1）。

### 4.3 补跑崩掉的 fold

```bash
bash scripts/rerun_dct_v313_double_w_failed.sh
```

只补跑 `double_w` 的 fold2/3/4，跳过 fold0/1（已成功的日志备份到 `.bak`）。

---

## 5. 监控项

每个 epoch 输出以下关键量，便于诊断重建项是否在退化：

```
v313_reconstruction_self           # self 重建当前损失
v313_reconstruction_cross          # cross 重建当前损失
v313_reconstruction_total          # 加权和
v313_reconstruction_ramp           # ramp 系数（0 → 1）
v313_reconstruction_weight         # 当前 λ_recon
v313_omics_available_fraction      # omics 可用样本比例（应=1.0）
v313_transported_slot_norm         # 运输槽 L2 范数（爆炸 → 异常）
active_stage_fraction              # 激活的 stage 比例
```

**异常信号**：`transported_slot_norm > 50` 或 `omics_available_fraction < 1.0` 时通常说明运输计划或数据缺失有问题。
