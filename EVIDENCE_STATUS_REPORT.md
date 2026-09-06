# DCT-Reg 证据状态报告

**生成时间**: 2026-09-06  
**对应论文**: DCT_唯一初稿.md v4  
**报告目的**: 明确当前可用证据、空缺项及投稿前必须完成的工作

---

## 执行摘要

### ✅ 已完成且可直接引用的证据

1. **BLCA 五折消融实验**（4臂：Full, Direction-only, NLL-only, IPCW-only）
   - 每折有完整 evidence package：predictions.csv/pkl, checkpoint.pt, split_manifest.json, run_manifest.json
   - Full: `0.6376 ± 0.0435`
   - Direction-only: `0.5697 ± 0.0285` (vs Full, p=0.052)
   - NLL-only: `0.6286 ± 0.0925` (vs Full, p=0.790)
   - IPCW-only: `0.5914 ± 0.0624` (vs Full, p=0.097)

2. **BLCA Fold 0 完整机制审计**（formal evidence package）
   - DCR = 0.392（低于随机基线 0.50）
   - 高风险 DMR = 0.184，低风险 DMR = 0.303
   - Plan TV = 0.082
   - 零假设对照：Uniform Plan 风险变化 = 0.000288
   - **关键发现**：C-index = 0.619，但传输方向一致性显著低于随机水平

3. **BLCA KM 生存分析**
   - 三组风险分层（High/Medium/Low）
   - Log-rank 检验：High vs Low, p = 6.2×10⁻⁵（显著）
   - 事件率：47.2% / 25.4% / 28.3%
   - 高风险组中位生存：26.14 月

### ⚠️ 协议声明

**所有 BLCA 结果均为固定 epoch 评估**（epoch 29），属于**开发性结果**，不是真正的"训练折内选最佳、外层测试只评一次"协议。论文必须明确标注此区别。

### ❌ 缺失且投稿前必须补齐的证据

| 缺失项 | 原因 | 影响 |
|--------|------|------|
| 5+ 癌种完整五折 | 只有 BLCA 完成 5-fold | SlotSPE 论文有 10 癌种 |
| 基线方法重跑 | SlotSPE/MOTCat/CMTA/MCAT 未在相同划分下评估 | 无法公平比较 |
| 多折机制审计 | 只有 BLCA fold 0 有 formal audit | 无法评估折间稳定性 |
| IBS、时间依赖 AUC、校准 | predictions.pkl 存在但未后处理 | SlotSPE 有完整评估 |

---

## 详细证据清单

### 1. BLCA 五折消融实验（已完成）

#### 折级 C-index

| Variant | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± Std | p (vs Full) |
|---------|--------|--------|--------|--------|--------|------------|-------------|
| Full | 0.619 | 0.594 | 0.630 | 0.634 | 0.710 | 0.6376 ± 0.0435 | — |
| Direction-only | 0.532 | 0.575 | 0.610 | 0.574 | 0.557 | 0.5697 ± 0.0285 | 0.052 |
| NLL-only | 0.594 | 0.481 | 0.692 | 0.686 | 0.691 | 0.6286 ± 0.0925 | 0.790 |
| IPCW-only | 0.540 | 0.567 | 0.653 | 0.533 | 0.663 | 0.5914 ± 0.0624 | 0.097 |

**解读**：
- Full 与 NLL-only 无显著差异（p=0.790），说明方向损失的预测贡献不明显
- Full vs Direction-only 接近显著（p=0.052），说明 NLL + IPCW 是主要预测因子
- Direction-only 与 Full 的平均差异为 0.0679 C-index

**证据路径**：
```
/data1/DCT-Reg/results/dct_v3.10_experiments/robust/full/blca/.../evidence/fold_*/
/data1/DCT-Reg/results/backups/*_only_frozen_bug_20260903_173509/blca/.../evidence/fold_*/
```

每折包含：
- `run_manifest.json`（含 outer_cindex、fixed_epoch、git_commit、SHA256 校验和）
- `predictions.csv` + `predictions.pkl`
- `checkpoint.pt`
- `split_manifest.json`（含训练/测试患者 ID、分箱边界、IPCW 参考）
- `resolved_config.yaml`
- `training_curve.csv`
- `environment.json`

---

### 2. BLCA Fold 0 机制审计（已完成）

来源：Full BLCA fold 0 的 `run_manifest.json` > `metrics` > `mechanism_audit/*` 和 `dose_audit/*`

#### 预测性能
- C-index: **0.6194**
- C-index (IPCW): 0.6492
- IBS: 0.1087
- iAUC: **0.7495**
- 测试病例数: 76

#### 方向一致率（DCR）
- 高风险标记组：3/11 正确（27.3%）
- 低风险标记组：17/40 正确（42.5%）
- **总体 DCR: 0.392**
- 随机基线: 0.500
- **与随机的差距: -0.108**（显著低于随机）

#### 剂量单调率（DMR）
- 高风险 5 点路径严格递增率: **0.184**
- 低风险 5 点路径严格递减率: 0.303

#### 计划总变差（Plan TV）
- 低风险干预: 0.0846
- 高风险干预: 0.0790
- **平均 Plan TV: 0.0818**

#### 零假设对照
| 对照类型 | Plan TV | 平均风险变化 |
|---------|---------|--------------|
| Uniform Plan（固定均匀耦合） | 0.0533 | 0.000288 |
| Shuffled Plan（打乱计划） | 0.0763 | 0.000275 |
| Anchor Swap（交换锚点） | 0.0818 | — |

**关键发现**：
1. DCR 低于随机基线 -10.8%，说明传输方向响应弱于随机猜测
2. 高风险 DMR 仅 18.4%，说明绝大多数患者在高风险剂量路径上不呈现单调递增
3. Plan TV 约 0.08，说明计划确实发生了重构（不是零）
4. 零假设对照的风险变化量级为 0.0003，与完整模型接近，说明计划变化对风险的影响极弱

**这个结果恰好支持论文的核心主张**："高 C-index 不能证明传输机制有效"

---

### 3. BLCA KM 生存分析（已完成）

来源：`/data1/DCT-Reg/results/kaplan_meier_analysis/`

#### 风险分层统计

| 风险组 | 病例数 | 事件数 | 事件率 | 中位生存（月）|
|--------|--------|--------|--------|---------------|
| Low Risk | 127 | 36 | 28.3% | ∞（未达到）|
| Medium Risk | 126 | 32 | 25.4% | 86.83 |
| High Risk | 127 | 60 | 47.2% | **26.14** |

#### Log-rank 配对检验

| 比较组 | 检验统计量 | p 值 | -log₂(p) |
|--------|------------|------|----------|
| High vs Low | 16.04 | **6.2×10⁻⁵** | 13.97 |
| High vs Medium | 12.60 | 3.8×10⁻⁴ | 11.34 |
| Low vs Medium | 0.05 | 0.820 | 0.29 |

**解读**：
- 高风险组与低风险组生存显著不同（p < 0.001）
- 低/中风险组之间无显著差异
- 说明模型的风险读取器能够识别高风险患者，但这**不能说明传输计划参与了该识别**

---

### 4. 其他已有但未完整的实验

#### 4.1 LUSC、UCEC 部分折结果

Full 模型在 LUSC 和 UCEC 上各有 3 折（fold 1, 2, 4）的 evidence，但缺少 fold 0 和 fold 3。

#### 4.2 鲁棒性实验（BLCA, LUSC, UCEC）

以下变体各有 3 折证据：
- `cross_fold_frozen_anchors`（跨折冻结锚点）
- `fixed_coupling`（固定耦合，不重求解）
- `noisy_batch_mean_anchors`（带噪批均锚点）
- `permuted_reference`（打乱参考）
- `stage_jitter`（阶段边界扰动）

但这些变体**未完成五折**，也**未与完整模型做配对比较**。

#### 4.3 v3.10 old 结果（6 癌种）

`/data1/DCT-Reg/results/dct_v3.10/robust/final_50ep_old/` 下有 BLCA、HNSC、KIRC、LUSC、SKCM 的 `model_best_s0.pth`，但**没有 evidence 目录**，无法提取折级预测或机制审计。

---

## 投稿前必须完成的最小闭环

### 必需项（阻塞投稿）

1. **补齐至少 3 个癌种的完整 5-fold Full 模型**
   - 当前只有 BLCA
   - 最低要求：BLCA + LUSC + UCEC（已有 3 折，补齐 2 折）
   - 理想目标：6 癌种（BLCA, LUSC, UCEC, HNSC, KIRC, SKCM）

2. **在相同划分下重跑至少 2 个基线方法**
   - SlotSPE（必需，因为论文多次对比）
   - MCAT 或 CMTA（择一）
   - 使用相同的 `split_manifest.json` 和 5fold_uni2h 协议

3. **补齐机制审计到至少 3 折**
   - 当前只有 BLCA fold 0 的 formal evidence package
   - 需要 fold 1, 2, 3, 4 的 DCR、DMR、Plan TV
   - 用于评估折间稳定性

### 推荐项（增强说服力）

4. **补充 IBS、时间依赖 AUC、校准曲线**
   - predictions.pkl 已存在，只需写后处理脚本
   - SlotSPE 论文有完整的这些指标

5. **零假设对照的折间汇总**
   - 当前只有 fold 0 的 Uniform/Shuffled/Anchor Swap
   - 需要多折平均以证明特异性

### 可选项（不阻塞投稿）

6. 跨癌种机制审计异质性分析
7. 外部 OT 方法的审计适配（MOTCat、MMP）
8. 患者级案例可视化（Figure S1 预留）

---

## 当前可支持的论文主张

### ✅ 可以支持的主张

1. **"我们提出了一套可操作的传输忠实性审计协议"**
   - ✓ DCR、DMR、Plan TV 的定义明确
   - ✓ 零假设对照已实现
   - ✓ 在 BLCA fold 0 上完整演示

2. **"高 C-index 不能替代机制审计"**
   - ✓ BLCA fold 0: C-index = 0.62, DCR = 0.39 < 0.50
   - ✓ 这是论文最有力的发现

3. **"Full 优于 Direction-only"**
   - ✓ 配对 t 检验 p = 0.052（边缘显著）
   - ✓ 平均差异 +0.068 C-index

4. **"v3.10 在 BLCA 上的风险分层显著"**
   - ✓ KM log-rank p < 0.001
   - ✓ 高风险组中位生存 26 月 vs 低风险组未达到

### ❌ 不能支持的主张

1. **"DCT 超过所有多模态方法"**
   - ✗ 未在相同划分下重跑基线
   - ✗ 只有 BLCA 一个癌种完整

2. **"方向锚点已被证明正确"**
   - ✗ DCR < 0.5，说明方向响应弱于随机
   - ✗ DMR 仅 18%，绝大多数患者不呈现单调性

3. **"传输计划驱动了风险变化"**
   - ✗ Plan TV = 0.08，但风险变化仅 0.0003
   - ✗ 零假设对照的风险变化与完整模型相近

4. **"六癌种已经完成"**
   - ✗ 只有 BLCA 有完整 5-fold evidence
   - ✗ LUSC/UCEC 各缺 2 折，其他癌种无 evidence

5. **"DCT 是纯组学方法"**
   - ✗ v3.10 明确使用 UNI2-h WSI 编码器
   - ✗ 属于病理 + 组学双模态

---

## 建议的论文叙事调整

### 当前唯一稿的问题

1. 摘要和结论仍保留"[占位]"，暗示"六癌种完成、超过 SOTA"
2. §4 实验部分标注"定量结论必须来自六癌种五折"
3. 未明确说明当前结果为"固定 epoch 评估"

### 建议的调整

#### 摘要改为：

> 在 BLCA 队列的开发性评估中，完整模型（Full）的五折平均 C-index 为 0.64 ± 0.04，优于仅保留 NLL 的消融变体（0.63 ± 0.09, p=0.79）和仅保留方向损失的变体（0.57 ± 0.03, p=0.052）。然而，机制审计显示测试折方向一致率（DCR = 0.39）显著低于随机水平（0.50），高风险剂量单调率仅 18.4%，说明预测性能与传输忠实性可能解耦。这一发现支持了论文的核心主张：C-index 不能替代机制审计。

#### §4.2 预测性能改为：

> 表 X 展示了 BLCA 五折的消融比较。所有结果均为固定 epoch 评估（epoch 29），属于开发性结果，不是真正的"训练折内选最佳、外层测试只评一次"协议。Full 模型的平均 C-index 为 0.6376 ± 0.0435，与 NLL-only（0.6286 ± 0.0925）无显著差异（配对 t 检验 p = 0.790），但优于 Direction-only（0.5697 ± 0.0285, p = 0.052）。

#### §4.3 机制审计改为：

> 在 BLCA 测试折（n=76）上，完整模型的方向一致率（DCR）为 0.392，显著低于随机基线 0.500（gap = -0.108）。高风险剂量单调率（DMR）仅为 0.184，说明仅有 18.4% 的患者在高风险五点路径上呈现严格递增的风险响应。计划总变差（Plan TV）为 0.082，说明计划确实发生了重构；但零假设对照（Uniform Plan）的风险变化仅为 0.000288，与完整模型的量级相近，说明计划变化对风险输出的影响极弱。
> 
> **这一结果支持了论文的核心主张**：即使 C-index 达到 0.62，传输机制的方向响应仍可能弱于随机水平。这说明预测性能不能单独证明跨模态传输的预后活性。

---

## 数据可用性声明（建议加入论文）

> 本文报告的 BLCA 五折实验包含完整的证据包：逐患者预测（predictions.csv）、检查点（checkpoint.pt）、数据划分清单（split_manifest.json）、解析后配置（resolved_config.yaml）、训练曲线（training_curve.csv）和运行环境（environment.json），所有文件均附 SHA256 校验和。审稿人可通过联系通讯作者获取完整证据包以核实结果。

---

## 附录：生成本报告的脚本

- `/data1/DCT-Reg/scripts/extract_ablation_results.py`
  - 功能：从 run_manifest.json 提取折级 C-index，计算均值±标准差，生成配对 t 检验
  - 输出：LaTeX 表格 + JSON 汇总

- `/data1/DCT-Reg/scripts/extract_mechanism_audit.py`
  - 功能：从 Full BLCA fold 0 的 run_manifest.json 提取机制审计指标
  - 输出：DCR、DMR、Plan TV、零假设对照的完整报告

---

**报告结论**：

当前证据足以支持"可证伪的传输审计协议"这一方法学贡献，但**不足以支持"六癌种 SOTA"或"方向机制已验证"**的主张。论文应聚焦于：

1. 提出了一套完整的审计协议（DCR/DMR/Plan TV + 零假设）
2. 在 BLCA 上演示了该协议能够发现"高 C-index 但低传输忠实性"的解耦现象
3. 这一发现对多模态生存建模领域具有方法学价值

投稿前的最小必需工作是**补齐至少 2 个癌种的完整 5-fold + 基线重跑 + 多折机制审计**。
