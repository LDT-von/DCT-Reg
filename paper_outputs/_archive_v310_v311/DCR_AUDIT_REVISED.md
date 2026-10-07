# DCR 审计修订报告

> 生成时间：2026-09-10
> 数据来源：`results/audit_best_epochs_blca/fold_*/audit_fold*.pkl`（已用新版 `direction_consistency` 重算）

---

## 一、修改说明

### 1.1 为什么修改 `direction_consistency`

原版 DCR 把所有低风险标签患者混在一起报告，但"低风险"包含了两种本质不同的患者：

| 亚组 | 定义 | 数量（fold 0）| 性质 |
|------|------|----------------|------|
| **LOW (Obs)** | 已观察到事件 + 事件时间 > 60% 分位 | 11/46 (24%) | 有真实观察证据 |
| **LOW (Cen)** | 删失 + 随访时间 ≥ 60% 分位 | 35/46 (76%) | **假设性**：假设他们如果没删失会继续生存 |

原版把这两类混在一起，导致：
- 删失患者的比例很大（BLCA 约 65% 删失率）
- 删失患者的"正确方向"标准是假设性的（不是真实观察到的）
- 删失患者的 `low_delta` 正确率单独只有 ~55%，拉低了整体 DCR

### 1.2 修改内容

**`scripts/audit_dct_reg.py`**：

```python
# 新增分组字段
low_observed_mask = observed & (event_time > upper)
low_censored_mask = ~observed & (event_time >= upper)

# 新增返回值
high_rate             = high_correct / high_total
low_rate_observed     = low_correct_observed / low_total_observed
low_rate_censored     = low_correct_censored / low_total_censored
correct_rate_no_censored  # 排除删失患者的汇总 DCR
```

**`scripts/extract_mechanism_audit.py`**：同步更新，新增分组打印输出。

---

## 二、新 DCR 报告（BLCA 5-Fold）

### Table 4a（修订版）：按患者类型分组的 DCR

| Fold | HIGH (Obs) n | LOW (Obs) n | LOW (Cen) n | DCR(混合) | DCR(去删失) | HIGH方向% | LOW_Obs方向% | LOW_Cen方向% |
|------|-------------|-------------|-------------|-----------|-------------|---------|-----------|------------|
| 0 | 11 | 11 | 35 | 0.421 | 0.545 | **72.7%** | 36.4% | 34.3% |
| 1 | 11 | 10 | 28 | 0.571 | 0.381 | **0.0%** | 80.0% | 71.4% |
| 2 | 11 | 11 | 30 | 0.365 | 0.364 | **27.3%** | 45.5% | 36.7% |
| 3 | 9 | 9 | 19 | 0.595 | 0.556 | **22.2%** | 88.9% | 63.2% |
| 4 | 10 | 10 | 30 | 0.680 | 0.550 | **30.0%** | 80.0% | 76.7% |
| **Mean±SD** | - | - | - | **0.527±0.116** | **0.479±0.088** | **30.4%±23.6%** | **66.1%±21.1%** | **56.4%±17.7%** |

> 注：DCR(混合) = 原来旧版报告的数字；DCR(去删失) = 排除 LOW(Cen) 后的新汇总

---

## 三、关键发现

### 发现 1：HIGH 方向完全反了

**这是最核心的问题。**

5-fold 合并后，HIGH 干预对 HIGH 患者的方向正确率只有 **30.4%**（16/52），远低于随机 50%。

这意味着：对于那些真正短生存（高风险）的患者，把 cost 推向 HIGH_RISK 锚点之后，模型的预测反而变得更安全（风险降低）了。

- Fold 1 最极端：**0% 正确**，HIGH 患者一个都没有出现正确的方向。
- Fold 0 是唯一超过 50% 的 fold（72.7%）。

### 发现 2：LOW Obs 方向还可以（66.1%），但 LOW Cen 方向弱（56.4%）

LOW 已观察组的方向正确率 66.1%，说明模型对"长生存已观察患者"的 low 干预响应还算合理。

但 LOW 删失组只有 56.4%（虽然高于 50%，但样本量小 + 方差大）。

### 发现 3：DCR(去删失) = 0.479，仍然低于 0.5

去掉删失患者后，DCR 从 0.527 下降到 0.479。这意味着**删失患者不是拉低 DCR 的主因**，真正的问题在高方向。

### 发现 4：HIGH 组样本太少（每个 fold 只有 9-11 个）

HIGH 组样本量太小导致波动极大（SD = 23.6%）。从 0% 到 72.7%，没有任何一个 fold 在 50% 附近稳定。

---

## 四、根本原因分析

### 4.1 Stage 3 LOW_RISK 锚点初始化失败（模型代码 bug）

通过检查 checkpoint（LUSC fold 1），发现：

```
risk_anchor_seen:
  [ True,  True]   ← stage 0
  [ True,  True]   ← stage 1
  [ True,  True]   ← stage 2
  [False,  True]]  ← stage 3: LOW_RISK = False（从未初始化！）

stage_edges: [-inf, 11.5, 20.6, 44.0, 173.8]
```

Stage 3 的 LOW_RISK 锚点永远不会被初始化（`_stage_membership_weights` 需要患者生存超过 173.8 个月才能贡献这个锚点，但 BLCA 整个数据集没有一个人做到）。

在反事实计算中：
```python
low_anchor = torch.where(low_seen, low_anchor, factual_costs)
# stage 3: low_anchor = factual_costs（锚点被替换）
```

这导致 LOW 反事实在 stage 3 上退化为 factual（完全不改变），使得 HIGH 和 LOW 干预在 stage 3 上的差异完全消失。

### 4.2 模型对反事实干预的响应极小

所有 fold 的 `|high_delta|` 和 `|low_delta|` 量级都在 0.0001~0.02，而 `factual_risk` 的取值范围约 3.9。Delta 是 factual 量级的 0.001%~0.5%。

### 4.3 OT 几何对预测的实际贡献接近 0

`scripts/test_ot_contribution.py` 的结果：No-OT 的 C-index ≥ Full（有 OT）的 C-index。OT 结构没有给预测带来正向贡献。

---

## 五、论文叙事建议

### 诚实报告（建议采用）

```markdown
### Direction Consistency Audit (Revised)

We revised our direction-consistency audit to distinguish between
observed-event patients and censored patients in the LOW-risk group.
The revised breakdown reveals:

- **HIGH group**: Only 30.4% of short-survival (observed-event)
  patients respond in the correct direction to the HIGH-risk anchor
  intervention. This is significantly below the chance level of 50%,
  indicating that the transport mechanism does not reliably shift
  risk in the expected direction for high-risk patients.

- **LOW (Obs) group**: 66.1% respond correctly, suggesting partial
  effectiveness for long-survival observed patients.

- **LOW (Cen) group**: 56.4% respond correctly; however, this is a
  hypothesis-driven subgroup and should be interpreted cautiously.

- **Aggregate DCR**: 0.527 (mixed) / 0.479 (excluding censored).

Conclusion: The transport intervention mechanism shows weak and
inconsistent direction-consistency, particularly for the HIGH group.
The OT geometry's contribution to predictive risk responses is
limited. This finding is consistent with the observation that
removing the OT module does not degrade C-index, suggesting that
the model's predictive power derives primarily from the slot-attention
hazard pathway rather than the transport geometry.
```

### 备选叙事（更强硬）

如果想更直接：

```markdown
The direction-consistency audit reveals that the OT-based counterfactual
intervention produces direction-correct responses in only 30.4% of
high-risk patients (vs. 50% chance). This is not a measurement error —
the HIGH anchor intervention systematically moves risk in the *opposite*
direction for short-survival patients, suggesting that the learned
transport cost tensor encodes information that is *inversely* related
to the desired intervention direction.
```

---

## 六、附录：与原 Table 4 对比

| 指标 | 旧版报告 | 新版报告 | 变化 |
|------|----------|----------|------|
| DCR (含删失) | 0.526 ± 0.116 | 0.527 ± 0.116 | 无变化 |
| DCR (去删失) | 未报告 | 0.479 ± 0.088 | 新增 |
| HIGH 方向正确率 | 未单独报告 | 0.304 ± 0.236 | 新增 |
| LOW(Obs) 方向正确率 | 未单独报告 | 0.661 ± 0.211 | 新增 |
| LOW(Cen) 方向正确率 | 未单独报告 | 0.564 ± 0.177 | 新增 |

**结论**：旧版 0.527 是被 LOW(Cen) 撑上去的（删失患者比例大，且删失组方向稍好 56.4%）。去掉删失后 DCR 降到 0.479。
