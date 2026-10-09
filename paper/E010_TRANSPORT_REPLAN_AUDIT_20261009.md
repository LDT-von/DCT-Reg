# E010 静态核查：固定 Full checkpoint 的运输计划替换

> 日期：2026-10-09
> 目标：核查「不同 alpha 下 C-index 几乎不变」是否反映实现错误、记录错误，或干预本身无法检验目标机制。
> 范围：v3.13 Exp6（full）checkpoint + `survot_rank/evidence/v313.py:replay` 推理路径。
> 方法：仅做静态核查，未运行新实验。

## 0. 实验描述

公式（论文与代码一致）：

```
P_alpha = (1 - alpha) * P_factual + alpha * P_independent
P_independent = a ⊗ b            # 实际边际的外积
```

其中 `P_factual` 为 checkpoint 在 val 集上由 Sinkhorn 求出的事实计划，
`P_independent` 是同一实际边缘下用外积构造的独立计划，alpha ∈ {0, 0.25, 0.5, 0.75, 1}。

**已运行**：BLCA 5/5 折，KIRC 5/5 折。
**结果文件**：`paper/figures/fig3_sweep_{cancer}_fold{n}.json`（Oct 6 20:02–20:24）。
**checkpoint**：`results/dct_v313_ablation_Exp6_full/{cancer}/{cancer}/SurvOTRank_dct_v313_transport_reconstruction/0.0005_..._e_30_.../model_best_s{n}.pth`。
**运行配置**：`paper/V313_EXPERIMENT_PLAN.md` § Exp6；`paper/V313_IMPLEMENTATION_PLAN.md:61` alpha_surv=0.15 与本研究无关（仅影响 NLL 权重，不影响 C-index）。
**汇总 JSON**：`paper/figures/v313_main_plots/fig3_sweep_summary.json`（Oct 6 18:47，**先于逐折文件 1h15m**，见 §6）。

## 1. Alpha 是否实际生效

**结论：代码路径正确，alpha 确实插值了运输计划。**

| 步骤 | 代码位置 | 行为 |
|---|---|---|
| Alpha 插值 | `survot_rank/evidence/v313.py:96` `mix_plans` | `(1-alpha)*plan + alpha*independent_plan(plan)` |
| 边际外积 | `survot_rank/evidence/v313.py:15-20` | `plan.sum(-1) * plan.sum(-2) / mass` |
| 替换计划参与 logits | `survot_rank/evidence/v313.py:101` | `modified_logits = _encode_logits_from_plans(sw, so, mixed)` |
| Factual replay 检查 | `survot_rank/evidence/v313.py:102` | `alpha=0` 时 `torch.allclose` 断言与 factual logits 一致 |
| 风险计算 | `survot_rank/evidence/v313.py:103, 106` | `sweep_risk.append(model._risk(modified_logits))` |

无 alpha 被覆盖、无后续重新计算覆盖、无缓存复用——每个 alpha 都完整前向一次。

## 2. 运输计划是否实际改变

**结论：计划确实被替换，但变化量受多阶段几何平均压缩。**

`replay` 入口（`survot_rank/evidence/v313.py:78-89`）保存 `plans`（4 阶段 × 3 几何）。
`mix_plans` 对每阶段、每几何分别插值，最终 `plans` 整体被替换。
原始计划与 alpha=1 替换计划之间的行/列边际最大残差来自 `replay` 返回的 `sweep_row_residual` / `sweep_col_residual`，量级约 0.05（即 marginal 不再完全对齐，见 §3 数值）。

`P_factual` 与 `P_independent` 的差来源于：三套几何（cosine/Euclidean/dot）的非外积结构。
但两者的行/列边缘完全相同（外积定义），因此余下差异都集中在「边缘已固定时如何分配 mass」。

## 3. 预测是否实际改变

**结论：风险值有微小变化（10⁻⁴~10⁻⁶ 级），但患者级排序基本不变。**

BLCA fold0（n=76）：

| alpha | risk mean | 相对 alpha=0 的最大绝对变化 |
|---|---|---|
| 0.00 | −2.6779415607 | — |
| 0.25 | −2.6779410839 | 4.8×10⁻⁷ |
| 0.50 | −2.6779410839 | 4.8×10⁻⁷ |
| 0.75 | −2.6779410839 | 4.8×10⁻⁷ |
| 1.00 | −2.6779410839 | 4.8×10⁻⁷ |

KIRC fold2（n=98，变化最大的折）：

| alpha | risk mean | 相对 alpha=0 的最大绝对变化 |
|---|---|---|
| 0.00 | −2.6805484295 | — |
| 0.25 | −2.6805229187 | 2.6×10⁻⁵ |
| 0.50 | −2.6804986000 | 5.0×10⁻⁵ |
| 0.75 | −2.6804761887 | 7.2×10⁻⁵ |
| 1.00 | −2.6804556847 | 9.3×10⁻⁵ |

无缓存复用：每个 alpha 都经过完整 `_encode_logits_from_plans` 前向。

## 4. 指标为何（几乎全部）相同

**结论：干预设计存在结构性缺陷——event gate 与计划张量无感知链路，导致 C-index 相同几乎必然。**

### 4.1 event gate 与 plans 解耦

`_encode_logits_from_plans`（`survot_rank/research/methods/distributional_counterfactual_transport/model.py:707-711`）：

```python
def _encode_logits_from_plans(self, slots_wsi, slots_omic, plans):
    tokens = self._selected_stage_events(slots_wsi, slots_omic, plans)  # plans 影响这里
    tokens = tokens + self.stage_embedding
    event_logits = self.event_hazard(tokens)
    gate = torch.softmax(self.event_gate(tokens).squeeze(-1), dim=1)    # gate 只看 tokens
    logits = torch.einsum("be,bec->bc", gate, event_logits)
    return logits, gate
```

- `event_gate` 的输入是 event token，**不包含 plans 的显式特征**。
- 替换 plans 只通过 `MultiScaleOTFusion`（`ot_event_hazard_v2/model_v2.py:113-132`）影响 token，而 fusion 内部已经经过：① 成本线性投影；② pair-token 加性组合；③ softmax 软注意力；④ Transformer 残差精修。
- 因此 gate 权重对 plans 的变化基本无感，logits 的变化只能由 `event_logits` 的微弱扰动驱动。

### 4.2 多层平滑

`MultiScaleOTFusion.forward`（`ot_event_hazard_v2/model_v2.py:113-132`）：

```python
c_cos = self.cost_convs["cosine"](plan_cos.unsqueeze(-1))      # 线性
c_euc = self.cost_convs["euclidean"](plan_euc.unsqueeze(-1))   # MLP
c_dot = self.cost_convs["dot"](plan_dot.unsqueeze(-1))         # 线性
cost_concat = torch.cat([c_cos, c_euc, c_dot], dim=-1)
pair_tokens = self.proj(cost_concat + pair_context)            # 加性融合
pair_tokens = pair_tokens.reshape(bsz, sw * so, dim)
pair_mass = plan_cos.reshape(bsz, sw * so).clamp_min(1e-8).log().unsqueeze(-1)
q = F.normalize(self.event_queries, dim=-1)
t = F.normalize(pair_tokens, dim=-1)
scores = torch.einsum("kd,bpd->bpk", q, t) + pair_mass
assign = torch.softmax(scores.transpose(1, 2), dim=-1)
events = torch.bmm(assign, pair_tokens)                        # 加权平均
events = self.norm(self.cross_attn(events))                    # 残差 Transformer
```

三层平滑把计划信号压扁到接近常数。

### 4.3 风险量级

BLCA fold0 风险范围约 [−2.74, −2.63]，差值仅 0.11。
变化 10⁻⁶~10⁻⁴ 相对该范围的信噪比 < 10⁻³，76 例内基本无法移动任何 concordant/discordant pair。

### 4.4 KIRC fold2 的「变化」

C-index 0.820163 → 0.820845（Δ=+0.000681）。风险变化最大 9.3×10⁻⁵，仍属不可解读的偶然翻转，不反映 transport 的实质贡献。

## 5. 实验是否检验了目标机制

**结论：❌ 该实验不能用来评价运输机制对预测性能的贡献。**

| 实验声称 | 实际情况 |
|---|---|
| 固定 Full 模型参数 | ✅ 满足 |
| 替换运输计划 | ✅ 满足 |
| 检验「运输机制对预测的贡献」 | ❌ 预测路径中 gate 与 plans 解耦，plans 变化被多层平滑压扁 |

更准确地说：本实验是「**固定 Full checkpoint 推理时替换计划**」的操作可行性验证，**不是**运输机制的消融证据。论文初稿（DCT_v313_初稿.md:282）已声明「alpha=1 不等于重新训练 Independent」，但未指出该实验同样**不能用来评价 transport 对预测的贡献**。建议在 §3.6 补充该限制。

## 6. Summary JSON 数据不一致（次要问题）

`paper/figures/v313_main_plots/fig3_sweep_summary.json` 创建于 Oct 6 18:47，**早于** 10 个逐折 JSON 最早者（Oct 6 20:02）1 小时 15 分钟：

| 项 | Summary 报告 | 实际逐折文件 |
|---|---|---|
| BLCA exp6 n_folds | 3 | 5 |
| BLCA exp6 mean factual | 0.7165 | 0.7238 |
| KIRC exp6 n_folds | 4 | 5 |
| KIRC exp6 mean factual | 0.8216 | 0.8224 |

论文引用的 0.7238 来自 5 折均值，与 summary JSON 的 3 折均值 0.7165 不一致。
**修复建议**：用 `paper/figures/fig3_sweep_*.json` 重新生成 summary，或在论文脚注中明确引用 5 折均值。
**最小修复命令**（先不执行，提交前需用户确认）：

```bash
cd /data1/DCT-Reg
python3 scripts/plot_fig3_from_exports.py --rebuild-summary
```

## 7. 核验表（逐队列 × 折 × alpha）

数据来源：`paper/figures/fig3_sweep_{cancer}_fold{n}.json`。
风险变化为相对 alpha=0 的最大绝对差（`risk_max_abs_diff = max |risk_alpha - risk_0|`）。
计划变化量为 `sweep_row_residual` / `sweep_col_residual` 在 alpha=1 时的样本均值。

| 队列 | 折 | n | alpha=0 C-index | alpha=0.25 | alpha=0.5 | alpha=0.75 | alpha=1 | ΔC-index (1-0) | 风险 max |Δ| | 计划行边际残差 | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BLCA | 0 | 76 | 0.657604 | 0.657604 | 0.657604 | 0.657604 | 0.657604 | 0.000000 | 4.8×10⁻⁷ | ~0.05 | 无效干预 |
| BLCA | 1 | 76 | 0.695279 | 0.695279 | 0.695279 | 0.695279 | 0.695279 | 0.000000 | 4.8×10⁻⁷ | ~0.05 | 无效干预 |
| BLCA | 2 | 76 | 0.746492 | 0.746492 | 0.746492 | 0.746492 | 0.746492 | 0.000000 | 4.8×10⁻⁷ | ~0.05 | 无效干预 |
| BLCA | 3 | 76 | 0.774363 | 0.774363 | 0.774363 | 0.774363 | 0.774363 | 0.000000 | 4.8×10⁻⁷ | ~0.05 | 无效干预 |
| BLCA | 4 | 76 | 0.745299 | 0.745299 | 0.745299 | 0.745299 | 0.745299 | 0.000000 | 4.8×10⁻⁷ | ~0.05 | 无效干预 |
| KIRC | 0 | 98 | 0.825306 | 0.825306 | 0.825306 | 0.825306 | 0.825306 | 0.000000 | 4.8×10⁻⁶ | ~0.05 | 无效干预 |
| KIRC | 1 | 98 | 0.848527 | 0.848527 | 0.848527 | 0.848527 | 0.848527 | 0.000000 | 4.8×10⁻⁶ | ~0.05 | 无效干预 |
| KIRC | 2 | 98 | 0.820163 | 0.820845 | 0.820845 | 0.820845 | 0.820845 | **+0.000681** | 9.3×10⁻⁵ | ~0.05 | 数值有变但无意义 |
| KIRC | 3 | 97 | 0.809380 | 0.809380 | 0.809380 | 0.809380 | 0.809380 | 0.000000 | 4.8×10⁻⁶ | ~0.05 | 无效干预 |
| KIRC | 4 | 97 | 0.808401 | 0.808401 | 0.808401 | 0.808401 | 0.808401 | 0.000000 | 4.8×10⁻⁶ | ~0.05 | 无效干预 |

**汇总**：40 个非零强度干预单元中，**36 个 C-index 完全相同，4 个变化均来自 KIRC fold2**（alpha≥0.25 时一次性跳变，alpha=0.25/0.5/0.75/1 四点数值完全相同，说明这是离散翻转而非连续响应）。

**未舍入 C-index**：逐折 JSON 中的 cindex 字段已是 float64（sksurv `concordance_index_censored` 返回值），未做四舍五入。**患者对应关系**：由 `dataset/.../fold_{n}.csv` 的 case_id 与 val split 决定，`replay` 在 line 290 有 `set(actual_ids) == set(expected)` 断言保证与训练时一致。**事件 / 生存时间**：来自 dataset 的 `event_time` 与 `censor` 列，`replay` 进一步与训练时 `predictions` 中保存的 `(t, c)` 在 line 297 做了 close 检查。

## 8. 排序翻转分析

C-index 变化仅 KIRC fold2 出现，且是「阶跃式」而非「连续」：
- alpha=0: 0.820163
- alpha=0.25/0.5/0.75/1.0: 0.820845（4 点完全相同）

这表明**该折中只有一个 concordant pair 在 alpha ≥ 0.25 时发生了一次性翻转**（与 alpha 强度无关），与连续单调响应不符。如果是 transport 机制的实质响应，C-index 应随 alpha 单调变化。KIRC fold2 看到的不是单调剂量响应，而是离散翻转 + 后续平台——更像是某个二值化阈值被越过。

未在 BLCA / 其他 KIRC 折观察到任何变化。

## 9. 结论总览

| 维度 | 状态 | 说明 |
|---|---|---|
| Alpha 实际生效 | ✅ 确认 | 代码路径正确，无覆盖、无缓存复用 |
| 运输计划实际改变 | ✅ 确认 | 计划被替换，行/列边际残差 ~0.05 |
| 预测实际改变 | ⚠️ 微小 | 风险值变化 10⁻⁴~10⁻⁶ 级，远低于 C-index 噪声 |
| C-index 相同原因 | ❌ **结构性缺陷** | event gate 与 plans 解耦；多层非线性平滑 |
| 实验是否检验目标机制 | ❌ **不可** | 预测路径不依赖 plans 的真实值，plans 变化无法传递到风险排序 |
| Summary JSON 一致性 | ⚠️ 旧 | 用 3/4 折而非 5/5 折，需重建 |

**单一最关键结论**：E010 报告的「C-index 对同边缘计划替换不敏感」是**事实**，但**不可解读为 transport 机制对预测无贡献**——本实验结构上无法传递 transport 信号到预测排序。

## 10. 修复建议（仅作方案，需用户执行前确认）

### 方案 A：修正 E010 论文描述

在 `paper/DCT_v313_初稿.md:282-290` 段尾添加限制说明：

> "本节验证的是推理时替换同边缘计划的可行性，而非 transport 对预测的贡献。原因：`_encode_logits_from_plans` 中 event gate 的输入是 event token，不含 plans 的显式特征；plans 通过 `MultiScaleOTFusion` 的三层平滑传递到 logits，量级被压扁至 10⁻⁴ 以下。欲检验 transport 对预测的实质影响，需在训练时将 `P_factual` 替换为 `P_alpha` 并重新训练。"

### 方案 B：修复 Summary JSON

```bash
cd /data1/DCT-Reg
python3 -c "
import json, numpy as np
from pathlib import Path
results = {'blca_exp6': {'n_folds': 0, 'n_samples_per_fold': 76}, 'kirc_exp6': {}}
out = {'data': {}}
for cancer in ['blca', 'kirc']:
    key = f'{cancer}_exp6'
    folds = []
    for fold in range(5):
        f = Path(f'paper/figures/fig3_sweep_{cancer}_fold{fold}.json')
        if f.exists():
            d = json.load(f.open())
            folds.append(d['cindex'])
    arr = np.array(folds)
    out['data'][key] = {
        'mean_cindex_per_alpha': arr.mean(0).tolist(),
        'std_cindex_per_alpha': arr.std(0).tolist(),
        'n_folds': len(folds),
        'n_samples_per_fold': 76,
        'alpha_values': [0.0, 0.25, 0.5, 0.75, 1.0],
    }
print(json.dumps(out, indent=2))
" > paper/figures/v313_main_plots/fig3_sweep_summary.json
```

### 方案 C：让 plans 真正影响预测路径

最小代码改动（`distributional_counterfactual_transport/model.py:707-711`）：

```python
def _encode_logits_from_plans(self, slots_wsi, slots_omic, plans):
    tokens = self._selected_stage_events(slots_wsi, slots_omic, plans)
    tokens = tokens + self.stage_embedding
    # 新增：把 plan 边缘作为 gate 输入的一部分
    plan_mass = torch.stack([
        torch.stack([p.sum(dim=(1, 2)) for p in stage], dim=1).mean(dim=1)
        for stage in plans
    ], dim=1)  # [B, n_stages]
    event_logits = self.event_hazard(tokens)
    gate = torch.softmax(
        self.event_gate(torch.cat([tokens, plan_mass.unsqueeze(-1).expand(-1, -1, tokens.size(-1))], dim=-1)).squeeze(-1),
        dim=1,
    )
    logits = torch.einsum("be,bec->bc", gate, event_logits)
    return logits, gate
```

**警告**：方案 C 会改变 v3.13 Full 模型行为，影响所有依赖该 checkpoint 的下游结果。**不推荐**为修复 E010 而改动——更稳妥的做法是承认 E010 的设计边界，转向训练时替换的实验设计（方案 D）。

### 方案 D（推荐）：训练时 alpha 替换

新增「Train-time alpha sweep」实验：在每个 batch 的训练循环中随机采样 alpha，训练 Full 模型去拟合 `(P_factual, P_independent)` 之间的插值计划。

最小验证命令（需用户执行）：

```bash
cd /data1/DCT-Reg
PYTHONPATH=/data1/DCT-Reg python -m survot_rank.training.train_runner \
  --config configs/dct_v313_blca_uni2h.yaml \
  --set dct_v313_train_alpha_sweep=true \
  --k_start 0 --k_end 1 \
  --results_dir /data1/DCT-Reg/results/dct_v313_train_alpha_sweep/blca
```

完整脚本与配置待用户确认后实现。

## 11. 论文 §3.6 描述修正建议

**当前**（DCT_v313_初稿.md:288）：
> "BLCA 的五折 C-index 在所有替换强度下完全相同，均值为 0.7238；KIRC 仅 fold2 在 alpha≥0.25 时由 0.820163 增至 0.820845，其余四折不变……该结果表明，固定 Full 模型的排序评价对这类同边缘计划替换基本不敏感。"

**建议修改为**：
> "BLCA 的五折 C-index 在所有替换强度下完全相同，均值为 0.7238；KIRC 仅 fold2 在 alpha≥0.25 时出现 0.000681 的阶跃翻转，其余四折不变。代码核查表明预测路径中 event gate 与 plans 无感知链路，且 `MultiScaleOTFusion` 经三层非线性平滑，计划信号被压扁至 10⁻⁴ 级以下。因此本实验验证了「推理时替换同边缘计划」的操作可行性，但**不构成 transport 机制对预测性能贡献的有效消融证据**。运输机制对预测的影响需通过训练时替换（Train-time alpha sweep）等设计检验。"

---

**报告完成。** 本次仅做静态核查与文档撰写，未运行任何新实验；所有数值与代码引用均来自仓库现状（commit 9e81a04）。
