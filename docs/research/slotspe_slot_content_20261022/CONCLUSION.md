# SlotSPE 槽内容与槽间相似度实测（BLCA，30 epoch，5 折，seed 3）

**日期**：2026-10-22
**问题**：病理（WSI）槽 8 个、基因（omic）槽 8 个，槽1↔槽8 的相似度各是多少？每个槽装了多少内容？
**答案**：**必须实测**——相似度是训练出来的，不是结构常量。但结构决定了相似度的**起点**。

## 一、结论速览

| 量 | WSI（病理）槽 | Omic（基因）槽 |
|---|---|---|
| 槽数 | 8 | 8 |
| 槽间余弦（8 槽两两，off-diagonal）均值 | **0.9166** | **0.9289** |
| **槽1 ↔ 槽8** 余弦（均值） | **0.9279** | **0.9276** |
| 槽1↔槽8 中位数 / 最小 / 最大 | 0.9381 / 0.1798 / 0.9981 | 0.9776 / 0.4002 / 0.9994 |
| 槽间余弦 >0.95 的比例 | 45.0% | 73.1% |
| **有效独立槽数**（余弦矩阵特征谱参与比） | **1.17**（上限 8） | **1.14**（上限 8） |
| 每槽"有效内容"（参与比，1=单点，N=完全均匀） | 241–246 / 256 | 310–317 / 329 |
| 归一化注意力熵（1.0=均匀铺满） | **0.9930** | **0.9943** |
| 单个最大注意力权重 | 0.0055（均匀=0.0039） | 0.0040（均匀=0.0030） |

**核心事实**：8 个槽的有效独立维数只有 **~1.15**。名义 8 槽，实际约等于 **1 个槽**。

## 二、"必须运行"与"结构必然"的分工

### 2.1 结构必然（读代码即知，不需跑）

`models/slot_attention.py` 中：

```python
self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))      # ← 形状 (1,1,256)
self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
```

**8 个槽共享同一个中心向量 `mu`，形状是 `(1,1,256)`**；槽之间只靠高斯噪声 `sigma * randn` 区分。
理论初始余弦 = \(|mu|^2 / (|mu|^2 + \sum \sigma^2)\)，实测验证：

| fold | 槽组 | \|mu\|² | Σσ² | 理论 init cos | 实测 init cos |
|---|---|---|---|---|---|
| 0 | wsi | 233.79 | 259.23 | 0.4742 | 0.4858 |
| 0 | omic | 254.45 | 257.08 | 0.4974 | 0.4969 |
| 3 | wsi | 235.28 | 258.24 | 0.4767 | 0.4882 |
| 3 | omic | 254.79 | 256.97 | 0.4979 | 0.4975 |

5 折全部吻合（预测 ~0.474/0.498，实测 ~0.486/0.497）。**初始槽相似度约 0.47–0.50 是结构决定的**。

### 2.2 必须实测（相似度是训练出来的）

最终槽余弦 0.92 与初始 0.48 的差距、以及"哪个槽装什么"，只能跑模型得到。

## 三、槽间相似度在 attention 内部哪一步产生

分阶段记录（5 折均值，off-diagonal 余弦）：

| 阶段 | WSI | Omic |
|---|---|---|
| `draw`（slots = mu + σ·noise） | 0.4858 | 0.4972 |
| `updates`（einsum(v, attn)，注意力加权平均） | **0.9343** | 0.7353 |
| `gru`（GRU 更新后） | 0.6270 | 0.5301 |
| `mlp`（残差 MLP 后 = 最终槽） | 0.9162 | 0.9299 |

**因果链**：注意力接近均匀（熵 0.993）→ 每个槽都对同一个 bag 做几乎相同的加权平均 → `updates` 阶段相似度跳到 0.93 → GRU 短暂拉开 → 残差 MLP 再把相似度推回 0.92。

### 3.1 排除两个替代解释

**排除 A：不是共享 mu 造成的**。把最终槽向量按共享 mu 方向分解（30 epoch）：

| 槽组 | 沿共享 mu 的范数占比 | 平行部分余弦 | **残差部分余弦** |
|---|---|---|---|
| wsi（5 折） | 0.3% – 1.9% | 0.09 – 0.12 | **0.80 – 0.99** |
| omic（5 折） | 0.2% – 2.5% | 0.09 – 0.12 | **0.85 – 0.98** |

共享 mu 只占槽范数的约 1%；把 mu 方向完全剔除后，残差部分自身余弦仍有 0.80–0.99。**相似性来自数据侧，不来自参数化。**

**排除 B：不是"看同一批 patch"造成的**。两槽 top-5 patch 的平均重叠数：

- WSI：slot1↔slot8 = **1.00 / 5**（几乎不重叠）；全部 28 对平均 0.77–1.36
- Omic：slot1↔slot8 = **0.56 / 5**；最高 slot6↔slot8 = 0.88

**槽看的东西明显不同，槽向量却高度相似**——这正是近均匀注意力 + 共享中心参数的联合后果。

## 四、每个槽装了多少内容

### 4.1 数量

参与比 \(1/\sum w^2\)（对注意力权重）：

- WSI 槽：241–246（满值 256）→ 每个槽几乎均匀铺满全部 256 个 patch
- Omic 槽：310–317（满值 329）→ 几乎均匀铺满全部 329 条通路

### 4.2 具体内容（omic 槽，各自权重最高的通路）

| 槽 | 最常出现的通路（患者数/379） |
|---|---|
| 1 | HALLMARK_HEME_METABOLISM(29), Sulfur_amino_acid_metabolism(26), N-glycan 前体生物合成(25) |
| 2 | Methylation(24), Sulfur_amino_acid_metabolism(22), N-glycan 前体(22), HALLMARK_E2F_TARGETS(21) |
| 3 | HALLMARK_XENOBIOTIC_METABOLISM(26), Sulfur_amino_acid_metabolism(25), HALLMARK_INTERFERON_GAMMA(23) |
| 4 | HALLMARK_HEME_METABOLISM(27), Methylation(23), Sulfur_amino_acid_metabolism(23) |
| 5 | HALLMARK_INTERFERON_GAMMA(24), N-glycan 前体(24), HALLMARK_IL6_JAK_STAT3(21) |
| 6 | Sulfur_amino_acid_metabolism(27), HALLMARK_HEME_METABOLISM(23), Methylation(22), Galactose_catabolism(21) |
| 7 | HALLMARK_APOPTOSIS(24), Sulfur_amino_acid_metabolism(22), HALLMARK_ADIPOGENESIS(22) |
| 8 | Sulfur_amino_acid_metabolism(25), HALLMARK_HEME_METABOLISM(22), Galactose_catabolism(21) |

**8 个槽的"签名通路"高度重叠**（Sulfur_amino_acid_metabolism 出现在全部 8 个槽，HALLMARK_HEME_METABOLISM 出现在 5 个槽）。

### 4.3 decoder 实际用了哪几个槽

`topk_ratio=0.25` → k = 8 × 0.25 = **2**。

- **WSI：379/379 患者都恰好保留 2 个槽**；保留的索引组合有 23 种，最常见 `(3,6)` 54 人、`(1,7)` 42 人、`(3,4)` 29 人
- **Omic：379/379 患者都恰好保留 2 个槽**；保留组合 25 种，最常见 `(1,6)` 56 人、`(1,5)` 54 人、`(0,4)` 35 人
- 各槽被保留的频率：WSI `[0.18, 0.29, 0.22, 0.36, 0.14, 0.23, 0.29, 0.30]`，Omic `[0.15, 0.35, 0.22, 0.14, 0.23, 0.37, 0.35, 0.20]` —— **没有哪个槽稳定主导**，8 个槽都在竞争进入 top-2

## 五、附带发现：omic 分支对部分患者整支失效（RNG 依赖）

decoder 的 gate 是

```python
slot_gate = torch.softmax(keep_score / self.temperature, dim=-1)   # T = 0.01
slot_gate = slot_gate * hard_keep
slot_gate = slot_gate / (slot_gate.sum(dim=1, keepdim=True) + 1e-8)
```

`keep` 由 **Gumbel 加噪后的 top-2** 决定，而 `softmax(score/0.01)` 极其尖锐。被保留的两个槽**未必**是承载原始 softmax 质量的那两个槽。

实测（5 折 379 患者，omic）：

| 量 | 值 |
|---|---|
| gate 行和 ≈ 1（正常） | 327 / 379 |
| gate 行和 < 0.5（omic 分支贡献近乎归零） | **29 / 379** |
| gate 行和 < 1e-6 | 17 / 379 |
| **坏例中被保留槽携带的原始 softmax 质量** | 中位数 **9.4e-18**，最大 7.97e-09 |
| 坏例中"原始 top-2 ∩ 被保留集合"为空 | 25 / 29 |
| 正常例中被保留槽的 softmax 质量 | 中位数 0.9986 |
| 正常例中原始 top-2 ∩ 被保留集合 | 平均 0.88 |

**机制**：当被保留的两个槽只携带 ~1e-17 的质量，归一化实际是除以 `1e-8` 这个 ε 而非真实质量，gate 整体塌到 ~0，**omic 分支对该患者的预测贡献为零**。

**RNG 依赖性**：同一批 379 患者，改变 RNG 抽样（s9 中 wsi+omic 共用一个随机流 vs s14 中只跑 omic）后，塌陷患者数从 38 变为 29。**这个失效的"发生率"本身是 RNG 抽样量**，与 P3（评估 RNG 噪声 0.0035）同源。

> 边界：这是"受控条件下观察到的发生率"，与 SPE-09 的 `natural_gate_observations`（自然数据 0/3080）口径不同——那是测"训练后自然预测时是否出现近零 gate"，这里是直接检查 decoder 内部的质量分配。

## 六、3 epoch 与 30 epoch 的差异（重要）

同一套测量方法、同一脚本，3 epoch checkpoint 对照：

| 量 | 3 epoch（fold0） | 30 epoch（5 折均值） |
|---|---|---|
| WSI off-diag 余弦 | **0.9973** | 0.9166 |
| WSI >0.95 比例 | **100%** | 45.0% |
| WSI 槽1↔槽8 | **0.9987** | 0.9279 |
| Omic off-diag 余弦 | 0.9928 | 0.9289 |
| Omic >0.95 比例 | 99.1% | 73.1% |

**训练会把槽适度拉开，但拉不回独立水平**：30 epoch 后有效独立槽数仍只有 ~1.15。

按折看差异极大（WSI off-diag 均值）：

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| WSI | 0.9801 | 0.8914 | 0.8057 | 0.9924 | 0.9108 |
| Omic | 0.8711 | 0.9765 | 0.9497 | 0.9095 | 0.9392 |

fold3 的 WSI 是 100% 配对 >0.95，fold2 只有 7.7%。**"槽冗余"的程度是 fold 相关的，不是单一常数**。

## 七、测量方法与可信度

| 脚本 | 作用 |
|---|---|
| `scripts/slotspe_diagnostics/s9_slot_content_cosine.py` | 槽余弦 + 槽内容（插桩 attention，保留原版前向） |
| `scripts/slotspe_diagnostics/s10_slot_mechanism.py` | 注意力熵、bag-mean 探针、随机子集对照 |
| `scripts/slotspe_diagnostics/s11_3epoch_control.py` | 3 epoch checkpoint 同法对照 |
| `scripts/slotspe_diagnostics/s12_stagewise_cosine.py` | attention 内部逐阶段余弦 |
| `scripts/slotspe_diagnostics/s13_mu_decomposition.py` | 共享 mu 平行/残差分解 |
| `scripts/slotspe_diagnostics/s14_gate_collapse_check.py` | gate 塌陷机制核查（omic 分支失效） |

**保真性**：s9 的插桩前向与原版 `MultiHeadSlotAttention.forward` 在同一 RNG 状态下 **max abs diff = 0.00e+00**（5 折全部为 0），即插桩未改变任何数值。

**对照设计**：同 checkpoint 同患者比较；通路名与 manifest 的 `omic_sizes` 逐一对齐（329 条，sizes 完全匹配）；5 折 val 患者共 379 人，折间不重叠。

**边界**：
- 测的是 BLCA、seed 3、30 epoch 末轮 checkpoint（`last.pt`），非峰值 checkpoint
- 测的是槽向量本身的余弦，不是"槽对预测的贡献"
- 未做训练：所有数字来自现有 checkpoint 的只读前向

## 八、产物

`docs/research/slotspe_slot_content_20261022/`
- `slot_content.json` — 每患者逐槽余弦/内容/gate（4.6 MB）
- `slot_mechanism.json` — 熵/bag-mean/随机子集对照
- `slot_content_3epoch.json` — 3 epoch 对照
- `stagewise_cosine.json` — 逐阶段余弦
- `mu_decomposition.json` — 共享 mu 分解
- `gate_collapse_check.json` — 每患者 gate 质量分配与失效判定
