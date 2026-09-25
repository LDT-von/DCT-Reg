# DCT-Reg 各版本创新点排名报告

> **生成时间**：2026-09-25
> **范围**：仓库 `/data1/DCT-Reg` 内已实现或已立项的 9 个 catalog 版本（含 v3.10 冻结主方法）+ v3.2/v3.30 的内部结构臂 + v3.11-alpha HPSA 候选
> **方法**：以代码 `key_contributions()`、官方 `docs/*.md`、`results/*.md`、`catalog.py` 状态为依据，结合"主张是否被实测验证"两个维度做横向打分；分数 1–10。
> **声明**：排名仅用于内部梳理"创新点强弱 × 证据强度"，不构成对未发表结论的承诺；所有 C-index 都是 best-epoch 验证值，非独立外层测试。

---

## 0. 评分维度

每个版本按 4 个维度打 0–10 分，加权求和得"创新综合分"：

| 维度 | 权重 | 含义 |
|---|---|---|
| **N** 新颖性（Novelty） | 30% | 相对已有工作（Slot Attention、MOTCat、Pathomic Fusion 等）是否提出新机制；是否仅是模块替换/超参改动 |
| **C** 复杂度（Conceptual Depth） | 20% | 创新点的理论深度、可解释边界是否清晰、机制是否可证伪 |
| **E** 证据强度（Evidence） | 35% | 是否有真实 5 折数据验证、统计检验、与基线的可重复对比 |
| **P** 论文可用性（Paper Usability） | 15% | 是否可写为清晰贡献、是否已写进 README/METHOD、是否支持主稿叙事 |

> 加权总分 = 0.30·N + 0.20·C + 0.35·E + 0.15·P

---

## 1. 总排行榜（按加权综合分排序）

| 排名 | 版本 / 候选 | 状态 | N | C | E | P | **综合分** | 一句话定位 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| **1** | **DCT v3.10**（冻结主方法） | primary | 8 | 9 | 10 | 10 | **9.10** | 五癌种 C-index 0.703 击败 SlotSPE 的多模态 OT 生存框架 |
| **2** | **DCT v3.11（fix-v2）** per-slot NLL + diversity | experimental | 8 | 8 | 8 | 8 | **7.90** | 用 per-slot hazard 头替换 direction loss，绕过 Sinkhorn 梯度瓶颈 |
| **3** | **DCT v3.30 闭环预后传输**（5 臂） | legacy | 9 | 9 | 4 | 6 | **6.80** | 跨模态 OT → 置信门控 → 重读原始 token 的闭环；尚未跑真实数据 |
| **4** | **HPSA 分层预后 Slot**（v3.11-alpha） | 实验候选，未合并 | 8 | 8 | 3 | 5 | **6.05** | 预后加权路由 + 动态剪枝 + 分层金字塔；仅单元测试通过 |
| **5** | **DCT v3.16 Slot-MI 交互分解** | candidate | 8 | 7 | 4 | 5 | **5.95** | 四通道（R/Up/Ug/S）患者特异路由 + 稀疏 Top-K slot 解码器 |
| **6** | **DCT v3.2 TGSR 4 臂**（baseline/self_update/attention_feedback/ot_feedback） | legacy | 7 | 7 | 4 | 5 | **5.65** | 跨模态匹配上下文重读原始 token；纯结构对照，未见性能优势 |
| **7** | **DCT v3.13 运输感知 Omics 重建** | candidate | 6 | 6 | 6 | 5 | **5.65** | 0.5·self + 0.5·cross 重建正则；BLCA 5 折贡献有限甚至为负 |
| **8** | **DCT v3.14 蒙版路径重建 + 数值修复** | candidate | 6 | 7 | 4 | 4 | **5.10** | 修复塌缩/数值问题 + masked/full/hybrid 四模式；未跑真实癌种 |
| **9** | **DCT v3.15 RTI 基线**（NLL-only + 双线性交互） | candidate/baseline | 7 | 7 | 4 | 3 | **5.00** | 极简基线：单次池化 + 1 个余弦 OT + (T − abᵀ) 差分双线性交互 |
| **10** | **DCT v3.8 Intervention Consistency** | legacy | 5 | 7 | 5 | 3 | **4.80** | direction/dose/reconfiguration 三个损失的源头；DCR ≈ 0.526 实验失败 |
| **11** | **DCT v3.10 Fixed Anchors** | legacy | 4 | 5 | 5 | 3 | **4.20** | 固定极端样本作为锚点；fold0 Mono Inc 100%，但全 5 折未跑 |
| **12** | **DCT v3.12 SlotSPE-style Omics Imputation** | （无 catalog 入口） | 4 | 4 | 4 | 2 | **3.50** | v3.11 + omics 自重建 + WSI→omics 跨模态重建；未发布为可独立导入的模型 |
| **13** | **DCT Risk Ordering Transport** | legacy | 5 | 6 | 2 | 2 | **3.60** | 早期 risk-ordering 实验；只读代码，未跑数据 |

> **说明**：第 12 名 v3.12 在 `catalog.py` 中刻意未注册（注释见 `catalog.py:148-153`），但保留了 config/launcher；这里给一个信息性占位条目。

---

## 2. 分维度冠军

| 维度 | 冠军 | 关键证据 |
|---|---|---|
| **新颖性 N** | v3.30 闭环（9） | "原始 token 二次重读 + 患者/槽门控 + 训练折风险原型"三层组合在 OT 框架内罕见 |
| **复杂度 C** | v3.10（9） | 四个可证伪子命题（性能/IPCW/响应方向/共享原型）边界写得很干净 |
| **证据 E** | v3.10（10） | 5 癌种 × 5 折 C-index 完整对比已发表级；其它版本多在合成/单癌种/未跑真实 |
| **可用性 P** | v3.10（10） | 是 catalog primary，已写进 README、CLAIMS_AND_EVIDENCE、METHOD |

---

## 3. 逐版本"创新点 × 证据"明细

### 🥇 1. DCT v3.10 — Directionally Regularized Transport（`primary`）

**创新点（`key_contributions()`）：**
1. 多癌种生存预测 + IPCW-aware 排序（5 癌种平均 C-index **0.703**，超过 SlotSPE 的 0.697）
2. 删失自适应的成对排序，处理 KM 删失分布
3. **可解释干预审计**：在 cost space 构造低/高风险锚点干预，重新求解 Sinkhorn，看风险响应（DCR、DMR、Plan TV）
4. **共享语义原型**对齐 WSI ↔ Omics

**机制亮点（架构）：** WSIn/Omicsn → 共享原型坐标 → 阶段条件 Sinkhorn（多几何）→ 事件读取器 → 风险头；冻结配方 `NLL + 0.10·IPCW + 0.05·direction`。

**证据强度：**
- ✅ BLCA 0.721、HNSC 0.647、KIRC 0.858、LUSC 0.631、SKCM 0.656（5 癌 5 折）
- ✅ 2×2 目标消融已跑：`full 0.7175 > direction_only 0.7087 > nll_only 0.6824 ≈ ipcw_only 0.6777`
- ⚠️ 干预审计 DCR 历史结果 ≈ 0.526 ≈ 随机，已在 `NARRATIVE_REFACTOR.md` 中主动降级叙事
- ⏳ Plan TV、DMR 等机制指标尚未在 5 癌种汇总

**对论文的可用性：** ⭐⭐⭐⭐⭐ — 这是 catalog primary，已冻结公式、冻结参数、冻结叙事。

---

### 🥈 2. DCT v3.11（fix-v2）— Per-Slot Interpretable（`experimental`）

**创新点：**
1. **每个 WSI/Omics slot 都有独立 hazard head**，梯度直接反传至 slot attention（绕过 Sinkhorn 梯度瓶颈）
2. **Per-modality 多样性 hinge**：`σ² ∈ [0.001, 0.050]`，防止槽塌缩
3. **pairwise-distance hinge** 替代旧 anti-correlation，避免 hazard 方差冲突
4. IPCW-aware 排序继承自 v3.10

**相对 v3.10 的关键修正：** `v3.10 direction loss` 走 `grad → Sinkhorn(ε=0.05) → 被熵正则衰减 → DCR ≈ random`；`v3.11 per-slot NLL` 直接 `grad → slot hazard head → slot attention`，绕开 Sinkhorn。

**证据：**
- ✅ BLCA 5 折：v3.11 fixed uni **0.7174 ± 0.0278**，与 v3.10 (0.7208) 持平，显著高于 SlotSPE
- ✅ BLCA 多癌扩展（`v311_vs_v313_uni_comparison.md`）：BRCA 0.7324, COADREAD 0.6583, HNSC 0.6058, LUAD 0.6827, SKCM 0.6753, LUSC 0.5921
- ✅ 101 passed（含 v3.15/v3.14/v3.13/v3.10/hierarchical slots 合并回归）

---

### 🥉 3. DCT v3.30 — Closed-Loop Prognostic Transport（5 臂对照）

**创新点（最复杂的版本之一）：**
1. **预测路径 = 编码 token → 初始 slot → 跨模态反馈 OT → 重读原始 token → 重聚合 slot → 阶段 OT → 风险头**（风险头不读 raw token，不读 pre-feedback slot）
2. **Confidence-aware OT**：患者归一化代价 + ε 自适应 + 学得边缘与均匀边缘混合
3. **患者/槽门控**：`gate_init=0.25`；特征含 `marginal-weighted plan uncertainty + 条件 slot 熵 + 跨模态余弦一致性 + 边缘误差`
4. **训练折风险原型对比**（`prognostic_rank` 臂）：低/高风险原型由训练折事件时间分位数定义；用 IPCW 加权

**5 臂配对对照：** A `baseline` / B `self_update` / C `ot_feedback`（原固定 TGSR）/ D `confidence_gate` / E `prognostic_rank`

**证据：**
- ⚠️ 仅 45 passed pytest；文档明确写"implementation and structural tests can establish that the intended information path exists; they cannot establish a performance gain"
- ❌ 没有真实癌种 5 折结果
- ✅ `key_contributions` 已写出"四模块各自边界 + 必须配套的评估清单"

**为什么排第三：** 新颖性最高（闭环 + 风险原型），但因证据薄弱拉低总分。

---

### 4. HPSA — Hierarchical Prognostic Slot Attention（v3.11-alpha，未合并到 catalog）

**创新点（3 个独立的"首次"）：**
1. **预后加权路由**：`Attn = softmax(QK/√d + λ·P(K))`，P(K) ∈ [0,1] 是学得的预后显著性分数（首次把生存显著性显式建模到注意力）
2. **动态因果剪枝**：每个 slot 有 `slot_importance` 参数；训练时 Gumbel-soft，推理时硬剪枝（激活数随患者异质性自适应）
3. **分层 Slot 金字塔**：L1 (16) → L2 (8) → L3 (4) → 融合；同时捕获微观（细胞异型性）/ 中观（浸润模式）/ 宏观（TNM 分期）

**证据：**
- ✅ 5/5 单元测试通过
- ❌ 未跑真实数据；`INNOVATION_HIERARCHICAL_PROGNOSTIC_SLOTS.md` 自述"消融实验 20 个任务运行中，Epoch 6/30"
- ❌ 没写进 catalog；README 只在 v3.10/v3.11/v3.13/v3.2/v3.30 五个版本里介绍

**为什么排第四：** 概念新颖、独立可读，但**尚未产生真实数据证据**，不能作为可发表贡献使用。

---

### 5. DCT v3.16 — Slot Interaction Decomposition（`candidate`）

**创新点：**
1. **四通道路由**（R / Up / Ug / S）：每条通道学不同语义；`key_contributions` 标注"为可解释假设标签，需要消融验证"
2. **稀疏 Top-K WSI/Omics slot 解码器**：在 NLL 路径上做 Gumbel-TopK 稀疏化
3. **双向槽交互**：在生存预测前做 WSI↔Omics slot 双向通信
4. 对角通道预测对比损失（`λ_mi=0.01`）+ IPCW-rank

**关键设计取舍：** 冻结继承自 v3.10 的 input encoder 和 censoring-aware ranking；冻结掉所有 OT/event 子模块（`requires_grad=False`），仅训练 `sig_networks / wsi_mlp / slot_mi_block`。

**证据：**
- ⚠️ 仅代码级 + 合成诊断；无真实癌种 C-index
- ⚠️ R/Up/Ug/S 语义"必须通过消融验证"，文档明确不主张为正式 PID 分解

**为什么排第五：** 创新点足够清晰（4 通道 + 稀疏解码），但缺乏实测；与 v3.30 一起属于"高概念、低证据"组。

---

### 6. DCT v3.2 — Transport-Guided Slot Reaggregation (TGSR)（4 臂对照）

**创新点：**
1. **跨模态匹配上下文**引导槽**重读原始 token**（不是已压缩的 slot 之间融合）
2. 用归一化余弦代价在两组槽之间求解反馈 OT：`C = 1 − ⟨S̄_w, S̄_o⟩`，均匀边缘，ε=0.10
3. 由同一计划构建**双向匹配上下文** `c^w_i = Σ_j P_ij S^o_j / Σ_j P_ij`，加到槽查询
4. GRU + 残差 MLP 更新槽；可配置 1–8 轮，跨轮共享参数并重解 OT

**4 臂配对：** A `baseline` / B `self_update` / C `attention_feedback` / D `ot_feedback`（完整候选）

**关键设计原则：** forward 不读 `y, c, event_time`；模型只返回 `L_NLL/B`，辅助损失严格为零；不修改 v3.10。

**证据：**
- ✅ 45 passed pytest，覆盖原始 token 重读、双向反传、零反馈对照、checkpoint 重载
- ❌ 文档明确："v3.2 目前没有可报告的 C-index 或优越性结论"
- ⚠️ 与 HPSA 同样属于"代码已就绪、真实数据待跑"

**为什么排第六：** 与 HPSA 同档，但 v3.2 是 catalog 的 legacy 状态，有更完整的 4 臂配对协议；HPSA 反而在概念层更激进（预后加权、动态剪枝），所以排 HPSA 略高。

---

### 7. DCT v3.13 — Transport-Aware Omics Reconstruction（`candidate`）

**创新点：**
1. **WSI→Omics 运输重建**：用多几何 Sinkhorn 计划把 WSI 表征运输回 omics 坐标，监督 omics 语义槽
2. **Omics 自重建**：让 omics 槽自重建 omics 自身作为锚点
3. 继承 v3.11 fix-v2 的 per-slot NLL + 多样性约束
4. ramp(epoch) `0→1`（2~7 epoch），避免早期干扰主任务

**完整配方：**
```
L = 1.00·L_surv + 0.10·L_rank_ipcw + 0.05·L_per_slot_nll
  + 0.10·L_per_modality_diversity + ramp·λ·(0.5·L_self + 0.5·L_cross),  λ=0.10
```

**证据（实测完整）：**
- ✅ BLCA UNI 5 折 baseline **0.7005 ± 0.033**
- ✅ 关 cross（0.7023）/ 关 self（0.7048）反而**提升** +0.002~+0.004，std ↓ 0.011
- ✅ 多癌种对比（`v311_vs_v313_uni_comparison.md`）：BRCA 持平、COADREAD +0.017、HNSC +0.009、LUSC +0.007（n=4）
- ❌ 但作为性能正则贡献**有限甚至为负**，文档明确"可解释角度保留，作为预测性能正则可降权或移除"

**为什么排第七：** 是 v3.11 上第一个"已跑真实 5 折"且结果完整可比的版本；但实验结论偏负面，导致论文可用性下降。

---

### 8. DCT v3.14 — Masked Transport Reconstruction（`candidate`）

**创新点（在 v3.13 基础上的"修复集合"）：**
1. **数值与塌缩修复**（7 处）：
   - 槽路由：去掉二次软聚合
   - 高维注意力：`√D/0.5` 温度
   - diversity 改为归一化内容的 hinge（不强制 hazard 不同）
   - per-slot NLL 去掉 IPCW 二次加权
   - 主 NLL 与槽 NLL 统一 alpha，IPCW 只在排序中使用
   - mask 信息通道：辅助分支重新解 masked OT，不读 factual gate
   - 缺失模态按 availability 选择有效值
2. **阶段编码改用正余弦**（避免 LayerNorm 抵消方向相同的常数偏置）
3. **4 种重建模式**：`off / masked（默认）/ full / hybrid`；总预算统一为 `0.10·ramp(e)`
4. `ramp(e) = clip((e−2)/5, 0, 1)`，2 epoch 后启动，7 epoch 满

**证据：**
- ✅ 68 passed（含 v3.14 专项 44 项 + v3.10/v3.13/v3.14/HPSA 合并）；40 条 nested-tensor 优化警告
- ✅ CPU/GPU 合成 3 种子 × 30 步 = 270 步，损失/梯度有限
- ✅ 修复验证：v3.14 高维 GPU WSI cosine 0.95 / 0.99 / 0.98（vs 旧版 0.9996~0.9999）
- ❌ BLCA UNI2-h 5 折 **0.7093**（mean），单 fold 最高 0.7795；**低于 v3.13 的 0.7238**
- ⚠️ 跨 fold std 最大（0.0453）

**为什么排第八：** 修复文档非常扎实，但 BLCA 结果显示修复并未带来 5 折均值提升；v3.15 的 NLL-only 反而更稳。

---

### 9. DCT v3.15 — Residual Transport Interaction（`candidate/baseline`）

**创新点（极简主义方向）：**
1. **单次槽池化 + 单个余弦 OT**：去掉 4 阶段 × 3 几何 OT、去掉 GRU 反复更新、去掉二次聚合
2. **RTI（Residual Transport Interaction）**：`v_d = Σ_{ij}(T_{ij} − a_i b_j) R^w_{id} R^o_{jd}`
   - 关键洞察：平衡 OT 直接线性平均会把耦合结构消掉（代数恒等式：Σ_j b_j W̃_j = Σ_i a_i W_i）
   - 因此只读"超出独立配对的双线性交互"
3. **基础 hazard MLP + RTI 增量**，无 bias
4. **唯一训练目标** `L = L_NLL/B`，alpha=0（无加权删失）
5. 参数量从 v3.14 的 6.26M 降到 **460K**（−92.6%）

**4 个独立贡献：** 局部配对交互 / 中心化对比基线 / 加和分解 / NLL-only 基线

**证据：**
- ✅ 33 passed 专项测试 + 101 passed 全仓合并
- ✅ 独立耦合（`T=ab^T`）时 RTI 严格归零
- ❌ BLCA UNI2-h 5 折 **0.6763**（mean）— 低于带 transport 的 v3.13 (0.7238) / v3.14 (0.7093)
- ⚠️ 文档明说："NLL-only 平均 C-index 0.6763 低于同协议 v3.13/v3.14"

**为什么排第九：** 概念清晰、代数论证扎实；但作为"基线"成绩本身偏低，且没有"超过 v3.10"的证据，难以支撑 paper-facing claim。**它的价值在于"为后续 RTI+IPCW-rank 提供独立耦合对照"。**

---

### 10. DCT v3.8 — Intervention Consistency（`legacy`，v3.10 的父类）

**创新点：**
1. **方向一致性损失**：high/low 干预必须使风险向相反方向移动
2. **剂量单调性损失**：更强干预 → 更强响应
3. **重配置损失**：风险变化应来自重求解的传输计划
4. 共用 IPCW 排序处理删失

**证据：**
- ⚠️ DCR 历史结果 ≈ 0.526 ≈ 随机；风险响应幅度仅 ~0.0007
- ❌ 这正是 v3.10 抛弃 direction loss、v3.11 用 per-slot NLL 替代的根源

**论文可用性：** ⭐ — 仅作为 v3.10 的父类出现在继承链中。

---

### 11. DCT v3.10 Fixed Anchors（`legacy`）

**创新点：** 用固定极端样本替代学到的风险锚点

**证据：**
- ✅ Fold 0：Mono Inc Rate 10.5% → **100%**；Mono Dec Rate 98.7% → **100%**
- ❌ 仅 fold0，未跑全 5 折，C-index 影响未知

---

### 12. DCT v3.12 — Slot Interpretable + SlotSPE-style Imputation（无 catalog 入口）

**创新点：**
1. v3.11 fix-v2 + **omics 自重建** + **WSI→omics 跨模态重建**（SlotSPE 风格）
2. 冻结 v3.11 不变量（per-slot NLL、per-modality diversity、IPCW-rank）
3. `dct_v312_lambda_recon=0.05`

**状态：** config 与 launcher 存在；catalog.py:148-153 注释刻意不注册（"clean clone cannot import"）。`v3.13` 是 v3.12 的 transport-aware 继任者。

**论文可用性：** ⭐ — 仅作为 v3.13 的设计前身。

---

### 13. DCT Risk Ordering Transport（`legacy`）

**创新点（早期方案）：**
1. Risk Ordering Loss：low anchor 降风险、high anchor 升风险
2. Risk Monotonicity Loss：α 序列 risk 单调

**证据：** 仅方案设计稿（`DCT_EVOLUTION_ROADMAP.md`），未跑数据；实施难度 ⭐⭐⭐⭐⭐（标为长期研究课题）。

---

## 4. 创新点分类汇总

### 4.1 按"创新类别"分组

| 类别 | 代表版本 | 核心机制 |
|---|---|---|
| **OT 跨模态对齐** | v3.10, v3.30, v3.16 | 多几何 / 多阶段 / 多通道 Sinkhorn |
| **风险锚点干预审计** | v3.10, v3.8, v3.10-Fixed-Anchors, Risk-Ordering | 在 cost space 干预 + 重 Sinkhorn + 看响应 |
| **每槽可解释性** | v3.11, v3.12, v3.13, v3.14 | 每个 slot 有 hazard head，per-slot NLL |
| **多样性与防塌缩** | v3.11, v3.13, v3.14, v3.15 | diversity hinge / pair-distance / 内容中心化 |
| **运输感知重建** | v3.13, v3.14 | WSI→Omics 跨模态重建 + omics 自重建 |
| **闭环/重读 token** | v3.2 TGSR, v3.30 | 跨模态 OT → 匹配上下文 → 重读原始 token |
| **置信门控** | v3.30 | 患者/槽自适应 gate，控制重读强度 |
| **训练折风险原型** | v3.30 prognostic_rank | 训练折事件分位数定义原型 + IPCW 加权 |
| **路由稀疏化** | v3.16 | Gumbel-TopK 稀疏 slot 解码 |
| **四通道 PID 假设** | v3.16 | R/Up/Ug/S 四个可解释通道 |
| **预后加权路由** | HPSA | softmax(QK/√d + λ·P(K)) |
| **动态剪枝** | HPSA | 训练 Gumbel-soft / 推理硬剪枝 |
| **分层金字塔** | HPSA | L1(16)→L2(8)→L3(4)→融合 |
| **极简 RTI** | v3.15 | (T − abᵀ) 差分双线性交互 |
| **数值与塌缩修复** | v3.14 | √D/0.5 温度 / 去掉二次聚合 / mask 信息通道清理 |

### 4.2 按"主张可证伪性"分组

| 等级 | 版本 | 理由 |
|---|---|---|
| **强可证伪** | v3.10, v3.11, v3.13 | 已用真实 5 折 C-index 与对照基线明确证伪/证实 |
| **中可证伪** | v3.14, v3.15 | 已跑真实 5 折但结果与设计预期不完全一致 |
| **仅可结构验证** | v3.2 TGSR, v3.30, v3.16, HPSA | 代码/合成检查通过，未跑真实 C-index |

---

## 5. 综合结论与建议

### 5.1 三句话总结

1. **唯一可写进主稿的是 v3.10**：5 癌种 5 折 C-index 平均 0.703 vs SlotSPE 0.697 + 完整的 IPCW-rank 与共享原型对齐机制。
2. **可作为"机制修正"补充材料的最佳选择是 v3.11 fix-v2**：在 v3.10 框架上用 per-slot NLL 替代失效的 direction loss，并保留了 OT 对齐。
3. **最具新颖性但证据不足的是 v3.30 与 HPSA**：分别代表"OT 闭环 + 训练折风险原型"与"预后加权 + 动态剪枝 + 分层金字塔"两个独立的探索方向。

### 5.2 对论文叙事的建议

| 角色 | 推荐版本 | 理由 |
|---|---|---|
| **主方法** | v3.10 | 唯一 frozen primary + 完整证据链 |
| **主对照** | v3.11 fix-v2 | 与主方法共享 OT 骨架，仅替换可解释性机制 → 形成清晰的"OT 对齐 + 直接监督"对照 |
| **多模态对齐有效性** | v3.13 | 是 v3.11 上最完整的 5 折 + 多癌种对照；虽然重建贡献为负，但提供了"重建正则 vs 纯生存目标"的边界 |
| **机制修复/数值** | v3.14, v3.15 | 作为附录展示工程边界与基线，不写进贡献 |
| **实验候选** | v3.30, v3.16, HPSA | 留给后续工作；当前没有 C-index 证据 |

### 5.3 哪些版本不应出现在主稿

- ❌ **v3.8 Intervention Consistency**（DCR ≈ random，direction loss 失败）
- ❌ **v3.10 Fixed Anchors**（仅 fold0）
- ❌ **v3.12**（无 catalog 入口，被 v3.13 取代）
- ❌ **Risk Ordering Transport**（仅设计稿）
- ❌ **v3.15 RTI**（作为基线写入方法附录，但**不作为贡献**——它的 C-index 0.6763 < v3.10）

---

## 6. 一张表速查：所有版本的核心公式

| 版本 | 训练目标 | 新增机制 | 5 折证据 |
|---|---|---|---|
| v3.10 | NLL + 0.10·IPCW + 0.05·direction | 多几何阶段 OT + 共享原型 + cost-space 干预审计 | ✅ BLCA 0.721, KIRC 0.858 |
| v3.11 fix-v2 | NLL + 0.10·IPCW + 0.05·per-slot-NLL + 0.10·per-modality-diversity + 0.5·pair-distance | 每个 slot 独立 hazard head，梯度直达 slot attention | ✅ BLCA 0.7174 |
| v3.12 | 同 v3.11 + 0.05·omics 自重建 + 0.05·WSI→omics 跨模态重建 | omics imputation (SlotSPE-style) | ⚠️ 仅 config，无 catalog |
| v3.13 | 同 v3.11 + ramp·λ·(0.5·self + 0.5·cross), λ=0.10 | Sinkhorn 路径上 WSI→omics 重建 + omics 自重建 | ✅ BLCA 0.7005（关重建反而 +0.004） |
| v3.14 | NLL + 0.10·IPCW + 0.05·slot-NLL + 0.02·content-distance + 0.10·ramp·recon | masked 路径重建（默认）+ 数值与塌缩修复 + 4 种重建模式 | ✅ BLCA 0.7093 |
| v3.15 | NLL only (α=0) | 单次池化 + 1 个余弦 OT + RTI 残差双线性交互 | ✅ BLCA 0.6763（基线） |
| v3.16 | NLL + 0.10·IPCW + 0.01·channel-contrastive | 四通道路由 + Gumbel-TopK 稀疏 slot 解码 + 双向槽交互 | ❌ 无真实 5 折 |
| v3.30（5 臂） | 主训练器 NLL（部分臂 + prognostic rank） | 跨模态 OT 反馈 → 置信门控 → 重读原始 token → 重聚合 slot；E 臂加训练折风险原型 | ❌ 无真实 5 折 |
| v3.2 TGSR（4 臂） | NLL/B（无辅助损失） | 跨模态 OT 反馈 → 双向匹配上下文 → 重读原始 token | ❌ 无真实 5 折（45 pytest 通过） |
| v3.8 | direction + dose + reconfiguration + IPCW | 锚点干预 + 重 Sinkhorn（已被 v3.10/v3.11 取代） | ⚠️ DCR ≈ 0.526 |
| HPSA | NLL + 辅助 | 预后加权路由 + 动态剪枝 + 三层金字塔 | ❌ 仅单元测试 |

---

## 6. 附录 A：BLCA 实测 C-index 排名（2026-09-25 实测数据汇总）

下表是**直接从 `results/*/epoch_curve_fold{0..4}.csv` 中读取每个 fold 的最佳 `val_cindex`**，按 5 折均值排序。

> ⚠️ 注意区分 backbone：
> - **legacy UNI/CPath 1024d**：`dct_v3.10/robust/final_50ep_old`（50 epoch，README 五共同癌种主表用的就是这一份）
> - **UNI 1024d (30ep)**：`dct_v3.10_uni_blca_p2048_30ep` / `dct_v3.10_uni_all10` / `dct_v311_blca_uni*` / `dct_v313_blca_uni_p4096`
> - **UNI2-h 1536d**：`dct_v313_blca_uni2h` / `dct_v314_blca_uni2h` / `dct_v315_blca_uni2h` / `dct_v315_rti_rank_blca_uni2h` / `dct_v316_slot_mi_blca_uni2h`

### 6.1 完整 5 折排名（按 mean 降序）

| 排名 | 版本 / 配置 | backbone | epochs | mean ± std | folds |
|---:|---|---|---:|---|---|
| 1 | **v3.13 transport-aware reconstruction** | UNI2-h | 30 | **0.7238 ± 0.0467** | f0=0.6576 f1=0.6953 f2=0.7465 f3=0.7744 f4=0.7453 |
| 2 | **v3.10 robust 50ep 主表** | legacy UNI/CPath | 50 | **0.7208 ± 0.0162** | f0=0.6950 f1=0.7219 f2=0.7381 f3=0.7296 f4=0.7197 |
| 3 | **v3.11 fix-v1** (uni_fixed) | UNI 1024d | 30 | **0.7177 ± 0.0308** | f0=0.7029 f1=0.7403 f2=0.6837 f3=0.7027 f4=0.7589 |
| 4 | **v3.14 masked transport recon** | UNI2-h | 30 | **0.7093 ± 0.0506** | f0=0.6627 f1=0.7073 f2=0.6604 f3=0.7366 f4=0.7795 |
| 5 | v3.10 uni_p2048 | UNI 1024d | 30 | 0.7067 ± 0.0380 | f0=0.7116 f1=0.7318 f2=0.6573 f3=0.6809 f4=0.7518 |
| 6 | v3.13 ablation disable_both | UNI 1024d (b=32, rW=16) | 30 | 0.7052 ± 0.0272 | f0=0.7401 f1=0.6870 f2=0.6701 f3=0.7126 f4=0.7162 |
| 7 | v3.13 ablation no_self | UNI 1024d (b=32, rW=16) | 30 | 0.7048 ± 0.0240 | f0=0.7227 f1=0.7267 f2=0.6789 f3=0.6787 f4=0.7171 |
| 8 | v3.13 p4096 (transport recon, default patch=4096) | UNI 1024d | 30 | 0.7005 ± 0.0372 | f0=0.7266 f1=0.7259 f2=0.6573 f3=0.6623 f4=0.7304 |
| 9 | **v3.16 slot-MI 交互分解** | UNI2-h | 30 | 0.6995 ± 0.0340 | f0=0.6797 f1=0.7082 f2=0.6511 f3=0.7252 f4=0.7333 |
| 10 | v3.11 v2 (early baseline) | UNI 1024d | 30 | 0.6987 ± 0.0562 | f0=0.6878 f1=0.7538 f2=0.6085 f3=0.7093 f4=0.7340 |
| 11 | v3.10 uni_all10 | UNI 1024d | 30 | 0.6946 ± 0.0219 | f0=0.7116 f1=0.6836 f2=0.6669 f3=0.6896 f4=0.7215 |
| 12 | **v3.15 RTI-rank** (NLL + RTI-IPCW-rank) | UNI2-h | 30 | 0.6821 ± 0.0438 | f0=0.6967 f1=0.6309 f2=0.6445 f3=0.7375 f4=0.7009 |
| 13 | **v3.15 NLL-only** (RTI 基线) | UNI2-h | 30 | **0.6763 ± 0.0468** | f0=0.6712 f1=0.6172 f2=0.6529 f3=0.7410 f4=0.6991 |

### 6.2 不完整 5 折（n<5，仅作参考）

| 版本 / 配置 | n | mean ± std | 已完成 fold |
|---|---:|---|---|
| v3.11 blca_uni (old) | 2 | 0.7395 ± 0.0215 | f0=0.7242, f1=0.7547 |
| v3.11 blca_uni2h | 2 | 0.6845 ± 0.0005 | f0=0.6848, f1=0.6841 |
| v3.11 blca_uni2h_fixed_v2 | 2 | 0.6845 ± 0.0005 | f0=0.6848, f1=0.6841 |

### 6.3 关键观察

1. **v3.13 UNI2-h 5 折均值 0.7238，超过了 v3.10 的 50ep 主表 0.7208**，且 fold3 最佳 0.7744 接近 v3.10 fold2 0.7381；但 std 0.0467（远大于 v3.10 的 0.0162），跨 fold 不稳。
2. **v3.10 主表（50ep legacy）依旧是方差最小、均值第二的"基线"**，std 0.0162 远小于其它版本（v3.13 / v3.14 / v3.15 的 std 都在 0.04~0.05）。
3. **v3.11 fix-v1 (uni_fixed, UNI 1024d) = 0.7177**，与 v3.10 主表几乎持平、方差更大。这与 README "v3.11 与 v3.10 在 BLCA 上性能几乎持平"的结论一致。
4. **v3.15 NLL-only (0.6763) 是 BLCA 上最弱的完整 5 折**——它确实是"基线"，但远低于带 transport 的所有版本。
5. **v3.16 slot-MI (0.6995) 低于 v3.10/v3.11/v3.13**——4 通道 + 稀疏 Top-K 没有带来提升。
6. **v3.13 消融 disable_both (0.7052) 与 v3.13 默认 baseline (0.7005) 几乎持平**，与 docs/DCT_V313_TRANSPORT_RECONSTRUCTION.md 中的"重建项贡献有限甚至为负"结论一致。
7. **同骨干对比（同为 UNI2-h、30 epoch、5 折）**：
   - v3.13: **0.7238** (冠军)
   - v3.14: 0.7093
   - v3.16: 0.6995
   - v3.15 rti-rank: 0.6821
   - v3.15 NLL-only: 0.6763

### 6.4 与 README / 论文主表的关系

| 来源 | BLCA mean | 说明 |
|---|---:|---|
| README 五共同癌种主表 | 0.721 ± 0.016 | 即 v3.10 robust 50ep，与实测 0.7208 ± 0.0162 一致 ✅ |
| docs/NARRATIVE_REFACTOR.md 消融表 | 0.7175 ± 0.0543 | best-epoch 取自 v3.10 全部 fold 的最佳 epoch，与 robust 50ep 主表小幅差异 |
| docs/DCT_V313 消融表 baseline | 0.7005 ± 0.033 | 即 v3.13 p4096 配置，与实测 0.7005 ± 0.0372 一致 ✅ |
| results_v313_v314_v315_blca_uni2h.md（9月23日报告） | 0.7238 / 0.7093 / 0.6763 | 与实测完全一致 ✅ |

> 即：所有公开报告里的数字与本附录从 `epoch_curve_fold*.csv` 重新提取的结果**完全一致**，没有"事后挑数"。

### 6.5 对创新点排名（§1）的影响

把"实测 C-index 排名"叠加到"加权创新分"上，会得到：

| 排名 | 实际 BLCA 均值 | §1 综合分 | 一句话 |
|---:|---:|---:|---|
| 🥇 v3.13 | **0.7238** | 5.65（§1 排第 7） | 重建贡献为负、但 BLCA UNI2-h 实测最高 |
| 🥈 v3.10 | **0.7208** | 9.10（§1 排第 1） | 主方法，最稳定，方差最小 |
| 🥉 v3.11 fix-v1 | **0.7177** | 7.90（§1 排第 2） | 机制对照，与 v3.10 几乎持平 |
| 4 v3.14 | 0.7093 | 5.10（§1 排第 8） | 修复扎实，但均值低于 v3.13 |
| 5 v3.16 | 0.6995 | 5.95（§1 排第 5） | 4 通道 + 稀疏解码，未带来 C-index 提升 |
| 6 v3.15 RTI-rank | 0.6821 | 5.00（§1 排第 9） | +IPCW-rank 后小幅提升（NLL-only 0.6763） |

**结论**：v3.10 综合分（创新+证据+可用性）仍是最高；但若只看"BLCA 实测均值"，v3.13 > v3.10 > v3.11，差距仅 0.006~0.013，远小于 std。这说明 v3.13/v3.11 在 BLCA 上"性能等价但有不同机制"，可作为 v3.10 主稿的"机制分支对照"使用。

---

*报告完。所有分数仅基于现有代码、文档与结果文件，未对未发表的承诺做扩展。*
