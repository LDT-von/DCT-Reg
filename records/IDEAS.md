# Idea 与创新边界记录

## 2026-08-26 评估

### 当前 idea 的价值

DCT-Reg 最有价值的部分不是“再做一个 OT 生存模型”，而是把病理—通路运输是否
真正承载预后语义，改写为可证伪的模型内部干预问题：改代价、保持边际、重新求解
Sinkhorn、使用同一风险读取器，再检查方向响应与计划变化。这比只报告 C-index 和
漂亮热图更有论文辨识度。

### 可守住的创新边界

- 可主张：方向正则化的 ground-cost intervention + re-Sinkhorn 风险响应，以及一套
  针对 transport necessity / specificity / dose structure 的审计协议。
- 不应主张：患者治疗反事实、临床因果效应、OT 计划天然可解释、或仅凭高 DCR 就证明
  factual prediction 依赖 transport。
- 继承编码器、slot、阶段和风险头不是新的核心贡献；论文需要把结构复用与 DCT-Reg
  新机制分开写。

### 当前最大的科学风险

方向损失可能训练出一个“对锚点扰动会按要求响应”的读取器，但 factual prediction
仍可能主要依赖共享表示或旁路。换言之，方向响应证明 intervention path 可控，不自动
证明 factual transport 是 load-bearing。

### 最值得补的 idea / 实验

1. **Factual-plan necessity test**：在持出集保持边际，置乱/均匀化 factual plan 后再读风险，
   检查预测与风险响应是否显著退化。这是核心主张最直接的补强。
2. **Cross-fitted frozen anchors**：锚点只由 inner-train 估计并冻结，在 outer-test 不更新；
   可进一步用另一折锚点做敏感性分析，排除 batch/EMA 特有捷径。
3. **Anchor swap sign test**：交换 high/low 锚点后响应方向应反转，且幅度应与原方向近似对称。
4. **Stage × geometry mediation**：逐阶段、逐几何单独干预，报告 Plan TV 和风险 delta，确认
   不是某一支路或数值尺度独占结果。
5. **Negative-control ladder**：从 fixed coupling、随机锚点、参考置乱到保边际计划置乱，形成
   由弱到强的零假设链，而不是只做一个随机对照。
6. **Anchor uncertainty**：对训练病例重采样获得锚点不确定性，并传播到 DCR/DMR/Plan TV；
   小事件癌种尤其需要这一层。

### 暂不建议新增模型模块

在 E1–E4 尚无正式结果时，不建议继续叠加新的损失、注意力或因果术语。当前最稀缺的
不是结构复杂度，而是能闭环证明“方向响应来自真实计划重构且可泛化”的证据。

## 新 idea 模板

### YYYY-MM-DD / idea 名称

- 要解决的可证伪问题：
- 与当前冻结方法的关系：补充实验 / 下一版本候选 / 冲突
- 预期观察：
- 关键零假设：
- 最小实验：
- 论文主张边界：
- 决定：保留 / 暂缓 / 拒绝

