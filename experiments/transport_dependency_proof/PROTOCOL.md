# 运输依赖预注册证明协议

## 核心命题

需要分别证明三个命题：

1. **计算依赖**：运输计划进入事实风险读取器且梯度有限；由 P0 回答。
2. **功能必要性**：保持输入、参数、slot 和边际不变时，替换事实计划会改变风险与预测；由 P3 回答。
3. **重求解必要性**：ground-cost 干预的方向响应依赖重新求解 Sinkhorn，而不是固定 coupling 或任意扰动；由 P2/P4/P5 回答。

计算依赖通过不等于功能必要性通过。

## P0：代码与数值门

- 相同输入在 `eval()` 下重复前向必须完全一致。
- factual、low、high 计划必须有限，最大边际误差 `<1e-3`。
- 方向目标必须向共享原型、阶段代价、证据边际、运输融合和风险头提供非零有限梯度。
- 正式配置、目标权重和 `outer_eval_only` 必须不可被队列静默改写。

P0 只支持“结构实现正确”，不支持性能或机制必要性结论。

## P1：BLCA 五折匹配 2×2 目标消融

固定相同 split、初始化、50 epoch 预算、预处理和读取器：

| 组别 | 目标 |
|---|---|
| nll_only | NLL |
| ipcw_only | NLL + 0.10 IPCW |
| direction_only | NLL + 0.05 direction |
| full | NLL + 0.10 IPCW + 0.05 direction |

报告每折 C-index、IPCW C-index、IBS、IAUC、DCR、DMR、Plan TV，使用折内配对差和 bootstrap 95% CI。完整方法相对 IPCW-only 的 C-index 非劣界为 `-0.01`。

## P2：re-Sinkhorn 必要性

在 BLCA/UCEC/LUSC 的 folds 1、2、4 比较 full 与 fixed-coupling。两者使用同样 cost intervention 和风险读取器；唯一差异是干预分支是否重新求解计划。

通过线：

- full 的平均 Plan TV `>=0.02`；
- 所有计划最大边际误差 `<1e-3`；
- fixed-coupling 相对 full 至少削弱 `50%` 的方向响应；
- 衰减使用患者级配对 bootstrap CI，不只比较宏平均。

## P3：factual-plan 功能必要性

冻结 full checkpoint，保持输入、slot、边际和所有参数不变，分别解码：

- factual Sinkhorn plan；
- 相同边际的独立计划 `mu outer-product nu`；
- 行列置乱后重新投影到相同边际的 shuffled plan。

报告患者级风险绝对变化、风险相关性、排序翻转率、NLL/C-index/IPCW C-index/IBS/IAUC 的配对差和 95% CI。

判定：若替换 plan 后风险和排序几乎不变，则不能主张 factual transport 是 load-bearing；若只影响校准而不影响排序，只能主张校准依赖。

## P4：锚点特异性

依次运行 noisy anchors、permuted reference、stage jitter 和 anchor swap。交换 high/low 锚点后，风险响应方向应反转；随机或置乱对照应向机会水平回落。

通过线：真实锚点的双向 DCR 95% CI 下界高于 `0.50`，且显著强于各零假设。anchor swap 的响应符号必须反转。

## P5：双向连续剂量

对 high 与 low 两侧分别在 `alpha={0,.25,.5,.75,1}` 重新求解 Sinkhorn。high 风险应单调上升，low 风险应单调下降。

通过线：两侧患者级 DMR 均 `>=0.70`，同时报告失败患者、Plan TV 和边际误差；端点正确但中间不单调不能算通过。

## P6：内部跨癌复现

冻结 full 方法后运行 BLCA、UCEC、KIRC、HNSC、SKCM、LUSC 各五折。不得按癌种调整目标权重、epoch 或通过线。报告癌种内与癌种间异质性、多重校正和资源消耗。

这只能支持内部跨队列泛化；独立机构数据 P7 完成前不得称外部临床泛化。

## 评估与工件规则

- 当前正式协议采用预注册的固定 50 epoch 预算，外层 fold 在训练过程中不可见并只评估一次；这是锁定交叉验证，不等同于独立机构测试。
- 所有正式运行要求 clean Git commit；若工作区脏，启动应失败而不是只记录 `git_dirty=true`。
- 每折必须保存 predictions、checkpoint、training curve、resolved config、split/environment/run manifest，以及 proof 子目录下的 audit cases/metrics。
- 正式结果出现后，不得依据结果修改阈值、主指标或选择性删除失败癌种。

