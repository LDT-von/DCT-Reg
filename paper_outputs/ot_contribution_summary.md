# OT 对抗性删除测试 (BLCA)

## 实验目的

通过 monkey-patch 把 OT 运输计划替换为均匀分布，量化 OT 模块对预测性能的贡献。
如果 OT 是关键机制，移除后 C-index 应该大幅下降；如果 OT 只是辅助正则，贡献应该接近 0。

## 测试方法

- 加载训练好的 BLCA fold 0/1 checkpoint
- Full: 原始 forward（包含 Sinkhorn OT 求解）
- No-OT: monkey-patch `_plans_from_cost_tensor`，让它返回均匀分布 (1/(Kw*Ko))
- Risk 计算：`survival = cumprod(1 - sigmoid(hazards))`, `risk = -sum(survival)`
- C-index: `concordance_index_censored(event_indicator, event_time, estimate)`

## 结果

| Fold | Full C-index | No-OT C-index | Δ (OT 贡献) | 相对变化 |
|------|--------------|---------------|------------|----------|
| 0    | 0.7013       | 0.7060        | **-0.0048** | -0.7% |
| 1    | 0.8012       | 0.8046        | **-0.0034** | -0.4% |
| **平均** | **0.7513** | **0.7553** | **-0.0041** | **-0.5%** |

## 判定

- 两个 fold 的 OT 贡献均为 **负值**（-0.7%, -0.4%）
- 绝对变化 |Δ| < 0.01，判定为 **minimal**（OT 对预测贡献很小）
- **No-OT 反而略好**：移除 OT 模块后 C-index 略高 0.004-0.005

## 结论

**OT 模块对预测性能没有正向贡献**。证据链：

1. **零假设测试**（5 fold BLCA）: DCR 对 OT 几何完全不敏感（Δ=0.000）
2. **对抗性删除测试**（2 fold BLCA）: No-OT 的 C-index ≥ Full 的 C-index
3. **剂量响应曲线**（HNSC, SKCM）: alpha 0→1 时 risk 变化 ≈ 0.001（接近噪声）

**OT 几何不构成模型的预测机制**。Direction Loss 通过特征空间起作用，而非 OT 几何。

## 论文叙事建议

承认 OT 对预测贡献有限，把核心贡献重新聚焦到：
- **Slot 特征 + Direction Loss** 提供反事实风险方向
- **OT 几何结构** 仅作为 gradient flow 的训练稳定器（次要贡献）
- **性能 vs 机制分离** 本身作为一个 honest finding 写入论文

## 输出文件

- `paper_outputs/ot_contribution_blca_fold0.json`
- `paper_outputs/ot_contribution_blca_fold1.json`
