# 最小架构

```text
WSI patches ──> slot encoder ─┐
                              ├─> stage-wise ground cost ─> Sinkhorn plan ─> shared risk head
Pathway groups ─> slot encoder┘                 │
                                                ├─ low-risk cost intervention ─> re-Sinkhorn ─> risk_low
                                                └─ high-risk cost intervention -> re-Sinkhorn ─> risk_high
```

训练时，主训练器提供生存 NLL；DCT-Reg 模型仅返回 `0.10 × IPCW-rank + 0.05 × direction`，两者合成冻结目标。

代码继承链保留原实现路径以降低迁移风险：

```text
DCTV310DirectionalRegularizedTransport
  -> DCTTransportInterventionConsistency
  -> DistributionalCounterfactualTransport
  -> FaithfulEvidenceTransport
  -> StagewisePrognosticTransport
  -> RankGuidedEventTransport / OTEventHazardV2
```

仓库的公开 catalog 只暴露最终 DCT-Reg 和消融父类。其余父类属于实现依赖，不是可单独宣称完成的论文方法。

`fixed_coupling` 控制的事实路径仍对当前 batch 新求解 Sinkhorn；只有低/高风险干预路径重放该 batch 的事实计划。这样才能把“是否重新求解”作为唯一差异，并避免跨 batch 缓存污染。

