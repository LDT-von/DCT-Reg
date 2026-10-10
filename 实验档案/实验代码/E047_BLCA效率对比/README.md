# E047 BLCA 效率对比实验

## 实验目的

在 BLCA 队列上测量 DCT v3.13 Full 与 8 个外部方法的显存和延迟，绘制性能-效率权衡图。

## 目标图表

**图 1：BLCA C-index vs 峰值 GPU 显存**
- 横坐标：峰值 GPU 分配显存 (MiB)
- 纵坐标：BLCA 5折平均 C-index

**图 2：BLCA C-index vs 前向延迟**
- 横坐标：中位前向wall-clock时间 (ms)
- 纵坐标：BLCA 5折平均 C-index

## 方法列表

| 方法 | BLCA 公开分数 | 权重状态 | 成本测量 |
|------|-------------|---------|---------|
| **DCT v3.13 Full** | 0.7238 | 已有 | 已有 |
| MCAT | 0.688 | 缺 | 待测 |
| MOTCat | 0.700 | 缺 | 待测 |
| CMTA | 0.702 | 缺 | 待测 |
| SurvPath | 0.666 | 缺 | 待测 |
| PIBD | 0.693 | 缺 | 待测 |
| MMP | 0.686 | 缺 | 待测 |
| Porpoise | 0.690 | 缺 | 待测 |
| LD-CVAE | 0.635 | 缺 | 待测 |

## 数据来源

- **DCT 实测成本**：来自 `results/v313_tradeoff_real_20261008/efficiency_tradeoff.json`
  - 显存：181.4 MiB
  - 延迟：493.0 ms
  - C-index：0.7238

- **外部方法公开分数**：来自 `paper/V313_OT_UNI2H_COMPARISON_INPUT.json`
  - 表01.csv / 表02.csv

## 测量协议

与 DCT v3.13 一致：
- 硬件：RTX 5090
- Batch size: 1
- Patches: 2048
- Feature dim: 1536 (UNI2-h)
- Repeats: 30 (计时)
- Warmup: 5
- Precision: float32
- Mode: eval_no_grad

## 文件清单

```
E047_BLCA效率对比/
├── 实验索引.json           # 实验元数据
├── SERVER_PROMPT.txt       # 服务器测量提示词
├── measure_e047_external_methods.py  # 测量脚本
├── plot_e047_efficiency.py # 绘图脚本
├── e047_efficiency_data.json.example  # 数据模板
├── e047_efficiency_data_draft.json   # 草稿数据（待填入测量值）
└── figures/               # 输出目录（待生成）
```

## 注意事项

1. **外部方法分数来源**：论文公开值，非同条件复现
2. **成本测量**：使用随机初始化权重测前向成本
3. **图表标注**：需明确标注"public score + measured cost"
4. **不重新训练**：仅测量推理成本

## 工作流程

1. [ ] 服务器测量 8 个外部方法的显存和延迟
2. [ ] 填写 `e047_efficiency_data.json` 中的 `memory_mib` 和 `latency_ms`
3. [ ] 运行 `plot_e047_efficiency.py` 生成图表
4. [ ] 验证图表质量和数据来源标注
5. [ ] 归档最终结果
