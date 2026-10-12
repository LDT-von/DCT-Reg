# E049：DCT v3.13 运输插件 MCAT / SurvPath 基线接入与对比

本实验目录用于回答两个问题：
1. **实用性**：完整插件（Learned OT + 重建）是否优于对应原生基线？
2. **收益归因**：在保持相同边际和模块容量的条件下，Learned OT 是否优于 Independent（独立耦合）？

DCT 内部已有的 v3.13 结果可以作为来源证据，但不能代替这一步。

## 实验臂设计

每种基线（MCAT / SurvPath）均有以下 6 臂：

| 臂名 | 插件 | 重建 | 作用 |
|---|---|---|---|
| `native` | 关闭 | 无 | 原生参考 |
| `independent` | 独立耦合（外积边际） | 无 | 控制新增结构与特征接入 |
| `learned` | Learned OT | 无 | 与 Independent 比较运输匹配本身 |
| `independent_three_decoders` | 独立耦合 | 独立三头 | 控制重建带来的独立贡献 |
| `learned_three_decoders` | Learned OT | 独立三头 | 完整插件设置 |
| `learned_shared_decoders` | Learned OT | 自/跨共享，WSI 独立 | 解码器分离控制（可选） |

**关键比较链**：
- `native` → `learned_three_decoders`：整体收益
- `independent` → `learned`：运输匹配本身的贡献（同等容量、同等边际）
- `independent_three_decoders` → `learned_three_decoders`：运输匹配 + 重建的联合效应
- `learned` → `learned_three_decoders`：辅助重建的独立贡献

Independent 和 Learned OT 共享所有可学习参数，只改变运输计划的计算方式。

## 配对比较原则

所有同基线、同臂的 5 折 × seed 比较均使用**配对差值**，不使用不同患者集的均值之差。
不同 seed 分别报告；不要将五折视为独立外部队列。

## 训练配方

- 默认同基线内部各臂共享：患者划分、终点、时间分箱、优化器（AdamW，lr=0.0005）、训练轮数（30 epoch）、checkpoint 选择规则（val C-index 最大选最优）
- MCAT 原生损失：事件分层 NLL；SurvPath 原生损失：负对数似然
- 插件辅助损失接在原生损失之后：`total_loss = native_loss + aux_loss`
- MCAT：六组基因分组；SurvPath：来自数据的实际通路数
- 两种基线的原生训练器分别独立实现，不混用损失函数

## 运行方式

```bash
# 生成实验计划（需填入真实 omic_sizes）
python scripts/prepare_transport_plugin_experiments.py \
    --baseline both \
    --omic-sizes 712,1359,102,1762,178,100 \
    --output /data1/DCT-Reg/results/e049_transport_plugin \
    --include-shared

# 调度全部任务
python run_experiments.py --batch-dir results/e049 --group all --protocol outer_test

# 汇总结果
python summarize.py --batch-dir results/e049
```

## 限制

- MCAT 和 SurvPath 均为单进程运行，不跨进程共享 `models` 命名空间
- 不从 DCT 内部已训练 checkpoint 直接迁移插件权重
- 不将插件关闭状态的结果宣称是完整插件训练收益
