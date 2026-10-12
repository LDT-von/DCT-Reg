# E049：DCT v3.13 运输插件 MCAT / SurvPath 基线接入与对比

本轮先只回答一个问题：在 **BLCA** 上，完整插件（Learned OT + 三个独立重建解码器）能否改善同一原生基线？默认只比较 `native` 与 `learned_three_decoders`。运输本身的收益归因需要后续另跑 `independent` 等消融臂。

DCT 内部已有的 v3.13 结果可以作为来源证据，但不能代替这一步。

## 实验臂设计

代码支持以下 6 臂，但默认只运行第一行和第五行：

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

所有同基线、同折、同 seed 的原模型与插件比较均使用**配对差值**，不使用不同患者集的均值之差。
不同 seed 分别报告；不要将五折视为独立外部队列。
汇总表的差值定义为“完整插件 − 原模型”，正值表示插件更好。仅五折的双侧 Wilcoxon 检验只能作为探索性描述，重点看逐折差值和平均差值。

## 训练配方

- 默认同基线内部各臂共享：患者划分、终点、时间分箱、优化器（AdamW，lr=0.0005）、训练轮数（30 epoch）、checkpoint 选择规则（val C-index 最大选最优）
- 本轮两种基线均使用仓库训练器的离散时间 `NLLSurvLoss`；MCAT 输出 hazard 后转换为 logit 再送入该损失。它是**同一基线内两臂匹配**的比较，不声称复现官方论文的完整训练配方。
- 插件辅助损失接在同一生存损失之后：`total_loss = survival_loss + aux_loss`。评估时不计算重建辅助项。
- MCAT：六组基因分组；SurvPath：来自数据的实际通路数
- 两种基线都以单患者为单位训练；同一基线的两臂使用同样的优化器、终点、划分和选模规则。

## 运行方式

```bash
# 在本目录执行，先核实服务器上的划分文件和数据路径。
# 默认 BLCA × MCAT/SurvPath × native/完整插件 × 5 折 × 1 seed = 20 个训练任务。
python run_experiments.py --batch-dir /data1/DCT-Reg/results/e049_blca --dry-run \
    --split-set 5fold_uni2h_outer_test_seed3

# 确认计划后，使用一个新的空 batch-dir 执行；任务逐个运行，失败即停止。
python run_experiments.py --batch-dir /data1/DCT-Reg/results/e049_blca_run --execute \
    --split-set 5fold_uni2h_outer_test_seed3

# 如果先只检查 fold 0：额外加 --folds 0，得到 4 个训练任务。
# 若只比较 MCAT：额外加 --baseline mcat，五折共 10 个训练任务。
python summarize.py /data1/DCT-Reg/results/e049_blca_run
```

不要对同一非空 `batch-dir` 重复执行。正式运行前，检查 `data/splits/<split-set>/blca/fold_0.csv` 的真实位置；`--data-path` 可覆盖配置中的 `/data1/dataset_csv`。清单生成默认**不会训练**，只有 `--execute` 会启动。

## 限制

- MCAT 和 SurvPath 均为单进程运行，不跨进程共享 `models` 命名空间
- 不从 DCT 内部已训练 checkpoint 直接迁移插件权重
- 不将插件关闭状态的结果宣称是完整插件训练收益
