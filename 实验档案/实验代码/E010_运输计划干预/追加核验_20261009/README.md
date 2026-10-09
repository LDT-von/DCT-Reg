# E010 运输计划干预：追加核验工具

本目录属于 v3.13 的 E010 补充诊断，不新增实验编号，不修改模型或既有实验成绩。

## 当前完成情况
- 已提供独立核验脚本 audit_saved_exports.py、17 项测试和服务器提示词。
- 本机只运行了人工构造的小数组和临时文件测试，17/17 通过。
- 未读取服务器真实患者结果，未启动推理或训练，尚不能给出模型故障或性能改善结论。
- 本目录尚未提交或 push；服务器不会仅靠已有远端提交自动获得它。

## 如何使用
1. 将本目录复制到服务器仓库的同名位置；若该目录已存在，先核对内容，不覆盖已有版本。
2. 读取 SERVER_PROMPT.txt，让服务器先核对现有导出和来源，生成真实路径清单。
3. 在仓库根目录运行离线核验。该步骤只读取已有结果，不导入或运行模型。
4. 只有缺少实际中间张量时才准备补充导出代码和推理命令，由用户执行模型任务。

清单格式（路径必须替换为实际核对后的目录，每个目录包含 export.json 和 patients.npz）：

```json
{"schema_version": 1, "exports": ["/actual/path/to/export_1", "/actual/path/to/export_2"]}
```

先检查少量折：
```bash
python '实验档案/实验代码/E010_运输计划干预/追加核验_20261009/audit_saved_exports.py' --input '/actual/path/to/exports.json' --output '/actual/path/to/a_new_output_directory'
```

BLCA 与 KIRC 各 5 折全部核验时，追加 --require-ten-folds。少于 10 折时不能写“完成十折”。输出目录必须尚不存在。

测试命令：
```bash
python '实验档案/实验代码/E010_运输计划干预/追加核验_20261009/test_audit_saved_exports.py'
```

## 输入与输出
仅接受 BLCA/KIRC、Exp6 Full、seed=3、legacy_val 的 v3.13 Pathways 配置。需要配置、checkpoint、曲线、最佳预测、划分以及训练/导出来源记录；检查记录的文件哈希、最佳 epoch、患者与结局，并重算 C-index。只读取研究者可信的预测 pickle，不加载 checkpoint。服务器环境需要 NumPy 和 PyYAML；本机测试使用合法的扁平 JSON 配置，不依赖 PyYAML。

输出：
- audit.json：逐 alpha 的逐患者风险差异汇总、可比较生存对变化、全部患者对的排序变化、中间层差异、缺失字段、文件来源及覆盖范围。
- patient_risk_changes.csv：按真实患者 ID 对应的逐 alpha 风险与差异，供内部核查；不要把患者级文件直接放入公开论文材料。

## 证据边界
- 由 plans 推算的混合计划和运行时实际导出的 sweep_plans 分开记录。只有后者可用于检查实际替换是否符合公式，且仍需核对导出代码来源。
- rows/cols 是求解器目标边际；与这些目标的残差，不等于混合计划相对原计划的边际变化。
- 比较 gate/logits 等运行时数组时，缺少字段会明确列出，不补造数值。
- C-index 相同可能伴随排序对变化互相抵消；均值风险差不能代替逐患者最大变化。
- 风险与排序发生变化也不能单独证明运输机制带来训练收益。训练贡献需要独立、同设置的训练消融。
- 不改既有主档、旧报告、模型和 checkpoint，不删除结果。新材料只能写到新的位置。

