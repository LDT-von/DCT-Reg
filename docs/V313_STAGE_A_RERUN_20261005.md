# Stage A 重跑修复与验收

依据 637e2b8 的审查记录，旧控制臂汇总不能作为五折性能证据。该审查中的“调度器已修复”结论需要补充：637e2b8 的任务列表有 fold 和目录标签，但实际子进程命令没有注入折范围和输出目录。

本次修复让每个子进程收到 `k_start=fold`、`k_end=fold+1`、对应的 `results_dir`、癌种、方法与评价协议。JSON 计划记录实际 GPU 分配。每个 GPU 使用独立队列，避免不同任务时长导致某张卡超过 `jobs-per-gpu`。补齐已跟踪源码缺少的 `exp6_kirc_uni2h.yaml`，其参数由现有配置生成器从 BLCA UNI2-h 基础配置继承。

## 服务器运行清单

先同步含上述修复的源码。以下命令只生成计划，不启动训练；新根目录保留旧 Stage A 产物。

```bash
cd /data1/DCT-Reg
export PYTHONPATH=/data1/DCT-Reg
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
$PY -m survot_rank.cli schedule --protocol legacy_val \
  --arm direct --arm independent --cancer kirc --cancer blca \
  --gpu 0 --gpu 1 --jobs-per-gpu 1 \
  --results-root results/v313_paper_v2
```

应有 20 条任务：4 组各 5 折，每条只运行一折，两个 GPU 各分配 10 条。确认配置均存在、真实 split 文件指纹有效后，由作者决定运行时间；在相同命令末尾添加 `--execute` 才会启动训练。本次修复与测试不运行模型训练。

## 结果验收

1. 保存每条任务的提交、配置、完整启动参数、fold、seed、真实划分文件 SHA-256 和实际输出目录。核对训练日志中的单折范围。
2. 将风险输出中的患者身份与该折验证集对应，核对训练/验证隔离和拟合参考；若产物仅用位置索引，保存其与患者身份的映射。
3. 对 CSV、最终预测与 checkpoint 的完整路径和内容作摘要核验；跨折字节重复应触发调查，不以不同摘要本身证明患者划分正确。
4. 从每折实际产物重算相同口径的 C-index，报告 5 折值、最佳 epoch、均值和样本标准差。`std=0` 不单独证明造假，`std>0` 也不单独证明有效。
5. 禁止将“分数在预期量级”或“Full 必须优于控制臂”作为合格条件。只有运行身份与协议可追溯的结果才用于机制结论。

70 个损失消融结果与 Stage A 分开处理。Stage B 的文件互异是审查线索，外部方法比较仍需要核对实际输入、划分、终点和选模/评分口径。当前审查文本没有提供 SlotSPE 四组的完整数值，不能据此补造比较表。
