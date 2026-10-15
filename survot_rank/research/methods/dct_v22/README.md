# 独立 DCT v2.2 过拟合修复候选

候选缩小现有病理和组学编码宽度、补上病理 dropout，保留 v2.1 的三项机制及六项损失。它不新增预测分支，不启用重构。患者效果尚未验证。定位以 `docs/DCT_V22_METHOD_CONTRACT.json` 为准。

三个配置沿用 E050 的 BLCA、UNI2-h、`5fold_uni2h`、seed3、legacy_val 和30轮训练。控制配置在相同种子下与 v2.1 的参数、预测、辅助目标和随机状态一致；容量配置只调整现有映射尺寸；正则化配置在容量调整基础上增加 dropout、权重衰减并降低学习率。三个配置分别写入新的结果目录。

```bash
python -m survot_rank.cli train --config configs/dct_v22_blca_legacy_val_regularized.yaml
```

需要机制归因时，再分别运行 `dct_v22_blca_legacy_val_control.yaml` 和 `dct_v22_blca_legacy_val_capacity.yaml`。组合配置的效果不能归因给其中某个单独参数。不要覆盖 E050 或 v3.13 结果。

主比较保留完整30轮，仍按验证 C-index 保存最早最佳权重。实际使用若希望减少后期训练，可单独加 `--set early_stop_patience=8 --set early_stop_warmup=12`；这一策略不能保证保留所有未来峰值，结果也不能冒充与完整30轮严格相同的对照。

监测区分训练过程目标与固定权重的训练集评估。`train_eval_main_nll`/`train_eval_cindex` 与 `val_main_nll`/`val_cindex` 使用评估模式和相同 patch 子采样策略。训练集仍然是原训练患者，模型不更新 IPCW 或排序记忆。这次额外评估增加一次训练集大小的无梯度前向；`--set dct_v22_evaluate_train=false` 可关闭。

槽诊断使用实际槽距离、相似度、范数、风险响应差异和饱和率。运输诊断使用实际质量、计划与同预算独立计划的距离、局部读取强度、部署增益和患者 NLL gap 的绝对均值。正则化配置每10次训练 forward 采样一次原始关系损失的成本网络梯度；未乘辅助权重，不写 `.grad`，不能等同于主预测的运输贡献。

保存曲线可以用下面的工具重算。输出目录必须是新目录；工具拒绝覆盖源表或已有输出。

```bash
python scripts/audits/audit_saved_training_curves.py --fold-dir /data1/DCT-Reg/实验档案/结果/E051_v22_20261014/regularized --output-dir /data1/DCT-Reg/实验档案/结果/E051_v22_20261014/recheck_regularized --write-corrected-csv
```

E050 汇总表只有验证 C-index。原诊断中的“多样性损失下降即坍缩”“运输缺额就是训练损失”“均值 gap 接近零即关系贡献为零”不能作为根因结论。legacy_val 用于调参和选权重；正式泛化证据还需在固定方案后使用独立 outer_test 协议验证。
