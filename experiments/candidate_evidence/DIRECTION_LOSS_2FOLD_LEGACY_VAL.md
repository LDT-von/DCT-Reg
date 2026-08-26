# Direction-loss 2-fold legacy validation report（隔离证据）

## 来源

- SurvOT-Rank commit：`271466d9d3375a061270218f0b1502ffa8f58723`
- 原文件：`docs/DCT_V310_DIRECTION_LOSS_EXPERIMENT.md`
- 拉取日期：2026-08-26
- 原报告范围：6 个癌种，legacy 5-fold 中的 folds 2、3，最佳验证 C-index 两折均值。

## 原报告数值（原样登记）

| Cancer | reported d=0.00 | reported d=0.05 | reported delta |
|---|---:|---:|---:|
| BLCA | 0.6756 | 0.7281 | +0.0525 |
| UCEC | 0.8095 | 0.8050 | -0.0046 |
| KIRC | 0.8069 | 0.8009 | -0.0060 |
| SKCM | 0.7141 | 0.7035 | -0.0106 |
| HNSC | 0.6780 | 0.6480 | -0.0300 |
| LUSC | 0.6361 | 0.6617 | +0.0256 |

这些数字只是在此保留来源记录，不进入 DCT-Reg 正式结果表。

## 隔离原因

1. 原提交只有汇总 Markdown，没有逐折预测、checkpoint、训练日志、解析后配置、split 哈希或运行 manifest，无法独立复核数值和运行身份。
2. 原报告声明两组均使用 `survot_method=dct_v310_directional_regularized_transport`，但该冻结类会在构造前后把 `dct_v38_lambda_direction` 强制设为 `0.05`。因此仅传入 `dct_v38_lambda_direction=0.00` 不能形成真正的 OFF 组。
3. 合法 OFF 消融必须使用可接受零权重的父类 `dct_transport_intervention_consistency`；本仓库的 `run_dct_v310_experiments.py` 已按此方式构建 2×2 队列。
4. 数值来自两个 legacy validation folds，不是预注册 `5fold_uni2h` 的五折配对结果，也不是嵌套外层测试。
5. 两折均值没有方差、置信区间或配对检验，不能使用“显著提升”表述。

## 当前判定

状态为 `quarantined`：它可用于定位需要重跑的癌种和核查旧实验，但不能支持“direction loss 已有效”或“核心主张已经证明”。只有找回完整运行工件并确认 OFF 组实际使用父类，或按当前正式协议重跑后，才能升级证据等级。

