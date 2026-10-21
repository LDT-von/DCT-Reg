# E053 v3.13 同口径 BLCA SlotSPE 三臂复刻

编号说明：本实验最初的工作目录名为 `experiments/e048_v313_aligned_blca/`，但 `E048` 已被
`实验档案/实验代码/E048_v313模块与单模态消融` 占用，故最终登记编号为 **E053**。

## 目的

在**与 v3.13 相同的数据口径**下，用 SlotSPE 插件跑三臂（native / learned / independent），
并与 v3.13 原产物逐 epoch 对照，确认复现链路的可信度与 transport 机制的实际作用幅度。

## 口径

| 项 | 值 |
|---|---|
| 癌种 | BLCA |
| 划分 | `/data1/dataset_csv/splits/5fold_uni2h/blca/fold_0.csv` |
| 特征 | `/data1/TCGA-UNI2-h-features/blca/uni2-h/pt_files`（.h5, UNI2-h） |
| train / val | 304 / 76 患者（excluded=0, overlap=0） |
| num_patches | 2048（短 bag 零填充，15 名患者） |
| encoding_dim | 1536 |
| label | `survival_months_dss` |
| epoch / seed / fold | 30 / 3 / 0 |
| 选点 | `legacy_val`（max val c-index over epochs） |
| eval_repeats | 5（v3.13 为 1，E053 取 5 次重复的均值） |

## 运行

```bash
cd /data1/DCT-Reg
export PYTHONPATH=/data1/DCT-Reg
/home/ubuntu/.conda/envs/trisurv/bin/python -u -m survot_rank.plugins.transport_only.slotspe_compare \
  --config experiments/e053_v313_aligned_blca/config.json \
  --output experiments/e053_v313_aligned_blca/out --run
```

**必须使用 `trisurv` conda 环境**：系统 `python3` 的 torch 编译不含 sm_120 内核，
在 RTX 5090 上会报 `CUDA error: no kernel image is available for execution on the device`。

## 结果摘要（fold 0, seed 3, val 76 患者）

| 臂 | val c-index (5 repeat 均值) | repeat std | NLL | 选中 epoch | 参数增量 | 训练秒 |
|---|---|---|---|---|---|---|
| native | 0.6338 | 0.0047 | 1.1160 | 4 | +0 | 715 |
| learned | 0.6148 | 0.0176 | 1.4022 | 8 | +600327 | 3094 |
| independent | 0.6688 | 0.0108 | 1.1218 | 5 | +600327 | 780 |

- learned − native = **−0.0190**
- learned − independent = **−0.0540**
- v3.13 原始产物：峰值 0.6576 @第 3 轮，末轮 0.5718
- E053 native：峰值 0.6338 @第 4 轮，末轮 0.5514
- 逐轮平均绝对差 0.0467，pearson r = 0.234

## 关键机制观察

learned 臂的 `same_weights` 对照（同一套权重，仅切换 plan_mode / 是否启用 transport）：

| 对照 | val c-index |
|---|---|
| transport ON, plan_mode=independent | 0.6148 |
| transport **BYPASS**（完全关闭） | 0.6150 |

逐患者风险差 max|Δrisk| = 1.076e-02，仅为 risk 标准差的 **0.79%**。
即：在 learned 权重下，推理时开关 transport 对排序与分数几乎没有影响。

## 限制

- 单折单 seed，非泛化性证据；未做统计显著性检验。
- eval_repeats 与 v3.13 不同（5 vs 1），对照列取自 v3.13 原始逐轮曲线。
- 峰值 epoch 接近但逐轮曲线相关性低（r=0.234），只能说明量级一致，不能说复现了原曲线。
- independent 臂同时改了 plan_mode 与边际控制，不可单独归因于某一项。

## 产物

- 图表与结果表：`实验档案/表格图片/E053_v313同口径_BLCA复刻/`
- 完整运行产物（prepared 缓存 4.6G + checkpoints）：`experiments/e053_v313_aligned_blca/out/`
- 归档只保留结果 JSON/CSV，不含特征缓存与权重。
