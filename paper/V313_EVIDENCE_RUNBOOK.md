# v3.13 证明实验与可视化：服务器运行顺序

这套工具只补需要修复或解释的证据。默认沿用 **legacy_val**，不会启动 110 个 outer_test 任务。训练必须显式加 `--execute`；checkpoint 推理必须显式运行 `export`。本次开发只用合成数据测试，没有启动真实实验。

## 0. 拉取与环境

```bash
cd /data1/DCT-Reg
git pull --ff-only origin main
export PYTHONPATH="$PWD"
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
TOOL=scripts/prepare_v313_evidence.py
EVIDENCE="$PWD/results/v313_evidence_v2"
```

使用已经能训练 DCT 的 `trisurv` 环境。绘图还需要 matplotlib；IPCW Brier 需要训练环境已有的 scikit-survival。工具不自动安装包、不修改旧结果。

运行入口包括 `plan / bind / scan / merge / audit / export / plot`，每个入口可加 `--help`。这里的 `$EVIDENCE` 保存清单和图，训练结果另存在 `$PWD/results/v313_controls_v2`。

## 1. 先补正确的 Direct / Independent：20 次

先生成并查看清单，**这一步不训练**：

```bash
$PY "$TOOL" plan --group controls \
  --gpu 0,1 --results-root "$PWD/results/v313_controls_v2" \
  --output "$EVIDENCE/controls_plan.json"
```

默认 BLCA、KIRC，fold 0–4，seed 3，30 epochs，共 **2 臂 × 2 癌种 × 5 折 = 20 次**。各 fold 的实际子进程参数包含正确的 `k_start/k_end` 和独立结果目录，两个 GPU 各跑一个任务，空闲后接着跑本 GPU 队列。

确认清单后，在服务器运行相同命令并加 `--execute`：

```bash
$PY "$TOOL" plan --group controls \
  --gpu 0,1 --results-root "$PWD/results/v313_controls_v2" \
  --output "$EVIDENCE/controls_plan.json" --execute
```

如果目标任务目录已有内容，工具会停止，避免覆盖。第一次运行失败后，先检查失败日志；需要重跑时用新的 `--results-root` 和 `--output`，可加 `--cancer blca --fold 2` 缩小范围。`plan` 用 `--group` 选臂；若只需单个控制臂，可用现有 `survot_rank.cli schedule --arm direct ... --execute`，之后另用 `scan` 建清单。

完成后解析实际含 hash 的训练目录，并核验：

```bash
$PY "$TOOL" bind --plan "$EVIDENCE/controls_plan.json" \
  --output "$EVIDENCE/controls.json"
$PY "$TOOL" audit --manifest "$EVIDENCE/controls.json" \
  --output "$EVIDENCE/controls_audit.json"
```

`bind` 要求每项只有一个对应 fold 的曲线文件。`audit` 检查患者划分、最佳 epoch、由最佳患者风险值重算的 C-index、预测文件/曲线/checkpoint 的 SHA-256；任何重复实验身份、重复字节结果或缺失文件都返回非零，不进入绘图。

## 2. 接入现有 Full / Exp0–6 / SlotSPE，先审计，不默认重跑

历史结果必须填写**真实训练版本、YAML、当时的 overrides、对应文件路径**。不能把当前 HEAD 填作历史训练版本。Full 的已报告 0.7238 / 0.8224 只有在原始结果核验通过后才进入图。

例如，以下占位路径与版本必须换成服务器上的实际值；没有确定前不要执行：

```bash
FULL_BLCA_ROOT=/实际/Full/blca/结果根目录
FULL_KIRC_ROOT=/实际/Full/kirc/结果根目录
FULL_COMMIT=实际Full训练commit
FULL_BLCA_YAML=/实际/当时使用的blca配置.yaml
FULL_KIRC_YAML=/实际/当时使用的kirc配置.yaml

$PY "$TOOL" scan --root "$FULL_BLCA_ROOT" --config "$FULL_BLCA_YAML" \
  --arm exp6 --cancer blca --seed 3 --source-commit "$FULL_COMMIT" \
  --output "$EVIDENCE/full.json"
$PY "$TOOL" scan --root "$FULL_KIRC_ROOT" --config "$FULL_KIRC_YAML" \
  --arm exp6 --cancer kirc --seed 3 --source-commit "$FULL_COMMIT" \
  --output "$EVIDENCE/full.json" --append
$PY "$TOOL" merge --manifest "$EVIDENCE/full.json" \
  --manifest "$EVIDENCE/controls.json" --output "$EVIDENCE/core.json"
$PY "$TOOL" audit --manifest "$EVIDENCE/core.json" \
  --output "$EVIDENCE/core_audit.json"
```

**`scan` 不替你猜消融开关。** 若历史命令用了 `--set`，必须逐个照填，例如 `--set dct_v313_recon_weighting=per_branch` 或历史的 `--set dct_v313_recon_weighting=legacy`。单分支开关和补偿 scale 也必须按当时的命令填写，不能把旧实验改标成新配方。

可以按同样方式把 Exp0–5 加入其他清单，再 `merge`。同一臂同一癌种同一 fold 同一 seed 只能保留一份明确选择的实验，不能逐折挑不同配方的最高分。

SlotSPE 原始曲线/预测文件若命名不同，手动按下面结构建立 manifest。工具支持它们的**结果审计和主表图**；`export` 只支持真实 v3.13 checkpoint，不冒充 SlotSPE 模型解释器。Stage B 的 20 次已有结果先恢复核验，默认不重训。

```json
{
  "schema_version": 1,
  "runs": [{
    "id": "slotspe_matched_blca_f0_s3",
    "arm": "slotspe_matched", "cancer": "blca", "fold": 0,
    "seed": 3, "protocol": "legacy_val", "source_commit": "真实训练commit",
    "config": "/实际/训练配置.yaml", "overrides": [],
    "curve": "/实际/curve.csv",
    "predictions": "/实际/best_predictions.pkl",
    "checkpoint": null,
    "split_csv": "/data1/dataset_csv/splits/5fold_uni2h/blca/fold_0.csv"
  }]
}
```

CSV 至少有 `epoch,val_cindex` 两列；PKL 是 `{patient_id: {risk: 单值, time: 单值, censor: 单值, ...}}`，`censor=1` 表示删失，高 risk 表示高风险。预测须来自 CSV 的最佳 epoch；**不要把 `split_*_results_final.pkl` 当最佳结果**。没有最佳患者预测时，需要先在原版 SlotSPE 运行路径导出，不用最终 epoch 文件替代。PKL 只读取自己可信的实验产物。

若数据集合法过滤了没有 RNA 的患者，需在 manifest 里显式提供 `expected_val_ids`，它必须是 split 中 val 的子集，并填写 `filter_reason`；默认严格要求全部 val 患者，工具不会自动忽略缺失患者。完整五折还会检查验证患者互斥、各 CSV 共享同一患者集合，以及原 val 五折是否恰好覆盖整个集合。

若 Full 原文件无法恢复或 replay 不符，再使用 `plan --group controls-full` 在新的目录下跑 **30 次**，得到同一批 Full/Direct/Independent。已有结果能正确恢复时不需要这一步。

## 3. 先做一个 fold 的导出检查，再导出全部

核心清单审计通过后，先选 **BLCA Full fold0**。这里是服务器真实推理，会载入病理特征和 RNA：

```bash
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL" export --manifest "$EVIDENCE/core.json" \
  --arm exp6 --cancer blca --fold 0 --device cuda:0 \
  --output "$EVIDENCE/export_check" --km --profile
$PY "$TOOL" plot --manifest "$EVIDENCE/core.json" \
  --exports "$EVIDENCE/export_check" --output "$EVIDENCE/figures_check"
```

逐项核验通过才会生成 `export.json`：严格加载所有 checkpoint 参数；检查训练折的参考 buffer；按最佳 epoch 恢复 OT 温度；每位患者的风险值与保存的最佳结果一致；`α=0` 重放原预测；eval 导出不更新模型 buffer。任何不符都会停止。源代码版本记录为训练 commit 与导出 commit 两列，旧训练版本不一致不静默当成同版本；逐患者 replay 是必要的执行检查。

全量导出用**新目录**，避免和烟测产物重复：

```bash
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL" export --manifest "$EVIDENCE/core.json" \
  --arm exp6 --arm direct --arm independent --device cuda:0 \
  --output "$EVIDENCE/exports_core" --km --profile
$PY "$TOOL" plot --manifest "$EVIDENCE/core.json" \
  --exports "$EVIDENCE/exports_core" --output "$EVIDENCE/figures_core"
```

`--km` 会额外在**该折训练患者**上用确定性评估采样推理，用训练风险中位数固定 KM 阈值；训练患者结局不用于选阈值。不同折模型的风险不直接混合，KM 按 fold 画。`--profile` 用当前 checkpoint 标准 eval forward，5 次 warmup + 30 次计时，CUDA 同步，记录设备、patch 数、参数量、延迟和显存；包含原模型自带解释分支，不包括特征提取/数据读取。

如只需要运输/重建/槽解释，可不加 `--km --profile`，减少推理。默认病例是该折患者 ID 排序后的前三位，不按生存结果挑选。也可用 `--case-id TCGA-...` 明确选择病例；选择时最好先限定癌种与 fold，确保属于该折。

## 4. 会生成哪些图、每张图证明什么

| 输出 | 所需数据 | 能回答的问题 |
|---|---|---|
| `scores_*.pdf/png` + `scores.csv` | 各臂真实 5 折最佳患者结果 | 整体 C-index 与折间波动；不是独立外部测试 |
| `paired_*` | Full 与对照同患者/结局/split/共同训练条件 | 每一折提高多少、平均提高多少个百分点 |
| `plan_sweep_*` | Full checkpoint 的验证患者推理 | 固定网络，逐渐替换运输计划是否改变风险、C-index、重建误差 |
| `slot_diagnostics_*` | WSI 与 Omics slots / hazard | 是否出现高度相似、距离小、预测差异小的槽坍缩 |
| `collapse_comparison_*` | 至少两组 Exp2/Exp3/Full 的完整 5 折导出 | 同折比较槽诊断，Exp2→Exp3 优先用于检验 diversity |
| `case_*_plans` | 全部 stage × geometry 的 OT 计划 | 耦合在患者/阶段/几何分支上的具体结构 |
| `case_*_pathways` | 真实 pathway 名、池化权重、重建误差 | 哪些通路参与槽表示，它们的潜在 token 重建误差 |
| `case_*_spatial` | **配对坐标**，可选 WSI 缩略图 | 不同 WSI 槽关注哪些实际采样位置 |
| `km_*` | `export --km` | 训练阈值固定后的验证患者高/低风险生存曲线 |
| `brier_*` / `calibration_*` | `--km` 与足够随访支持 | 离散 bin 时间点的删失校正误差与校准 |
| `efficiency_*` + `profiles.json` | `--profile` | 同设备、输入规模与 forward 定义下的计算成本 |

图同时输出矢量 PDF 与 300 dpi PNG。`figure_index.json` 逐图说明证据边界并列出未生成原因；没有资源或统计支持的图会明确跳过，不填假值。完整 5 折之前不输出该臂的“5 折均值”。

运输 sweep 定义 `Tα=(1−α)T+α[T1·(Tᵀ1)ᵀ/ΣT]`，使用**实际 factual plan 的边缘质量**，每个 stage/geometry 都替换，`α=0,.25,.5,.75,1`。近似 Sinkhorn 的数值边缘与理论目标可能有极小偏差，使用实际边缘保证替换前后相同。它是同一模型里的干预，不等同于重新训练 Independent，也不预设曲线必须单调。Direct 的主预测仍用 OT，只改变训练 cross 重建输入；单在 eval 切 `cross_mode` 不能证明主预测收益。

重建目标是**编码后的 pathway token**，不是原始基因表达，且目标编码器可随训练变化。因此不同模型的重建误差不直接当作谁更准确的排名；优先报告 Full 内部的计划替换响应，再结合 Direct/Independent 训练对照的 C-index。

## 5. 病理空间图的坐标准备

每个特征文件必须有对应的 `N×2` NumPy `.npy` 坐标，行顺序与当时生成该 `.pt` 特征时的 patch 顺序完全一致。不要用重新切片且顺序不同的坐标。病例导出已记录实际特征路径、文件 hash、多切片拼接顺序及采样 patch 行号；padding 用 −1 标记并排除。

建立 `coordinates.json`（可放在 `$EVIDENCE`），每个 feature 一项：

```json
{
  "features": [{
    "feature_sha256": "从该病例export.json读取的真实feature哈希",
    "coords": "/实际/slide_coords.npy",
    "coords_sha256": "该npy文件的sha256",
    "thumbnail": "/实际/slide_thumbnail.png",
    "thumbnail_scale": [0.03125, 0.03125]
  }]
}
```

`thumbnail_scale` 是**坐标单位到缩略图像素**的 x/y 乘数，必须按实际缩略比例填写；坐标约定是 level-0 patch 左上角 `(x,y)`，坐标与缩略图使用同一原点。没有原始 WSI/缩略图，可以省略 `thumbnail` 和 `thumbnail_scale`，输出真实 patch 坐标散点图，不伪造病理背景。H5 `coords` 可在服务器转换为 `.npy`，同时记录原始 H5 的路径/版本，确保特征与坐标来自同一次提取。

文件 hash 可以这样查看：

```bash
sha256sum /实际/slide_coords.npy
```

重新绘图时使用新输出目录：

```bash
$PY "$TOOL" plot --manifest "$EVIDENCE/core.json" \
  --exports "$EVIDENCE/exports_core" --coordinates "$EVIDENCE/coordinates.json" \
  --output "$EVIDENCE/figures_with_spatial"
```

注意：共享 prototype 对局部槽的 `K×K` 权重不是 patch 热图。本工具捕获局部 Slot Attention 最后一次 pooling 权重，再与 prototype 权重组合成 `K×N` 的解释图。该图是描述性 attention rollout，不是每个像素/patch 的因果贡献；阶段标签也是潜在运输阶段，不自动等同于临床分期。

## 6. 有需要再跑的补充项

旧 Exp4/Exp5 单分支如果用了 0.025，不能与 Full 的每支 0.05 混在一起宣称单支贡献或协同作用。确需回答这个问题时，生成**权重匹配的另外 20 次**清单：

```bash
$PY "$TOOL" plan --group matched-recon --gpu 0,1 \
  --results-root "$PWD/results/v313_matched_recon_v2" \
  --output "$EVIDENCE/matched_recon_plan.json"
```

确认后同命令加 `--execute`；完成后 `bind` → `audit` → `merge` → `export` → `plot`。新 Exp4 仅 self=0.05，Exp5 仅 cross=0.05，关闭另一分支；需要 Full 同样每支 0.05、共同训练条件一致。绘图总览也保留 source/overrides，因此不要把旧单分支曲线更名成新实验。

若要验证 diversity 的可视化，先恢复已有 Exp2/Exp3 的 checkpoint 与最佳患者结果，导出 5 折即可，不默认重训。多 seed 可用 `plan --seed 5 --seed 7`，会按所选臂展开；当前 1 个 seed 的图不声称多 seed 稳健。

**推荐实际执行顺序：20 次控制臂 → 核验现有 Full 和 SlotSPE → Full 单折导出检查 → 全折导出/绘图 → 需要时补权重匹配单支与 Exp2/Exp3 可视化。** 不需要重新投放整个 110 项矩阵。
