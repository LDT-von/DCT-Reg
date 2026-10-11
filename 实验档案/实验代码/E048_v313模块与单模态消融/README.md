# E048：DCT v3.13 模块与单模态消融

本实验目录用于回答三个问题：病理和基因分别提供多少预测信息；v3.13 的运输与读出结构是否有贡献；两条重建支路及辅助损失是否必要。这里交付的是可运行代码和检查，不是新的实验成绩。**本次未启动真实数据训练，也没有生成新的真实 C-index。**

这是新增的 E048，历史实验和代码保持原位置。每次运行必须使用新的批次名，脚本拒绝覆盖或续跑已有批次。训练由使用者显式启动。

## 实验臂与比较含义

默认 `--group first` 是以下 7 臂。默认队列 BLCA、KIRC，折号 0–4，随机种子 3，共 70 个任务。所有任务使用同一批配对患者和原有五折划分。

| 臂名 | 有效输入与改动 | 回答的问题 |
|---|---|---|
| `full` | 完整 v3.13；在 E048 中重新训练 | 当前批次的配对参照 |
| `wsi_only` | 仅病理参与预测和训练目标；保留病理编码、局部槽、原型、逐槽 NLL、多样性和排序；无重建 | 病理单模态能达到多少 |
| `omics_only` | 仅基因参与预测和训练目标；保留组学编码、局部槽、原型、逐槽 NLL、多样性、排序及系数 0.05 的 self 重建 | 基因单模态能达到多少 |
| `concat` | 两模态表征简单拼接风险预测，跳过 OT 和事件读出；保留两模态槽、原型、排序、逐槽 NLL、多样性、self 及 direct cross 重建 | 完整融合管线比简单拼接管线多贡献多少 |
| `no_self` | 从 Full 关闭 self 重建，cross 的系数仍为 0.05 | self 支路的独立作用 |
| `no_cross` | 从 Full 关闭 cross 重建，self 的系数仍为 0.05 | cross 支路的独立作用 |
| `no_reconstruction` | 从 Full 同时关闭两条重建支路，保留其余结构和辅助目标 | 全部重建监督的作用 |

单模态臂是只使用一种模态的模型，不是把训练完成的双模态模型另一种输入置零。病理单模态不加入 WSI 自重建，因为这不属于 v3.13 的训练配方。`concat` 有辅助监督，是融合管线对照；不能将它写成“没有任何辅助损失的 plain concatenation”。它的逐槽 NLL 系数为 0.05、多样性为 0.10、self 为 0.05、direct cross 为 0.05。

`--group structure` 包含 `full` 和下面 7 臂；`--group loss` 包含 `full` 和下面最后 3 臂。`--group all` 合并去重，共 17 臂；默认两队列、五折、seed 3 共 170 个任务。

| 分组 | 臂名 | 精确对照 |
|---|---|---|
| structure | `direct` | cross 重建直接读取病理槽，检验运输对齐后的重建输入 |
| structure | `independent` | 使用独立耦合，同时影响预测和重建；这是联合耦合控制，不是仅删除一条损失 |
| structure | `no_prototypes` | 直接使用患者局部槽，跳过共享原型坐标 |
| structure | `shared_stage_plan` | 四个阶段复用 stage 0 的成本和边际；仍保留 4 个事件 token、风险头和 4 个时间区间输出 |
| structure | `single_geometry` | 只使用 cosine 几何的信息，将该通道复制为三个接口；检验多几何信息，不用于宣称减少了三分之二计算量 |
| structure | `no_event_encoder` | 去掉事件 Transformer，保留阶段嵌入、LayerNorm、gate 和 hazard 读出 |
| structure | `pooled_transport` | 均匀阶段 gate 汇聚运输，将 WSI 运输到组学槽后与组学槽均值拼接，以 MLP 预测 |
| loss | `no_rank` | 从 Full 关闭 IPCW 排序辅助项 |
| loss | `no_slot_nll` | 从 Full 关闭逐槽生存监督 |
| loss | `no_diversity` | 从 Full 关闭槽多样性辅助项 |

这 17 个名称是固定接口。可以通过重复 `--arm` 或逗号列表选择更小的实验组合，但不支持任意未知臂名。

## 固定配方与配对原则

基础配置是 `configs/base.yaml`：UNI2-h，病理特征维度 1536，投影维度 256，最多 2048 个 patch，8 个病理槽和 8 个组学槽，4 个生存时间区间，DSS 终点，batch size 8，30 个 epoch，学习率 0.0005，AdamW、cosine scheduler、seed 3。事件分层采样和只用训练患者拟合分箱的开关也显式写入配置。

重建采用 `dct_v313_recon_weighting: per_branch`。Full 的 self 和 cross 原始系数各为 `0.10 × 0.50 = 0.05`，两臂共用从 epoch 2 开始、5 个 epoch 的渐增日程。去掉一条支路时，剩余支路仍为 0.05，**不会因只剩一条而重新归一化到 0.10，也不会降到 0.025**。

Full 的逐槽 NLL 系数为 0.05，多样性系数为 0.10，排序系数为 0.10。历史 Full 配置、隐式默认值或多样性系数可能不同，因此必须在 E048 中重新训练 Full；不能把历史已报告均值直接复制成这批实验的参照。

所有同队列、同折、同 seed 的臂使用相同的划分文件。即使训练单模态，也保留与 Full 相同的配对患者集，不为某个臂单独换成更容易的患者子集。调度器在启动前检查全部 CSV，记录 SHA256；每个任务启动前再次检查哈希，患者重复或跨分区泄漏会阻止执行。TCGA 样本或切片条码统一到患者条码检查泄漏。

比较时应使用相同队列、折号、seed、终点、特征、训练配方及 checkpoint 选择规则。`paired_differences.csv` 报告同折 Full 与消融臂的差值，不能用两个不同患者集上的均值之差代替配对差值。不同 seed 分别报告；不要将五折视为独立外部队列。

## 验证分数和独立测试分数

默认 `--protocol legacy_val` 读取：

```text
<data-path>/splits/<split-set>/<cancer>/fold_<fold>.csv
```

CSV 至少包含非空 `train,val` 两列，患者互斥；若已有 `test` 列，也检查与前两者互斥。该协议报告的是按验证集选择 checkpoint 后的 **best validation C-index**，不能称为未参与选择的独立测试成绩。

`--protocol outer_test` 必须明确提供已有的内外划分：`inner_train,inner_val,outer_test` 三列均非空且患者三方互斥。训练和分箱只使用 inner train，checkpoint 选择使用 inner val，选定后在 outer test 评估。缺少这些列时立即拒绝。仅将 `train,val,test` 改一个协议名称，或者拿已有 best validation 分数当 test 分数，都不满足这个协议。

`--split-set` 可以是划分集名称，也可以是包含 `<cancer>/fold_<fold>.csv` 的绝对目录。代码不会自动生成新的外部测试划分。**best validation 和 outer test 的数值、摘要、差值不允许混在一张均值表中。**

## Linux 服务器运行

以下命令在项目已有依赖的环境运行。路径按实际服务器位置修改。先查看计划；默认模式不会启动 worker，也不会创建结果批次。选用 `--plan-out` 才会保存独立的计划 JSON，并拒绝覆盖既有文件。

```bash
cd /data1/DCT-Reg
EXP="实验档案/实验代码/E048_v313模块与单模态消融"

python "$EXP/run_experiments.py" \
  --group first --cancer blca,kirc --fold 0,1,2,3,4 --seed 3 --gpu 0 \
  --batch-name e048_first_uni2h_seed3 \
  --results-root /data1/DCT-Reg/results/e048_v313_ablation \
  --data-root /data1/TCGA-UNI2-h-features \
  --data-path /data1/dataset_csv --split-set 5fold_uni2h \
  --protocol legacy_val
```

确认数据和划分就绪后，以下命令才显式启动首批 70 个任务：

```bash
python "$EXP/run_experiments.py" \
  --group first --cancer blca,kirc --fold 0,1,2,3,4 --seed 3 --gpu 0 \
  --batch-name e048_first_uni2h_seed3 \
  --results-root /data1/DCT-Reg/results/e048_v313_ablation \
  --data-root /data1/TCGA-UNI2-h-features \
  --data-path /data1/dataset_csv --split-set 5fold_uni2h \
  --protocol legacy_val --execute
```

仅打印过计划不会占用这个批次名；执行过的批次，无论成功、失败还是中断，都需要换新名字，脚本没有自动续跑机制。调度器串行启动任务；`--gpu 0,1` 只是轮流指定物理 GPU，始终最多一个训练进程。worker 内使用逻辑 GPU 0，外部设置 `CUDA_VISIBLE_DEVICES`，避免物理和逻辑编号混用。

如需全部 17 臂，将 `--group first` 改为 `--group all`，并使用新的批次名。按结构或损失单独运行则使用 `--group structure`、`--group loss`。每组都包含重新训练的 Full。

仅做部分任务时可以使用 `--arm full,wsi_only,omics_only`、`--cancer blca`、`--fold 0`。这种批次不构成五折证据。`--max-epochs 1` 是供使用者自行排查运行环境的短程选项，必须使用独立的调试批次名；不能将其成绩混入 30-epoch 论文比较。

独立测试模式的示例先只打印计划。假定下面目录已有按训练协议准备好的三分区 CSV：

```bash
python "$EXP/run_experiments.py" \
  --group first --cancer blca,kirc --fold 0,1,2,3,4 --seed 3 --gpu 0 \
  --batch-name e048_outer_test_seed3 \
  --results-root /data1/DCT-Reg/results/e048_v313_ablation \
  --data-root /data1/TCGA-UNI2-h-features --data-path /data1/dataset_csv \
  --split-set /data1/dataset_csv/splits/v313_inner_outer \
  --protocol outer_test
```

需要实际训练时自行增加 `--execute`。未提供真实独立测试划分前，不能执行上述协议。

## 产物与汇总

每个新批次保存完整 `plan.json`，明确所选臂、队列、seed、折号和任务划分哈希。每项任务保存独立 `config.yaml`；首次构造模型后，worker 生成 `task.json` 记录继承默认值和实验开关生效后的公共参数及来源。`model_audit.json` 记录实际损失系数与模型参数量，`label_audit.json` 记录实际评估患者及结局，完整训练和核验成功后才写 `completion.json`。训练日志位于批次的 `logs/`；任务输出按 `protocol/arm/cancer/foldN/seedN/` 分开。

结局核验同时保留数据表原始生存时间和数据加载器实际使用的 float32 目标时间；预测逐项核对后者，避免把正常精度转换误判为标签不同。真实的患者、结局或删失状态差异仍会拒绝。

进程非零退出、划分改变、缺少完成文件、完成状态不等于 `complete`，或 `training_complete` 不为布尔值 `true`，都会判定失败并停止后续任务。出现失败需检查对应日志，不能拿部分训练产物填入完整实验表。

训练完成后执行汇总。汇总输出目录必须全新：

```bash
python "$EXP/summarize.py" \
  --batch-dir /data1/DCT-Reg/results/e048_v313_ablation/e048_first_uni2h_seed3 \
  --output-dir /data1/DCT-Reg/results/e048_v313_ablation/e048_first_uni2h_seed3_summary
```

汇总生成 `report.json`、`fold_results.csv`、`five_fold_summary.csv`、`paired_differences.csv`、`paired_five_fold_summary.csv` 和说明文件。它核验实际评估患者与结局、配置和划分来源及预测文件哈希，并由预测重新计算 C-index。只要同队列、seed、协议的任一选中臂缺折或核验不通过，该组五折均值和成对均值保持空值，返回非零状态；不会将已完成两三折包装成五折均值。

## 代码检查与证据边界

调度检查只读取临时 CSV、构造计划并模拟子进程，没有加载真实特征或训练模型：

```bash
python -m unittest discover -s "$EXP/tests" -p test_schedule.py -v
```

模型、worker 和结果汇总的测试在同目录 `tests/` 中。结构测试或合成输入上的梯度检查，只能说明代码路径可以执行、相应开关生效；不能证明模块提升了真实患者上的生存预测。最终论文应根据本批公平训练和真实预测的配对结果决定保留哪些贡献表述。

完整的本次检查命令如下，包含 v3.13 原有回归测试；截至 2026-10-11，在最新远端代码的独立工作副本上 **106 项通过**。细节见 `代码核验_20261011.md`。

```bash
python -m pytest "$EXP/tests" tests/test_dct_v313_transport_reconstruction.py -q
```
