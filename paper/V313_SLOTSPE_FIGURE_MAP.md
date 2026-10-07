# DCT v3.13：SlotSPE 风格论文图执行计划

更新日期：2026-10-07。目标是用**已有 v3.13 checkpoint 和真实患者数据**补齐解释图与总体 KM。代码最低入口基线为包含 3efe824 的 main；实际运行记录最新完整 commit，不把 3efe824 当作永久最新版本。

## 1. 范围与成果

Full 必须由训练配置、overrides、实际模型与 checkpoint 共同确认，完整配方为 transport + learned。exp6/full 只是可能的实验标签。沿用已有 legacy_val、UNI2-h、seed=3 的 BLCA/KIRC 五折，不混其他编码器、split 或版本。

允许只读核对、已有 checkpoint 推理、attention/KM 导出、真实作图及必要代码修复。禁止新训练、110 次 outer_test、reset、强制覆盖、删除旧实验目录或修改旧结果。V313_EVIDENCE_RUNBOOK.md 中的训练命令不属于本次授权。

v3.13 的 Exp0–Exp6、Direct、Independent 和 SlotSPE 基线比较属于本论文实验，可保留。此次解释主图与 KM 使用核实后的 Full。旧图清理见 V313_FIGURE_CLEANUP_20261007.md；服务器已有旧产物仍须保留。

| 成果 | 必需数据 | 用途 |
|---|---|---|
| BLCA/KIRC 每癌种总体 KM | 五折验证预测、结局、各折训练风险中位数 | 展示高低预测风险组的生存曲线 |
| 逐槽 Top-3/Bottom-3 通路及通路×槽矩阵 | 真实 attention_omic、真实通路名称 | 展示组学槽关注的通路 |
| 学习 OT、同事实边际独立计划及差值 | 同病例 factual plans、stage gate | 展示运输关联结构 |
| 四风险组 Top-10 通路并集图 | 完整五折 attention_omic、折内风险百分位 | 展示通路关注与预测风险的队列关联 |
| 真实组织—通路病例图 | WSI/缩略图、坐标或索引图片、病例导出 | 连起组织位置、槽与命名通路 |
| 其他癌种总体 KM（可选） | 同协议、现成完整五折 v3.13 Full 产物 | 扩展癌种覆盖 |

前五项优先。组织资源缺失时继续完成其余图，并明确病例主图未完成。已有 v3.13 架构、七臂损失消融、逐折机制对照、transport sweep 保留；旧解释图与合成排版样例不计为本次新图。

## 2. SlotSPE 参考与适配

以下沿用前一轮对用户提供的 Structural Prognostic Event Modeling for Multimodal Cancer Survival Analysis（36 页）的阅读记录。

| SlotSPE 图 | 原图内容 | DCT 对应 |
|---|---|---|
| Fig.3 / Fig.6 | 多癌种总体高低风险 KM | 每癌种一张 Full 总体图，每名患者一次 |
| Fig.5 | 槽空间分区、Top-5 组织块、Top/Bottom-3 通路 | 真实组织—通路病例主图 |
| Fig.12 / Fig.13 | 原切片、两类槽分区、通路×槽矩阵、逐槽通路 | 病例主图和逐槽通路补充图 |
| Fig.14 | 四组 Top-10 通路并集热图 | 按预测风险划分的四组通路图 |
| Fig.1 / Fig.2 | 总架构与模块细节 | 保留实际 v3.13 OT 和重建架构 |
| Fig.4 / Fig.10 / Fig.11 | 计算成本、校准、DCA | 需要独立可核验数据，此次不默认扩展 |

原文 Fig.14 按生存时间分组，并用 MoE retention 选槽。当前 DCT 使用真实组学 attention 跨槽等权平均，按折内预测风险 midrank 分组，不能称为原实验复现。

## 3. 安全更新与执行记录

1. 检查 /data1/DCT-Reg 的 branch、HEAD、本地修改、已跟踪和未跟踪产物，保存状态记录。不要 git add .。
2. git fetch origin，确认 origin/main 含 3efe824，查看待更新文件与删除项。
3. 只有无冲突、可安全快进且不会移除需保留的实验产物时，才更新原工作区。若有修改、分叉，或新版删除路径与服务器现存旧图/产物相交，保留原目录，在新的 /data1/DCT-Reg-v313-figures-<本次标识> 创建基于 origin/main 的独立代码工作区。不默认 stash，不 reset，不覆盖；数据仍从 /data1/DCT-Reg 读取。
4. 使用实际代码目录运行，记录 CODE_ROOT、DATA_ROOT、HEAD、Python/PyTorch/CUDA 环境，优先 trisurv。缺依赖记录具体缺项。
5. 新记录、导出和图像采用新的本次目录。冲突则换新的标识，不能清空旧目录。保存命令、日志和真实失败状态。

以下变量由执行者根据服务器实际情况填写，必须是绝对路径：

~~~bash
CODE_ROOT=/data1/DCT-Reg
DATA_ROOT=/data1/DCT-Reg
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
SESSION=v313_slotspe_20261007_v2
TASK_RECORD="$DATA_ROOT/results/$SESSION"
FRESH_EXPORT_ROOT="$TASK_RECORD/exports_new"
CASE_EXPORT_ROOT="$TASK_RECORD/cases_new"
COHORT_EXPORT_ROOT="$TASK_RECORD/approved_exports"
FIG_ROOT="$DATA_ROOT/paper/figures/$SESSION"

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT"
~~~

CODE_ROOT 可以是独立代码工作区。TASK_RECORD 等按冲突规则处理；任何命令执行前先确认变量已正确设置。

## 4. 锁定真实 Full 与病例所属折

读取 /data1/DCT-Reg/results/v313_evidence_v2/full_manifest.json。若缺失，先只读寻找现成清单和原始产物，按真实路径恢复**新清单**；不训练，不编造来源。

逐 run 核对：

- id、实际 arm、cancer、fold、seed、protocol；
- source_commit、当时 config、overrides、实际构造的模型和 Full 开关；
- checkpoint、最佳 epoch 曲线、**最佳 checkpoint**的患者预测、split_csv；
- 患者集合、结局、SHA-256、UNI2-h 特征来源；
- 实际 v3.13、cross_mode=transport、plan_mode=learned，不能将控制臂标成 Full。

遗留 v311 参数名可能被 v3.13 继承，不能只按参数前缀判版本。最佳患者预测不能由 final epoch 文件替代。

只选 BLCA/KIRC、fold0–4、seed=3、legacy_val 的十个唯一 Full run，保存 "$TASK_RECORD/full_v313_selected.json"，保留原标签与来源。缺项写明，不借其他 seed/版本/编码器补齐。

~~~bash
V313_MANIFEST="$TASK_RECORD/full_v313_selected.json"
"$PY" scripts/prepare_v313_evidence.py audit \
  --manifest "$V313_MANIFEST" \
  --output "$TASK_RECORD/full_v313_audit.json"
~~~

audit passed 还须配合独立完整性核对：十个 run 齐全、各癌种五折唯一、验证患者无重复、总患者集合一致。不能用不完整清单的 passed=True 宣称完成。

以 split_csv 与最佳患者预测 ID 交叉确认 TCGA-2F-A9KP，记录真实 fold、run ID 和验证集归属；**不预设 fold1，不从图片目录反推**。不在当前 Full 验证集合则该病例停止，其他图继续。

KIRC 固定选择规则：按验证患者 ID 升序，选择首个 replay 通过且组织资源可核验的病例，记录资源筛选依据，不按热图或最高分挑人。资源未确认时先完成 BLCA 非组织图及两个癌种队列图。

## 5. 复用或补齐 attention / KM

合格的既有导出需满足：export.json 的身份与选定 run 一致，checkpoint/config/split/预测来源与 hashes 对齐，验证患者齐全，patients.npz 有真实 attention_omic、pathway_names、risk/time/censor，以及 km_train_median 和对应 km_high。病例还需 case NPZ、attention_wsi、plans、gate、原始 patch/slide 索引与特征来源记录。

缺训练阈值不能用验证/全队列中位数代替。总 attention=1 柱图、重建误差或 hazard 乘权不替代真实 attention。

只补缺失或不合格 run，使用新输出根。以下 BLCA_ARM、BLCA_FOLD 由来源核验填写，选定清单须恰好命中一个 run：

~~~bash
"$PY" scripts/prepare_v313_evidence.py export \
  --manifest "$V313_MANIFEST" --output "$FRESH_EXPORT_ROOT" \
  --arm "$BLCA_ARM" --cancer blca --fold "$BLCA_FOLD" \
  --device cuda:0 --alphas 0 --km
~~~

若十折均无合格导出，才一次处理选定十折：

~~~bash
"$PY" scripts/prepare_v313_evidence.py export \
  --manifest "$V313_MANIFEST" --output "$FRESH_EXPORT_ROOT" \
  --cancer blca --cancer kirc --device cuda:0 --alphas 0 --km
~~~

alphas 0 只导出 factual 数据，不重做已有 sweep。km 包含该折训练患者阈值推理，不是新训练。

若 A9KP 未在 cases 中，只对核实所属折补病例：

~~~bash
"$PY" scripts/prepare_v313_evidence.py export \
  --manifest "$V313_MANIFEST" --output "$CASE_EXPORT_ROOT" \
  --arm "$BLCA_ARM" --cancer blca --fold "$BLCA_FOLD" \
  --case-id TCGA-2F-A9KP --device cuda:0 --alphas 0 --km
~~~

病例 ID 不能传给全部五折或两个癌种。RUN_BLCA 设置为实际包含 export.json 的精确目录，不写死 exp6_blca_f1_s3，不取 glob 第一个目录。

cohort/KM 共用只有十个已核验 run 的 COHORT_EXPORT_ROOT。可直接复用恰好完整的旧根；若分散则创建新聚合目录，只放十个选定 run 的必要文件，保留原文件与来源映射。不混重复补病例、备份或其他 seed。采用文件链接时确认扫描能看到 export.json，不假定 rglob 遍历目录软链。相对路径按原来源解释，不能聚合后指向错误文件。

记录每 run 的复用/补导出、原目录、使用目录及 hashes。

## 6. 先生成不需要组织图片的图

PLOT_ARM_BLCA/PLOT_ARM_KIRC 根据真实标签填写。当前聚合入口把元数据 full 识别为 exp6，故 full 标签的绘图参数为 exp6；其他可识别标签保留原值，不悄悄改 manifest。

先 check-only，再生成 BLCA 通路与 OT：

~~~bash
"$PY" scripts/plot_v313_slotspe_style.py pathways \
  --export "$RUN_BLCA" --case-id TCGA-2F-A9KP --check-only \
  --output "$FIG_ROOT/blca_a9kp_pathways"
"$PY" scripts/plot_v313_slotspe_style.py coupling \
  --export "$RUN_BLCA" --case-id TCGA-2F-A9KP --check-only \
  --output "$FIG_ROOT/blca_a9kp_transport"

"$PY" scripts/plot_v313_slotspe_style.py pathways \
  --export "$RUN_BLCA" --case-id TCGA-2F-A9KP \
  --output "$FIG_ROOT/blca_a9kp_pathways"
"$PY" scripts/plot_v313_slotspe_style.py coupling \
  --export "$RUN_BLCA" --case-id TCGA-2F-A9KP \
  --output "$FIG_ROOT/blca_a9kp_transport"
~~~

输出 slot_pathways、pathway_slot_map、transport_association 的 PNG/PDF/来源 JSON。attention 均匀、OT 差值小也保留，不人为增强。

两个癌种完整五折分别运行：

~~~bash
"$PY" scripts/plot_v313_slotspe_style.py cohort \
  --exports "$COHORT_EXPORT_ROOT" --cancer blca \
  --arm "$PLOT_ARM_BLCA" --seed 3 --top 10 --check-only \
  --output "$FIG_ROOT/blca_cohort"
"$PY" scripts/plot_v313_slotspe_style.py cohort \
  --exports "$COHORT_EXPORT_ROOT" --cancer kirc \
  --arm "$PLOT_ARM_KIRC" --seed 3 --top 10 --check-only \
  --output "$FIG_ROOT/kirc_cohort"

"$PY" scripts/plot_v313_slotspe_style.py cohort \
  --exports "$COHORT_EXPORT_ROOT" --cancer blca \
  --arm "$PLOT_ARM_BLCA" --seed 3 --top 10 \
  --output "$FIG_ROOT/blca_cohort"
"$PY" scripts/plot_v313_slotspe_style.py cohort \
  --exports "$COHORT_EXPORT_ROOT" --cancer kirc \
  --arm "$PLOT_ARM_KIRC" --seed 3 --top 10 \
  --output "$FIG_ROOT/kirc_cohort"
~~~

不默认 allow-partial；缺折则列为缺失，不用于完整论文队列图。

四组按折内风险 midrank，Q1 较低风险、Q4 较高风险，ties 保留相同百分位。通路值是 attention_omic 跨槽等权平均，各组 Top-10 取并集，最多 40 条，不保证并集只有 10 条。输出真实组均值、相对四组均值的差值图、patient_groups.csv、pathway_group_means.csv；后者可保留全通路均值供核对。

## 7. 真实组织资源与病例主图

只读查找 WSI、缩略图、坐标或带原始行号的 RGB patch 图片，优先沿病例特征路径与提取记录定位，不下载新数据。

按 configs/v313_slotspe_assets.example.json 契约填新的 "$TASK_RECORD/real_wsi_assets.json"，不覆盖模板，核对：

- 病例、slide_id 与 exported feature_sha256 匹配；
- 坐标顺序与 UNI2-h 特征行一致，有 extraction_provenance.evidence_path；
- 行数相等不足以证明顺序一致，需同次提取的索引/日志/文件证据；
- 坐标原点/轴/level、patch 读取 size、downsample、缩略图尺寸齐全；
- 原始 patch 行可定位真实组织块，padding/重复采样按导出索引处理；
- 文件 hashes 可核对，没有真实病理标注不制造 Tumor/Stroma 标签。

~~~bash
ASSETS="$TASK_RECORD/real_wsi_assets.json"
"$PY" scripts/plot_v313_slotspe_style.py case \
  --export "$RUN_BLCA" --case-id TCGA-2F-A9KP \
  --assets "$ASSETS" --slide-index "$BLCA_SLIDE_INDEX" --slots 0,1,2 \
  --check-only --output "$FIG_ROOT/blca_a9kp_tissue"

"$PY" scripts/plot_v313_slotspe_style.py case \
  --export "$RUN_BLCA" --case-id TCGA-2F-A9KP \
  --assets "$ASSETS" --slide-index "$BLCA_SLIDE_INDEX" --slots 0,1,2 \
  --output "$FIG_ROOT/blca_a9kp_tissue"
~~~

BLCA_SLIDE_INDEX 按真实映射填写，不固定取 0。槽 0/1/2 为预先固定展示顺序；实际槽数或真实 patch 不满足时记录失败。

KIRC 用固定 KIRC_CASE_ID、RUN_KIRC、KIRC_SLIDE_INDEX 同样生成 pathways、coupling、case；必要时单折补病例导出并记录，不随结果更换病例。

目标内容为原切片、两类槽分区、逐槽空间热图、Top-5 真实组织块和命名通路。输出 wsi_slot_assignments、tissue_pathway_case 的 PNG/PDF/来源 JSON。若现代码缺原切片视图或版式不足，据真实资源修复展示。

投影遵循 v3.13 cross 分支：各阶段平均几何计划，按组学列质量归一化，再按 factual stage gate 汇总为 B；omics_spatial = B.T @ attention_wsi，WSI 槽到通路关联为 B @ attention_omic。相同编号两模态槽不天然一一对应。

缺资源时列实际查找位置、缺文件/映射、受影响病例，继续其他图。不拿合成组织、随机坐标或空白背景充当真实 WSI 解释图。

## 8. 每癌种总体 KM

每癌种一张 Full，仅高风险/低风险两条曲线。按所属折训练患者风险中位数分组后合并五折结局，每名患者一次。risk=-sum(survival)，越大（越接近0）风险越高，等于阈值归高风险。

~~~bash
"$PY" scripts/plot_fig5_km_curves.py \
  --exports "$COHORT_EXPORT_ROOT" --arm "$PLOT_ARM_BLCA" \
  --cancer blca --seed 3 --check-only
"$PY" scripts/plot_fig5_km_curves.py \
  --exports "$COHORT_EXPORT_ROOT" --arm "$PLOT_ARM_KIRC" \
  --cancer kirc --seed 3 --check-only

"$PY" scripts/plot_fig5_km_curves.py \
  --exports "$COHORT_EXPORT_ROOT" --arm "$PLOT_ARM_BLCA" \
  --cancer blca --seed 3 --output "$FIG_ROOT/km"
"$PY" scripts/plot_fig5_km_curves.py \
  --exports "$COHORT_EXPORT_ROOT" --arm "$PLOT_ARM_KIRC" \
  --cancer kirc --seed 3 --output "$FIG_ROOT/km"
~~~

包含删失标记、名义95% CI、在险人数、实际组人数、PNG/PDF、患者 CSV 和阈值/来源 JSON。不同折风险尺度不直接拼接计算 pooled C-index。legacy_val 用验证集选 checkpoint，称“五折开发集验证预测汇总”，不能称独立外层测试。

show-logrank 仅为可选探索性统计，CI/检验未校正交叉验证模型依赖。其他癌种只在现成、同协议、完整五折 v3.13 来源齐全时扩展；缺数据不重训。

## 9. 失败处理、验收与交付

旧 rerun_v313_export_for_attention.py 在备份已存在时可能直接返回，但仍打印 [backup]，不证明当前目录已搬移。非空检查针对实际 output_root/run_id，兄弟目录不会触发。正式入口配合新根目录，不放宽 rtol=1e-5、atol=2e-5 最佳预测对齐判据。

记录具体 run、命令、完整异常、现存文件和 commit；部分输出保留，重试换新根或独立目标清单。replay 不符查 checkpoint/最佳 epoch、config/overrides、数据顺序、patch 采样和环境，未通过 run 不计为核验成果。

实际打开所有 PNG，修复遮挡、截断、字号和色标，核对实际数值、病例信息与真实组织块。检查 PDF 能打开且文字/色条完整。空图、合成图和“脚本能运行”不替代真实交付。

阶段记录放 TASK_RECORD，最终报告保存 paper/V313_SLOTSPE_FIGURES_REPORT_<本次标识>.md，包括：

- 实际 commit、环境、执行命令、日志；
- 十折来源表、复用/补导出状态、replay 结果；
- 各癌种实际患者数、fold 覆盖、缺失与重复核验；
- 病例 ID、fold/slide、选择规则、资源与坐标对齐证据；
- 风险方向、KM 阈值、四组定义及组人数；
- 每张真实图 PNG/PDF 的绝对路径、预览、用途、来源 JSON/CSV；
- 已完成与缺失清单，不预填性能或生物学发现。

通路图完成不代表组织图完成，attention 五折齐全不代表 KM 阈值齐全，audit passed 不代表十折/患者全集齐全。

必要代码或文档修复通过相关检查后单独 commit、正常 push，验证远端 ref/tree。只提交相关修复、报告、核验后的轻量图和统计；大型 checkpoint、患者/病例 NPZ、WSI、特征、敏感原始数据和大日志不加入 Git。

## 10. 解释边界

attention 是模型注意力分配，OT 是运输关联，不能直接证明因果或模块预测收益。真实数值均匀、差值小也保留；不用重建误差、hazard 乘权或逐行 max 缩放制造差异。

四预测风险组与 SlotSPE 生存时间组不同。同编号 WSI/组学槽不默认对应，不同折同编号槽也不默认语义对齐；队列图用跨槽平均 attention。通路名称不是病理组织类型，低 attention 不等于生物学无关。模块有效性仍由匹配消融/对照结果支持，不由解释图美观程度替代。
