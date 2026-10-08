# v3.13：参考表格和两类图的对应实现（2026-10-08）

本轮范围：只使用 v3.13；核对已存统计、补代码及作图，不启动真实患者推理或训练。参考图来自用户上传的两个截图，不照抄其他论文的方法分数或模块名称。已有 main 诊断修复 a1a4584 保留。

## 1. 消融表：已生成真实记录版

当前结果：paper/tables/v313_training_variants_20261008/ablation_two_cohorts.png（另有 PDF/SVG）。展示 BLCA/KIRC 七组训练配方及 Direct/Independent 两个训练控制；红色第一、下划线第二。五折标准差使用 ddof=1；两癌种宏平均独立标为 Mean (2)。Full 在这两个癌种的九组记录中均值最高，但这是单 seed 最佳验证结果，不能自动推出每项损失独立有效或统计显著。

十癌种覆盖版：同目录 ablation_ten_cohort_coverage.png（另有 PDF/SVG）。Full 有全部 50 个折值；其余八行目前只有 BLCA/KIRC。缺失保持 NR，不能把其他模型的公开成绩填作 DCT 的消融。Mean (10) 只对完整十癌种行计算；只有一个可用数值的列不标第一或第二。

溯源：从正文附录 A 提取 70 个损失消融折值；Full 的 BLCA/KIRC 与十癌种输入逐项一致；读取已通过的 controls_audit.json 的 20 个机制控制折值。原值不改。输出 ablation_tables.json 保留逐折数据、源文件 LF 哈希和产物哈希，当前完整 Full/control 患者与配置配对审计仍待补齐。

| 我们的行 | 实际改变 | 能支持的结论 |
|---|---|---|
| Exp0–Exp3 | 同一 v3.13 架构上逐步添加排序、逐槽监督、多样性目标 | 训练配方收益；Exp0 不能称 Vanilla Slots |
| Exp3 | 同 Full 比较时无 self/cross 重建 | 联合重建目标的整体作用 |
| Exp4/Exp5 | 历史 self-only/cross-only，保留支路为 0.025 | 配方对照；不能直接称严格同权重 w/o cross/self |
| Full | self/cross 各 0.05，其余目标和实际结构保留 | 完整配方 |
| Direct | cross 解码输入为原始病理槽，主预测仍是学习 OT | 重建是否受益于运输对齐 |
| Independent | 边际乘积计划用于预测和 cross 重建 | 非独立联合耦合的作用，不能称完全无交互 |

参考截图里的 Selective Slot Attention、Iterative Cross-attention 是对方模块，不能直接换标题当成我们的消融。共享原型、阶段条件、多几何分支等结构如要分别证明，需要各自真实、同条件的控制实验。当前不预设这些缺失实验已完成。

复现现有表（新输出目录）：

```bash
python scripts/plot_v313_reference_panels.py ablation --output paper/tables/v313_training_variants_<new-id>
```

如服务器找到其他八癌种真实消融，只汇入同协议输入 JSON 的 variants[*].folds，保留 source，并经 --input 指定。历史 0.025 配方与后续 matched 0.05 配方分行，不混合。不能修改旧原始记录或正文成绩。

## 2. 槽—通路病例图：已有数据入口，已补组合布局

现有真实 BLCA TCGA-2F-A9KP 图位于 paper/figures/v313_slotspe_20261007_v2/blca_a9kp_pathways/：slot_pathways 与 pathway_slot_map。329 条通路、8 个组学槽。组合图新增为左侧完整通路×槽矩阵，右侧全部槽 Top-3/Bottom-3，并按槽着色；长通路名换行、行高适应，显示条形数值。

```bash
python scripts/plot_v313_slotspe_style.py pathways --export "$RUN_BLCA" --case-id TCGA-2F-A9KP --scale raw --output "$FRESH_RAW"
python scripts/plot_v313_slotspe_style.py pathways --export "$RUN_BLCA" --case-id TCGA-2F-A9KP --scale percentile --output "$FRESH_PERCENTILE"
```

两个输出目录都必须是新目录。RUN_BLCA 是服务器已核验包含该患者的 export.json 所在精确目录，不能凭患者图片目录猜 fold。只读已存真实 attention，无需重训。组合产物：pathway_case_panel_raw / pathway_case_panel_percentile 的 PNG/PDF/JSON；原有两类图继续输出。

默认保留原始注意力。percentile 使用每个槽内部的 midrank，所有 ties 同值；均匀注意力均为 0.5，不会人为分出 Top=1/Bottom=0。两张组合图都保存完整 raw_attention 和各槽原始 range。百分位展示排序，不能当绝对注意力强度；微小原始差别也可能占据大的百分位差，解释时必须核对原始值。Top/Bottom 叫高/低关注通路，不自动叫生物学相关/无关通路，更不能当因果贡献。

当前本机未保存病例完整 NPZ，因此不能以 Top/Bottom 的六条值编造完整矩阵。真实新版组合图需服务器读取旧 NPZ 后绘制；本机仅合成布局 QA。KIRC 及其他癌种可用同入口，需事先锁定合法病例，不能按最漂亮热图挑人。WSI/Top-5 组织图另需真实切片与行序坐标，第二个截图中的通路子图无需组织资源。

## 3. 性能—显存/时间图：代码就绪，实测结果待补

现有 profile_model 已测量同步墙钟耗时中位数/p95 和 GPU 峰值 allocated；新增输入指纹、特征维数、精度、CUDA/cuDNN 与 TF32/cuDNN 设置、实际测量患者。标准 eval+no_grad 前向，包含模型原生解释分支；不是额外运输 sweep。CPU 不伪造 GPU 显存；total peak allocated 以 MiB 表示，含驻留模型权重和输入。

DCT 可复用已有入口的 export --profile --alphas 0；该步骤会加载 checkpoint 并执行真实患者评估，仅由用户在服务器决定运行。外部方法必须是已锁定的 UNI2-h、DSS、同患者/划分的 checkpoint；可以用模型原生前向的薄适配器调用同一 profile_model，不手工缩减其计算图。当前 load_run 仅支持 DCT，外部原生模型加载/输入适配需在各官方实现中完成，不能声称当前 DCT export CLI 可加载所有基线。

输入模板 configs/v313_efficiency_tradeoff.example.json 包含 DCT、MOTCat、MCAT 的两癌种五折空记录。路径和哈希都是待填示例，无法充当测量结果。可加入已有合格方法，排除 SlotSPE 比较行。

每个测量点的 profile_json 按现有 export.json 格式包含：run.cancer/fold/seed/protocol/split_csv，hashes.predictions/checkpoint/split_csv，profile 为 profile_model 返回字段并加 benchmark_case_ids。外部 native payload 须包含共同 x_wsi 张量供基准函数记录；适配器仅转换接口，不加另一模型或省去计算。测量患者预先固定为该折验证 ID 字典序首位，所有方法、checkpoint 一致。DCT 旧导出的首位如不同则不可直接混合，须统一基准后另存新的来源记录。

作图器读取每折真实患者预测 pkl，重算 C-index，并检查完整五折、患者/结局/训练集合、profile 与预测关联、文件哈希、测量病例和实际 WSI 输入指纹；逐模型硬件、精度、patch 数、batch、特征维数、计时设置必须相同。未通过即不写图。横坐标为各折固定病例峰值显存/耗时中位数的宏平均，纵坐标为相同队列集合的五折 C-index 宏平均；每个方法一颗点，DCT 是星形。不会预设 DCT 更省或更好。

```bash
python scripts/plot_v313_reference_panels.py tradeoff --input "$REAL_MEASURED_JSON" --output "$FRESH_COST_FIGURES"
```

只有参考论文的成绩、其他 GPU 上的耗时或另一 patch 设置的测量时，不能拼成正式图；缺少同条件 checkpoint 则先报告缺项，不自动训练。少量固定病例是受控微基准，不能宣称整个患者群的临床吞吐；正文需如实注明。

## 4. 本轮验证与交付边界

55 项 focused tests 通过，包括现有证据、修正版诊断、解释和新增面板。两张真实记录表的 PNG 与单页 PDF 视觉核验通过；组合图两种 scale 和效率图只做明确标注的合成 QA，不属于模型结果。真实服务器推理/训练次数为 0；正文和既有外部比较表未改。

服务器提示词：paper/V313_REFERENCE_PANELS_SERVER_PROMPT_20261008.txt。旧诊断 v2 检索修复仍按另一个 DIAGNOSTIC_BUGFIX 提示词完成，本次注意力图与检索问题不是同一指标。

推送前整合远端 7bfec1f（十癌种 KM）和 acb7cbc（环境配置）。KM 的 10 个源 JSON LF 哈希和图片文件已只读核对，状态记录更新为 10/10 交付；这不改变当前消融仅两癌种完整的覆盖，也不代表本轮完成患者/checkpoint/split 级 KM 审计。
