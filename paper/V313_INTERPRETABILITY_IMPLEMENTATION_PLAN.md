# v3.13 可解释性与论文图：实施计划

状态：**设计稿；新增接口尚未实现。** 现有可运行入口是 scripts/prepare_v313_evidence.py；本文第 9 节给出现有命令，第 7 节是拟开发接口。本文不启动训练、推理或切片，也不要求重新跑 110 个 outer_test 任务。

目标：沿用当前 legacy_val、UNI2-h、seed=3 的已核验结果，把“有多少提升”“槽关注什么”“运输是否参与预测”分开呈现。先完成 BLCA/KIRC，再考虑扩展癌种。最低方案新增训练次数为 **0**；需要读取现有 checkpoint，在服务器执行导出与绘图。

## 1. 参考 SlotSPE 的哪些图，为什么这样改

参考文件：C:/Users/栋栋/Desktop/研一/多模态癌症生存预测论文清单/03_槽注意力与结构化事件/STRUCTURAL PROGNOSTIC EVENT MODELING FOR.pdf，ICLR 2026。页码是 PDF 页码。

| SlotSPE 图 | 原图对应实验 | 我们的对应方案 |
|---|---|---|
| Fig.1–2，p4–5 | 框架与组件示意 | DCT 框架图，呈现真实 shared slots、运输、预测、self/cross 重建；不作为性能证据 |
| Fig.5、12–13，p10、34–35 | 病例 WSI、槽空间图、Top-5 patches、组学通路 | 病例解释大图；增加实际 OT 矩阵和由 OT 投影的组学槽空间图 |
| Fig.14，p36 | 队列通路热图：按观察生存时间分四组，使用最高保留概率槽 | 队列通路热图：改为训练阈值定义的预测风险四组；明确这是不同分组方案 |
| Fig.3、6，p8、24 | 高低风险 KM，原文使用验证集风险中位数 | 使用每折训练风险中位数，在该折验证患者画 KM |
| Fig.4、10，p9、31 | 内存、时间、C-index 与模块开销 | 同硬件、同输入、同 forward 定义的效率图；暂列补充 |
| Fig.7–9，p27–28 | 编码器、槽数、迭代数、稀疏选择、温度消融 | 都需要额外训练；当前不列入最低方案 |
| Fig.11，p32 | 校准、DCA、临床分析 | 有时间支持和临床变量时再做，不默认画 60 月图 |

SlotSPE 的主性能与组件消融主要使用表格。我们保留主表，增加逐折连线图展示 Full 的提升。其专家标注不能照搬；没有病理专家审核时，我们只展示原始 patch 和模型注意力，不写“已识别肿瘤/免疫细胞”等诊断结论。

## 2. 交付图清单与优先级

| 编号 | 图与用途 | 所需输入 | 当前状态 | 优先级 |
|---|---|---|---|---|
| A | 架构图：解释完整 transport + learned | 模型实际计算路径与损失公式 | 需设计矢量图 | P1 |
| B | Full/Direct/Independent/Exp0/SlotSPE 主表与逐折提升图 | 经核验的最佳患者预测、split、历史配置 | 现有审计/绘图可复用 | P0 |
| C | 2–4 个病例解释大图 | Full checkpoint、同源坐标、WSI 或 patch 图、通路名 | 需补组合图与 OT 空间投影 | P1 |
| D | BLCA/KIRC 队列通路热图 | Full 全部验证患者的 pathway attention、训练风险阈值 | 需补全队列导出与聚合 | P1 |
| E | 运输替换干预图 | Full checkpoint、全部验证患者 | 已有 sweep，需完善展示 | P1 |
| F | 验证患者高低风险 KM | 验证风险、结局、训练风险中位数 | 现有实现可复用 | P1 |
| G | Exp2/Exp3/Full 槽诊断 | 对应 checkpoint 与最佳患者结果 | 现有实现可复用，先确认原始产物 | P2 |
| H | 效率、校准、临床扩展 | 匹配运行环境与时间/临床支持 | 部分已有；外部模型需适配 | P2 |

优先完成 B → C 单病例 → D/E/F → C 其余病例 → A。B 先固定真实提升；C 的单病例用于验证空间映射，不以画面漂亮作为通过标准。

推荐正文放 A、B、C、D、E；F 可放正文或补充，G/H 放补充。不要为了凑图重复呈现同一结果。

## 3. 开始前：核验输入与产物覆盖

### 3.1 实验清单

服务器建立 core.json，至少包含：

- Full：arm=exp6，BLCA/KIRC 各 5 折，共 10 项。
- Direct、Independent：BLCA/KIRC 各 5 折，共 20 项。
- Exp0、SlotSPE native/matched：按已恢复的真实结果添加，不能根据截图反推路径或训练配置。
- 每项记录训练 commit、历史 YAML、所有 overrides、seed、protocol、split、最佳 checkpoint、最佳预测、epoch curve。
- 训练 commit 与解释导出 commit 分列，不能把当前代码版本写成历史训练版本。

新 preflight 必须检查 **预期矩阵覆盖**：core 的 30 个唯一身份全部到齐，missing 为空；预期身份为 arm/cancer/fold/seed/protocol。单独 audit passed 不替代覆盖检查。现有 merge 只合并 runs，不保留所有 missing 信息，因此覆盖检查要读原清单和 expected_matrix。

配对比较要求同一折的患者、结局、split、编码器、采样、训练预算、seed 和 checkpoint 选择一致；列出每个对照允许改变的开关。有效损失权重也要记录。条件不同则标记为另一组实验，不画作同折机制对照。

### 3.2 解释所需资源

| 资源 | 用途 | 缺失时 |
|---|---|---|
| 最佳 checkpoint + 原配置 + 特征/RNA | 重放并导出实际 attention、计划、风险 | 只保留已审计的性能表，不制造解释 |
| 与特征同行序的 coords | 将实际采样 patch 放回切片 | 不能画病理空间图 |
| WSI 缩略图 | 空间图背景 | 可画坐标散点，但不称为病理叠加图 |
| 原始 WSI 或逐行 patch 图 | Top-5 高分辨率组织块 | 仅有缩略图时，不把放大截图当原始 patch |
| pathway 名称/成员清单 | 通路热图和条形图 | 不能只靠列序猜通路名 |
| train/val 风险及结局 | 风险分组、KM | 不从验证结果寻找最漂亮的阈值 |
| 病理专家标注 | 组织类别解释 | 展示未标注 patch，保留模型解释边界 |

保存资源清单 assets.json，示例见 configs/v313_interpretability_assets.example.json。示例中的 null 必须由真实元数据填写；它不是可直接运行的输入。

坐标需要同时确认“行数一致”和“来自同一次特征提取”。单靠 N 相同或文件 hash 不能证明对应关系；还需提取日志/H5 来源/patch 清单。特征、坐标、图像的 SHA-256 用于冻结已确认的对应关系，不替代来源核验。

## 4. 图 C：病例解释大图如何得到

### 4.1 病例选择与冻结

先列出每个 fold 的验证患者，检查资源完整性。在看结局、预测是否正确以及热图之前，用“资源合格 → 患者 ID 排序”的规则，BLCA/KIRC 各选 1–2 名患者。记录规则、全部合格候选和最终病例；资源不足写明原因。

病例清单按 run_id 绑定 case_id，不能把同一 case_id 无差别传给全部 fold。每名患者必须属于该 checkpoint 的验证折。多张切片各自定位，不能把不同切片的坐标叠在一起。正文病例不是最佳案例筛选；如果另列代表性案例，要说明选择理由并提供全部候选图。

如比较三个 arm，必须使用相同病例和可核验的采样位置；不能各挑各的“最好看”病例。

### 4.2 导出与重放

复用 v313.py 的 export_run / replay / indexed_sample：

1. 严格载入最佳 checkpoint，恢复最佳 epoch 的 OT 温度，核对训练参考 buffer。
2. 验证每名验证患者的 risk 与已保存最佳预测一致。
3. 导出 attention_wsi、attention_omic、全部 stage×geometry 的 plans、stage_gate、slots、reconstruction error。
4. 导出每个实际采样 token 的 feature SHA、slide_index、patch_index；padding 保留为 -1。
5. 全程 eval，不更新模型 buffer；记录训练与导出代码版本。
6. 保存逐患者数值，绘图只读这些产物，不在绘图阶段重新采样。

现有 attention 是局部槽 pooling 权重与共享 prototype 权重组合后的 rollout。它描述模型聚合关系，不是 patch 的因果贡献。重建误差针对编码后的 pathway token，不是原始基因表达误差。

### 4.3 WSI 槽空间图

对每张切片取该切片实际采样行，在 coords 中找回 level-0 左上角位置。用于排名/画图时排除 padding，合并同一原始行的重复采样后再选 Top-5，禁止用 padding 或重复行补满五张。

同时保留每个槽在有效 patch 上的总质量 valid_mass；显示归一化只用于图像，不能覆盖原始 attention。patch assignment 图使用槽权重最大者着色，标为“主要关联槽”；无效/无质量位置用灰色。不会把 assignment 当分割标签。

同一图不同槽使用共同的颜色范围，或明确标注每槽归一化。只覆盖实际采样区域，不插值成整张切片都得到预测。若画 patch 矩形，尺寸也来自提取元数据。

### 4.4 组学槽通过真实 OT 投影到病理区域

DCT 不照搬 SlotSPE 的“组学槽重新查询原始 patch”操作。使用本模型实际运输关系，得到组学槽关联的 WSI attention。

令 Aw 为 Kw×N 的 WSI attention；Tsg 为第 s 个潜在阶段、第 g 个几何分支的 Kw×Ko 运输计划；qs 为 stage_gate。与模型 _transport_wsi_to_omic 完全一致：

    Cs = mean_g(Tsg)
    Bs[o,w] = Cs[w,o] / max(sum_w Cs[w,o], 1e-8)
    Hs = Bs @ Aw
    q = stage_gate / max(sum(stage_gate), 1e-8)
    H = sum_s q[s] * Hs

重要顺序：先平均各几何计划，再按列质量归一化；不能先归一化每个几何计划再平均。数值核验使用未屏蔽 padding 的原始 Aw；得到 H 后再生成排除 padding 的显示版本，保存两者及有效质量。

核心函数设计：

    project_omic_attention(plans, stage_gate, attention_wsi)
        -> projected_raw, per_stage_raw, column_mass
    prepare_display_attention(raw, slide_index, patch_index)
        -> per_slide_maps, valid_mass, unique_patch_rows
    select_top_patches(display, unique_patch_rows, top_k=5)
        -> stable_ranked_patch_rows

这是“经运输关联的组学槽空间图”，不是基因表达在组织中的实测分布，也不是因果归因。零列质量与 padding 占比高的槽必须明确标记。

### 4.5 Top-5 组织块

两种资产来源：原始 WSI，或特征提取时保存、可逐行索引的 patch 文件。优先复用当时 patch，保证视野一致；用 WSI 读取时严格使用原提取的 level、size 和坐标。

OpenSlide read_region 的 location 是 level-0 左上角，size 是目标 level 的像素尺寸；不能把两种坐标单位混用。[官方 API](https://openslide.org/api/python/)

缩略比例使用实际 thumbnail 尺寸 / slide level-0 尺寸，不能假设固定 1/32。不从文件名猜 20×、patch size 或倍率。缺少 MPP 时不添加微米标尺；缺少提取尺寸时先恢复元数据，再读取 Top-5。

每张 patch 记录 case_id、slide_id、原始行号、level-0 坐标、提取 level/size、槽编号、rank、attention 和图像 hash。并列权重按原始行号稳定排序。

### 4.6 通路与版式

病例大图建议：

- A：WSI 缩略图与实际采样覆盖。
- B：WSI 槽空间图 + 所选槽 Top-5 patches。
- C：OT 投影的组学槽空间图 + 对应 Top-5 patches。
- D：槽×通路 attention，以及所选槽 Top-3 高权重和 Bottom-3 低权重通路。
- E：实际 OT 矩阵；主图可显示逐阶段几何平均计划，补充保留全部 stage×geometry。

没有 DCT 的 MoE 保留概率就不使用 SlotSPE 的保留概率选槽。第一版按每种模态槽编号均匀选择最多 3 个槽；补充图展示全部槽，规则冻结。Top/Bottom 通路表示表示聚合权重，不写成已证明的促进/抑制通路。

输出 case_id/figure.pdf、figure.png、source_arrays.npz、patches.csv、panel_metadata.json。构图完成后，人工核验热图位置、Top-5 视野与切片相符；不只检查文件是否生成。

## 5. 图 D：队列通路热图如何得到

### 5.1 补齐全患者导出

当前 patients.npz 不保存所有患者 attention；只有选中病例保存 attention，因此不能拿 3 个病例冒充队列热图。

新增 cohort_pathway_attention.npz，存储每名验证患者的 Ko×P attention、case_id、run_id、fold、pathway_names；用 float32 分片保存，不保存全队列 Kw×N patch attention。病例空间 attention 仍只存选定病例，控制磁盘和内存。

BLCA/KIRC Full 共 10 个 checkpoint，每折推理全部验证患者；复用同一批风险、sweep、slot 与 attention 导出，避免为每张图重新跑一次。导出 --km 已计算训练风险；新增保存训练风险向量及其 q25/q50/q75，而不是重新从验证风险算四分位数。

### 5.2 分组协议

主方案使用每折模型的训练风险四分位阈值，把该折验证患者分成 Q1–Q4；高 risk 对应高风险。训练样本用确定性评估采样得到的模型风险，属于训练内参考，不声称训练分组是独立验证。

训练风险阈值重复时保留相同边界，不按患者结局打散；出现空组或过小组就报告，不能为了画满四组临时调阈值。不能直接拼接不同 fold 的原始 risk 再算统一四分位。

五折验证患者应互斥；每名患者只贡献一次。删失患者仍可按预测风险分组。图同时标注各组人数、事件/删失数；不把删失随访时间当作死亡时间。

SlotSPE 原 Fig.14 按观察生存时间分组。我们的风险分组避免把删失样本当作已知死亡时间，并与预测任务对应；图注必须说明方案不同。

### 5.3 槽与通路聚合

跨 fold 的 slot 编号通常没有自动对齐保证，因此不将“fold0 的 slot1”和“fold4 的 slot1”当同一种生物学事件。

第一版对每名患者：
1. 按真实 pathway_names 对齐通路，检查签名定义/成员一致。
2. 对 Ko 个归一化组学槽 attention 等权平均，得到 P 维通路参与权重。
3. 在风险组内按患者等权平均，记录 n；不能让病例切片数量决定患者权重。
4. 同时保存各折组均值，检查趋势是否跨折一致。

明确标注“平均槽聚合权重”，不叫风险贡献。若以后使用其他槽加权，必须说明模型真实可用的权重及其意义，不能凭空添加 MoE 风险权重。

主图选择各组 Top-10 通路的并集，稳定去重；若行数过多，正文展示按组间极差排序的前 30 个，其余放补充。选择规则先冻结，不按 p 值挑行。完整 P 通路矩阵一并发布。

主图显示原始平均 attention，共用色标。可另画逐行 z-score 图展示相对变化，清楚标注 z-score，不能把它解释为表达量变化。第一版不自动生成 GO 富集或通路显著性结论。

### 5.4 输出

每癌种：cohort_pathways.pdf/png、patient_groups.csv、group_pathways.csv、fold_group_pathways.csv、selection.json。selection 记录阈值来源、聚合方式、行筛选和色标；任何组缺失都进入 figure_index.json。

## 6. 图 E/F/G/H：机制与效果证据

### 6.1 固定 checkpoint 运输替换图 E

复用 Full 的 alpha sweep，alpha=0、0.25、0.5、0.75、1。

    a = T @ 1
    b = T.T @ 1
    Tind = a @ b.T / sum(T)
    Talpha = (1-alpha)*T + alpha*Tind

逐 stage、逐 geometry 使用实际 factual plan 的边缘质量，检查边缘残差。alpha=0 必须重放原预测。网络按替换计划重新计算后续输出，包括对应的门控；不要把当前实现写成“固定所有门控、只换矩阵”。

每癌种画：
- 各折 C-index 随 alpha 的曲线，以及相对 alpha=0 的变化。
- 患者风险变化 |risk_alpha-risk_0| 的分布；注明 risk 单位为模型分数。
- 同一 Full 模型中的 cross pathway token 重建误差变化。
- 固定病例的 learned/independent 计划，以及 alpha=1 空间图。

不要求曲线单调，不预设结果支持方法。若预测几乎不变，如实报告该模型对这类计划替换不敏感；不能靠删掉折来改变结论。alpha=1 是固定 Full 内的计划替换，不等于重新训练的 Independent。

Direct 仍在主预测使用 OT，只改变 cross 重建输入；Independent 使用独立耦合，并不表示“两模态独立训练、不交换信号”。Full 与这两个训练对照的差异，要结合图 B 看；两个对照彼此接近不能证明 cross 重建无用。

### 6.2 KM 图 F

2026-10-07 作者将主图形式确定为每个癌种一张总体高低风险 KM。复用 export --km 的训练风险中位数：各折先将验证患者分组（risk >= train_median 为高风险），再按患者 ID 汇总全五折组别及结局。不同模型的原始 risk 不用于计算跨折统一中位数。严格检查同一队列、五折齐全及患者只出现一次；缺训练阈值时报错。默认 Full，控制臂可分别出图。验证集选 checkpoint 和 CV 模型依赖须保留说明，不能伪装成单模型外部队列。

scripts/plot_fig5_km_curves.py 已改为总体图入口，survot_rank/evidence/km_oof.py 负责导出来源与五折分组校验。支持删失标记、名义置信区间和在险人数；--show-logrank 可显示探索性 p 值，但未校正 CV 模型依赖。RMST 尚未实现，截断时间需由随访支持预先选择。代码检查不等于真实曲线已经运行。

SlotSPE KM 要在其真实模型/原环境导出训练参考风险，不能交给 DCT checkpoint loader。

### 6.3 槽诊断图 G

恢复 Exp2、Exp3、Full 原 checkpoint 后，比较 WSI/omics 的槽余弦相似度、距离、hazard 方差；每折展示，结合 Exp2→Exp3 的真实 diversity 开关。不会把槽看起来不同直接当作提升原因；还要对应性能表。

epoch0 是第一轮训练后的评估，不能自动称为随机初始化预测。标准差内的差异不自动代表“不显著”；p>0.05 也不代表等价。图 B 默认展示逐折变化，不给未经有效统计支持的显著性星号。

### 6.4 效率与临床图 H

profile 单独在空闲 GPU 做；记录型号、软件、batch、patch 数、预热、同步、forward 范围。DCT 的现有 forward 含重建相关开销，不能与 SlotSPE 的最小预测分支直接比较。解释导出 replay 的额外计算也不能当普通预测延迟。

校准使用有支持的离散 bin 时间，不自动外推 60 月；DCA、临床 Cox 等另立数据/统计方案。有现成 C-index 的 SlotSPE 表不代表已具备其校准和临床分析所需产物。

## 7. 代码设计：复用与新增边界

以下是实施时的目标模块，**不是当前已实现文件/命令**。

| 文件 | 设计职责 |
|---|---|
| survot_rank/evidence/v313.py（扩展） | 复用严格 replay；增加全患者组学 attention、训练风险四分位导出、run-specific 病例选择 |
| survot_rank/evidence/spatial.py（新增） | 精确 OT 投影、有效 patch 映射、稳定 Top-K；纯数值函数 |
| survot_rank/evidence/assets.py（新增） | 坐标/patch/WSI 清单验证、来源检查、缩略图和 patch 读取；OpenSlide 懒加载 |
| survot_rank/evidence/case_figures.py（新增） | 病例分面图、通路条形图、矩阵图、共享图例及元数据 |
| survot_rank/evidence/cohort.py（新增） | 训练风险阈值、跨折患者去重、通路对齐、组聚合与热图 |
| survot_rank/evidence/clinical.py（按需） | KM 扩展、log-rank、RMST、at-risk 表；时间支持检查 |
| scripts/v313_interpretability.py（新增） | 分阶段入口、dry-run 清单、阶段产物与错误提示 |
| tests/test_v313_interpretability_*.py（新增） | 数值投影、资产错配、队列聚合与状态恢复的必要测试 |

模型 forward 和训练损失不为绘图改写；已有 replay 接口负责产生真实中间量。绘图模块只读冻结导出，输入缺失时跳过对应面板并明确记录，不填示例数据。

### 7.1 拟新增入口

示意接口，完成代码和 --help 后才可执行：

    v313_interpretability.py preflight --plan PLAN --output CHECK
    v313_interpretability.py select-cases --plan PLAN --output CASES
    v313_interpretability.py export --plan PLAN --cases CASES --device cuda:0 --output EXPORTS
    v313_interpretability.py case-figures --plan PLAN --exports EXPORTS --output FIGURES
    v313_interpretability.py cohort-figures --plan PLAN --exports EXPORTS --output FIGURES
    v313_interpretability.py summary --plan PLAN --exports EXPORTS --figures FIGURES

默认 preflight/select-cases 仅做清单检查，不读取全部特征、不运行 checkpoint；元数据不足时要求用户执行明确的资产检查阶段。export 是显式真实推理入口；patch 图读取发生在 case-figures，清单列明读取量。第一版不提供隐式“自动训练缺失 fold”。

病例选择是确定性规则；export 使用 run_id→case_id 映射。断点恢复以 run 目录为单位：只跳过 hash/版本/参数完全匹配且 complete=true 的产物；不同导出参数使用新目录。用临时目录写完后原子发布，避免半成品被绘图读取。

### 7.2 导出契约

在原产物旁新增，不覆盖原字段：

    export_root/run_id/
      export.json
      patients.npz
      cohort_pathway_attention.npz
      train_reference.npz
      case_000.npz
      cases.json

export.json 记录 schema_version、训练/导出 commit、checkpoint/config/split hash、病例/通路顺序、特征 fingerprint、alphas、采样规则、有效损失权重、artifact hashes 和 complete 状态。

cohort_pathway_attention.npz：
- case_ids[N]、run_id、fold、pathway_names[P]、attention_omic[N,Ko,P]。
- attention 的归一化轴、构造方式与 dtype；不混入 pathway expression。
- 若不同模型 Ko 不同，逐 run 存储，不用 padding 冒充共享槽。

train_reference.npz：
- train_case_ids、risk、thresholds=[q25,q50,q75]、median、quantile_method、split hash。
- 仅供阈值与参考计算；图用验证患者，输出身份明确。

cases.json：
- 每项明确 run_id、case_id、slide_ids、选择规则、资源资格和选择时是否接触预测/结局。
- 关联 case_*.npz 的实际 feature/row provenance。

汇总 source_data 采用 CSV/JSON，保留 patient/fold 单位；不以图片里的数字作为唯一数据来源。

## 8. 实施阶段与验收

| 阶段 | 工作 | 验收结果 |
|---|---|---|
| S0 | 恢复 Full、controls、基线清单；准备资产清单 | 覆盖检查通过，资源缺口明列，历史配置明确 |
| S1 | 实现 projection/assets；单折单病例 | 原预测重放一致；投影等价；切片坐标与 Top-5 视野人工核验 |
| S2 | 扩展全队列 attention 与训练阈值 | 10 个 Full run 完成；五折验证患者互斥；通路来源一致 |
| S3 | 生成 D/E/F 和逐折图 | 每图有 source_data、分组/单位/折数、缺失原因；没有伪造组或阈值 |
| S4 | 2–4 个病例大图与架构图 | 面板可读；槽/patch/通路可追溯；架构与代码一致 |
| S5 | 复核文稿、图注与目录 | 图表数字同源；增益、机制、关联、临床结论分清 |

必要代码检查：
1. 非均匀边缘、多个 stage/geometry 下，空间投影与模型运输函数的输出一致；专门覆盖“先平均后归一化”。
2. alpha=0 风险重放一致，所有替换计划保留实际边缘，eval buffers 不变。
3. padding、重复 patch、多切片、零列质量、并列排名、缺失坐标均正确处理。
4. 错误坐标行数/hash/来源不允许出空间图；图片尺寸不能由猜测填充。
5. 队列重复患者/路径名称错配/重复训练阈值/空组触发明确错误或跳过原因。
6.病例 ID 不属于验证折时停止；省略病例不能静默换成另一位。
7. 中断导出不是完成状态；恢复不能混用不同 checkpoint、alphas 或采样配置。
8. 用小型合成数据检查模块，再由用户在服务器执行单折真实 replay。合成测试通过不等于模型效果成立。

图像验收：输出矢量 PDF 与 300 dpi PNG；双栏宽约 180 mm，最终正文字号至少 8 pt；固定 Full/Direct/Independent 配色；每幅有 A/B/C 面板号、单位、共享或明确独立色标。论文 PDF 中按实际大小检查，不能只看放大的 PNG。

## 9. 当前可运行的服务器步骤

下面只使用现有 scripts/prepare_v313_evidence.py 的参数。core.json 必须先按原 runbook 恢复；不存在时不要直接执行。所需训练数据/模型实际在服务器，示例路径需确认。

    cd /data1/DCT-Reg
    export PYTHONPATH="$PWD"
    PY=/home/ubuntu/.conda/envs/trisurv/bin/python
    TOOL=scripts/prepare_v313_evidence.py
    EVIDENCE="$PWD/results/v313_evidence_v2"

先审计真实核心清单：

    $PY "$TOOL" audit --manifest "$EVIDENCE/core.json" \
      --output "$EVIDENCE/core_audit_for_figures.json"

另外人工确认 core 的 30 个预期身份齐全；新增 preflight 以后再自动化此项。

先由用户执行 Full/BLCA/fold0 的检查导出。export_check_interpretability 必须是未使用的新目录：

    CUDA_VISIBLE_DEVICES=0 $PY "$TOOL" export \
      --manifest "$EVIDENCE/core.json" --arm exp6 --cancer blca --fold 0 \
      --device cuda:0 --output "$EVIDENCE/export_check_interpretability" --km

已有绘图可先生成运输、通路、槽诊断、KM 等基础图：

    $PY "$TOOL" plot --manifest "$EVIDENCE/core.json" \
      --exports "$EVIDENCE/export_check_interpretability" \
      --output "$EVIDENCE/figures_check_interpretability"

资源完整且明确指定单折病例时，可在 export 加 --case-id 真实患者ID。现有默认病例为患者 ID 排序前三名；它不会自动按资产完整性选病例。不能把同一 --case-id 用于全折导出。

核验通过后，导出 Full 的 BLCA/KIRC 全 10 个 checkpoint，使用新目录：

    CUDA_VISIBLE_DEVICES=0 $PY "$TOOL" export \
      --manifest "$EVIDENCE/core.json" --arm exp6 \
      --cancer blca --cancer kirc --device cuda:0 \
      --output "$EVIDENCE/exports_full_interpretability_v1" --km

这一步能得到已有 Full sweep/KM/slot 图输入，**当前不会得到新设计的全队列 attention、训练四分位和病例组合图**。新 exporter 完成后再使用其入口，或复用已导出的风险并补所需字段；禁止把现有 patients.npz 当作已具备全队列 attention。

coords 清单确认后，现有 plot 可加 --coordinates 生成 WSI 槽空间图，schema 使用原 runbook 第 5 节；本文 assets.json 是更完整的新设计，不能直接传给旧 --coordinates。新增 assets 模块应提供显式转换，保留该旧接口兼容性。

## 10. 工作量、运行量与边界

第一版核心为 Full 10 个 checkpoint 的全验证推理；--km 另加对应训练患者的确定性参考推理。五个 alpha 和重建诊断有额外计算，不能把它估算成一次普通前向。运行时长要在单折检查后实测，不沿用训练耗时猜测。

若做训练对照的解释图，再加已有 Direct/Independent 20 个 checkpoint 的读取；若只是主表/逐折图，读取已审计预测即可。SlotSPE 原结果能作主表，解释/KM 则需其独立导出适配。Exp2/3 槽图视原 checkpoint 是否完整再安排。

第一阶段无需新增训练，也不重复已完成的 controls。只有已有 Full 无法严格重放、原始权重丢失，或明确要回答权重匹配单分支贡献时，才另列训练计划；不能为出图悄悄补训练。

最终目录建议：

    results/v313_interpretability_v1/
      inputs/plan.json assets.json cases.json
      checks/preflight.json replay_checks.json
      exports/run_id/...
      figures/main/ figures/supplement/
      source_data/...
      figure_index.json
      FIGURE_REPORT.md

figure_index.json 逐图记录生成状态、输入 hash、方法、证据边界和跳过原因。FIGURE_REPORT.md 收录最终图、对应实验、结论与限制。

交付判断：病例图解释模型关联；队列图解释通路参与模式；运输 sweep 检验固定模型对计划替换的响应；Full 与训练对照的配对结果回答性能收益。四者组合支持方法讨论，但不能把注意力热图、单 seed legacy_val 或不显著检验写成因果、等价或外部临床验证。
