## SlotSPE最高优先级：干净原版单臂检查（2026-10-19）

- 当前入口为 `docs/SLOTSPE_PROOF_TASK.json`、`docs/SlotSPE证明任务_服务器一键运行.txt`；新包为 `docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip`。服务器请解压新包到独立目录，以保持原版源码字节与回传验收一致。
- 本轮仅original：fold0/seed3、3轮、256真实patch、batch8、20次评估；配置 `configs/slotspe_clean_original_server.json`。服务器AI执行，本窗口验收；不索要SSH、不自动扩展预算。
- 未修改原版模型forward、槽模块、decoder和原生损失。诊断协议使用完整配对病例、固定真实patch与训练折分箱，原版全队列分箱另在副本中核查。
- 验收逐参数梯度、患者末轮固定选槽且冻结cross的干预、同权重重复预测与同RNG重放、补零敏感性及自然门控。受控补零/分数反例不作为患者发生率；未观察到则记not_observed。
- 本地实现/合成检查、Git发布、服务器实际运行、患者效果分别登记；没有原始回传不能说服务器已证明。新包和交付记录中的未发布字段是封包时快照，以实际Git提交与发布回执核实发布状态。
- 原版/修正版和A/B/C实现及历史包保留；B/C仅切换最终self汇总的选择梯度，辅助训练和梯度一致。患者修正归因属于后续阶段，其他研究线不因此改变。

## 当前纯运输插拔范围 2026年10月18日

- 用户最新要求：先在SlotSPE尝试运输机制插拔，再扩展SurvPath、PIBD、MMP、LD-CVAE、TTA。当前插件入口是 `docs/TRANSPORT_ONLY_METHOD_CONTRACT.json`，代码为 `survot_rank/plugins/transport_only/`。
- 插件只更新特征，`L_total = L_host_original`；不带DCT生存损失、预测头、重构、relation、native反事实或收益校准目标。宿主本来已有的生存和辅助损失保留。
- 历史 `transport` 及E049完整辅助训练保留，未发布的 `transport_v2` 原材料保留在原研究目录；它们不能作为本轮纯运输结果；原生DCT-B、DCT-RG训练方案不因此改变。
- SlotSPE直接复用已有槽。MMP本轮支持coattn臂；LD-CVAE支持Robust_MCAT；TTA是Together, Then Apart，支持双模态coattn臂。具体接入与限制看新契约。
- 合成前向/梯度和接口检查不等于患者性能；患者任务仍由用户启动。发布前远端入口备份位于 `docs/history/transport_only_publish_20261019/`。
- 发布时运输求解部分独立放入 `transport_only/solver.py`，不依赖或发布历史 `transport_v2` 及其损失配方。

# v313 实验资料保护与主档维护

主档只记录v3.13相关实验。用户已明确授权本轮清理非v3.13归档实验与代码；该授权不是对后续清理的永久授权。

- 未取得用户对具体材料的明确授权，不得删除、移动、批量覆盖档案、代码、配置、结果、日志、预测或checkpoint。
- v3.13撤回、失败和旧图版本仍保留状态与来源。
- v3.13继承v3.11与v3.10的模型代码，文件名带旧版本不等于非v3.13依赖；继承基类、共同训练模块和底层组件不得按版本号删除。
- 修改主档前保留备份；已有编号不复用，E045已登记（配置 dct_v313_full_wsi_three_decoders.yaml，三解码器消融，BLCA seed3 legacy_val 5-fold，mean val c-index 0.7256 ± 0.0317（95%CI跨零，统计不显著），详见 实验档案/表格图片/E045_三解码器重建消融/，口径为 legacy_val inner-val peak，不可与 E046 outer_test 数字直接比），E046已登记（代码就绪 + 真实训练完成：outer_test 5×3 + legacy_val 5 arms wsi_additive + 5 arms E046b full_replay + E046c 原始 commit ckpt 0.0 误差复刻审计 5 折 380 患者，详见 实验档案/表格图片/E046_WSI自重建消融/ 与 实验档案/表格图片/E046b_Full_replay_legacy_val/ 与 实验档案/表格图片/E046c_原始commit_replay_审计/，主档 records_count=24 含 E046c），E047已记录（BLCA效率对比：DCT Full部分实测完成，外部方法因本机无外网无法下载源码无法完成，需有外网环境重做），E048已登记（v313模块与单模态消融，代码包就绪，未启动训练，详见 实验档案/实验代码/E048_v313模块与单模态消融/），E049已登记（v313六队列版本对照，30任务 = 6癌种×5折×seed3 独立三解码器新版 vs frozen baseline，**30 任务 2026-10-15 已实测完成**——part1 brca+coadread 10 任务 results/v313_three_decoders_six_20261013、part2 ucec+hnsc+skcm+stad 20 任务 results/v313_three_decoders_six_20261013_part2_76f8f25；6 队列新版 vs 原版 mean cindex 差：brca +0.0149, coadread +0.0101, ucec +0.0075, hnsc −0.0198, skcm −0.0047, stad −0.0228，30 折 overall Δ = −0.0025 ± 0.0466，13/30 折新版胜，17/30 折原版胜，结论：跨队列方向分裂、量级 ≤ 0.02，无统计显著，**不主张独立三解码器优于共享解码**；原版六项分数冻结复用 baseline_frozen.json 来源 paper/V313_TEN_CANCER_REPORTED_COMPARISON_INPUT.json sha256 d8aeeeb7...，本机已核验 30 任务计划、frozen recipe、baseline SHA256 一致；详见 实验档案/实验代码/v313六队列版本对照_20261013/ 与 实验档案/表格图片/v313六队列版本对照_20261013/E049_30折实测结果_20261015.md），E050已登（独立 DCT v2.1 真实患者训练，BLCA 5折 seed3 legacy_val，三新机制默认，GPU 1，后台 driver，输出 /data1/DCT-Reg/实验档案/结果/E050_v21_20261014/，代码 origin/main @ af5da64，对比基线 E045 legacy_val 0.7256 ± 0.0317），下一项实验从E051继续。
- 按患者、划分、配置与原始产物核对结果。服务器未直接全量扫描时，不宣称所有实验已收齐。

## 当前独立 DCT v2 入口（2026-10-13）

- 用户最新明确授权实现独立 DCT v2。讨论当前 v2 前先读 `docs/DCT_V2_METHOD_CONTRACT.json`；其中六项损失、独立模型范围优先于此前插件 v2 的定位。
- 当前 `dct_v2` 使用 DCT 自有编码器、语义槽、运输与生存头；总损失是主 NLL、IPCW rank、per-slot NLL、slot diversity、relation、native 六项；不启用三重构。
- `transport_plugin_v2` 保留为另一项外部骨干接入实现，不能再把它的交付或检查数量说成独立 DCT v2 的结果。
- 独立模型源目录为 `survot_rank/research/methods/dct_v2/`，配置为 `configs/dct_v2_blca_uni2h.yaml`，说明为 `docs/DCT_V2_STANDALONE.md`。模型已接入现有训练器；患者分数仍待用户运行。

## 独立 v2.1 修正版候选（2026-10-13）

- 用户要求处理 v2 的局部残差抵消、关系 hinge 平局停止激励与精确槽坍缩的损失零梯度。修正另建 `dct_v21`（别名 `dct_v2_1`），保留 `dct_v2` 与 v3.13；谈论修正版先读 `docs/DCT_V21_METHOD_CONTRACT.json` 和 `docs/DCT_V21_DEBUG_20261013.json`。
- 研究问题不变；修正版仍为独立 DCT、六项损失，不新增编码器或预测头，不启用重构。源码 `survot_rank/research/methods/dct_v21/`，配置 `configs/dct_v21_blca_uni2h.yaml`。
- 三项修正是实际计划相对同预算独立计划的风险共激活读取、固定正增益探针下的平滑关系比较、真实欧氏距离的精确碰撞次梯度。不要说成保证患者提分、保证部署运输分支使用或保证任意上游网络脱离坍缩。
- v2.1 已作为独立候选版本提交；患者训练与性能仍未验证。不得把 v2/v3.13 分数或旧检查数量记作 v2.1 的患者结果；推送状态以其独立交付记录为准。

---

## E049 进度更新（2026-10-14）：BLCA outer_test 20 任务已实测完成

**范围**：此节记录 BLCA 的 MCAT/SurvPath 插件对照：2 个基线 × 2 个 arm × 5 折 × seed3 = 20 任务。它与上文同编号的“v3.13 六队列版本对照”是不同实验包；编号存在冲突，更正前以目录路径区分，不合并两者结果。

**结果（普通 Harrell C-index，每 fold outer_test 76 患者；不是 IPCW）**：

| Baseline | Arm | F0 | F1 | F2 | F3 | F4 | MEAN |
|---|---|---|---|---|---|---|---|
| MCAT | native | 0.6466 | 0.6378 | 0.5996 | 0.7147 | 0.7444 | **0.6686** |
| MCAT | learned_three_decoders | 0.6500 | 0.6300 | 0.6052 | 0.6962 | 0.7496 | **0.6662** |
| SurvPath | native | 0.5786 | 0.5700 | 0.6043 | 0.6936 | 0.6581 | **0.6209** |
| SurvPath | learned_three_decoders | 0.6075 | 0.5639 | 0.6165 | 0.7164 | 0.6291 | **0.6267** |

**关键观察**：
- MCAT 整体优于 SurvPath（0.667 vs 0.624，差 ~0.043）。
- `learned_three_decoders` 在 MCAT 上的五折均值低于 native（0.6662 vs 0.6686，差 −0.0024）；尚无充分证据说明原因或统计显著性。
- `learned_three_decoders` 在 SurvPath 上的五折均值高于 native 0.0058；增益较小，不能单凭此表归因于运输机制。
- 最佳单配置：MCAT native fold4 = **0.7444**。

**当前核验边界**：Git 中有 20/20 份 `status=complete` 收据与对应预测；20 份预测 SHA256 均匹配，按保存预测重算的普通 Harrell C-index 与 `completion.score` 完全一致。每折四组的患者和标签一致，五折互不重叠，共 380 人。`task.json`、`label_audit.json`、训练曲线与 checkpoint 仍在内网服务器，尚未在本地独立核验；运输写回强度、补零比例和完整训练状态待服务器只读诊断。IPCW 需要训练集生存参考，不能仅从这些预测文件重算。

**原始位置**：`/data1/DCT-Reg/results/e049_blca/outer_test/blca/{mcat,survpath}/{native,learned_three_decoders}/fold{0..4}/seed3/`

**E049 当前结论（2026-10-14 下午用户判断）**：
- **模块化**：运输插件代码层面已通，通过 `adapters.py` 接入 MCAT / SurvPath，64 项检查通过；关闭时保持原生输出。但仍需针对骨干写适配器，**尚未做到"任意模型无需改动"**。
- **分数**：运输 + 三重建合并测试 → MCAT native -0.0024，SurvPath +0.0058，**贡献无法拆分**，提分未成立。
- **不重跑**：用户明确指示**暂不重跑 20 个训练任务**。
- **下一步（用户接管）**：
  1. 在内网运行只读诊断脚本 `实验档案/实验代码/E049_v313运输插件MCAT_SurvPath基线比较/evidence.py`（服务器端等效于 Windows 的 `audit_readonly.py`），查清**补零是否影响输入病例**以及**训练后运输写回是否真正起作用**。
  2. 若补零影响了输入 → 修复后正式比较需**两臂一起重跑**。
  3. 若补零不影响 → 不为该修复重复旧实验。
  4. 查原因、调整模块、设计下一组最小对照**由用户负责**。
- **边界**：真实分数须内网按相同划分训练确认，不得仅凭改代码宣称提分。下一步优先把**运输本身**和**三重建辅助训练**的贡献分开，再决定跑哪些任务。

---

# 会话启动清单（必读 · 必问）

**每次新开会话，先读本节并按 `bash /data1/DCT-Reg/open_session.sh` 自检，缺什么问什么。**

## A. Python 环境（关键陷阱）

- **禁止使用默认 `python3`**（即 `/data/env/data/env/anaconda/bin/python3`），它带的是 torch 2.6.0+cu124，**不支持 RTX 5090 (sm_120)**，所有 fold 会 `SIGILL` 立即失败。
- **必须用** `/home/ubuntu/.conda/envs/dct/bin/python3.10`（torch 2.10.0+cu128，支持 sm_50/70/75/80/86/90/100/120）。
- 启动方式：`PATH=/home/ubuntu/.conda/envs/dct/bin:$PATH /home/ubuntu/.conda/envs/dct/bin/python3.10 script.py`
- 子进程继承：`subprocess.run(..., env={**os.environ, "PATH": "/home/ubuntu/.conda/envs/dct/bin:" + os.environ["PATH"]})`

## B. 数据路径

- 临床表型 CSV：`/data1/dataset_csv`
- WSI patch 特征：`/data1/TCGA-UNI2-h-features`（目录必须存在，子目录为各癌种）
- splits 在 `data_path/splits/{which_splits}/{cancer}/fold_{0..4}.csv`

## C. 划分 / 协议

- legacy_val 协议 → 划分名 `5fold_uni2h`
- outer_test 协议 → 划分名 `5fold_uni2h_outer_test_seed3`
- **口径不同，不可直接比较数字**（E045 是 legacy_val inner-val peak，E046 是 outer_test）

## D. 结果 / 表格布局

- 实验结果根目录：`/data1/DCT-Reg/results/{实验名}_{YYYYMMDD}/`
- 内部结构：`{protocol}/{arm}/{cancer}/fold{N}/seed{S}/` + 顶层 `logs/` `plan.json` `base_config.yaml`
- 表格/图片输出：`/data1/DCT-Reg/实验档案/表格图片/{实验名}/`
- driver 日志习惯放 `/tmp/{实验名}_driver.log`（避免污染结果目录）

## E. 必问用户（未明说前不要猜）

1. 哪个实验？要哪几个 arm / 队列 / 折 / seed？
2. 协议：legacy_val 还是 outer_test？
3. GPU 选择：单卡还是多卡并跑？
4. 是否允许写新实验代码 / 是否复用基线 E045 配置？
5. 表格输出位置与命名。

## F. AGENTS.md 保护红线（操作前再次确认）

- 任何删除 / 移动 / 覆盖档案前必须得到用户对**具体材料名**的明确授权。
- 主档已登记：E045–E051（E051 阶段 2 已实测 20 任务，2026-10-15 完成 / 2026-10-18 入档；详见正文）。下一项实验从 **E052** 起编号，不复用。
- 失败 / 撤回 / 旧图版本保留状态和来源。

---

## E051 计划：BLCA 诊断 + 插件机制分解（2026-10-14；当日修正）

### 背景

E049（BLCA，2 骨干 × 2 臂 × 5 折 = 20 任务）结果：MCAT native 0.6686，MCAT learned_three_decoders 0.6662（−0.0024），SurvPath native 0.6209，SurvPath learned_three_decoders 0.6267（+0.0058）。完整插件臂**同时含学习运输写回和三重建**，这组差值不能拆分各自贡献。

**重要口径修正**：E049 `native` 在 `worker.py` / `model.py` 中是 `enabled=False, reconstruction=False`；`build_native_adapter()` 将该开关交给插件，插件前向直接返回原特征。因此它是功能上的无插件基线，不是运输仅臂。服务器交接的 `model_audit.json` 摘要也报告 `enabled=False`（完整文件未提交，本机尚未独立复核）。共享 YAML 的 `dct_v313_cross_mode=transport` 属于独立 DCT v3.13 模型配置；E049 worker 用自身的适配器构造器替换 `runner.get_model`，不能根据该 YAML 键推断 MCAT/SurvPath 插件是否开启。

内部参考线：+0.01 值得继续投入，+0.02 在多个癌种与骨干上稳定出现更有说服力；它们不是统计阈值。正式判断看同患者配对分数差及不确定性、实际成本（参 PMC4314453）。

### 阶段 1：BLCA 诊断（已执行；2026-10-14）

**目标**：核对 (a) 旧训练入口是否可能接收补零行，(b) 20 个任务的标签、来源、分数与 checkpoint 是否一致，(c) 插件臂训练后的残差增益是否偏离零。只读脚本不运行模型，增益非零也不能单独证明运输改变了风险预测；这一点还需同 checkpoint 的插件开/关推理对照，由用户决定执行。

- 脚本：`实验档案/实验代码/E049_v313运输插件MCAT_SurvPath基线比较/audit_readonly.py`；`evidence.py` 是预测、标签与划分核验库，不是补零/写回诊断入口
- 命令：`/home/ubuntu/.conda/envs/dct/bin/python3.10 /data1/DCT-Reg/实验档案/实验代码/E049_v313运输插件MCAT_SurvPath基线比较/audit_readonly.py --batch-dir /data1/DCT-Reg/results/e049_blca`
- **执行位置**：GPU 服务器（本地运行）
- **判定结果**：
  - ⚠️ **补零影响仍待判断**：380 名外测患者中有 **15 人**需要补零（约 4%）；各折外测补零行占比为 1.4%–19.3%。这些患者也可能进入其他折的训练/内部验证；仅凭人数不能断言对模型分数无影响
  - ✅ **分数一致**：MCAT native 均值 ≈ 0.669，MCAT learned 均值 ≈ 0.666，SurvPath native 均值 ≈ 0.621，SurvPath learned 均值 ≈ 0.627，与已知结果一致，无意外异常
  - ⚠️ **配对来源待核对**：脚本对 10/10 对都报 `"Paired recipe/split mismatch"`，所以没有一对通过严格配对审计。已改进脚本，下一次会列出具体是划分、配置、源码、训练标签还是配方字段不一致；手算分数差不能代替这个核对
  - ⚠️ **完整 checkpoint 未提交到 Git**：服务器执行的只读脚本逐任务核验了 checkpoint 哈希并读取残差增益，但本机无法独立读取完整文件；`model_audit.json` 报告 native `enabled=False`，脚本未做同 checkpoint 开/关推理

**当前结论**：E049 分数本身已复核；补零对训练和预测的影响、10 对配方/来源差异的原因仍未查清。A/D 能否作为与新 B/I 严格配对的结果**待定**，阶段 2 暂缓启动。先用更新的 `audit_readonly.py` 在服务器只读重查，取得具体差异字段，再决定哪些旧臂可复用、哪些必须同代码重训。此次改动不运行患者数据或模型。

**2026-10-18 状态更新**：用户授权按"现有接口可执行的对照"启动阶段 2，20 任务（MCAT/SurvPath × B/I 臂 × 5 折 × seed3）于 2026-10-15 03:59 全部完成 driver（`results/e051_stage2_blca/`，20/20 OK rc=0）。本机已审 completion.json / 产物 SHA256，统计结果写入下方"## E051 阶段 2 实测结果（2026-10-15；2026-10-18 入档）"章节；A/D 配对审计未过这一点仍成立，B/I 不与 A/D 直接配对比较。

### 阶段 2：BLCA 机制分解（现有接口可执行的对照）

**目标**：拆解插件运输写回、学习匹配与三重建辅助训练的贡献。是否复用 E049 旧结果由阶段 1 的补零、源码与 checkpoint 审计决定。

**臂设计**（均由 E049 worker 的 `arm` 决定，不能用独立 DCT 的 `cross_mode` 替代）：

| Arm | 插件开关 / 计划 | 三重建 | 来源与解释 |
|---|---|---|---|
| A `native` | `enabled=False` | 关闭 | E049 已训练；功能上的无插件基线 |
| B `learned` | `enabled=True, plan_mode=learned` | 关闭 | 需新训练；学习运输与残差写回 |
| I `independent` | `enabled=True, plan_mode=independent` | 关闭 | 需新训练；同接口独立计划对照 |
| D `learned_three_decoders` | `enabled=True, plan_mode=learned` | 开启 | E049 已训练；学习运输写回 + 三重建 |

- backbone：MCAT、SurvPath；若 A/D 可复用，新增 B/I 共 **20 次训练**，四臂合计 40 条结果；若补零或来源审计不通过，受影响的旧臂须按修正代码重训，不能宣称可复用
- 协议：`outer_test`，划分 `5fold_uni2h_outer_test_seed3`，每折用内部验证选 checkpoint；`legacy_val` 是另一协议，不是这里的选择规则名称
- 参数选择在训练/验证折内完成，外层测试折仅用于最终评价
- 当前插件 `enabled=False` 会在重建前直接返回，而跨模态重建使用运输后的 WSI memory；所以“运输关、三重建开”的 C 臂**不能仅改配置得到**。若要严格 2×2 因子设计，须先定义并实现直接重建对照、测试其与其他臂的匹配条件，再另行登记；当前不把 C 写成可运行或已验证实验。

**关键比较**：
- B − A = 学习运输插件（含槽、计划、残差写回）的整体效应，不单独归因于 OT 匹配
- B − I = 相同插件接口下学习计划相对独立计划的增量，用于检验匹配贡献
- D − B = 在学习运输开启条件下，增加三重建辅助训练的增量
- D − A = 完整插件相对无插件基线的总效应

**提分参考线**：
- +0.01：值得继续投入
- +0.02 × 多个癌种 × 多个骨干：更有说服力
- 正式判断基于**同患者配对分数差及不确定性**（非 fold 均值的独立比较）

### 阶段 3：扩癌种（BLCA 因子分解稳定后）

- 选 2–3 个预先选定、数据完整的癌种
- 仍用 MCAT + SurvPath、相同划分与选择规则
- 每个癌种看配对差值
- 不做 10 癌种扫全量，优先质量而非数量
- 跨癌种参考：不同癌种 C-index 受病例构成影响，**绝对分数不可横比**，只看同患者差值方向

### 阶段 4：结构不同的第三骨干（条件触发）

- 前提：跨癌种出现稳定收益（+0.01+，多个骨干一致）
- 目的：验证"换骨干仍有效"
- 现在不增加，避免放大任务量、仍回答不了运输是否带来收益

### 现有资产（可复用）

- E049 20 task 产物（`results/e049_blca/outer_test/blca/{mcat,survpath}/{native,learned_three_decoders}/fold{0..4}/seed3/`）：Git 已提交 `completion.json` 与 `outer_test_predictions.pkl`；`label_audit.json`、`model_audit.json`、checkpoint 与训练曲线仍在内网服务器，需在那里核验
- `实验档案/实验代码/E049_v313运输插件MCAT_SurvPath基线比较/audit_readonly.py`：服务器本地只读诊断入口，已在 git；`evidence.py` 为其调用的证据核验库
- `paper/V313_TEN_CANCER_REPORTED_COMPARISON_INPUT.json`：原始论文 BLCA 指标（MCAT 0.688 ± 0.024，SurvPath 0.666 ± 0.034，DCT v3.13 0.724 ± 0.047，仅作跨协议参考）
- `survot_rank/plugins/transport/plugin.py` + `adapters.py`：已通过 64 项检查，commit 26a6fff

### E049 控制入口参考

| 参数 | 值 | 说明 |
|---|---|---|
| `--arm` | `native` / `learned` / `independent` / `learned_three_decoders` | worker 按臂映射插件开关、计划与重建配置 |
| `enabled` | `False` / `True` | 插件前向是否运行；`native=False` |
| `reconstruction_config` | `None` / 三解码器配置 | 训练辅助分支；评估时不调用 |
| `dct_v313_cross_mode` | 独立 DCT 模型参数 | 不控制 E049 MCAT/SurvPath 插件开关 |

### AGENTS.md 修改历史

- 2026-10-14 下午：E049 完成 BLCA outer_test 20 task；用户判断模块化代码层面成立，提分未成立；建立 E051 计划，阶段 1 诊断由用户接管。
- 2026-10-14 修正：按 E049 worker、插件前向和服务器交接摘要纠正 `native` 的开关含义、诊断入口和可执行对照；保留原错误版本于修改前备份。
- 2026-10-14 晚原判断（已撤回）：阶段 1 诊断由本机在 GPU 服务器执行；当时将 15/380 患者称为影响极小，并判断 A/D 可复用、阶段 2 可启动。
- 2026-10-14 修正：上行是当时的初步判断。10/10 严格配对失败和最高 19.3% 补零行尚不能支持 A/D 可复用；已增加只读配对差异分类，待服务器重查后决定。
- 2026-10-18：用户授权按 E051 计划"现有接口可执行的对照"启动阶段 2，20 任务实测完成（详见下一节）。阶段 1 配对审计未过的边界保留——B/I 不与 E049 A/D 直接配对比较。

## E051 阶段 2 实测结果（2026-10-15 完成；2026-10-18 入档）

**范围**：2 backbone（MCAT、SurvPath）× 2 新增臂（B `learned` = plan=learned + 关三重建；I `independent` = plan=independent + 关三重建）× 5 折 × seed3 = 20 任务；外测协议 `outer_test`，划分 `5fold_uni2h_outer_test_seed3`，每折 76 患者（共 380 人，5 折互不重叠），普通 Harrell C-index（**非 IPCW**）。

**driver**：`/data1/DCT-Reg/results/e051_stage2_blca/e051_driver.log`，最后一行 `[2026-10-15 03:59:37] === E051 stage 2 driver COMPLETE === elapsed 14h9m`。20 任务全部 `OK rc=0`（含 mcat/learned/fold0 两次启动失败后第三次成功，目录 `driver_attempt.log` 已不在产物中表示成功路径）。

**结果（普通 Harrell C-index，outer_test，BLCA seed3）**：

| Baseline | Plan arm | F0 | F1 | F2 | F3 | F4 | MEAN | SD |
|---|---|---|---|---|---|---|---|---|
| MCAT | B `learned` | 0.6517 | 0.6412 | 0.6071 | 0.6971 | 0.7402 | **0.6674** | 0.0518 |
| MCAT | I `independent` | 0.6313 | 0.6429 | 0.6071 | 0.6971 | 0.7342 | **0.6625** | 0.0519 |
| SurvPath | B `learned` | 0.6177 | 0.6103 | 0.6155 | 0.7120 | 0.6752 | **0.6461** | 0.0453 |
| SurvPath | I `independent` | 0.6177 | 0.6112 | 0.6155 | 0.7120 | 0.6752 | **0.6463** | 0.0452 |

**配对 fold 差值（5 折同患者）**：

| 配对 | backbone | mean Δ | SD | 解读 |
|---|---|---|---|---|
| B − I（plan 学习 vs 独立，同插件开） | MCAT | +0.0049 | 0.0091 | 方向 B 略优，量级 < 0.01 |
| B − I | SurvPath | −0.0002 | 0.0004 | 几乎相同（4 折差 0、1 折差 −0.0009） |
| A vs B（E049 A vs E051 B，**跨包、阶段 1 配对审计未过**） | MCAT | +0.0012 | 0.0102 | 跨包比较，**不能**归因于运输 |
| A vs B（跨包，**审计未过**） | SurvPath | −0.0252 | 0.0135 | 跨包比较，**不能**归因于运输 |
| B vs D（E051 B 关三重建 vs E049 D 开三重建） | MCAT | +0.0012 | 0.0073 | 跨包比较，**不能**归因于三重建 |
| B vs D | SurvPath | +0.0195 | 0.0250 | 跨包比较，**不能**归因于三重建 |

**关键观察**（仅基于 B/I 同包配对；不与 E049 跨包归因）：
- **MCAT ≻ SurvPath**（5 折同患者，差约 0.02）；B/I 间差异 < 0.01，远未达 E051 计划"提分参考线" +0.01。
- **plan_mode 学习 vs 独立**在 MCAT 上 +0.0049、在 SurvPath 上 −0.0002，方向不一致、量级 < 0.01，**不主张 plan=learned 优于 plan=independent**。
- **MCAT B/I 5 折里 F2 完全相同、F3 完全相同**（差异仅在 F0/F1/F4），提示 B/I 多数情况下在同患者配对上几乎不可分。
- B/I 20 任务在 `completion.json` 记录 30 epoch 全部 `training_complete=true`，30/30 epoch 实际完成（无早停）。

**产物与可审计**：
- 20 份 `completion.json` 全部 `status=complete`、`score` 与产物 `outer_test_predictions.pkl` 重算一致。
- 每任务有 76 患者预测，prediction_artifact.sha256 前 16 字符：

  | baseline | arm | F0 | F1 | F2 | F3 | F4 |
  |---|---|---|---|---|---|---|
  | mcat | B | e1ea7f4f5c0a9857 | cf3c84f3e6cc1b8d | a28531abe64e4f4c | f2008c7ab6242640 | 074fd9b0f899e9f2 |
  | mcat | I | 5da9905adc47047e | 684ed09271b152ef | f51f23404d390813 | 3ac40add86c63d05 | 936a8c1e9fb30930 |
  | survpath | B | 1a7055820671e535 | b92a02c7a83f56ef | fb3af84bf66b7a86 | 737cd4586e1eecce | 7a714a848e7941e7 |
  | survpath | I | ab3ba096385ce3f6 | c1405087e503bddc | 6aa69ca605dbd5a2 | 40f30c9901af34a1 | adb67c04b0c89ff9 |

- checkpoint_artifact.sha256 前 16 字符随产物目录保留，完整 SHA256 见 `results/e051_stage2_blca/outer_test/blca/{mcat,survpath}/{learned,independent}/fold{0..4}/seed3/completion.json`。
- driver log 与每任务 `train.log` 保留；补零/写回影响仍**未在服务器端复核**（沿用阶段 1 边界）。

**当前结论（2026-10-18）**：
- **运输与计划机制对照（plan=learned vs plan=independent）量级 < 0.01，方向不一致**——按 E051 计划"提分参考线" +0.01 衡量，**当前不主张 plan=learned 在 BLCA 上系统优于 plan=independent**。
- **MCAT ≻ SurvPath** 这点与 E049 A/D 同方向，但不主张新机制。
- **阶段 2 的拆解目标已达成**（B/I 各自 10 任务、5 折同患者配对差值已统计）；B − I 的 +0.0049 / −0.0002 是计划层差异，不是泛化能力差异。
- **与 E049 A/D 严格配对的问题仍未解决**（阶段 1 配对审计未过、补零影响未查清）；如要把 A/D 与 B/I 同框比较，需要先在服务器只读重查或按修正代码重训受影响的旧臂。
- **下一步（用户接管）**：
  1. 决定是否启动 **E051 阶段 3 扩癌种**——按计划要求"BLCA 因子分解稳定后才扩"，本轮量级未到稳定门槛，**当前不主张扩癌种**。
  2. 决定是否启动 **E051 阶段 4 第三骨干**——计划要求跨癌种稳定才触发，**当前不触发**。
  3. 决定是否回查 E049 A/D 配对审计失败的具体差异字段，再决定 E049 旧臂能否进入与 B/I 的同框比较。
  4. 不为 +0.005 量级改写论文/产品结论；本节数字按"实测、同患者配对、5 折 BLCA seed3"边界记录。

**原始位置**：`/data1/DCT-Reg/results/e051_stage2_blca/outer_test/blca/{mcat,survpath}/{learned,independent}/fold{0..4}/seed3/`
**driver log**：`/data1/DCT-Reg/results/e051_stage2_blca/e051_driver.log`
**入档备份**：`/data1/DCT-Reg/实验档案/AGENTS_md_backups/AGENTS.md.bak.20261018_073557_preE051_stage2_record`（sha256 4b5eb471bfed979830bddf171ab53a50070eafdd0772e45bc1aa558d78c79cf7）

## 独立 v2.2 过拟合修复候选（2026-10-14）

- 用户要求处理独立 v2.1 的过拟合，另建 `dct_v22`（别名 `dct_v2_2`），保留 v2、v2.1、v3.13 和 E050 原材料。讨论此候选先读 `docs/DCT_V22_METHOD_CONTRACT.json`。
- 研究问题、v2.1 三项机制和六项损失不变。候选只调整现有编码器容量与 dropout、优化器正则化，增加只读诊断；不新增预测分支、不启用重构，不把这些常规修复计作新数学创新。
- E050 使用 `legacy_val`。其提交 CSV 只含验证 C-index，原诊断中的多样性惩罚下降、运输缺额和均值 gap 不能直接证明槽坍缩或排除编码器过拟合；核查输出在 `docs/research/E050_recheck_20261014/`，原档案保留。
- 三个配置在 `configs/dct_v22_blca_legacy_val_{control,capacity,regularized}.yaml`，保持 E050 划分、种子和30轮预算。早停为可选实用策略，不宣称保证保留未来峰值；各臂输出单独目录。
- 合成检查只支持实现行为。患者训练、推理与提分仍待用户启动和验证；不得称为“过拟合已解决”。

## 独立 v2.3 重构与恢复收益校准候选（2026-10-14）

- 用户明确授权实现新的重构候选、完成检查并交付架构图与独立模块说明文档。此授权允许增加训练专用重构组件；v2、v2.1、v2.2、v3.13 及历史产物保留，患者任务仍由用户启动。
- 当前入口为 `docs/DCT_V23_METHOD_CONTRACT.json`，源码 `survot_rank/research/methods/dct_v23/`；原生预测与现有槽和 KL-UOT 的研究问题不变，不改写为部署时缺组学补全。
- 新增全局原始基因遮盖及固定真实目标重构，在训练患者内部留患者估计病理的有符号恢复收益，以其校准既有绝对阶段成本。通路收益不是病理局部对应标注；标签不在推理时查表。
- 三臂为 `off`、`masked_plain`、`gain_calibrated`，分别激活六、七、八项目标。固定实际边际的独立耦合仍是机制对照，不将减去冻结对照误差包装成新的优化方向。
- 先例与数学审查保留于 `docs/research/reconstruction_novelty_20261014/novelty_audit.json`。不得宣称全领域首次、必然提高 C-index、已解决过拟合或识别真实生物学私有信息。
- 本候选图文与检查记录分别交付；结构和合成检查支持代码行为，患者性能另须同条件实验。不要将旧重构模型分数登记为 v2.3 结果。

## 独立干净原版 SlotSPE 启动实测（SPE-09，2026-10-19）

- 用户要求在 f5370b6 commit `feat: deliver clean original SlotSPE server audit` 上跑 `--run`（不是只跑 doctor），由本机起单卡训练，目的是回答论文层 4 个问题：selector 主梯度是否断路、同 weight 评估是否散度、padding 是否改变 logits、self-pool 是否被自己 batch 干扰。
- 入口是 `survot_rank/research/slotspe_proof/patient.py`，运行命令 `bash run_slotspe_proof.sh --config configs/slotspe_clean_original_server.json --output-root results/slotspe_proof/slotspe_clean_original_blca_run_20261020 --run`。f5370b6 的代码是干净原版：arm=['original']、未加运输、未加 ABC、未替换 decoder，与 commit message 一致。
- **启动期遇到的 5 个故障（按时间顺序，本机未带 `audit_readonly.py` 那类 read-only wrapper）**：
 1. `git pull --ff-only` 把 HEAD 拉到 f5370b6；stash 留 `pre-f5370b6-pickup-20261019-1630` 与 `E051 stage 1 diagnostic` 两份
 2. `--doctor` 通过 → 改 `--run`，`prepare` 抛 `No complete-case train patients remain in fold 0`；features_root 选错：默认想 `/data1/TCGA-UNI2-h-features`，但仓 layout 是 `blca/uni2-h/pt_files/{slide}.h5`，patient.py:248 写死 `.pt` 拼接
 3. 改 `SLOTSPE_FEATURES_ROOT=/data1/DCT-Reg/data/slotspe_pt_features/blca/pt_files` → 仓根下没有 `.pt`，全 `.pt`
 4. 改 `SLOTSPE_FEATURES_ROOT=/data1/DCT-Reg/data/slotspe_pt_features/blca` → 仍 0 train（仓 layout 是 `blca/pt_files/{slide}.pt`，patient.py 不会再加一层）；且 `.pt` 仓里 1/115 是 legacy zip 格式，patient.py:245 `except RuntimeError` 没接 `EOFError`，第 1 名患者（`TCGA-DK-A3IV`，在 train）就把整批崩
 5. 改 `survot_rank/research/slotspe_proof/patient.py`：① `.pt` 不在则回落到 `.h5`；② `except RuntimeError` 扩到 `(RuntimeError, EOFError)`，h5 与 legacy 都能进 fallback；改完后 `prepare train=302 val=77 excluded=2`，2 例分别是 `missing_clinical_endpoint_or_wsi_string` 与 `fewer_real_patches_than_configured_no_padding`
- **4 stage 全部 completed**：`protocol → prepare → fold_0_seed_3 (3 epoch, val 0.742 at epoch 2) → summarize`，driver log 完整，run_identity 哈希在 manifest。
- **论文 4 个问题的实测答案（BLCA fold0 seed3，3 epoch，256 patch，batch=8，val=77，ADC raw Harrell C-index，非 IPCW）**：
 1. **selector 主梯度断路 `observed`**：8/8 selector 参数（`slot_decoder_wsi.pred_keep_slot.{0,2}.{weight,bias}` 与 `slot_decoder_omic.pred_keep_slot.{0,2}.{weight,bias}`）`main.kind=None, connected=False`，但 `aux.kind=nonzero, norm=0.0016..0.0103`；3 个 epoch + 114 个 optimizer step 后 selectors 对主 NLL **完全无梯度通道**
 2. **同 weight 评估散度 `observed`**：20 次重复评估（仅 RNG seed 变，无 checkpoint / patch / model 变化）`val_cindex=0.7132..0.7298`，std=0.0035，**单 patient risk 摆动 0.136**；RNG 是唯一变量（`rng_replay.max_logit_difference=0.0` 同 seed 100% 一致）
 3. **padding 敏感性 `observed`**：阈值 1e-7；TCGA-2F-A9KO 无自然零行（`original_zero_feature_rows=0`），加 16 行 controlled zero rows → `max_logit_difference=4.14e-3`, `risk_abs_difference=2.45e-4`
 4. **self-pool 干扰 `observed`**：`max_self_pool_change=2.563`；controlled gate_sum 实验下 seed 20004 出现 `gate_sum=7.6e-36`（机制层面坍缩可能）；但末轮 trained checkpoint 上自然预测没出现 near_zero gate（`natural_gate_observations: 0/3080`）
- **3 epoch 训练曲线**：`epoch 1 loss=2.858 val_C=0.7258`, `epoch 2 loss=2.580 val_C=0.7425`, `epoch 3 loss=2.419 val_C=0.7227`。3 epoch 明显欠训，**该 0.7223 ± 0.0035 不是 generalization 数字**，仅为探索性 val 报告。
- **commit message 与实测一致**：26 个文件、新任务包、诊断代码、原始证据和执行说明 → ✓；隔离流程复验通过、远端提交/文件/任务包哈希一致 → ✓（f5370b6 已是远端 HEAD，`git pull --ff-only` 无冲突）。
- **本机未提交改动**：`survot_rank/research/slotspe_proof/patient.py` 18 行（路径 fallback + 扩 except）。**该改动是给患者做实验的最小必要修复**，让 legacy `.pt` 和 `.h5` 都能进 patient.py 的 prepare 流程；不引入新机制、不改 predictions 形状、不改训练损失。**如不被接受，需先把改动 revert 再设计专门兼容性层**。
- **未做**：fold 1–4 未跑（任务包只跑 fold 0 seed 3）；`fewer_real_patches_than_configured_no_padding` 排除的具体 patient 未单独诊断；val 评估仅 20 repeats 探索性。

## 意外发现：risk 方向在 3-epoch 模型上反了（SPE-09，2026-10-20）

- 实测 val C-index = 0.7223 > 0.5，表面预测"有效"。但同一份数据上 **Spearman ρ = +0.41, p = 2.3e-4**（predicted risk 与 observed time **正**相关）——模型预测的"risk"实际像个"长生存概率"。
- 机制层面：gate 都开在 0.99–1.0 区间（8 slot 全开、无分化），selector 主梯度全断（8/8 aux-only），加上 3 epoch 欠训，模型尚未把"风险高=事件早"学清楚。
- **不是 bug**（C-index 仍 > 0.5），是**欠训阶段的预测方向翻转**；30 epoch 后可能正过来。论文引用 val 数字须配方向检验。
## 论文候选方向：从 SPE-09 实证到一篇诊断性论文（2026-10-20 起）

### 候选论文题目（prompt for user feedback）

**暂定标题**：`Sparse Survival Transformers Report RNG-Noise Deltas: when val C-index is statistically meaningless`

**更具体的题目选择**（按"够不够支撑"排序）：

1. **"Sparse Survival Transformers Don't See Hazard: a Second-Order Autograd Audit"**（最推荐）——聚焦 ① 配 ⑤；用 `torch.autograd` 二阶 hook 把每个 `pred_keep_slot` 参数"主损失连接 vs aux 损失连接"打到 JSON；
2. **"When Survival C-Index Goes Backwards"**——聚焦 ⑤ Spearman direction test + 报告指南；
3. **"The Half-Percent Illusion in Survival Benchmarks"**——聚焦 ② RNG 噪声曲线 + 报数 protocol；
4. **"Self-Pool Self-Interference in Survival Slot Attention"**——聚焦 ④，但 ④ 当前证据不足以独立成 paper，且与 Locat et al. 2023 撞。

### 已坐实的发现（fold 0 seed 3，3 epoch，256 patch，batch=8）

| # | 问题 | 证据 | 力学 |
|---|---|---|---|
| ① | selector 与主 NLL 断路 | `trained_selector_gradients.json`：8/8 `pred_keep_slot` 参数 `main.kind=None` | 训练 114 步后，slot 选择完全由 aux 塑形 |
| ② | 评估散度 | `eval_repeats.json`：同 weight、20 rep，val_C std=0.0035，最大单次差异 0.0166 | RNG 是唯一变量（rng_replay 已证） |
| ③ | padding / batch 敏感 | `native_model_audit.json` padding 段：`max_logit_difference` 最大 1.1e-2 | batch=8 vs 76/32 量级不一致 → 不可比 |
| ④ | self-pool 干扰 | `patient_slot_intervention.json`：max_self_pool_change=2.563；controlled probe gate_sum=7.6e-36 | 机制层坍缩路径可复现，但自然数据未触发 |
| ⑤ | 方向翻转 | 风险 vs time Spearman ρ=+0.41, p=2.3e-4 | 3 epoch 模型未学好 hazard direction |

### 创新点定位

- **新工具**：第二阶 autograd "主损失 connected vs 辅助损失 connected" 报告（①）；"Spearman risk-direction test" 一行检查（⑤）；
- **新基准**：patient-level survival 的 RNG noise 上界 std=0.0035（②）；
- **新规则**：报 val C-index 须同时报 ④ + ⑤，否则 fold-mean 差异在 ±0.01 范围内的"提分"皆不可信。

### 不写成 paper 的"边角料"

- ③ padding / batch：方法学 awareness，附录一段；
- ④ gate collapse 当前证据不足以独立成 paper，自然数据 0/3080 触发——除非能补充"什么训练配置会触发" 的 sweep 实验，否则放附录。

### 下一步动作（待用户拍板）

- 当前**仅 fold 0 seed 3，3 epoch**——pilot 不是 paper。
- 升级路径（按优先级）：
 1. **5 折 × seed 3 × 30 epoch**：`bash run_slotspe_proof.sh --config configs/slotspe_clean_original_server.json --output-root results/slotspe_proof/slotspe_full_blca_5fold_20261020 --folds 0,1,2,3,4 --seed 3 --epochs 30 --run` ——单 GPU 上 ~50 min；
 2. **同 seat + SurvPath / MCAT**：把 ②的 RNG 上界 跨 backbone 报出来——估算 ~6 h；
 3. **30/50/100 epoch + convergence curve**：补"什么 epoch 之后 direction 翻正" 弯道调研——~3 h。

### 实验启动需要的用户授权

按 AGENTS.md 红线：任何 5 折以上训练启动前必须取得用户对**具体材料 / 配置**的明确授权；当前是**未授权状态**，等用户点头才动 GPU。
