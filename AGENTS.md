# v313 实验资料保护与主档维护

主档只记录v3.13相关实验。用户已明确授权本轮清理非v3.13归档实验与代码；该授权不是对后续清理的永久授权。

- 未取得用户对具体材料的明确授权，不得删除、移动、批量覆盖档案、代码、配置、结果、日志、预测或checkpoint。
- v3.13撤回、失败和旧图版本仍保留状态与来源。
- v3.13继承v3.11与v3.10的模型代码，文件名带旧版本不等于非v3.13依赖；继承基类、共同训练模块和底层组件不得按版本号删除。
- 修改主档前保留备份；已有编号不复用，E045已登记（配置 dct_v313_full_wsi_three_decoders.yaml，三解码器消融，BLCA seed3 legacy_val 5-fold，mean val c-index 0.7256 ± 0.0317（95%CI跨零，统计不显著），详见 实验档案/表格图片/E045_三解码器重建消融/，口径为 legacy_val inner-val peak，不可与 E046 outer_test 数字直接比），E046已登记（代码就绪 + 真实训练完成：outer_test 5×3 + legacy_val 5 arms wsi_additive + 5 arms E046b full_replay + E046c 原始 commit ckpt 0.0 误差复刻审计 5 折 380 患者，详见 实验档案/表格图片/E046_WSI自重建消融/ 与 实验档案/表格图片/E046b_Full_replay_legacy_val/ 与 实验档案/表格图片/E046c_原始commit_replay_审计/，主档 records_count=24 含 E046c），E047已记录（BLCA效率对比：DCT Full部分实测完成，外部方法因本机无外网无法下载源码无法完成，需有外网环境重做），E048已登记（v313模块与单模态消融，代码包就绪，未启动训练，详见 实验档案/实验代码/E048_v313模块与单模态消融/），E049已登记（v313六队列版本对照，30任务 = 6癌种×5折×seed3 独立三解码器新版，原版六项分数冻结复用 baseline_frozen.json 来源 paper/V313_TEN_CANCER_REPORTED_COMPARISON_INPUT.json sha256 d8aeeeb7...，本机已核验 30 任务计划、frozen recipe、baseline SHA256 一致；未启动训练，详见 实验档案/实验代码/v313六队列版本对照_20261013/ 与 实验档案/表格图片/v313六队列版本对照_20261013/），E050已登（独立 DCT v2.1 真实患者训练，BLCA 5折 seed3 legacy_val，三新机制默认，GPU 1，后台 driver，输出 /data1/DCT-Reg/实验档案/结果/E050_v21_20261014/，代码 origin/main @ af5da64，对比基线 E045 legacy_val 0.7256 ± 0.0317），下一项实验从E051继续。
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
- 主档已登记：E045–E050。下一项实验从 **E051** 起编号，不复用。
- 失败 / 撤回 / 旧图版本保留状态和来源。

---

## E051 计划：BLCA 诊断 + 插件机制分解（2026-10-14；当日修正）

### 背景

E049（BLCA，2 骨干 × 2 臂 × 5 折 = 20 任务）结果：MCAT native 0.6686，MCAT learned_three_decoders 0.6662（−0.0024），SurvPath native 0.6209，SurvPath learned_three_decoders 0.6267（+0.0058）。完整插件臂**同时含学习运输写回和三重建**，这组差值不能拆分各自贡献。

**重要口径修正**：E049 `native` 在 `worker.py` / `model.py` 中是 `enabled=False, reconstruction=False`；`build_native_adapter()` 将该开关交给插件，插件前向直接返回原特征。因此它是功能上的无插件基线，不是运输仅臂。服务器交接的 `model_audit.json` 摘要也报告 `enabled=False`（完整文件未提交，本机尚未独立复核）。共享 YAML 的 `dct_v313_cross_mode=transport` 属于独立 DCT v3.13 模型配置；E049 worker 用自身的适配器构造器替换 `runner.get_model`，不能根据该 YAML 键推断 MCAT/SurvPath 插件是否开启。

内部参考线：+0.01 值得继续投入，+0.02 在多个癌种与骨干上稳定出现更有说服力；它们不是统计阈值。正式判断看同患者配对分数差及不确定性、实际成本（参 PMC4314453）。

### 阶段 1：BLCA 诊断（用户接管）

**目标**：核对 (a) 旧训练入口是否可能接收补零行，(b) 20 个任务的标签、来源、分数与 checkpoint 是否一致，(c) 插件臂训练后的残差增益是否偏离零。只读脚本不运行模型，增益非零也不能单独证明运输改变了风险预测；这一点还需同 checkpoint 的插件开/关推理对照，由用户决定执行。

- 脚本：`实验档案/实验代码/E049_v313运输插件MCAT_SurvPath基线比较/audit_readonly.py`；`evidence.py` 是预测、标签与划分核验库，不是补零/写回诊断入口
- 运行位置：内网（服务器）
- 命令：`/home/ubuntu/.conda/envs/dct/bin/python3.10 /data1/DCT-Reg/实验档案/实验代码/E049_v313运输插件MCAT_SurvPath基线比较/audit_readonly.py --batch-dir /data1/DCT-Reg/results/e049_blca`
- 判定：
  - 补零影响输入 → 修复后**两臂一起重跑**原生与插件组
  - 补零不影响 → 不为该修复单独重复旧实验

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
