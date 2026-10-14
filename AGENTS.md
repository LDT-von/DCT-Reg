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
- 主档已登记：E045–E050。新实验从 **E051** 起编号，不复用。
- 失败 / 撤回 / 旧图版本保留状态和来源。
