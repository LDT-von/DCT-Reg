# v313 实验资料保护与主档维护

主档只记录v3.13相关实验。用户已明确授权本轮清理非v3.13归档实验与代码；该授权不是对后续清理的永久授权。

- 未取得用户对具体材料的明确授权，不得删除、移动、批量覆盖档案、代码、配置、结果、日志、预测或checkpoint。
- v3.13撤回、失败和旧图版本仍保留状态与来源。
- v3.13继承v3.11与v3.10的模型代码，文件名带旧版本不等于非v3.13依赖；继承基类、共同训练模块和底层组件不得按版本号删除。
- 修改主档前保留备份；已有编号不复用，E045已登记（配置 dct_v313_full_wsi_three_decoders.yaml，三解码器消融，BLCA seed3 legacy_val 5-fold，mean val c-index 0.7256 ± 0.0317（95%CI跨零，统计不显著），详见 实验档案/表格图片/E045_三解码器重建消融/，口径为 legacy_val inner-val peak，不可与 E046 outer_test 数字直接比），E046已登记（代码就绪 + 真实训练完成：outer_test 5×3 + legacy_val 5 arms wsi_additive + 5 arms E046b full_replay + E046c 原始 commit ckpt 0.0 误差复刻审计 5 折 380 患者，详见 实验档案/表格图片/E046_WSI自重建消融/ 与 实验档案/表格图片/E046b_Full_replay_legacy_val/ 与 实验档案/表格图片/E046c_原始commit_replay_审计/，主档 records_count=24 含 E046c），E047已记录（BLCA效率对比：DCT Full部分实测完成，外部方法因本机无外网无法下载源码无法完成，需有外网环境重做），E048已登记（v313模块与单模态消融，代码包就绪，未启动训练，详见 实验档案/实验代码/E048_v313模块与单模态消融/），E049已登记（v313六队列版本对照，30任务 = 6癌种×5折×seed3 独立三解码器新版，原版六项分数冻结复用 baseline_frozen.json 来源 paper/V313_TEN_CANCER_REPORTED_COMPARISON_INPUT.json sha256 d8aeeeb7...，本机已核验 30 任务计划、frozen recipe、baseline SHA256 一致；未启动训练，详见 实验档案/实验代码/v313六队列版本对照_20261013/ 与 实验档案/表格图片/v313六队列版本对照_20261013/），E050已登记（独立 DCT v2.1 真实患者训练，待定范围，代码 origin/main @ af5da64，GPU 1，dct env；具体癌种/划分/epoch/输出目录待用户逐项授权后启动），下一项实验从E051继续。
- 按患者、划分、配置与原始产物核对结果。服务器未直接全量扫描时，不宣称所有实验已收齐。

## E049 进度更新（2026-10-14）：BLCA outer_test 20 任务已实测完成

**范围**：E049 v313六队列版本对照首批仅 1 队列 = BLCA × 2 arm × 5 fold × seed3 = 20 任务。其它 5 队列（BRCA/LUAD/COADREAD/KIRC/STAD）尚未启动，原 30 任务计划仍冻结。

**结果（IPCW cindex，每 fold outer_test 76 患者）**：

| Baseline | Arm | F0 | F1 | F2 | F3 | F4 | MEAN |
|---|---|---|---|---|---|---|---|
| MCAT | native | 0.6466 | 0.6378 | 0.5996 | 0.7147 | 0.7444 | **0.6686** |
| MCAT | learned_three_decoders | 0.6500 | 0.6300 | 0.6052 | 0.6962 | 0.7496 | **0.6662** |
| SurvPath | native | 0.5786 | 0.5700 | 0.6043 | 0.6936 | 0.6581 | **0.6209** |
| SurvPath | learned_three_decoders | 0.6075 | 0.5639 | 0.6165 | 0.7164 | 0.6291 | **0.6267** |

**关键观察**：
- MCAT 整体优于 SurvPath（0.667 vs 0.624，差 ~0.043）。
- `learned_three_decoders` 在 MCAT 上**未带来增益**（0.6662 vs 0.6686，差 −0.0024，**非显著**），WSI 特征重构插件在 MCAT 已能学习 WSI 模式时冗余。
- `learned_three_decoders` 在 SurvPath 上 +0.0058 极弱增益。
- 最佳单配置：MCAT native fold4 = **0.7444**。

**已验证的产物完整度**（20/20 task 全部 `status=complete`）：
- `completion.json` 含 `score`、`prediction_patient_count=76`、`label_audit/checkpoint/prediction` 三个 SHA256（防篡改）。
- `outer_test_predictions.pkl`（76 患者 × `{risk, censor, time, logits}` 4-key dict），可重算任意 cindex 变种（naive / IPCW / Uno's / Harrell）。
- `label_audit.json` 76 位患者 time/censor/source_time 三元组。
- `model_audit.json`：`omic_sizes=[81,302,497,418,1404,422]`、`wsi_input_dim=1536`、`enabled=true`、`plan_mode=learned`、`reconstruction=true/false`、`decoder_mode=separate`。
- `model_best_s{fold}.pth`（~46 MB），plugin-only keys 在 `learned_three_decoders` 臂下多 46 个（`adapter.plugin.reconstruction.{self_decoder,cross_decoder,wsi_decoder}.*`），shared keys L2 差异 < 6（两实验独立训练，非数据泄漏）。
- 4 个 orchestrator 修复 commit 已 push 到 main（详见下一节）。

**原始位置**：`/data1/DCT-Reg/results/e049_blca/outer_test/blca/{mcat,survpath}/{native,learned_three_decoders}/fold{0..4}/seed3/`

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
