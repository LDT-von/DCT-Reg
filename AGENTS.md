# v313 实验资料保护与主档维护

主档只记录v3.13相关实验。用户已明确授权本轮清理非v3.13归档实验与代码；该授权不是对后续清理的永久授权。

- 未取得用户对具体材料的明确授权，不得删除、移动、批量覆盖档案、代码、配置、结果、日志、预测或checkpoint。
- v3.13撤回、失败和旧图版本仍保留状态与来源。
- v3.13继承v3.11与v3.10的模型代码，文件名带旧版本不等于非v3.13依赖；继承基类、共同训练模块和底层组件不得按版本号删除。
- 修改主档前保留备份；已有编号不复用，E045已登记（配置 dct_v313_full_wsi_three_decoders.yaml，三解码器消融，BLCA seed3 legacy_val 5-fold，mean val c-index 0.7256 ± 0.0317（95%CI跨零，统计不显著），详见 实验档案/表格图片/E045_三解码器重建消融/，口径为 legacy_val inner-val peak，不可与 E046 outer_test 数字直接比），E046已登记（代码就绪 + 真实训练完成：outer_test 5×3 + legacy_val 5 arms wsi_additive + 5 arms E046b full_replay + E046c 原始 commit ckpt 0.0 误差复刻审计 5 折 380 患者，详见 实验档案/表格图片/E046_WSI自重建消融/ 与 实验档案/表格图片/E046b_Full_replay_legacy_val/ 与 实验档案/表格图片/E046c_原始commit_replay_审计/，主档 records_count=24 含 E046c），E047已记录（BLCA效率对比：DCT Full部分实测完成，外部方法因本机无外网无法下载源码无法完成，需有外网环境重做），E048已登记（v313模块与单模态消融，代码包就绪，未启动训练，详见 实验档案/实验代码/E048_v313模块与单模态消融/），E049已登记（v313六队列版本对照，30任务 = 6癌种×5折×seed3 独立三解码器新版，原版六项分数冻结复用 baseline_frozen.json 来源 paper/V313_TEN_CANCER_REPORTED_COMPARISON_INPUT.json sha256 d8aeeeb7...，本机已核验 30 任务计划、frozen recipe、baseline SHA256 一致；未启动训练，详见 实验档案/实验代码/v313六队列版本对照_20261013/ 与 实验档案/表格图片/v313六队列版本对照_20261013/），下一项实验从E050继续。
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
- v2.1 是本地候选；患者训练与性能仍未验证。不得把 v2/v3.13 分数或旧检查数量记作 v2.1 的患者结果；推送状态以其独立交付记录为准。
