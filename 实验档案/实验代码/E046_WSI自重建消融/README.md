# E046：v3.13 可选 WSI 自重建消融

状态：代码与合成检查就绪，未运行真实训练/患者推理，未产生新 C-index 或图表。
这是 v3.13 的候选辅助分支，默认权重为 0，不替换已有 Full，也不将既有结果改名为新结果。

## 模块做什么

WSI 原始特征经原有投影器得到 patch token，原有 WSI 语义槽作为 decoder 的 key/value。
对 detached patch token 做固定随机线性投影生成 patch query，再通过交叉注意力解码 patch 特征。
decoder 输出没有 query/target 的残差直通。重建目标和 query 输入都 detach；随机投影不训练，
但槽与 decoder 可训练，WSI 重建梯度经槽反传到 WSI 编码器。

这属于**以目标特征为 query 条件的自重建**，不是预测未观测 patch，也不是把 WSI 图像像素还原。
该辅助任务不单独证明生物学解释、OT 对应有效或模型创新性。v3.13 原有创新假设仍是：
事实 OT 计划同时用于预测和 WSI→omics 的 transport-aware cross 重建。

loss 为逐 patch 的 1−cosine，先对每位有效患者的有效 patch 求均值，再对有效患者求均值。
原始 WSI 全零行按当前数据集约定视为补零；额外支持 wsi_patch_mask（True=有效），
wsi_available 和 wsi_missing。无有效 patch 时返回有计算图的 0。
这些掩码只用于新增辅助损失，未改动原有预测编码器的 padding 行为。

query 按 chunk_size（默认 256）分块，减少单次注意力工作空间；训练仍保留各块反传激活，
不能据此宣称显存与 patch 数无关。eval 完全跳过辅助 decoder；训练时增加参数与计算量，
推理 checkpoint/模型加载时仍包含启用分支的参数，不能直接声称模型体积不变。

## 三组对照

以下系数为 ramp 达到 1 后；患者 NLL、IPCW、逐槽 NLL、多样性、encoder、划分、
patch 数、训练轮数、优化器均沿用同一 base-config，不人为修改其既有实际权重。

| arm | omics self | transport cross | WSI self | 重建系数和 |
|---|---:|---:|---:|---:|
| full | 0.05 | 0.05 | 0 | 0.10 |
| wsi_additive | 0.05 | 0.05 | 0.05 | 0.15 |
| wsi_fixed_total | 0.033333… | 0.033333… | 0.033333… | 0.10 |

fixed_total 将原有两支与原始 WSI 权重按共同倍率重缩放，保持原有**有效系数和**，
不保证损失数值、梯度范数或 decoder 参数量相同。此对照能排除单纯增加系数和的解释，
但不能完全排除参数量变化，也不能将其收益只归因于 WSI loss。
如手动关闭一个原有重建分支，fixed_total 保持原有剩余预算 0.05；
原有两个分支都关闭时该模式报错。新增分支启用必须选 per_branch，legacy 开启会报错。
默认关闭时不增加参数、不消耗额外初始化 RNG，旧 Full checkpoint 可严格加载。
启用后的 checkpoint 必须按保存的 WSI weight/budget 构建再严格加载，不可用默认关闭的模型加载。

训练日志新增 v313_reconstruction_wsi、三个实际分支系数和总系数；
总 reconstruction 已包含三个加权项，再只乘一次原有 ramp（epoch≤2 为 0，epoch4 为0.4，epoch≥7为1）。
objective_weights_effective()/effective_recon_coefficients() 同样记录实际权重与策略。

## 运行

在仓库根目录运行。默认仅输出计划，不加载模型/患者数据，不创建结果目录。

~~~bash
python "实验档案/实验代码/E046_WSI自重建消融/run_wsi_reconstruction_ablation.py" --results-root /data1/results/v313_e046_outer_20261009 --cancers blca kirc --seeds 3 13 23 --protocol outer_test
~~~

核对计划中的数据路径、划分哈希、三组参数，再执行同一命令并加入 --execute：

~~~bash
python "实验档案/实验代码/E046_WSI自重建消融/run_wsi_reconstruction_ablation.py" --results-root /data1/results/v313_e046_outer_20261009 --cancers blca kirc --seeds 3 13 23 --protocol outer_test --execute
~~~

默认配置为 configs/dct_v313_uni2h.yaml，WSI 权重 0.05，5折，单GPU串行。
BLCA/KIRC×3组×5折×3seed=90次训练；--cancers all 是十癌种共450次训练。
可指定 --gpu 2（子进程可见设备仍为0）、--data-path、--data-root-dir、--which-splits、
--base-config；实验启动前固定权重，不能按各折结果调参。
如只复现已有开发协议，明确选 --protocol legacy_val，使用新的 results-root，
不可将最佳验证结果当独立测试结果。outer_test 需预先准备
5fold_uni2h_outer_test_seed3 对应划分，使用 inner_val_fraction=0.20、split_seed=3。
现有 evidence/manifest.py 审计工具仅支持 legacy_val；不要用它将 outer_test 强行标为 legacy_val。

执行前检查全部 split CSV 和特征根目录、拒绝非空结果目录；保存 base-config 副本、
原始配置/划分/关键源码 SHA-256、源 commit、逐任务覆盖参数、单fold范围与日志。
源码与划分在运行中改变会终止。所有 arm/fold/seed/protocol 各有独立目录。
中断或失败后保留全部材料，再用新批次目录重启；本入口不自动覆盖或续跑。

## 结果怎么判断

先按同癌种、同折、同seed、同协议匹配患者/结局/划分与训练配置，
确认最佳checkpoint由同一规则选取，再比较配对 ΔC-index，报告均值、样本标准差与逐折结果。
outer_test 使用内层验证选出的 checkpoint 对外层测试的一次评估，不按测试分数选 epoch。
legacy_val 仅描述开发验证收益。多seed应按seed分别汇总五折，不能把相关的15折当15个独立样本。
若收益不稳定、只 additive 好或完全无收益，都如实保留；WSI 重建不是必须模块。

图表入口专用目录：实验档案/表格图片/E046_WSI自重建消融/。
代码留在本目录；模型实现仍在原 v313 包内，避免复制出第二套模型。
建议结果图：每癌种配对增益图、三支重建损失与 C-index 学习曲线、实测训练显存/耗时。
只用真实产物作图，图与数表必须保留 source/hash/protocol/seed。

72项合成/配置回归检查通过，不代表真实模型性能。实验目录 JSON 已登记 E046，
Word 主档未修改；等真实结果到位再备份主档并更新，下一编号 E047。
