# 按 SlotSPE 原论文补充 DCT 图像

核对日期：2026-10-07。参考文件：用户本地 `03_槽注意力与结构化事件/STRUCTURAL PROGNOSTIC EVENT MODELING FOR.pdf`，标题为 *Structural Prognostic Event Modeling for Multimodal Cancer Survival Analysis*，36 页，正文和附录共 14 个 Figure 编号。以下来自完整图页的视觉检查与对应章节，不按旧 DCT 八面板病例图推断原文。

## 原论文有什么图

| 原图 | PDF 页 | 实际内容 | DCT 如何对应 |
|---|---|---|---|
| Fig.1 | 4 | SlotSPE 总架构：双模态槽、选择激活、模态交互、生存预测 | 保留我们自己的架构图，突出共享语义坐标、分阶段多几何 OT、事件预测、运输重建 |
| Fig.2 | 5 | MoE 槽选择和跨模态重建细节 | 展示我们实际的 OT 运输和通路查询重建，不照搬 MoE |
| Fig.3 / Fig.6 | 8 / 24 | 多癌种整体高低风险 KM；方法/缺失模态为行，癌种为列；另显示 RMST | 完成每癌种一组 pooled OOF 高低风险曲线。已有 `plot_fig5_km_curves.py`，缺的是完整 `--km` 导出 |
| Fig.4 / Fig.10 | 9 / 31 | C-index 与显存/时间散点；训练成本柱图；模块成本比例 | 作为补充材料，使用统一硬件/输入规模的真实 profile 数据 |
| Fig.5 | 10 | 主文解释图：WSI 槽分区、组学槽空间投影、每槽 Top-5 组织块、Top/Bottom-3 通路 | **本次重点：替换原来的总 attention 柱图，生成真实组织—通路病例图** |
| Fig.7 | 27 | ResNet50/CONCH/TITAN/UNI 编码器比较 | 有同协议训练结果时画；不同编码器的旧结果不直接混合 |
| Fig.8 | 27 | WSI/组学槽数量的组合消融箱线图 | 可选训练扩展，与本次从 checkpoint 出解释图分开 |
| Fig.9 | 28 | 槽迭代次数、MoE Top-K、温度消融 | 只借鉴本架构实际存在的参数，不凭空增加 SlotSPE MoE 参数 |
| Fig.11 | 32 | 60 月校准、DCA、KIRC 整体 KM | KM 优先；校准和 DCA 在统一时间点预测概率、删失处理齐全后扩展 |
| Fig.12 / Fig.13 | 34 / 35 | BRCA/UCEC 完整病例：原切片、两类槽分区、六槽 Top-5、通路×槽矩阵、逐槽 Top/Bottom-3，组织块有病理医师标注 | BLCA/KIRC 各一例主图，其余病例为补充；组织类型标签需要真实标注 |
| Fig.14 | 36 | 四个组的 Top-10 通路并集热图；每个组一列，通路一行 | **本次重点：四组通路图与真实数值 CSV，不再用重建误差或 hazard 乘权替代 attention** |

模块消融主结果在 Table 3/8；临床变量增量分析在 Table 10。缺失组学比较主要在 Table 2/6 和 KM 的额外行。

## 优先交付的图

1. **病例解释主图**：同一真实切片的 WSI 槽分区和 OT 投影组学槽分区；固定槽 0/1/2 的空间热图与 Top-5 组织块；对应组学槽的 Top-3/Bottom-3 命名通路。BLCA 先使用 `TCGA-2F-A9KP`，KIRC 使用已记录的验证病例。图标题和随访信息按 `export.json` 的病例 ID 映射获取。
2. **逐槽通路补充图**：全部组学槽的 Top/Bottom-3，以及通路×槽矩阵。保留 raw attention，不把所有柱子强制拉成 1，不把低关注称为生物学无关。
3. **DCT 运输关联图**：同一病例的学习 OT、同事实边际的独立计划、二者差值。突出 DCT 的运输结构；图中行列仍分别是 WSI 槽和组学槽。
4. **四风险组 Top-10 通路图**：BLCA/KIRC 各一图。左侧显示真实组均值，右侧显示该通路相对四组均值的差值，色标保留实际数量级。默认要求完整五折，显式 `--allow-partial` 才允许带 PARTIAL 水印的探索图。
5. **每癌种总体 KM**：沿用已有 pooled KM 入口。全部患者使用所属折的训练集风险中位数分组后合并；不从逐折 PNG 拼出总体曲线。

## 从 SlotSPE 借鉴到 DCT 的具体差异

原文 Fig.14 的 I.2 节写的是：取 MoE retention 最高的槽，按生存时间分成四个 bin，再平均组内通路 attention。它不是当前 DCT 的 hazard 加权患者排序图。

当前 DCT 导出没有这个 MoE retention 字段。我们的明确适配是：**真实 `attention_omic` 跨槽等权平均；按每折验证风险的 midrank 分四组**，Q1 较低风险、Q4 较高风险；它是预测风险关联的探索分析，不宣称复现原文的生存时间分组实验。风险相同的患者使用相同 midrank，不制造不同风险组。未来有训练参考的四分位数时可以增加独立分组协议，但不混用两套定义。

WSI 到组学槽的空间投影按当前 v3.13 cross 分支执行：每阶段先平均几何计划、按组学列质量归一化，再按 factual stage gate 汇总；`omics_spatial = B.T @ attention_wsi`。WSI 槽到命名通路为 `B @ attention_omic`。各轴使用不同符号，支持 WSI 与组学槽数量不相等，避免旧投影索引 bug。

病例组合图每行的 WSI 槽和组学槽编号仅为固定展示顺序，**不表示相同编号的两模态槽一一对应**。真正的跨模态关联看运输矩阵；右侧通路属于该行的组学槽。病例选择不根据最漂亮的热图或最高 C-index 临时更换。

## 新代码与输出

- `survot_rank/evidence/slotspe_style.py`：病例 ID 对应、正确 OT 投影、原始 patch 行定位、真实图片资源核验、五折通路汇总。
- `scripts/plot_v313_slotspe_style.py`：`case`、`pathways`、`coupling`、`cohort` 四个入口。只读取已导出的数据，不运行模型。
- `configs/v313_slotspe_assets.example.json`：真实切片/坐标/组织块资源契约。
- `tests/test_v313_slotspe_style.py`：异尺寸槽轴、padding/重复 patch、病例 ID、风险 ties、跨折通路重排、完整覆盖等合成验证；测试不代表真实图已生成。

病例输出为 `wsi_slot_assignments.{png,pdf,json}`、`tissue_pathway_case.{png,pdf,json}`；补充输出为 `slot_pathways.*`、`pathway_slot_map.*`、`transport_association.*`；队列输出为 `cohort_top_pathways.*`、`patient_groups.csv`、`pathway_group_means.csv`。既有图文件保留，使用新的输出目录。

## 服务器运行顺序

先恢复完整 attention 导出。如还缺 KM，直接使用正式 exporter 的 `--km`，**选新的输出根目录**，不移动已有目录、不放宽 replay 容差：

```bash
cd /data1/DCT-Reg
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
$PY scripts/prepare_v313_evidence.py export \
  --manifest results/v313_evidence_v2/full_manifest.json \
  --output results/v313_slotspe_style_exports_v1 \
  --arm exp6 --cancer blca --cancer kirc --device cuda:0 --km
```

先检查指定的新根目录不存在或每个 run 输出为空。上面的 `--km` 本身就补齐该次导出的训练阈值；不使用写死 `km=False` 的 attention 重跑脚本。f0/f1 的旧目录非空问题只涉及函数参数 `output_root/run['id']`，兄弟目录不会触发 `any(out.iterdir())`。

源码中另有一个可确定的问题：旧 `rerun_v313_export_for_attention.py` 的 `backup_old()` 在备份目录已存在时直接返回，调用者却仍打印 `[backup]`。因此该行日志不证明本次已搬走当前目录。不能仅凭事后的空目录状态判断报错当时为空；也不能据此认定 manifest 将输出根改写。这份新流程直接使用正式 exporter 和新根目录。

如需要额外选择 `TCGA-2F-A9KP`，先对它实际所属的一折单独导出到新的病例导出根，不能把这个 `--case-id` 传给全部五折；其他四折没有该病例，会被 exporter 正确拒绝。

不需要组织图片即可先得到逐槽通路图与运输关联图。把 `RUN` 设置为精确含 `export.json` 的目录，不按第一个子目录猜 run：

```bash
RUN=results/v313_slotspe_style_exports_v1/exp6_blca_f1_s3
$PY scripts/plot_v313_slotspe_style.py pathways \
  --export "$RUN" --case-id TCGA-2F-A9KP \
  --output paper/figures/slotspe_style/blca_a9kp_pathways
$PY scripts/plot_v313_slotspe_style.py coupling \
  --export "$RUN" --case-id TCGA-2F-A9KP \
  --output paper/figures/slotspe_style/blca_a9kp_transport
$PY scripts/plot_v313_slotspe_style.py cohort \
  --exports results/v313_slotspe_style_exports_v1 --cancer blca \
  --output paper/figures/slotspe_style/blca_cohort
$PY scripts/plot_v313_slotspe_style.py cohort \
  --exports results/v313_slotspe_style_exports_v1 --cancer kirc \
  --output paper/figures/slotspe_style/kirc_cohort
```

注意：`exp6_blca_f1_s3` 是用户报告的 run ID，必须以 `full_manifest.json` 及实际 `export.json` 为准；如果病例不在此次 export 的 `cases` 清单中，需要在该次 exporter 增加 `--case-id TCGA-2F-A9KP`。

仅在核对病例确属 BLCA fold1 后，可用下面命令补其病例导出；它不会修改前面的全队列导出：

```bash
$PY scripts/prepare_v313_evidence.py export \
  --manifest results/v313_evidence_v2/full_manifest.json \
  --output results/v313_slotspe_style_a9kp_export_v1 \
  --arm exp6 --cancer blca --fold 1 --case-id TCGA-2F-A9KP \
  --device cuda:0 --alphas 0 --km
RUN=results/v313_slotspe_style_a9kp_export_v1/exp6_blca_f1_s3
```

如 manifest 的 Full 标签为 `full`，将上述 exporter 的 `--arm exp6` 改成实际标签 `--arm full`；新的队列画图入口会将 `full` 标签对应到 exp6。

填好真实资源清单后画原论文形式的切片图：

```bash
$PY scripts/plot_v313_slotspe_style.py case \
  --export "$RUN" --case-id TCGA-2F-A9KP \
  --assets /data1/DCT-Reg/results/v313_real_wsi_assets.json \
  --slide-index 0 --slots 0,1,2 \
  --output paper/figures/slotspe_style/blca_a9kp_tissue
$PY scripts/plot_fig5_km_curves.py \
  --exports results/v313_slotspe_style_exports_v1 \
  --cancer blca --cancer kirc \
  --output paper/figures/slotspe_style/km
```

真实切片图所需资源只有：本病例 exported feature SHA-256；与原特征行一致的坐标；原切片/缩略图和提取 level、size、downsample；或者带原始 patch 行号的 RGB 图片清单。支持 NPY/HDF5/CSV 坐标，支持 OpenSlide 读取或按索引保存的图片。资源路径目前尚未提供，因此本次没有生成冒充真实病例的组织图片。

## 完成标准

完成的病例图能逐项定位：真实组织位置 → 原始 patch 行 → 选中组织块 → 对应槽 → 命名通路。所有导出和源图片保留可核对的记录。队列图标明实际覆盖、分组方式、原始数值和探索性性质。真实结果不佳或 attention 接近均匀时保留该结果，不通过 hazard 乘权或逐行 max 缩放制造差异。

本次准备验证：9 项合成单元检查通过；四个入口共六种版式均用明确标记的合成数据渲染并检查。真实患者图、临床比较和模块有效性结论不由这些检查产生。服务器仍需按以上顺序执行现有 checkpoint 导出和画图，不要求重跑 110 次训练。
