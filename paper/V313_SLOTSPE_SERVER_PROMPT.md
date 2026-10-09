# 服务器执行提示词：仅 DCT v3.13 真实论文图

请在服务器 /data1/DCT-Reg 使用已有 v3.13 checkpoint，完成 SlotSPE 风格真实论文图。先阅读当前 main 的 paper/V313_SLOTSPE_FIGURE_MAP.md，按其顺序执行。3efe824 是最低入口基线；使用包含本次计划更新的最新 main 并记录完整 HEAD，不把旧 SHA 当永久最新版本。

本次授权只读核验、已有 checkpoint 推理、导出、真实绘图和必要代码修复。不得新训练、跑110次 outer_test、reset、强制覆盖、删除旧实验目录或改写旧产物。旧 runbook 的训练命令不在本次授权内。

1. 安全更新。检查 /data1/DCT-Reg 的 branch、HEAD、修改和产物，fetch 后检查更新路径。能安全快进且不损失修改/旧产物才更新原目录；否则保留原目录，创建 origin/main 的独立代码工作区，数据仍用 /data1/DCT-Reg 绝对路径。记录 commit、环境和日志，新输出用新目录，冲突不清空。

2. 锁定十折真实 v3.13 Full。读取 results/v313_evidence_v2/full_manifest.json，核对真实 arm、训练 config/overrides/source_commit、checkpoint、最佳 epoch 预测、split 和 UNI2-h 来源。Full 必须是实际 v3.13 transport+learned，标签 exp6/full 本身不能证明版本。选 BLCA/KIRC、seed3、legacy_val 各 fold0–4 十个唯一 run，保存新清单并 audit，另检查完整患者集合和五折覆盖。缺数据只读溯源，不拼其他版本或重训。

3. 核实病例。用 split 与最佳预测确认 TCGA-2F-A9KP 的真实验证 fold/run，禁止预设 fold1或按图片目录推断，病例 ID 只能传所属折。KIRC 按验证患者 ID 升序固定首个 replay 通过且真实组织资源可核验的病例，记录筛选原因，不按图好看或分数挑选。

4. 补导出。优先复用身份、hashes、验证患者、真实 attention_omic/pathway_names 及 km_train_median/km_high 均齐全的 export。缺失 run 用 scripts/prepare_v313_evidence.py export 在新根补推理，启用 --km --alphas 0；病例缺 case NPZ 时单折补 --case-id。不放宽 replay 判据。cohort/KM 输入只有十个已核验 run，不混重复病例导出/备份/其他 seed，不假定 rglob 遍历目录软链接。

5. 先画无需组织资源的图。用 scripts/plot_v313_slotspe_style.py 的 pathways、coupling、cohort，先 check-only 再生成。包含逐槽 Top-3/Bottom-3 命名通路、通路×槽矩阵、学习 OT/同事实边际独立计划/差值、BLCA/KIRC 完整五折四风险组 Top-10 通路并集。四组按折内风险 midrank，通路为真实 attention_omic 跨槽等权平均，保存患者分组和通路组均值 CSV。不默认 allow-partial，不增强不存在的差异。

6. 真实组织病例图。只读寻找 WSI/缩略图、坐标或带原始特征行号的 RGB patch，核实病例与 feature SHA-256、坐标顺序证据、level/size/downsample 和 slide 索引。按 configs/v313_slotspe_assets.example.json 写新的真实资源清单。用 case 生成原切片、两类槽分区、逐槽热图、Top-5 真实组织块和命名通路。BLCA 优先 A9KP，KIRC 用固定病例；没有真实标注不制造组织标签。缺资源列实际路径和缺项，继续其他图，不用合成图冒充真实结果。

7. 总体 KM。用 scripts/plot_fig5_km_curves.py 先 check-only 再画。每癌种一张 Full 高/低风险两条曲线；按所属折训练风险中位数分组后合并五折，每人一次。risk=-sum(survival)，越大风险越高，等于阈值算高风险。附删失标记、名义95% CI、在险人数、真实组人数、PNG/PDF、患者 CSV、来源/阈值 JSON。先 BLCA/KIRC；其他癌种只有现成同协议完整五折 v3.13 产物才扩展。

8. 验收交付。实际打开生成 PNG，修复截断、重叠、字号和色标，核对 PDF。在仓库写自包含运行报告，附真实预览、绝对路径、每图用途、十折来源、患者数/覆盖、病例选择、风险/分组定义、命令、异常和缺失项。通路图完成不能算组织图完成，合成排版测试不能算实验结果。代码/文档修复通过相关检查后单独 commit 并正常 push，验证远端 ref/tree；大型 checkpoint、NPZ、WSI、特征和大日志不进 Git。

解释必须准确：四组按预测风险，与 SlotSPE 生存时间组不同；同编号 WSI/组学槽不天然对应，不同折同编号槽也不默认对齐。attention/OT 展示关联，不直接证明因果或模块收益。保留真实数值，不用重建误差或 hazard 乘权替代 attention，不预写生物学结论。legacy_val 结果不能称独立外层测试，KM 的可选 log-rank 仅作探索性统计。

失败时记录具体 run、命令、完整异常和部分输出，重试用新目录。旧脚本打印 [backup] 不证明当前目录已搬移；非空检查针对实际 output_root/run_id，不是兄弟目录。不删除旧目录解决冲突，不放宽 replay 容差。

持续完成所有已授权且不依赖缺失资源的图。最终明确哪些真实图完成、哪些缺什么，不把“代码准备好”或预计时间写成“所有图已完成”。
