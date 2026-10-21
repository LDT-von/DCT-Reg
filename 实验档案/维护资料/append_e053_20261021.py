"""Append E053 to the v3.13 master ledger (docx) following the E045 precedent.

Backup must already exist and must still match the current master bytes.
Existing experiment material is never modified or removed.
"""
import copy
import hashlib
import json
import shutil
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path("/data1/DCT-Reg")
MASTER = ROOT / "实验档案/DCT实验总档案.docx"
BACKUP = ROOT / "历史备份/实验档案/20261021_E053_续写前/DCT实验总档案.docx"
FIG = ROOT / "实验档案/表格图片/E053_v313同口径_BLCA复刻"
RECEIPT = ROOT / "实验档案/维护资料/E053_续写核验.json"


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paragraph(doc, text, style=None):
    p = doc.add_paragraph(text, style=style)
    for run in p.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(10.5 if style is None else 9.5)
        run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    return p


def table(doc, headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.autofit = False
    t._tbl.remove(t._tbl.tblPr)
    t._tbl.insert(0, copy.deepcopy(doc.tables[0]._tbl.tblPr))
    for i, (name, width) in enumerate(zip(headers, widths)):
        t.columns[i].width = Inches(width)
        t.rows[0].cells[i].text = name
    for values in rows:
        cells = t.add_row().cells
        for c, text in zip(cells, values):
            c.text = str(text)
    for row_index, row in enumerate(t.rows):
        for i, c in enumerate(row.cells):
            c.width = Inches(widths[i])
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(3)
                p.paragraph_format.space_before = Pt(3)
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.font.name = "Calibri"
                    run.bold = row_index == 0
                    run.font.color.rgb = RGBColor.from_string(
                        "FFFFFF" if row_index == 0 else "111111")
                    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
            shade = OxmlElement("w:shd")
            shade.set(qn("w:fill"),
                      "254B61" if row_index == 0 else ("F1F5F8" if row_index % 2 == 0 else "FFFFFF"))
            c._tc.get_or_add_tcPr().append(shade)
    return t


if not BACKUP.is_file():
    raise RuntimeError("Backup missing; author it before running this script")
if checksum(BACKUP) != checksum(MASTER):
    raise RuntimeError("Master changed after the backup was taken; re-backup first")

doc = Document(MASTER)
if any(p.style.name == "Heading 1" and p.text.startswith("E053") for p in doc.paragraphs):
    raise RuntimeError("E053 already registered")
original_paragraphs = len(doc.paragraphs)
original_tables = len(doc.tables)

# --- sync the two counter / next-number paragraphs -------------------------
COUNT_OLD = ("当前保留 20 个 v3.13 相关实验与分析条目，新增 E045 为既有模块对照再分析及待运行可视化入口。"
             "复用 Full 模型的图表不累计为独立训练次数；公开文献参照不计作本人训练实验。")
COUNT_NEW = ("当前本 docx 保留 21 个 v3.13 相关实验与分析条目，新增 E053 为单折单 seed 同口径三臂复刻。"
             "复用 Full 模型的图表不累计为独立训练次数；公开文献参照不计作本人训练实验。"
             "说明：实验目录_当前.json 另有 E046、E046b、E046c、E047 四条仅登记在 JSON 目录、尚未写入本 docx，"
             "属既有缺口，编号不因此重用。")
count_changed = False
for p in doc.paragraphs:
    if p.text == COUNT_OLD:
        p.text = COUNT_NEW
        count_changed = True
    if p.text.startswith("本轮按用户明确要求，将主档范围收窄到 v3.13："):
        if "2026-10-21 E053 已续写" not in p.text:
            p.add_run(" 2026-10-21 E053 已续写，本 docx 当前共 21 条，下一编号 E054。")
    if p.text == "新增实验时复制本页，放在待补项之前，改成下一个未使用编号。标题使用一级标题并保留另起页；完成后更新目录。下一项新增从 E046 开始。":
        p.text = ("新增实验时复制本页，放在待补项之前，改成下一个未使用编号。标题使用一级标题并保留另起页；"
                  "完成后更新目录。下一项新增从 E054 开始（E048–E052 已被其他实验占用，不得重用）。")
    if p.text == "E046 实验名称":
        p.text = "E054 实验名称"
    if "E046_实验名称" in p.text:
        p.text = p.text.replace("E046_实验名称", "E054_实验名称")
if not count_changed:
    raise RuntimeError("Counter paragraph not found; refusing to append without a count")

# --- E053 record -----------------------------------------------------------
doc.add_page_break()
doc.add_heading("E053 v3.13 同口径 BLCA SlotSPE 三臂复刻", level=1)
paragraph(doc, "类型：训练实验    状态：完成（单折单 seed 探索性验证）")
paragraph(doc, "实验目的：在与 v3.13 相同的数据口径下，用 transport_only 插件跑 native / learned / independent 三臂，"
               "并与 v3.13 原产物逐轮对照，检验复现链路的可信度，以及 transport 机制在推理时的实际作用幅度。")
paragraph(doc, "版本与提交：ad07836daf564f0e0e1afe9064b7d1da473810b4    运行日期：2026-10-21    "
               "负责记录人：刘栋天（Cursor 协助）")
paragraph(doc, "数据与终点：BLCA，TCGA 生存终点 survival_months_dss，特征编码器 UNI2-h（.h5, 1536 维），"
               "纳入样本与患者隔离：train 304 / val 76，排除 0，train∩val = 0。")
paragraph(doc, "划分与 seed：splits/5fold_uni2h/blca/fold_0.csv，seed 3，30 epoch；"
               "epoch 与 checkpoint 选择：legacy_val，取逐 epoch val c-index 最大者（与 v3.13 选点规则一致）。")
paragraph(doc, "配置文件与实际 overrides：experiments/e053_v313_aligned_blca/config.json；"
               "num_patches=2048、encoding_dim=1536、lr=5e-4、batch=8、pad_short_bags=true、"
               "eval_repeats=5、device=cuda:0、mechanism transport_mode=balanced、eps=0.05。"
               "运行环境为 /home/ubuntu/.conda/envs/trisurv/bin/python（torch 2.10.0+cu128）；"
               "系统 python3 的 torch 不含 sm_120 内核，在 RTX 5090 上会报 no kernel image is available。")
paragraph(doc, "有效目标系数与对照差异：native 臂 transport 关闭、参数增量 0；learned 与 independent 臂各新增 600327 参数。"
               "learned 用可学习边际的计划，independent 固定参考边际（realized marginal control）。"
               "learned 臂另做 same_weights 对照：同一套权重下，plan_mode=independent 与完全 bypass transport 两种推理。")

paragraph(doc, "结果表", style="Caption")
table(doc, ["臂", "val c-index（5 repeat 均值）", "repeat std", "NLL", "选中 epoch", "参数增量", "训练秒"],
      [["native", "0.6338", "0.0047", "1.1160", "4", "+0", "715"],
       ["learned", "0.6148", "0.0176", "1.4022", "8", "+600327", "3094"],
       ["independent", "0.6688", "0.0108", "1.1218", "5", "+600327", "780"]],
      [1.0, 1.5, 0.8, 0.8, 0.9, 0.9, 0.8])
paragraph(doc, "差值：learned − native = −0.0190；learned − independent = −0.0540。"
               "对照 v3.13 原始逐轮曲线：v3.13 峰值 0.6576（第 3 轮）、末轮 0.5718；"
               "E053 native 峰值 0.6338（第 4 轮）、末轮 0.5514；逐轮平均绝对差 0.0467，pearson r = 0.234。")

paragraph(doc, "图片与图注：", style="Normal")
doc.add_picture(str(FIG / "E053_fig1_epoch_curves_vs_v313.png"), width=Inches(6.7))
paragraph(doc, "图 E053a 三臂逐轮 val c-index 与 v3.13 对照（左），及逐轮散点与恒等线（右）。"
               "轮次统一按 1 起算显示：v3.13 原始 CSV 的 epoch 列是 0 起算，E053 curve 的 epoch 列是 1 起算，"
               "两者按训练轮序一一对应。虚线为 v3.13 原始产物。可见峰值轮次接近（3 对 4），峰值量级接近，"
               "但逐轮曲线相关性低（r=0.234），说明是同量级而非同轨迹。", style="Caption")
doc.add_picture(str(FIG / "E053_fig2_arm_summary.png"), width=Inches(6.7))
paragraph(doc, "图 E053b 三臂 val c-index（5 repeat 均值，误差棒为 repeat 标准差）对照 v3.13 峰值 0.6576（左）；"
               "三臂训练损失曲线（右）。native 与 independent 训练损失接近，learned 因运输计划求解开销，"
               "训练时间为 native 的 4.3 倍。", style="Caption")
doc.add_picture(str(FIG / "E053_fig3_transport_inference_effect.png"), width=Inches(6.7))
paragraph(doc, "图 E053c learned 臂 same_weights 推理对照（左）与逐患者风险差分布（右）。"
               "同一套 learned 权重下，transport 开启（plan_mode=independent）c-index 0.6148，"
               "完全 bypass transport 为 0.6150；逐患者风险差 max|Δ| = 1.076e-02，"
               "仅为 risk 标准差的 0.79%。", style="Caption")

paragraph(doc, "相关代码：实验代码／E053_v313同口径_BLCA复刻／代码索引.json（入口 "
               "survot_rank/plugins/transport_only/slotspe_compare.py，绘图 make_figures.py）")
paragraph(doc, "表格图片：表格图片／E053_v313同口径_BLCA复刻／（含 E053_结果表.md、三张图与三臂 summary/curve/predictions）")
paragraph(doc, "成果与可支持结论：在与 v3.13 对齐的数据口径下，transport_only 插件三臂可稳定跑通，"
               "native 臂峰值 0.6338 与 v3.13 峰值 0.6576 量级一致（差 0.019），达峰轮次接近（4 对 3），"
               "收敛末轮 0.5514 对 0.5718，量级一致；说明该插件在 v3.13 口径下的复现链路是可信的。"
               "独立训练的独立臂（independent）在本折优于 native 与 learned。"
               "机制层面：learned 权重下推理时开关 transport 对排序与分数几乎无影响（Δc-index = −0.0002，"
               "逐患者风险差仅为 risk 标准差的 0.79%），说明在本设置下 transport 的作用主要发生在训练期，"
               "而非推理期的对应精细度。")
paragraph(doc, "限制与不能支持的结论：单折单 seed，未做统计显著性检验，不能作泛化性或优越性主张；"
               "eval_repeats=5 而 v3.13 为 1，对照列取自 v3.13 原始逐轮曲线，两者评估口径不完全相同；"
               "逐轮 r=0.234 只能支持量级一致，不能称复现了原曲线；"
               "independent 臂同时改变计划模式与边际控制两处，不能单独归因于其中一项；"
               "bypass 对照只覆盖推理期，不能据此推断训练期 transport 无贡献；"
               "本条只覆盖 fold 0 seed 3，不覆盖其余四折。")
paragraph(doc, "原始产物 来源路径 SHA256 与配对核对状态："
               "归档目录 实验档案/表格图片/E053_v313同口径_BLCA复刻/，26 个产物文件的 SHA256 清单见 "
               "实验档案/维护资料/E053_产物SHA256.txt。关键产物："
               "out/paired_summary.json 45717f2048f72ab6e8e8e95cfd43e1058e628da1dbf4e52d80358256e1222c7e；"
               "out/prepared/manifest.json 769ea87e229fc9b8e1352232d8eefd8823a5896c0b75a8b0454a7d3cebad5fb8；"
               "out/native/summary.json 29db5f82341d288cbe8db1e1be1b56bf7f8f4eb051f7fc032943bec070a11780；"
               "out/learned/summary.json 143645d72a315d0daa556413254833f463adf1b040e7b44aea389e83e58805cf；"
               "out/independent/summary.json a5852ff5ab6b9b50a17a1d5c94c4d580d03805587a1b04d070f4554de37d8335；"
               "out/learned/same_weights_controls.json 6d7a87795eb31a844cc141962fd93c55662b901ff1816f438505ec102bd27b27。"
               "配对核对状态：患者与划分已逐项核对，train/val 集合与 fold_0.csv 完全一致，train∩val = 0，"
               "excluded = 0；prepared 缓存与 v3.13 使用的同一 features_root、同一 split 源、同一 num_patches 与 encoding_dim。"
               "checkpoint 哈希（未入归档，位于 experiments/e053_v313_aligned_blca/out/）："
               "native/best.pt cb8f25d65ec955ba6038df376cebe097bbd34d856b32d820c1d7b2c70de04579、"
               "learned/best.pt 755a5be432f1e0de7fdeae319962cbcd47f42a0f7c65a799dd8c0348cbd26be1、"
               "independent/best.pt ad7428cf9cba040da82377f4c7ffab31c7579302b02363e479ff49f9d25a2b8a。")
paragraph(doc, "与既有实验的复用关系：直接对照 v3.13 原产物 "
               "results/dct_v313_blca_uni2h/blca/blca/SurvOTRank_dct_v313_transport_reconstruction/"
               "…_e_30_…_sp_dct_v313_blca_uni2h/epoch_curve_fold0.csv（该文件 fold0 峰值 0.6576@第3轮、末轮 0.5718）；"
               "与 E046c 的 0.7238 五折均值不可直接比较：E046c 是 legacy_val 五折均值且逐患者 0.0 误差复刻，"
               "E053 是单折三臂的探索性对照。"
               "编号说明：本实验早期工作目录名为 experiments/e048_v313_aligned_blca/，"
               "但 E048 已被「E048_v313模块与单模态消融」占用（代码包与结果均真实存在），故最终登记编号 E053，"
               "早期目录已改名，未复用被占用编号。")
paragraph(doc, "后续动作与修订记录：待授权项为 fold 1–4 与多 seed 扩展（当前仅 fold 0 seed 3）、"
               "以及在 E046c 风格下对 learned 臂做 cudnn.deterministic 复刻审计。"
               "2026-10-21 首次登记。已知缺口：E046、E046b、E046c、E047 四条仍只登记在 "
               "实验目录_当前.json，未写入本 docx，属既有缺口，待后续补录。")

# --- ask Word to refresh the table of contents on open --------------------
settings = doc.settings.element
if settings.find(qn("w:updateFields")) is None:
    node = OxmlElement("w:updateFields")
    node.set(qn("w:val"), "true")
    settings.append(node)

doc.save(MASTER)

receipt = {
    "date": "2026-10-21",
    "experiment": "E053",
    "backup": str(BACKUP.relative_to(ROOT)),
    "backup_sha256": checksum(BACKUP),
    "master_sha256_before": checksum(BACKUP),
    "master_sha256_after_authoring": checksum(MASTER),
    "docx_records_before": 20,
    "docx_records_after": 21,
    "json_catalog_records_before": 24,
    "json_catalog_records_after": 25,
    "new_training_runs": 3,
    "new_patient_inference_runs": 380,
    "old_paragraphs": original_paragraphs,
    "new_paragraphs": len(doc.paragraphs),
    "old_tables": original_tables,
    "new_tables": len(doc.tables),
    "toc": "updateFields flag set; Word refreshes TOC on open",
    "known_gap": "E046/E046b/E046c/E047 remain JSON-catalog only; not written into this docx",
}
RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print("E053 appended to master ledger.")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
