"""Draw the implemented DCT v3.13 using SlotSPE's two-modality visual grammar.

This is a source-grounded vector illustration, not an experiment. No patient
data, model imports, inference or training are used. All mini tensors are
schematic. English and Chinese PDF/SVG/PNG exports share exactly one topology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess

import pypdfium2 as pdfium
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Ellipse, Line, Polygon, Rect, String
from reportlab.lib.colors import Color, toColor as HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 1760, 1170
INK, GRAY = '#263547', '#6A7585'
PINK, GREEN = '#B95686', '#538348'
PURPLE, ORANGE = '#7457AC', '#B97025'
M1, M2, M3, M4 = '#EFF3FB', '#F4EEFB', '#FFF6E9', '#EDF7F4'
SOURCE_FILES = [
    'survot_rank/research/methods/distributional_counterfactual_transport/model.py',
    'survot_rank/research/methods/ot_event_hazard_v2/model_v2.py',
    'survot_rank/research/methods/legacy/experimental/dct_v311_slot_interpretable/model.py',
    'survot_rank/research/methods/legacy/experimental/dct_v313_transport_reconstruction/model.py',
]


def register_fonts(font_dir: Path) -> None:
    for name, filename in [('DctSans', 'arial.ttf'), ('DctBold', 'arialbd.ttf'),
                           ('DctCN', 'simhei.ttf')]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))


def draw(language: str) -> Drawing:
    cn = language == 'zh'
    d = Drawing(W, H)
    def tr(en, zh):
        return zh if cn else en

    def rect(x, y, w, h, fill='white', edge=None, radius=6, dash=False, sw=1):
        r = Rect(x, H-y-h, w, h, rx=radius, ry=radius,
                 fillColor=HexColor(fill) if fill else None,
                 strokeColor=HexColor(edge) if edge else None, strokeWidth=sw)
        if dash:
            r.strokeDashArray = [6, 4]
        d.add(r)

    def text(x, y, value, size=17, color=INK, bold=False, anchor='middle'):
        font = 'DctCN' if cn else ('DctBold' if bold else 'DctSans')
        d.add(String(x, H-y, value, fontName=font, fontSize=size,
                     fillColor=HexColor(color), textAnchor=anchor))

    def arrow(points, color=INK, dash=False, head=True, width=1.8):
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            line = Line(x1, H-y1, x2, H-y2, strokeColor=HexColor(color), strokeWidth=width)
            if dash:
                line.strokeDashArray = [6, 4]
            d.add(line)
        if head:
            x1, y1 = points[-2]; x2, y2 = points[-1]
            a = math.atan2(y2-y1, x2-x1)
            p = [x2, H-y2]
            for angle in (a+2.65, a-2.65):
                p += [x2+9*math.cos(angle), H-y2-9*math.sin(angle)]
            d.add(Polygon(p, fillColor=HexColor(color), strokeColor=None))

    def dot(x, y, radius, color):
        d.add(Circle(x, H-y, radius, fillColor=HexColor(color), strokeColor=None))

    def ref(x, y, name, color, radius=13):
        """Named tensor reference; repeated names mean the identical tensor.

        References expose fan-out without wrapping a connector around panels.
        They are data ports, not additional computational modules.
        """
        dot(x, y, radius, color)
        text(x, y+5, name, 16, 'white', True)

    def box(x, y, w, h, lines, color=INK, fill='white', size=17):
        rect(x, y, w, h, fill, color)
        for i, value in enumerate(lines):
            line_font = 'DctCN' if cn else ('DctBold' if i == 0 else 'DctSans')
            line_width = pdfmetrics.stringWidth(value, line_font, size)
            fitted_size = size * min(1.0, (w-24) / max(line_width, 1.0))
            text(x+w/2, y+h/2+(i-(len(lines)-1)/2)*22+6, value,
                 fitted_size, color, bold=i == 0)

    def panel(x, y, w, h, tag, title, fill, color):
        rect(x, y, w, h, fill, color, 10, sw=1.1)
        rect(x+13, y+13, 40, 29, color, radius=4)
        text(x+33, y+34, tag, 17, 'white', True)
        text(x+64, y+34, title, 19, color, True, 'start')

    def tokens(x, y, color, n=6, w=9, h=24, gap=4):
        base = HexColor(color)
        for i in range(n):
            v = .30+.65*i/max(1, n-1)
            fill = Color(1-(1-base.red)*v, 1-(1-base.green)*v, 1-(1-base.blue)*v)
            d.add(Rect(x+i*(w+gap), H-y-h, w, h, rx=2, ry=2,
                       fillColor=fill, strokeColor=HexColor(color), strokeWidth=.5))

    slot_colors = ['#91BDE5','#D2B5E5','#EACB85','#99C7B4',
                   '#E6A7AF','#B4C87E','#A7AFDC','#D6B79F']
    def slots(x, y, n=8, step=14, size=10):
        for i in range(n):
            rect(x+i*step, y, size, 21, slot_colors[i % len(slot_colors)], '#758092', 1, sw=.6)

    def heatmap(x, y, size, seed):
        # Deliberately schematic. The manifest and caption disclose this.
        rng = random.Random(seed)
        base = HexColor(PURPLE)
        for r in range(6):
            for c in range(6):
                v = .08+.67*math.exp(-((c-(r+seed)%6)/1.6)**2)+.13*rng.random()
                col = Color(1-(1-base.red)*v, 1-(1-base.green)*v, 1-(1-base.blue)*v)
                d.add(Rect(x+c*size/6, H-y-(r+1)*size/6, size/6-.35, size/6-.35,
                           fillColor=col, strokeColor=None))

    rect(0, 0, W, H)
    text(25, 36, tr('DCT v3.13: transport-coupled prediction and pathway reconstruction',
                    'DCT v3.13：由同一传输计划连接生存预测与通路重建'), 28, bold=True, anchor='start')
    rect(25, 52, 16, 13, '#F1F3F6', '#A2ABB8', 2)
    text(50, 64, tr('Retained slot-encoding foundation', '沿用的槽编码基础'), 15, GRAY, anchor='start')
    rect(382, 52, 16, 13, M2, PURPLE, 2)
    text(407, 64, tr('M1-M4: DCT modifications relative to SlotSPE', 'M1-M4：相对 SlotSPE 的实际改动'), 15, PURPLE, anchor='start')
    ref(1023, 59, 'T', PURPLE, 10)
    text(1043, 64, tr('Repeated badges = the same tensor', '同名标记 = 同一个张量'), 15, GRAY, anchor='start')
    arrow([(1425, 59), (1460, 59)], GREEN, True)
    text(1472, 64, tr('Training only', '仅训练时使用'), 15, GREEN, anchor='start')

    # Preserve SlotSPE's recognizable twin input/encoder/compression rows.
    panel(25, 90, 510, 610, 'A', tr('Two-modality slot encoding', '病理与组学双路槽编码'),
          '#FBFCFD', '#A2ABB8')
    text(105, 153, tr('Histology', '病理输入'), 19, PINK, True)
    rng = random.Random(313)
    for r in range(4):
        for c in range(4):
            x, y = 49+c*25, 170+r*25
            rect(x, y, 23, 23, '#F6DEE9', '#CFABC0', 1, sw=.4)
            for _ in range(6):
                px, py = x+3+rng.random()*16, y+3+rng.random()*16
                d.add(Ellipse(px, H-py, 1.4, 1.0, fillColor=HexColor('#A67AA3'), strokeColor=None))
    text(98, 289, tr('WSI patch bag', 'WSI 图块集合'), 16, PINK)
    arrow([(150, 220), (170, 220)], PINK)
    box(172, 166, 74, 108, [tr('Patch', '图块'), tr('features', '特征'), '+ MLP'], PINK, '#FCEAF1', 16)
    arrow([(246, 220), (259, 220)], PINK)
    tokens(262, 185, PINK, n=3, w=8, h=70, gap=3)
    arrow([(294, 220), (307, 220)], PINK)
    d.add(Polygon([309,H-167, 365,H-197, 365,H-243, 309,H-275],
                  fillColor=HexColor('#ECD9ED'), strokeColor=HexColor('#9F78A6'), strokeWidth=.8))
    text(335, 211, tr('Slot', '槽'), 16, PURPLE, True)
    text(335, 233, tr('attention', '注意力'), 14, PURPLE)
    text(335, 157, tr('Iterative', '迭代压缩'), 13, GRAY)
    arrow([(365, 220), (381, 220)], PINK)

    text(105, 344, tr('Genomics', '组学输入'), 19, GREEN, True)
    for r in range(5):
        tokens(51, 366+r*17, GREEN, n=5+(r%2), w=11, h=12, gap=3)
    text(98, 474, tr('Pathway vectors', '通路基因向量'), 16, GREEN)
    arrow([(150, 407), (170, 407)], GREEN)
    box(172, 353, 74, 108, [tr('Pathway', '通路'), tr('encoder', '编码器')], GREEN, '#EEF5E9', 16)
    arrow([(246, 407), (259, 407)], GREEN)
    tokens(262, 372, GREEN, n=3, w=8, h=70, gap=3)
    text(276, 364, 'X_o', 15, GREEN)
    arrow([(294, 407), (307, 407)], GREEN)
    d.add(Polygon([309,H-354, 365,H-384, 365,H-430, 309,H-462],
                  fillColor=HexColor('#ECD9ED'), strokeColor=HexColor('#9F78A6'), strokeWidth=.8))
    text(335, 398, tr('Slot', '槽'), 16, PURPLE, True)
    text(335, 420, tr('attention', '注意力'), 14, PURPLE)
    arrow([(365, 407), (381, 407)], GREEN)

    # M1 is AFTER local slot attention; no prototype is inserted into raw tokens.
    rect(384, 150, 135, 341, M1, '#6D8AB7', 8)
    text(452, 174, 'M1', 19, '#4F709F', True)
    text(452, 196, tr('Prototype', '原型坐标'), 16, '#4F709F', True)
    text(452, 217, tr('coordinates', '重分配'), 16, '#4F709F', True)
    for i in range(4):
        dot(412+i*25, 245, 5, PINK)
    text(452, 267, tr('WSI bank', '病理原型库'), 14, PINK)
    slots(401, 280, n=8, step=14, size=10)
    text(452, 325, 'S_w', 17, PINK, True)
    for i in range(4):
        dot(412+i*25, 358, 5, GREEN)
    text(452, 380, tr('Omics bank', '组学原型库'), 14, GREEN)
    slots(401, 427, n=8, step=14, size=10)
    text(452, 477, 'S_o', 17, GREEN, True)
    text(278, 598, tr('Separate prototype banks; reused across patients',
                      '两模态原型库分开学习，并在患者间复用'), 16, GRAY)
    text(278, 627, tr('S = (S_w, S_o): the two semantic slot sets',
                      'S = (S_w, S_o)：两模态语义槽集合'), 16, INK)
    text(278, 680, tr('Local slot attention -> prototype coordinate reassignment',
                      '局部槽注意力 → 原型坐标重分配'), 15, GRAY)

    # The overview has ONE horizontal prediction spine. Every connector is a
    # single straight segment; explicit data references replace cross-panel buses.
    arrow([(519, 292), (542, 352)], PINK)
    arrow([(519, 438), (542, 378)], GREEN)
    ref(547, 365, 'S', INK)

    panel(565, 90, 530, 610, 'M2', tr('Stage-conditioned multi-geometry OT', '阶段条件化的多几何传输'),
          M2, PURPLE)
    text(829, 188, tr('Construct an explicit joint transport plan', '构造显式跨模态联合传输计划'), 18, PURPLE)
    text(829, 216, tr('Learned costs and marginals for each latent stage', '每个潜在阶段学习代价与证据边际'), 16, GRAY)
    arrow([(560, 365), (588, 365)], INK)
    box(590, 290, 264, 150,
        [tr('Cost construction', '代价构造'),
         tr('Cosine / Euclidean / positive-dot', '余弦 / 欧氏 / 正点积'),
         tr('+ learned stage & evidence costs', '+ 学习阶段代价与证据代价'),
         tr('Evidence-conditioned marginals', '证据条件化的两侧边际')], PURPLE, 'white', 15)
    arrow([(854, 365), (883, 365)], PURPLE)
    box(885, 338, 175, 54, ['Sinkhorn OT'], PURPLE, '#E9DFF5', 19)
    arrow([(1060, 365), (1069, 365)], PURPLE, head=False)
    ref(1082, 365, 'T', PURPLE)
    text(828, 481, tr('T_(s,m): prediction plans', 'T_(s,m)：预测所用传输计划'), 18, PURPLE, True)
    for g, label in enumerate([tr('Cosine', '余弦'), tr('Euclidean', '欧氏'), tr('Positive-dot', '正点积')]):
        text(790+g*117, 514, label, 15, PURPLE)
    for s in range(4):
        y = 529+s*34
        text(661, y+23, tr(f'Latent stage {s+1}', f'潜在阶段 {s+1}'), 15, PURPLE)
        for g in range(3):
            heatmap(775+g*117, y, 29, s*3+g+1)
    text(830, 685, tr('4 learned latent stages x 3 geometries', '4 个学习潜在阶段 × 3 种几何'), 15, GRAY)

    panel(1120, 90, 615, 610, 'M3', tr('Transport-event survival prediction', '传输事件生存预测'),
          M3, ORANGE)
    text(1428, 188, tr('Transport mass + slot-pair content', '传输质量 + 槽对内容'), 18, ORANGE)
    text(1428, 216, tr('Both inputs jointly define each event', '两类输入共同构建事件表示'), 16, GRAY)
    ref(1190, 292, 'S', INK)
    text(1213, 298, tr('Same S_w, S_o', '复用 S_w、S_o'), 16, INK, anchor='start')
    arrow([(1190, 305), (1190, 333)], INK)
    arrow([(1095, 365), (1148, 365)], PURPLE)
    box(1150, 335, 555, 60,
        [tr('Transport-event fusion', '传输事件融合'),
         tr('T + S -> event tokens', 'T + S → 事件 tokens')], ORANGE, 'white', 18)
    arrow([(1428, 395), (1428, 420)], ORANGE)
    box(1180, 422, 500, 50,
        [tr('Stage embedding + event Transformer', '阶段嵌入 + 事件 Transformer')], ORANGE, 'white', 18)
    arrow([(1310, 472), (1310, 518)], ORANGE)
    arrow([(1570, 472), (1570, 518)], ORANGE)
    rect(1220, 520, 180, 46, 'white', ORANGE)
    text(1298, 548, tr('Stage gate', '阶段门控'), 16, ORANGE, True)
    ref(1375, 543, 'g', ORANGE, 12)
    box(1480, 520, 180, 46, [tr('Event logits z_s', '事件 logits z_s')], ORANGE, 'white', 16)
    arrow([(1310, 566), (1310, 588)], ORANGE)
    arrow([(1570, 566), (1570, 588)], ORANGE)
    box(1180, 590, 500, 45,
        [tr('Weighted logits: z = sum_s g_s z_s', '加权 logits：z = sum_s g_s z_s')], ORANGE, '#FFF0D9', 18)
    arrow([(1428, 635), (1428, 649)], ORANGE)
    box(1180, 651, 500, 34,
        [tr('Hazards -> survival curve / patient risk', '风险概率 → 生存曲线 / 患者风险')], ORANGE, 'white', 17)

    # Cross and self reconstruction use two horizontal lanes with aligned
    # decoder input/output ports. The shared decoder is called twice; its two
    # memories are never concatenated or jointly fed to one reconstruction.
    panel(25, 740, 1710, 330, 'M4', tr('Transport-coupled pathway reconstruction', '传输耦合的通路潜变量重建'),
          M4, GREEN)
    text(1713, 774, tr('TRAINING ONLY', '仅训练时使用'), 16, GREEN, True, 'end')
    ref(331, 806, 'T', PURPLE)
    text(354, 812, 'T_(s,m)', 16, PURPLE, anchor='start')
    arrow([(331, 819), (331, 840)], PURPLE, True)
    ref(492, 806, 'g', ORANGE)
    text(515, 812, 'SG(g)', 16, ORANGE, anchor='start')
    arrow([(492, 819), (492, 840)], ORANGE, True)
    box(47, 854, 184, 48, [tr('WSI slots S_w', '病理槽 S_w')], PINK, 'white', 18)
    box(47, 985, 184, 48, [tr('Omics slots S_o', '组学槽 S_o')], GREEN, 'white', 18)
    arrow([(231, 878), (264, 878)], PINK, True)
    box(266, 842, 360, 72,
        [tr('Barycentric transport + stage aggregation', '重心传输 + 阶段加权聚合'),
         tr('Mean geometries / column normalization / SG(g)', '几何均值 / 列归一化 / SG(g) 加权')], PURPLE, 'white', 16)
    arrow([(626, 878), (768, 878)], PURPLE, True)
    text(698, 864, tr('Cross memory', '跨模态记忆'), 15, PURPLE)
    arrow([(231, 1009), (768, 1009)], GREEN, True)
    text(490, 995, tr('Self memory', '自重建记忆'), 15, GREEN)
    box(770, 842, 260, 194,
        [tr('Shared pathway decoder D', '共享通路解码器 D'),
         tr('Learned pathway queries', '学习的通路查询'),
         tr('Two calls, same parameters', '调用两次，参数共享')], GREEN, 'white', 17)
    arrow([(1030, 878), (1058, 878)], PURPLE, True)
    arrow([(1030, 1009), (1058, 1009)], GREEN, True)
    box(1060, 854, 195, 48, [tr('Cross reconstruction', '跨模态重建结果')], PURPLE, 'white', 16)
    box(1060, 985, 195, 48, [tr('Self reconstruction', '自重建结果')], GREEN, 'white', 16)
    arrow([(1255, 878), (1308, 878)], PURPLE, True)
    arrow([(1255, 1009), (1308, 1009)], GREEN, True)
    box(1310, 854, 170, 48, [tr('Cross loss', '跨模态重建损失')], PURPLE, 'white', 16)
    box(1310, 985, 170, 48, [tr('Self loss', '自重建损失')], GREEN, 'white', 16)
    box(1530, 900, 180, 88,
        [tr('Target: SG(X_o)', '目标：SG(X_o)'),
         tr('Encoded pathway', '编码后的通路'), tr('tokens', 'tokens')], GREEN, '#DDEFE8', 16)
    arrow([(1530, 923), (1482, 878)], PURPLE, True)
    arrow([(1530, 965), (1482, 1009)], GREEN, True)

    rect(25, 1090, 1710, 40, '#F4F6F8', '#C8D0D9', 6)
    text(880, 1116,
         tr('Training: survival NLL + IPCW rank + per-slot NLL + diversity + ramped self / cross reconstruction',
            '训练目标：生存 NLL + IPCW 排序 + 逐槽 NLL + 多样性约束 + 逐步启用的 self / cross 重建'),
         18, INK, True)
    text(27, 1157,
         tr('Repeated S, T and g badges denote tensor reuse. SG = stop-gradient. Stages and tensors are learned / schematic, not annotated disease stages.',
            '同名 S、T、g 标记表示复用同一个张量；SG 表示停止梯度。阶段由模型学习，图内张量为示意，不对应已标注的疾病分期。'),
         15, GRAY, anchor='start')
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'paper/figures/v313_architecture_redesign')
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--font-dir', type=Path, default=Path('C:/Windows/Fonts'))
    args = parser.parse_args()
    register_fonts(args.font_dir)
    # Capture source identity without importing or executing research code.
    hashes = {}
    for name in SOURCE_FILES:
        p = args.source_root / name
        if not p.is_file():
            parser.error(f'Cannot verify source file {p}; pass --source-root for the v3.13 checkout')
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for language in ('en', 'zh'):
        drawing = draw(language)
        stem = args.output_dir / f'fig1_v313_framework_{language}'
        renderPDF.drawToFile(drawing, str(stem.with_suffix('.pdf')))
        renderSVG.drawToFile(drawing, str(stem.with_suffix('.svg')))
        with pdfium.PdfDocument(str(stem.with_suffix('.pdf'))) as document:
            if len(document) != 1:
                raise RuntimeError('Architecture figure must have exactly one page')
            bitmap = document[0].render(scale=2)
            bitmap.to_pil().save(stem.with_suffix('.png'))
            bitmap.close()
        outputs += [stem.with_suffix(ext).name for ext in ('.pdf', '.svg', '.png')]
    git = subprocess.run(['git', '-C', str(args.source_root), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        'method': 'dct_v313', 'source_commit': git, 'source_sha256': hashes,
        'baseline_reference': 'https://arxiv.org/html/2512.01116v2#S3',
        'layout_reference': 'SlotSPE Figure 1; original vector redraw with DCT topology',
        'new_relative_to_slotspe': ['prototype coordinate reassignment',
                                   'stage-conditioned multi-geometry transport',
                                   'transport-event survival readout',
                                   'prediction-plan-coupled reconstruction'],
        'reconstruction_target': 'stop-gradient encoded pathway tokens',
        'reconstruction_gate': 'stop-gradient prediction gate',
        'shared_decoder': True, 'training_only_reconstruction': True,
        'connector_layout': 'single-segment connectors; named tensor references for fan-out',
        'tensor_reference_semantics': {'S': '(S_w, S_o)', 'T': 'same factual prediction plans',
                                       'g': 'same prediction gate; stop-gradient in reconstruction'},
        'schematic_not_patient_data': True, 'outputs': outputs,
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Created {len(outputs)} figure files from source {git[:12]}')


if __name__ == '__main__':
    main()
