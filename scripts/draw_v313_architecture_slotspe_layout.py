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
W, H = 1660, 970
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

    def box(x, y, w, h, lines, color=INK, fill='white', size=17):
        rect(x, y, w, h, fill, color)
        for i, value in enumerate(lines):
            text(x+w/2, y+h/2+(i-(len(lines)-1)/2)*22+6, value,
                 size, color, bold=i == 0)

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
                    'DCT v3.13：由同一传输计划连接生存预测与通路重建'), 26, bold=True, anchor='start')
    rect(25, 52, 16, 13, '#F1F3F6', '#A2ABB8', 2)
    text(50, 64, tr('Retained slot-encoding foundation', '沿用的槽编码基础'), 15, GRAY, anchor='start')
    rect(382, 52, 16, 13, M2, PURPLE, 2)
    text(407, 64, tr('M1-M4: DCT modifications relative to SlotSPE', 'M1-M4：相对 SlotSPE 的实际改动'), 15, PURPLE, anchor='start')
    arrow([(1000, 59), (1040, 59)])
    text(1050, 64, tr('Prediction', '预测路径'), 15, anchor='start')
    arrow([(1200, 59), (1240, 59)], GREEN, True)
    text(1250, 64, tr('Training-only branch', '仅训练时启用的分支'), 15, GREEN, anchor='start')

    # Preserve SlotSPE's recognizable twin input/encoder/compression rows.
    panel(25, 90, 510, 470, 'A', tr('Two-modality slot encoding', '病理与组学双路槽编码'),
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
    text(278, 522, tr('Separate prototype banks; reused across patients',
                      '两模态原型库分开学习，并在患者间复用'), 16, GRAY)
    text(278, 546, tr('Coordinates are learned; biological semantics need validation',
                      '槽坐标由模型学习；生物学含义需实验验证'), 14, GRAY)

    # M2 replaces the baseline's direct slot-interaction prediction path.
    panel(557, 90, 560, 470, 'M2', tr('Stage-conditioned multi-geometry OT', '阶段条件化的多几何传输'),
          M2, PURPLE)
    arrow([(519, 292), (547, 292), (547, 165), (580, 165)], PINK)
    arrow([(519, 438), (551, 438), (551, 213), (580, 213)], GREEN)
    box(582, 146, 242, 72,
        [tr('Slot-pair costs', '槽对代价'), tr('Cosine / Euclidean / positive-dot', '余弦 / 欧氏 / 正点积')],
        PURPLE, 'white', 15)
    box(841, 146, 252, 72,
        [tr('Learned stage + evidence costs', '学习阶段代价 + 证据代价'),
         tr('Evidence-conditioned marginals', '证据条件化的两侧边际')], PURPLE, 'white', 15)
    arrow([(703, 218), (703, 241), (787, 241), (787, 259)], PURPLE)
    arrow([(967, 218), (967, 241), (899, 241), (899, 259)], PURPLE)
    box(647, 261, 388, 45, ['Sinkhorn OT  ->  T_(s,m)'], PURPLE, '#E9DFF5', 19)
    arrow([(841, 306), (841, 320)], PURPLE)
    for g, label in enumerate([tr('Cosine', '余弦'), tr('Euclidean', '欧氏'), tr('Positive-dot', '正点积')]):
        text(778+g*100, 344, label, 15, PURPLE)
    for s in range(4):
        y = 354+s*42
        text(664, y+23, tr(f'Latent stage {s+1}', f'潜在阶段 {s+1}'), 15, PURPLE)
        for g in range(3):
            heatmap(760+g*100, y, 33, s*3+g+1)
    text(835, 548, tr('4 learned latent stages x 3 geometries; schematic plans',
                      '4 个学习潜在阶段 × 3 种几何；矩阵为示意'), 15, GRAY)

    # M3 consumes BOTH plans AND the semantic slot content. Gate and logits are
    # parallel outputs; the gate does not generate the logits.
    panel(1139, 90, 496, 470, 'M3', tr('Transport-event survival prediction', '传输事件生存预测'),
          M3, ORANGE)
    # Draw after the M3 panel so its background cannot cover this input.
    arrow([(519, 292), (539, 292), (539, 325)], PINK, head=False, width=1.3)
    arrow([(519, 438), (539, 438), (539, 325)], GREEN, head=False, width=1.3)
    dot(539, 325, 3.2, INK)
    arrow([(539, 325), (539, 135), (1128, 135), (1128, 168), (1171, 168)], INK, width=1.3)
    arrow([(1117, 420), (1154, 420), (1154, 183), (1169, 183)], PURPLE)
    box(1171, 150, 442, 73,
        [tr('Transport-event fusion', '传输事件融合'),
         tr('Plans + slot-pair content -> event tokens', '传输计划 + 槽对内容 → 事件 tokens')], ORANGE, 'white', 17)
    arrow([(1392, 223), (1392, 260)], ORANGE)
    box(1192, 263, 400, 43, [tr('Stage embedding + event Transformer', '阶段嵌入 + 事件 Transformer')], ORANGE, 'white', 17)
    arrow([(1392, 306), (1392, 322), (1272, 322), (1272, 337)], ORANGE)
    arrow([(1392, 322), (1512, 322), (1512, 337)], ORANGE)
    box(1192, 340, 160, 46, [tr('Stage gate g', '阶段门控 g')], ORANGE, 'white', 16)
    box(1412, 340, 180, 46, [tr('Event logits z_s', '事件 logits z_s')], ORANGE, 'white', 16)
    arrow([(1272, 386), (1272, 402), (1375, 402)], ORANGE)
    arrow([(1502, 386), (1502, 402), (1410, 402)], ORANGE)
    box(1378, 390, 30, 26, ['+'], ORANGE, '#F4E2C4', 18)
    text(1387, 440, tr('Weighted logits: z = sum_s g_s z_s', '加权 logits：z = sum_s g_s z_s'), 17, ORANGE, True)
    arrow([(1392, 447), (1392, 464)], ORANGE)
    box(1184, 467, 412, 42, [tr('Hazards -> survival curve / patient risk', '风险概率 → 生存曲线 / 患者风险')], ORANGE, 'white', 17)
    text(1387, 529, tr('Slot content and plans jointly define each event', '槽内容与传输计划共同构建事件'), 14, GRAY)
    text(1387, 547, tr('Latent stages are not annotated disease stages', '潜在阶段不等同于已标注的疾病分期'), 14, GRAY)

    # An explicit bridge exposes the central difference from baseline recon:
    # the EXACT plans used above are reused; the gate is stop-gradient here.
    arrow([(1055, 354), (1070, 354), (1070, 506)], PURPLE, head=False, width=1.2)
    arrow([(1055, 513), (1070, 513), (1070, 506)], PURPLE, head=False, width=1.2)
    dot(1070, 506, 3.5, PURPLE)
    arrow([(1070, 506), (1070, 587), (490, 587), (490, 719)], PURPLE, True, width=2.4)
    rect(658, 575, 392, 24, 'white', radius=2)
    text(854, 594, tr('Reuse the same prediction plans T_(s,m)',
                      '复用预测实际使用的同一组 T_(s,m)'), 18, PURPLE, True)
    arrow([(1272, 386), (1158, 386), (1158, 612), (620, 612), (620, 719)], ORANGE, True, width=1.7)
    rect(754, 602, 345, 23, 'white', radius=2)
    text(925, 620, tr('Prediction gate -> stop-gradient SG(g)',
                      '预测门控 → 停止梯度 SG(g)'), 16, ORANGE)

    panel(25, 644, 1610, 220, 'M4', tr('Transport-coupled pathway reconstruction', '传输耦合的通路潜变量重建'),
          M4, GREEN)
    text(1614, 678, tr('TRAINING ONLY', '仅训练时使用'), 16, GREEN, True, 'end')
    box(47, 712, 184, 49, [tr('WSI slots S_w', '病理槽 S_w')], PINK, 'white', 18)
    box(47, 792, 184, 49, [tr('Omics slots S_o', '组学槽 S_o')], GREEN, 'white', 18)
    arrow([(231, 736), (273, 736)], PINK, True)
    box(276, 712, 398, 54,
        [tr('Barycentric transport + stage aggregation', '重心传输 + 阶段加权聚合'),
         tr('Mean geometries -> normalize columns -> SG(g)', '几何均值 → 按列归一化 → SG(g) 加权')], PURPLE, 'white', 16)
    arrow([(674, 738), (700, 738)], PURPLE, True)
    box(703, 712, 198, 49, [tr('Cross memory', '跨模态重建记忆')], PURPLE, 'white', 17)
    arrow([(231, 816), (700, 816)], GREEN, True)
    box(703, 792, 198, 49, [tr('Self memory', '自重建记忆')], GREEN, 'white', 17)
    arrow([(901, 736), (927, 736), (927, 766), (952, 766)], PURPLE, True)
    arrow([(901, 816), (927, 816), (927, 791), (952, 791)], GREEN, True)
    box(955, 726, 258, 106,
        [tr('Shared pathway decoder D', '共享通路解码器 D'),
         tr('Learned pathway queries', '学习的通路查询'),
         tr('Two calls, same parameters', '调用两次，参数共享')], GREEN, 'white', 17)
    arrow([(1213, 750), (1240, 750)], PURPLE, True)
    arrow([(1213, 815), (1240, 815)], GREEN, True)
    box(1243, 714, 165, 49, [tr('Cross loss', '跨模态重建损失')], PURPLE, 'white', 16)
    box(1243, 792, 165, 49, [tr('Self loss', '自重建损失')], GREEN, 'white', 16)
    box(1442, 733, 171, 94,
        [tr('Target: SG(X_o)', '目标：SG(X_o)'),
         tr('Encoded pathway', '编码后的通路'),
         tr('tokens', 'tokens')], GREEN, '#DDEFE8', 16)
    arrow([(1442, 754), (1408, 739)], PURPLE, True)
    arrow([(1442, 809), (1408, 816)], GREEN, True)
    text(1530, 709, tr('From pathway encoder', '来自通路编码器'), 14, GRAY)

    rect(25, 883, 1610, 47, '#F4F6F8', '#C8D0D9', 6)
    text(830, 912,
         tr('Training: survival NLL + IPCW rank + per-slot NLL + diversity + ramped self / cross reconstruction',
            '训练目标：生存 NLL + IPCW 排序 + 逐槽 NLL + 多样性约束 + 逐步启用的 self / cross 重建'),
         18, INK, True)
    text(27, 955,
         tr('SG = stop-gradient. Reconstruction targets encoded tokens; it does not claim raw gene imputation. All tensors are schematic.',
            'SG 表示停止梯度。重建目标为编码后的通路 tokens；当前实现不代表原始基因补全。图内张量均为结构示意。'),
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
        'schematic_not_patient_data': True, 'outputs': outputs,
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Created {len(outputs)} figure files from source {git[:12]}')


if __name__ == '__main__':
    main()
