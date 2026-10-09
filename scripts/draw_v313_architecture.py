"""Publication figure for the full v3.13 model, with editable vector exports.

All visual tensors are schematic, not measured patient data.
"""
from pathlib import Path
import math
import random

import pypdfium2 as pdfium
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Circle, Ellipse
from reportlab.lib.colors import toColor as HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'figures'
W, H = 1220, 752
INK = '#22334A'
MUTED = '#67758A'
BLUE = '#4874AE'
TEAL = '#1A8D8B'
VIOLET = '#8264B2'
ORANGE = '#BF783C'
LINE = '#CAD4DF'


def build_architecture():
    for name, file in [('FigSans', 'arial.ttf'), ('FigBold', 'arialbd.ttf'), ('FigItalic', 'ariali.ttf')]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/' + file))
    d = Drawing(W, H)

    def rect(x, y, w, h, fill='white', edge=None, radius=8, sw=1):
        d.add(Rect(x, H-y-h, w, h, rx=radius, ry=radius,
                   fillColor=HexColor(fill) if fill else None,
                   strokeColor=HexColor(edge) if edge else None, strokeWidth=sw))

    def text(x, y, s, size=18, color=INK, bold=False, anchor='start', italic=False):
        d.add(String(x, H-y, s, fontName='FigBold' if bold else 'FigItalic' if italic else 'FigSans',
                     fontSize=size, fillColor=HexColor(color), textAnchor=anchor))

    def mathtext(x, y, runs, size=18, color=INK, anchor='start'):
        # Explicit sub/superscript placement avoids unsupported Unicode glyphs.
        widths=[pdfmetrics.stringWidth(s,'FigItalic',size*scale) for s,scale,offset in runs]
        if anchor=='middle':
            x-=sum(widths)/2
        elif anchor=='end':
            x-=sum(widths)
        start=x
        for (s,scale,offset),width in zip(runs,widths):
            text(x,y+offset,s,size*scale,color,italic=True)
            x+=width
        return start

    def path(points, color=INK, dash=False, width=1.6, head=True):
        for (x1,y1),(x2,y2) in zip(points, points[1:]):
            line = Line(x1,H-y1,x2,H-y2,strokeColor=HexColor(color),strokeWidth=width)
            if dash:
                line.strokeDashArray = [5,4]
            d.add(line)
        if head:
            (x1,y1),(x2,y2) = points[-2:]
            a = math.atan2(y2-y1, x2-x1)
            p = [x2,H-y2]
            for t in [a+2.68,a-2.68]:
                p.extend([x2+8*math.cos(t),H-(y2+8*math.sin(t))])
            d.add(Polygon(p,fillColor=HexColor(color),strokeColor=None))

    def dot(x,y,r,color,edge=None):
        d.add(Circle(x,H-y,r,fillColor=HexColor(color),strokeColor=HexColor(edge) if edge else None))

    def panel(x,y,w,h,letter,title,fill='#FAFBFD'):
        rect(x,y,w,h,fill,LINE,12,.8)
        rect(x+15,y+14,25,25,INK,None,5)
        text(x+27.5,y+32,letter,17,'white',True,'middle')
        text(x+51,y+33,title,18,INK,True)

    def module(x,y,w,h,title,sub=None,color=INK,fill='white',size=17):
        rect(x,y,w,h,fill,color,7,.9)
        text(x+w/2,y+h/2+(1 if sub else 6),title,size,color,True,'middle')
        if sub:
            text(x+w/2,y+h/2+21,sub,14,MUTED,anchor='middle')

    def tokens(x,y,color,count=8,w=8,h=30,gap=4):
        base=HexColor(color)
        for i in range(count):
            mix=.20+.65*i/(count-1)
            col=Color(1-(1-base.red)*mix,1-(1-base.green)*mix,1-(1-base.blue)*mix)
            d.add(Rect(x+i*(w+gap),H-y-h,w,h,rx=2,ry=2,fillColor=col,strokeColor=None))

    def heatmap(x,y,size,seed):
        rng=random.Random(seed)
        base=HexColor(VIOLET)
        for r in range(8):
            for c in range(8):
                value=.10+.67*math.exp(-((c-(r+seed)%8)/1.8)**2)+.18*rng.random()
                col=Color(1-(1-base.red)*value,1-(1-base.green)*value,1-(1-base.blue)*value)
                d.add(Rect(x+c*size/8,H-y-(r+1)*size/8,size/8-.45,size/8-.45,
                           fillColor=col,strokeColor=None))

    rect(0,0,W,H)
    text(20,38,'DCT',27,INK,True)
    text(89,37,'Stage-conditioned transport with shared pathway reconstruction',21,INK)
    rect(1082,17,118,31,'#EEF2F7',None,6)
    text(1141,38,'Full v3.13',16,INK,True,'middle')

    panel(20,70,392,367,'A','Semantic slot encoding')
    panel(430,70,375,367,'B','Multi-geometry transport','#FAF9FD')
    panel(823,70,377,367,'C','Survival prediction')
    panel(20,472,1180,223,'D','Transport-aware reconstruction','#F6FBFA')
    text(1179,503,'TRAINING ONLY',14,TEAL,True,'end')

    # A. Schematic tissue patches and pathway vectors; separate prototype banks.
    text(40,138,'WSI patches',17,BLUE,True)
    rng=random.Random(13)
    for r in range(3):
        for c in range(3):
            x,y=42+c*19,155+r*19
            rect(x,y,17,17,['#F3DCE8','#EED0E0','#F6E3EB'][(r+c)%3],None,1)
            for k in range(5):
                px=x+2+rng.random()*12; py=y+2+rng.random()*12
                d.add(Ellipse(px,H-py,1.1+rng.random(),.7+rng.random(),
                              fillColor=HexColor('#9D729B'),strokeColor=None))
    text(69,233,'UNI2-h',14,MUTED,anchor='middle')
    path([(104,183),(123,183)],BLUE)
    module(125,156,88,57,'Projection','1536 → 256',BLUE,'#F0F5FC',15)
    path([(213,183),(231,183)],BLUE)

    text(40,272,'Omics pathways',17,TEAL,True)
    for r in range(6):
        for c in range(6):
            v=(r*5+c*3)%11/11
            base=HexColor(TEAL)
            col=Color(1-(1-base.red)*(.16+.72*v),1-(1-base.green)*(.16+.72*v),1-(1-base.blue)*(.16+.72*v))
            d.add(Rect(42+c*9,H-289-(r+1)*8,8,7,fillColor=col,strokeColor=None))
    text(69,363,'Pathway vectors',13,MUTED,anchor='middle')
    path([(104,317),(123,317)],TEAL)
    module(125,290,88,57,'Pathway','encoder',TEAL,'#EDF8F5',15)
    mathtext(169,369,[('X',1,0),('o',.65,-6)],16,TEAL,'middle')
    path([(213,317),(231,317)],TEAL)

    for y,color,notation in [(148,BLUE,'Sʷ'),(292,TEAL,'Sᵒ')]:
        rect(233,y,153,103,'white',color,7,.9)
        text(309.5,y+21,'Local slot attention',15,color,True,'middle')
        for i in range(8):
            dot(268+i*12,y+33,2.5,'#9EAAB8')
        path([(310,y+37),(310,y+47)],color,width=1)
        rect(247,y+49,126,20,'#F4F7FA',None,3)
        text(310,y+64,'Prototype routing',13,color,True,'middle')
        tokens(262,y+73,color,h=12)
        text(309.5,y+99,'8 semantic slots  '+notation,13,color,anchor='middle')
        # Separate prototype bank above each slot block, a schematic coordinate basis.
        for i in range(4):
            dot(260+i*26,y-12,3.2,color)
        text(309.5,y-25,'Prototype bank',13,MUTED,anchor='middle')
        path([(338,y-12),(398,y-12),(398,y+59),(375,y+59)],color,width=1)
    text(216,414,'Modality-specific prototypes reused across patients',13,MUTED,anchor='middle')
    text(216,431,'Per-slot NLL + slot diversity during training',13,MUTED,anchor='middle')

    # B. The main transport plans remain separate over stage and geometry.
    path([(386,238),(440,238),(440,169),(459,169)],BLUE)
    path([(386,382),(447,382),(447,189),(459,189)],TEAL)
    module(461,139,323,65,'Stage-conditioned costs','Geometry + learned pair cost',VIOLET,'#F2EFF8',18)
    mathtext(617.5,227,[('Learned marginals  a',1,0),('s',.65,4),(', b',1,0),('s',.65,4),('  →  Sinkhorn',1,0)],16,VIOLET,'middle')
    path([(618,204),(618,212)],VIOLET)
    cols=[549,624,699]
    for i,x in enumerate(cols):
        text(x+22,253,'G'+str(i+1),15,VIOLET,True,'middle')
    for s in range(4):
        y=264+s*34
        text(489,y+20,'Stage '+str(s+1),14,MUTED)
        for g,x in enumerate(cols):
            heatmap(x,y,28,s*3+g+3)
    mathtext(765,328,[('T',1,0),('s,m',.6,5)],19,VIOLET,'middle')
    path([(735,264),(742,264),(742,394),(735,394)],VIOLET,head=False,width=1)
    text(617.5,409,'4 latent stages × 3 geometries',14,VIOLET,anchor='middle')
    text(617.5,426,'Learned coupling',14,VIOLET,anchor='middle')

    # C. Transport mass and slot content both enter the event representation.
    path([(742,307),(781,307),(781,293),(813,293),(813,176),(840,176)],VIOLET)
    rect(843,141,334,77,'white',VIOLET,7,.9)
    tokens(862,157,VIOLET,6,w=14,h=21,gap=5)
    text(1087,174,'Event tokens',18,INK,True,'middle')
    text(1010,202,'Transport mass + slot-pair content',15,MUTED,anchor='middle')
    path([(1010,218),(1010,235)],INK)
    module(843,238,334,43,'Event transformer',None,INK,size=18)
    path([(1010,281),(1010,291),(920,291),(920,300)],INK)
    module(843,303,155,47,'Stage gate g','',ORANGE,'#FBF4EA',16)
    module(1023,303,154,47,'Hazard head h','',INK,'white',16)
    path([(998,326),(1023,326)],INK)
    # Schematic survival curve; no claim of measured output.
    path([(1100,350),(1100,371),(1120,371)],INK)
    path([(1033,406),(1174,406)],MUTED,head=False,width=.8)
    path([(1033,406),(1033,368)],MUTED,head=False,width=.8)
    path([(1033,370),(1058,370),(1058,377),(1084,377),(1084,386),
          (1122,386),(1122,395),(1166,395)],BLUE,head=False,width=2.3)
    text(846,380,'Hazards → survival',16,INK,True)
    text(846,405,'Patient risk',16,MUTED)
    text(846,427,'Survival NLL + IPCW rank',13,MUTED)
    text(1169,422,'Time',12,MUTED,anchor='end')

    # A single central connector visually emphasizes plan reuse.
    dot(742,394,4,VIOLET)
    path([(742,394),(742,452),(406,452),(406,534)],VIOLET,True,width=2)
    rect(501,441,221,23,'white',None,3)
    text(611.5,458,'Reuse the prediction plans',14,VIOLET,True,'middle')

    # D. Self branch bypasses transport; cross branch maps WSI memory to omics slots.
    module(45,542,143,46,'Omics slots Sᵒ',None,TEAL,'white',17)
    module(45,607,143,46,'WSI slots Sʷ',None,BLUE,'white',17)
    path([(188,565),(545,565)],TEAL,True)
    text(349,584,'Self memory',15,TEAL,anchor='middle')
    path([(188,630),(263,630)],BLUE,True)
    rect(267,590,241,76,'white',VIOLET,7,.9)
    text(387.5,614,'Transport into omics slots',17,VIOLET,True,'middle')
    mathtext(387.5,638,[('Column-normalize mean',1,0),('m',.65,4),('(T',1,0),('s,m',.65,4),(')',1,0)],14,MUTED,'middle')
    text(387.5,657,'Fuse stages with SG(g)',14,ORANGE,anchor='middle')
    # Geometry averaging is specific to reconstruction, not main prediction.
    path([(406,534),(406,590)],VIOLET,True)
    rect(300,524,211,26,'#F2EFF8',None,4)
    text(405.5,542,'Mean over geometries',14,VIOLET,anchor='middle')
    path([(508,630),(545,630)],VIOLET,True)
    rect(548,536,248,130,'white',TEAL,8,1.3)
    text(672,563,'Shared pathway decoder',18,TEAL,True,'middle')
    text(672,588,'Learned queries Q',15,MUTED,anchor='middle')
    text(672,612,'Cross-attention to slot memory',15,INK,anchor='middle')
    text(672,644,'One decoder · two memory inputs',14,MUTED,anchor='middle')
    path([(796,568),(896,568)],TEAL,True)
    path([(796,627),(896,627)],VIOLET,True)
    for x,y,branch,color in [(847,555,' self',TEAL),(846,652,' cross',VIOLET)]:
        start=mathtext(x,y,[('X',1,0),('o',.65,-6),(branch,1,0)],15,color,'middle')
        path([(start+2,y-13),(start+5,y-16),(start+8,y-13)],color,head=False,width=.8)
    rect(900,555,276,83,'#EAF5F0',TEAL,7,.9)
    mathtext(1038,590,[('L',1,0),('self',.65,5),(' + L',1,0),('cross',.65,5)],24,TEAL,'middle')
    text(1038,616,'Normalized cosine + Smooth L1',14,MUTED,anchor='middle')
    rect(928,514,220,28,'white',ORANGE,5,.9)
    text(1038,534,'Target = SG(Xᵒ)',16,ORANGE,anchor='middle')
    path([(1038,542),(1038,554)],ORANGE,True)
    text(50,684,'Transport plans remain differentiable; SG stops gradients only through the target and stage fusion weights.',14,MUTED)

    # Compact, paper-scale legend.
    path([(22,725),(57,725)],INK,width=1.5)
    text(65,730,'Prediction',15,MUTED)
    path([(188,725),(223,725)],TEAL,True,width=1.5)
    text(231,730,'Training branch',15,MUTED)
    rect(386,713,37,24,'#FBF4EA',ORANGE,4,.8)
    text(404.5,730,'SG',14,ORANGE,True,'middle')
    text(432,730,'Stop gradient',15,MUTED)
    text(1197,730,'Visual tensors and survival curve are schematic',13,MUTED,anchor='end')

    OUT.mkdir(parents=True,exist_ok=True)
    base=OUT/'v313_architecture'
    renderPDF.drawToFile(d,str(base.with_suffix('.pdf')))
    renderSVG.drawToFile(d,str(base.with_suffix('.svg')))
    # SVG keeps independent editable objects and uses available standard fonts.
    svg=base.with_suffix('.svg')
    source=svg.read_text(encoding='utf-8')
    source=source.replace('font-family: FigBold;', 'font-family: Arial; font-weight: bold;')
    source=source.replace('font-family: FigItalic;', 'font-family: Arial; font-style: italic;')
    source=source.replace('font-family: FigSans;', 'font-family: Arial;')
    svg.write_text(source,encoding='utf-8')
    pdf=pdfium.PdfDocument(str(base.with_suffix('.pdf')))
    pdf[0].render(scale=4).to_pil().save(base.with_suffix('.png'),dpi=(288,288))
    pdf[0].render(scale=1.8).to_pil().save(OUT/'v313_architecture_preview.png')
    pdf.close()
    return d


if __name__ == '__main__':
    build_architecture()
    print('Architecture exported as editable SVG, vector PDF and 4880 x 3008 PNG.')
