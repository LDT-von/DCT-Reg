"""Build the source-aligned DCT v2.3 figure and Chinese method note.

Only artifact authoring is performed. This script never loads patient data,
executes a model, starts a training job, or changes experiment results.
Use the Codex bundled Python runtime. Rendering is handled separately.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from reportlab.lib.colors import toColor as HexColor
from reportlab.pdfgen import canvas


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO / "docs" / "research" / "dct_v23_artifact_spec_20261014.json"
NODE = Path(r"C:\Users\栋栋\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
SHARP = Path(r"C:\Users\栋栋\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules\sharp")
COLORS = {"ink": "#222222", "pink": "#f4c3d4", "green": "#d5e7c9",
          "purple": "#dec4ea", "blue": "#d9e8f3", "yellow": "#fff0c9",
          "teal": "#7ca4ac", "red": "#bf4b4b", "muted": "#646464",
          "border": "#b2b2b2", "light": "#fafafa"}


class Scene:
    """A small vector scene with identical SVG and PDF geometry."""
    def __init__(self, width=1800, height=1100):
        self.width, self.height = width, height
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<rect width="100%" height="100%" fill="white"/>']
        self.ops = []

    def rect(self, x, y, w, h, fill="white", stroke="#222222", radius=0, lw=1.8, dash=None):
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"{dashed}/>')
        self.ops.append(("rect", (x, y, w, h, fill, stroke, radius, lw, dash)))

    def line(self, points, color="#222222", lw=1.8, dash=None, arrow=False):
        pts = " ".join(f"{x},{y}" for x, y in points)
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{lw}" stroke-linejoin="round"{dashed}/>')
        self.ops.append(("line", (points, color, lw, dash)))
        if arrow:
            x, y = points[-1]
            px, py = points[-2]
            angle = math.atan2(y-py, x-px)
            a = 10
            tri = [(x, y), (x-a*math.cos(angle-.43), y-a*math.sin(angle-.43)),
                   (x-a*math.cos(angle+.43), y-a*math.sin(angle+.43))]
            self.polygon(tri, color, color, 0)

    def polygon(self, points, fill, stroke, lw=1):
        pts = " ".join(f"{x},{y}" for x, y in points)
        self.svg.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
        self.ops.append(("polygon", (points, fill, stroke, lw)))

    def circle(self, x, y, r, fill="white", stroke="#222222", lw=1.5):
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
        self.ops.append(("circle", (x, y, r, fill, stroke, lw)))

    def text(self, x, y, s, size=20, color="#222222", anchor="middle", weight="normal"):
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{html.escape(s)}</text>')
        self.ops.append(("text", (x, y, s, size, color, anchor, weight)))

    def label(self, x, y, lines, size=20, color="#222222", weight="normal"):
        for i, s in enumerate(lines):
            self.text(x, y+i*(size+6), s, size, color, weight=weight)

    def block(self, x, y, w, h, lines, fill, size=20, stroke="#222222", dash=None):
        self.rect(x, y, w, h, fill, stroke, 10, dash=dash)
        first = y+h/2 - (len(lines)-1)*(size+6)/2+size/3
        self.label(x+w/2, first, lines, size)

    def write(self, svg_path, pdf_path):
        svg_path.write_text("\n".join(self.svg+["</svg>"]), encoding="utf-8")
        scale = 1152 / self.width
        c = canvas.Canvas(str(pdf_path), pagesize=(self.width*scale, self.height*scale))
        c.setTitle("DCT v2.3 architecture")
        c.scale(scale, scale)
        H = self.height
        for kind, p in self.ops:
            if kind == "rect":
                x,y,w,h,fill,stroke,radius,lw,dash = p
                c.setFillColor(HexColor(fill)); c.setStrokeColor(HexColor(stroke)); c.setLineWidth(lw)
                c.setDash([float(v) for v in dash.split()] if dash else [])
                c.roundRect(x,H-y-h,w,h,radius,stroke=1,fill=1)
            elif kind == "line":
                points,color,lw,dash=p
                c.setStrokeColor(HexColor(color)); c.setLineWidth(lw)
                c.setDash([float(v) for v in dash.split()] if dash else [])
                path=c.beginPath();path.moveTo(points[0][0],H-points[0][1])
                for x,y in points[1:]:path.lineTo(x,H-y)
                c.drawPath(path,stroke=1,fill=0)
            elif kind == "polygon":
                points,fill,stroke,lw=p
                c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(stroke));c.setLineWidth(lw);c.setDash([])
                path=c.beginPath();path.moveTo(points[0][0],H-points[0][1])
                for x,y in points[1:]:path.lineTo(x,H-y)
                path.close();c.drawPath(path,stroke=int(lw>0),fill=1)
            elif kind == "circle":
                x,y,r,fill,stroke,lw=p
                c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(stroke));c.setLineWidth(lw);c.setDash([])
                c.circle(x,H-y,r,stroke=1,fill=1)
            elif kind == "text":
                x,y,s,size,color,anchor,weight=p
                c.setFillColor(HexColor(color));c.setFont("Helvetica-Bold" if weight=="bold" else "Helvetica",size)
                if anchor=="middle":c.drawCentredString(x,H-y,s)
                elif anchor=="end":c.drawRightString(x,H-y,s)
                else:c.drawString(x,H-y,s)
        c.showPage();c.save()


def slot_stack(sc, x, cy, modality):
    palette=["#d4e2ef", "#e9b7d1", "#f5d899", "#b9d6b0", "#d3c2e5", "#bedada"]
    sc.rect(x-7,cy-86,45,178,"white","#777777",7,lw=1,dash="4 4")
    for i,c in enumerate(palette):sc.rect(x,cy-76+i*28,31,23,c,"#555555",1,lw=1)
    sc.text(x+14,cy+120,f"{modality} slots",18)


def draw_architecture(out_dir, spec):
    out_dir.mkdir(parents=True,exist_ok=True)
    sc=Scene()
    pink,green,purple,blue=[COLORS[k] for k in ["pink","green","purple","blue"]]
    sc.text(900,40,"DCT v2.3",30,weight="bold")
    # Main two-modality model. Inputs are schematic, not patient images.
    rng=random.Random(23)
    sc.rect(35,181,153,153,"#f6e9f0",radius=2)
    for _ in range(100):
        x,y=rng.uniform(44,179),rng.uniform(190,325)
        sc.circle(x,y,rng.uniform(1,4),rng.choice(["#cf85aa","#b489c2","#ddbad2"]),"#c89bbc",.25)
    sc.label(111,364,["Whole slide","image"],19)
    for i in range(5):
        sc.rect(228,179+i*33,28,28,["#e1a7bf","#d998b5","#e6b6cb","#bf9ab8","#e0b2c6"][i],lw=1)
    sc.label(242,382,["Patch features","UNI2-h"],18)
    sc.line([(188,257),(218,257)],arrow=True)
    sc.block(290,182,130,150,["WSI","projection"],pink,23)
    sc.line([(259,257),(286,257)],arrow=True)
    for i in range(5):sc.rect(456,191+i*29,19,20,pink,lw=1)
    sc.line([(420,257),(450,257)],arrow=True)
    sc.block(512,182,136,150,["Slot Attention","Prototype","coordinates"],purple,20)
    sc.line([(476,257),(508,257)],arrow=True)
    slot_stack(sc,687,257,"WSI")
    sc.line([(648,257),(676,257)],arrow=True)

    sc.circle(111,538,73,"#f4f7f1")
    genes=[("G1",63,507),("G2",120,490),("G3",82,551),("G4",132,552),("G5",96,590)]
    for text,x,y in genes:sc.rect(x,y,37,23,green,lw=.8);sc.text(x+18,y+17,text,13)
    sc.label(111,642,["Paired omics"],19)
    for i in range(5):
        for j in range(3):sc.rect(218+j*14,465+i*32,13,19,["#b9d6b0","#d6e4c9","#edd58e"][i%3],lw=.8)
    sc.label(242,661,["Pathway groups"],18)
    sc.line([(188,538),(209,538)],arrow=True)
    sc.block(290,463,130,150,["Pathway","encoders"],green,23)
    sc.line([(260,538),(286,538)],arrow=True)
    for i in range(5):sc.rect(456,472+i*29,19,20,green,lw=1)
    sc.line([(420,538),(450,538)],arrow=True)
    sc.block(512,463,136,150,["Slot Attention","Prototype","coordinates"],purple,20)
    sc.line([(476,538),(508,538)],arrow=True)
    slot_stack(sc,687,538,"Omics")
    sc.line([(648,538),(676,538)],arrow=True)

    sc.block(777,327,156,148,["Pair features","+ latent","stages"],"#efe5f4",21)
    sc.line([(725,257),(752,257),(752,370),(773,370)],pink,2.5,arrow=True)
    sc.line([(725,538),(752,538),(752,432),(773,432)],"#7ca467",2.5,arrow=True)
    sc.rect(987,292,178,214,blue,radius=10)
    sc.text(1076,327,"Cost + budgets",21)
    sc.text(1076,365,"KL-relaxed UOT",21)
    sc.line([(933,401),(983,401)],arrow=True)
    # Actual-plan thumbnail.
    heat=[[.8,.17,.06,.03],[.16,.67,.22,.04],[.03,.07,.42,.23],[.04,.13,.21,.59]]
    for i,row in enumerate(heat):
        for j,v in enumerate(row):
            k=int(255-170*v); sc.rect(1012+j*23,398+i*17,23,17,f"#{k:02x}{min(255,k+15):02x}ff","white",lw=.5)
    sc.text(1130,439,"T",23)
    sc.block(1225,311,161,177,["Mass-gated","transport","risk readout"],"#e3eef5",21)
    sc.line([(1165,401),(1221,401)],arrow=True)
    sc.block(1454,312,121,177,["Native +","transport","fusion"],"#f2f2f2",20)
    sc.line([(1386,401),(1450,401)],arrow=True)
    sc.block(1628,322,132,158,["Survival","head"],"#fff0c9",23)
    sc.line([(1575,401),(1624,401)],arrow=True)
    # Native slot summaries bypass transport and remain available.
    sc.circle(801,146,22,"white");sc.text(801,154,"P",23)
    sc.line([(706,164),(706,146),(775,146)],pink,2.5,arrow=True)
    sc.line([(823,146),(1514,146),(1514,308)],pink,2.5,arrow=True)
    sc.text(1105,130,"Native WSI summary",20,"#9a4a6a")
    sc.circle(801,653,22,"white");sc.text(801,661,"P",23)
    sc.line([(725,538),(747,538),(747,653),(775,653)],"#7ca467",2.5,arrow=True)
    sc.line([(823,653),(1514,653),(1514,493)],"#7ca467",2.5,arrow=True)
    sc.text(1102,640,"Native omics summary",20,"#4c7240")
    for x,col,txt in [(1010,pink,"WSI token"),(1158,green,"Pathway token"),(1342,purple,"Slots")]:
        sc.rect(x,568,17,17,col,lw=.8);sc.text(x+24,582,txt,15,anchor="start")

    # Separate training-only panel, faithful to the masked auxiliary pass.
    sc.rect(35,717,1725,322,"white","#777777",14,lw=1.7,dash="9 7")
    sc.text(65,748,"Training only",21,anchor="start",weight="bold")
    sc.block(61,800,143,139,["Raw gene","inputs"],green,21)
    sc.block(243,799,150,140,["Global gene","mask"],"#edf4e7",21)
    sc.line([(204,868),(239,868)],"#7ca467",2,arrow=True)
    sc.block(438,799,169,140,["Masked view","same encoders","and UOT"],"#ebe3f4",20)
    sc.line([(393,868),(434,868)],"#7ca467",2,arrow=True)
    sc.block(654,779,151,57,["Native"],"#edf4e7",20)
    sc.block(654,849,151,57,["Learned T"],blue,20)
    sc.block(654,919,151,57,["Q diagnostic"],"#f0f0f0",19)
    sc.line([(607,868),(630,868),(630,807),(650,807)],arrow=True)
    sc.line([(630,868),(650,878)],arrow=True)
    sc.line([(630,868),(630,947),(650,947)],dash="6 5",arrow=True)
    sc.block(850,800,149,140,["Shared trunk","+ pathway","gene heads"],"#fff0c9",20)
    for y in [807,878,947]:sc.line([(805,y),(827,y),(827,869),(846,869)],dash="6 5" if y==947 else None,arrow=True)
    sc.block(1050,800,155,140,["Masked gene","reconstruction"],"#ecf3e7",20)
    sc.line([(999,869),(1046,869)],arrow=True)
    sc.line([(132,939),(132,1008),(1128,1008),(1128,944)],"#7ca467",1.7,dash="6 5",arrow=True)
    sc.text(626,1000,"Held-out values are targets only",17,"#5a7251")
    sc.block(1268,784,177,81,["Held-patient","OOF gains"],"#f4e7ec",21)
    sc.block(1268,898,177,60,["RNA + WSI mean","Ridge teachers"],"#f7f2f5",16)
    sc.line([(1356,898),(1356,869)],COLORS["red"],1.7,dash="6 5",arrow=True)
    sc.block(1500,784,222,81,["Stage-cost","calibration"],"#edf1f7",20)
    sc.line([(1445,824),(1496,824)],COLORS["red"],2,dash="6 5",arrow=True)
    sc.line([(1611,784),(1778,784),(1778,226),(1076,226),(1076,288)],COLORS["red"],2,dash="6 5",arrow=True)
    for i, label in enumerate(["Solid: model information flow", "Dashed: supervision / audit", "P: mean pooling"]):
        sc.text(1500,930+i*22,label,14,COLORS["muted"],anchor="start")
    prefix=out_dir/"dct_v23_architecture_20261014"
    sc.write(prefix.with_suffix(".svg"),prefix.with_suffix(".pdf"))
    # Rasterize SVG at 300 dpi equivalent for a 16-inch exported figure.
    cmd=[str(NODE),"-e",f"require({json.dumps(str(SHARP))})({json.dumps(str(prefix.with_suffix('.svg')))})"+
         f".resize(4800).png().withMetadata({{density:300}}).toFile({json.dumps(str(prefix.with_suffix('.png')))})"+
         ".catch(e=>{console.error(e);process.exit(1)})"]
    subprocess.run(cmd,check=True)
    return {k:str(prefix.with_suffix("."+k)) for k in ["svg","png","pdf"]}


def set_font(run, size=11, bold=False, color="000000"):
    run.font.name="Calibri";run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=RGBColor.from_string(color)
    rpr=run._element.get_or_add_rPr()
    fonts=rpr.find(qn("w:rFonts"))
    if fonts is None:fonts=OxmlElement("w:rFonts");rpr.insert(0,fonts)
    for k,v in {"ascii":"Calibri","hAnsi":"Calibri","eastAsia":"宋体"}.items():fonts.set(qn("w:"+k),v)


def para(doc,text,boldlead=None):
    p=doc.add_paragraph();p.paragraph_format.space_after=Pt(8);p.paragraph_format.line_spacing=1.22
    if boldlead and text.startswith(boldlead):set_font(p.add_run(boldlead),bold=True);set_font(p.add_run(text[len(boldlead):]))
    else:set_font(p.add_run(text))
    return p


def heading(doc,text,level=1):
    p=doc.add_paragraph(style=f"Heading {level}");set_font(p.add_run(text),size=15 if level==1 else 12,bold=True)
    p.paragraph_format.space_before=Pt(12);p.paragraph_format.space_after=Pt(8)
    return p


def table(doc,headers,rows,widths):
    t=doc.add_table(rows=1, cols=len(headers));t.autofit=False
    for i,w in enumerate(widths):t.columns[i].width=Cm(w)
    for i,text in enumerate(headers):t.rows[0].cells[i].text=text
    for row in rows:
        cells=t.add_row().cells
        for i,text in enumerate(row):cells[i].text=text
    tblpr=t._tbl.tblPr
    borders=OxmlElement("w:tblBorders")
    for name in ["top","left","bottom","right","insideH","insideV"]:
        el=OxmlElement("w:"+name);el.set(qn("w:val"),"single");el.set(qn("w:sz"),"4");el.set(qn("w:color"),"D9D9D9");borders.append(el)
    tblpr.append(borders)
    for ri,row in enumerate(t.rows):
        for ci,cell in enumerate(row.cells):
            cell.width=Cm(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=cell._tc.get_or_add_tcPr();mar=OxmlElement("w:tcMar")
            for tag,val in [("top",85),("bottom",85),("left",100),("right",100)]:
                el=OxmlElement("w:"+tag);el.set(qn("w:w"),str(val));el.set(qn("w:type"),"dxa");mar.append(el)
            pr.append(mar)
            shade=OxmlElement("w:shd");shade.set(qn("w:fill"),"E7EEF5" if ri==0 else ("F7F9FA" if ri%2==0 else "FFFFFF"));pr.append(shade)
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing=1.12;p.paragraph_format.space_after=Pt(2)
                for r in p.runs:set_font(r,size=10.5,bold=ri==0)
    repeat=OxmlElement("w:tblHeader");t.rows[0]._tr.get_or_add_trPr().append(repeat)
    doc.add_paragraph().paragraph_format.space_after=Pt(2)
    return t


def hyperlink(doc,text,url):
    p=doc.add_paragraph();p.paragraph_format.space_after=Pt(6)
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    link=OxmlElement("w:hyperlink");link.set(qn("r:id"),p.part.relate_to(url,RT.HYPERLINK,is_external=True))
    r=OxmlElement("w:r");rpr=OxmlElement("w:rPr");color=OxmlElement("w:color");color.set(qn("w:val"),"235E8B");rpr.append(color)
    size=OxmlElement("w:sz");size.set(qn("w:val"),"20");rpr.append(size);r.append(rpr)
    txt=OxmlElement("w:t");txt.text=text;r.append(txt);link.append(r);p._p.append(link)


def build_document(path, spec):
    doc=Document();sec=doc.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7)
    sec.top_margin=Cm(2.0);sec.bottom_margin=Cm(1.8);sec.left_margin=Cm(2.1);sec.right_margin=Cm(2.1)
    for name in ["Normal","Title","Subtitle","Heading 1","Heading 2"]:
        st=doc.styles[name];st.font.color.rgb=RGBColor(0,0,0);st.font.name="Calibri"
        st.element.get_or_add_rPr().append(OxmlElement("w:rFonts"))
        st.element.rPr.rFonts.set(qn("w:eastAsia"),"宋体")
    doc.styles["Normal"].font.size=Pt(11)
    # Bundled default templates may carry an inherited blue Title rule.
    for root in [doc.styles.element,doc._element]:
        for border in list(root.iter(qn("w:pBdr"))):
            border.getparent().remove(border)
    doc.core_properties.title="DCT v2.3 方法与模块说明";doc.core_properties.subject="选择性跨模态运输与原始组学遮盖重构"
    doc.core_properties.author="DCT Research"
    title=doc.add_paragraph(style="Title");set_font(title.add_run("DCT v2.3 方法与模块说明"),24,True)
    para(doc,"版本开始日期 2026年10月14日  交付核查日期 2026年10月15日",boldlead="版本开始日期")
    para(doc,"本版本把严格遮盖的原始基因恢复接入独立 DCT，并用训练折内部的患者外预测收益监督既有运输成本。重构回答“病理能否补回隐藏的分子信息”，收益校准回答“哪些跨模态交换值得优先考虑”。最终任务仍是患者生存预测。")
    heading(doc,"1 要解决的问题")
    para(doc,"我们只有患者级病理和组学配对，没有病理区域与通路的局部对应标注。两种模态也只部分对应：病理可能帮助解释一部分分子状态，却不必解释全部组学。模型既要学习有用的匹配，又要保留每种模态自己的预后证据。")
    para(doc,"此前的生存标签给每位患者提供的监督很少。运输分支可以拟合训练标签，却没有直接证据说明它送入的病理信息能够恢复未见分子内容。仅让完整组学参与运输后再恢复组学，还可能利用答案生成计划，使较低误差不能证明真正的跨模态恢复能力。")
    heading(doc,"2 这一版的 idea")
    para(doc,"先遮住部分真实基因，再让病理沿现有运输关系帮助补全；同时用未参与教师拟合的训练患者测量补全的额外收益。把这种收益用于校准现有槽对成本，让交换优先考虑可验证的帮助。",boldlead="先遮住部分真实基因")
    para(doc,"这不是增加解码器数量来计算创新。候选贡献在于严格遮盖、患者外收益估计、局部运输选择三者的连接。患者提分与新颖性都需要独立证据，当前实现交付不等于已经证明性能优势。")
    heading(doc,"3 一句话理解")
    para(doc,"组学自己能补一部分答案；这一版要学习病理还能补回什么，并把这份额外帮助交给运输选择。重构是训练辅助任务，推理时仍输入患者的完整病理和组学，输出生存风险。")

    doc.add_page_break();heading(doc,"4 保留的模型基础")
    table(doc,["组件","方法与用途"],[
        ["WSI 投影与通路编码器","把预提取病理特征和各通路输入映射到共同维度；沿用 v2.2 的容量与正则化控制。"],
        ["语义槽与原型坐标","把大量病理 token 和通路 token 汇总为有限语义槽，并为运输提供可比较坐标。"],
        ["阶段条件 UOT","在各阶段学习成本，以 KL 松弛允许实际质量偏离参考预算；保留部分对应的建模能力。"],
        ["原生读取与质量控制残差","原生槽汇总直接参与预测；运输更新按实际质量接入，减少交换时原生路径仍可工作。"],
        ["生存头与既有六项损失","延续主 NLL、IPCW rank、槽 NLL、多样性、关系与 native 监督，不把常规正则化视作新数学贡献。"]
    ],[4.0,12.7])
    heading(doc,"5 新组件各自有什么用")
    table(doc,["新组件","具体工作","解决的问题"],spec.get("new_modules",[
        ["全局基因遮盖","先按唯一基因身份遮盖，再映射到所有通路入口。","避免重叠通路把隐藏答案重新带入模型。"],
        ["共享原始基因解码器","用同一解码参数比较原生补全、学习运输和独立混合。","恢复固定真实目标，并控制解码器能力差异。"],
        ["患者外收益工具","训练折内部划分教师拟合患者与留出患者。","减少模型记住患者后给自身匹配打高分。"],
        ["运输收益校准","读取患者外收益，监督现有绝对槽对成本。","让可泛化的恢复帮助影响运输选择。"]
    ]),[4.0,6.2,6.5])
    para(doc,"架构图用实线表示模型中的信息流，用虚线表示训练目标或校准监督。底部的遮盖重构与患者外收益路径只在训练中使用。图中的病理与基因图标为示意绘制，不代表任何患者实验结果。")

    doc.add_page_break();heading(doc,"6 严格遮盖怎样防止看答案")
    para(doc,"遮盖发生在原始基因值进入任何通路编码、成本网络和预算网络之前。同一个基因如果出现在多个通路，所有副本必须一起遮盖。重构的查询只能使用通路身份及可见上下文，不能使用隐藏基因值生成查询、运输计划或读取门。")
    para(doc,"模型重构目标是外层训练折均值和标准差标准化后的固定基因表达值，而不是随编码器一起变化的潜表示。这里的原始值指输入文件中的表达数值，沿用已有上游处理，不表示未经处理的测序计数。目标值只参与误差计算；验证折、测试折的统计不能进入标准化和收益教师。")
    heading(doc,"7 为什么还需要患者外收益")
    para(doc,"同一个模型可以记住训练患者，在这些患者上给出很低的重构误差。重新抽一组遮盖位置仍不足以排除这种记忆。因此收益教师在训练折内部轮换留出患者：某位患者的收益，只由没有用该患者拟合的教师估计。教师用固定原始组学坐标和预提取 UNI 病理特征的均值，不使用随 DCT 更新的潜表示。")
    para(doc,"收益记录需要携带患者归属、教师拟合患者列表、标准化统计来源、遮盖设置和来源校验。校准工具不得把验证或测试患者混入教师拟合，也不得使用生存测试结果选择收益阈值。")
    heading(doc,"8 收益怎样解释")
    para(doc,"收益教师先测可见组学补全的误差，再测加入固定病理均值条件后的误差。原生误差减去条件误差就是额外恢复收益：正值表示病理条件有帮助，负值表示可能有害。这是患者与通路层面的恢复效用代理，尚不是 DCT 槽对的真实匹配收益，更不是病理与基因的因果对应。")
    para(doc,"教师采用小型 ridge 回归，不使用生存标签。默认将可见 RNA 固定随机投影到 32 维，再给条件教师加入 16 维固定病理均值投影；正则强度为 10，训练折内部做 3 折交叉拟合，使用 8 组遮盖与种子 3。每个全局基因至少被隐藏一次。教师标准化仅使用各自内部拟合患者的均值和标准差；跨教师收益是归一化误差差，不能解释为统一原始单位的风险。")
    para(doc,"该收益经可见通路到组学槽的归属映射，形成槽级成本校准目标；映射停止梯度。收益经单调的 sigmoid 变换转换为成本目标：正收益倾向较低成本，负收益倾向较高成本。该项只更新指定既有成本参数，不反传至编码器或解码器。高成本可能减少 UOT 中的交换，但不保证运输质量单调变化，也不能解释为生物学私有信息比例。")
    para(doc,spec.get("oof_detail","患者外收益由独立工具生成并保存，模型读取已冻结的收益标签。生成收益涉及真实患者训练，需由用户启动；本轮只完成实现与合成接口检查。"))

    doc.add_page_break();heading(doc,"9 训练时实际发生什么")
    for text in spec.get("training_steps",[
        "完整视图沿 v2.2 的编码、槽、UOT 与生存头计算主生存任务。",
        "同一批数据构造严格遮盖视图；隐藏值只作为重构目标，前向读取可见组学和病理。",
        "共享解码器恢复隐藏基因；按原生、学习运输和同预算独立混合记录误差。",
        "读取训练折患者外收益标签，经停止梯度的可见通路归属映射后，校准已有的阶段槽对成本。",
        "既有生存目标与新训练项共同更新；诊断分开记录误差、收益及运输响应。"]):
        para(doc,text)
    heading(doc,"10 推理时实际发生什么")
    para(doc,"输入完整的患者病理特征和组学，走原生与运输融合生存路径。不会遮住真实基因，也不需要先运行重构解码器或收益教师。训练辅助组件不能成为部署时输入额外患者信息的通道。")
    heading(doc,"11 独立耦合比较有什么用")
    para(doc,"学习计划与独立计划使用相同的实际行边际、列边际和总质量。这样可以比较具体匹配安排的作用，而不是把更大交换量误当成更好匹配。解码器也共享，防止一个分支靠更多参数占优。")
    para(doc,"独立耦合只作对照与诊断，没有新增同预算重构差额训练项。固定基准后的差额可能仅重新加权普通重构梯度，因此这一版没有把它宣传为全新的优化方向。")
    heading(doc,"初始配方")
    para(doc,"共同维度 128，WSI 隐层 256，dropout 0.30；重构解码隐层 64，基因遮盖比例 0.30。重构权重 0.05，收益校准权重 0.02，校准热身 3 轮，收益温度 0.10。这些数值是待验证起点，不能称为最优配方。遮盖视图增加一次辅助前向，离线收益生成也有额外拟合开销。")

    doc.add_page_break();heading(doc,"12 损失项为什么存在")
    table(doc,["目标","训练作用"],spec.get("loss_rows",[
        ["主生存 NLL 与 IPCW rank","拟合患者生存事件并约束风险排序。"],
        ["槽 NLL 与 slot diversity","为槽提供生存信息和差异约束；惩罚下降本身不能证明槽坍缩。"],
        ["关系与 native","固定实际预算比较生存匹配；为少参与交换的原生路径提供监督。"],
        ["严格遮盖重构","共享 trunk 和通路输出头计算原生与条件补全误差；有效患者与通路均权，两路合为一项。"],
        ["运输收益校准","用训练患者外的恢复效用代理校准已有阶段槽对成本，训练之外不增加收益预测头。"]
    ]),[5.0,11.7])
    para(doc,"完整候选共八项训练目标：六项既有生存与槽目标，加一项重构和一项收益校准。普通重构对照为七项，无重构基础臂为六项。两路重构共用解码器参数，不是三个解码器或三项重构损失。")
    para(doc,"遮盖按全局基因身份同步，但误差按有效患者与通路平均。同一基因跨通路重复出现时，可以重复计分，因此该数值不能称为全局唯一基因等权 MSE。")
    heading(doc,"13 成功需要哪些证据")
    para(doc,"确实重构：在未参与教师或模型拟合的患者上，隐藏基因恢复优于原生补全。匹配有效：学习匹配优于同预算独立混合，打乱配对病理后优势减弱。预后有效：在同患者划分、训练预算和模型选择规则下，生存结果优于无重构及普通遮盖重构。")
    para(doc,"损失下降、结构检查通过、最佳验证分数提高分别回答不同问题。不能用合成检查替代患者泛化，也不能只挑一个折或一个早期峰值宣布过拟合解决。")
    heading(doc,"14 当前交付和检查状态")
    para(doc,spec.get("checks_text","结构与合成检查数字待最终冻结记录填写。尚未启动真实患者训练、收益教师训练或患者推理，没有本版本的患者分数。"))
    para(doc,"v2、v2.1、v2.2、v3.13 与 E050 原材料保留。本候选采用独立版本与独立输出目录，重构收益生成和患者训练应使用新的产物路径。")

    doc.add_page_break();heading(doc,"15 相关工作与创新边界")
    para(doc,"遮盖重构、跨模态重构、最优运输和交叉拟合各有先例。我们应围绕真实问题和可验证机制描述贡献：在局部对应未知且部分对应的病理组学任务中，以严格遮盖恢复为训练信号，用训练患者外的额外恢复收益校准现有运输选择，并检验是否带来生存泛化。")
    para(doc,"当前版本的实现不是全领域首次证明，也没有患者性能优势证明。与相关工作区别需要在正文、源码和公平消融中持续核实。新增组件应按其实际作用说明，不能按损失或解码器数量计算创新。")
    for title,url in spec.get("references",[
        ["MORPHEUS 病理与可见组学恢复遮盖组学","https://arxiv.org/html/2508.00969v2"],
        ["LUCID SAE 运输计划与跨模态重构","https://arxiv.org/html/2602.07311"],
        ["MCSP OTMR 作者代码","https://github.com/AI4MyBUPT/MMOTR"],
        ["Statistical Matching via Schrödinger Bridge 固定边际与补全","https://arxiv.org/html/2606.22770"],
        ["Intent Controlled Partial OT 局部参与选择","https://arxiv.org/html/2605.20030"]
    ]):hyperlink(doc,title,url)
    heading(doc,"16 后续实验的判断顺序")
    para(doc,"先检查严格遮盖及患者归属是否成立，再检查未见患者上的原始基因恢复与收益校准，最后比较生存性能。基础修复、普通重构与收益校准应分臂，保持特征、患者划分、种子、训练预算和检查点选择一致。")
    para(doc,"如果基因恢复改善而生存不改善，说明辅助任务学到的信息尚未转化为预后价值。如果训练收益很高但患者外收益消失，说明它仍可能依赖患者记忆。如果收益教师估计不稳定，则先修正数据量和估计协议，不能继续堆叠模块解释。")
    para(doc,"这一版最重要的研究问题是：可以跨患者成立的恢复帮助，是否能够教会运输做出更有用的交换。")

    doc.add_page_break();heading(doc,"17 使用顺序与运行入口")
    para(doc,"在已有 DCT Reg 仓库中安装本候选，再修改三份配置的数据路径与 GPU 编号。校准文件按训练折分别保存；已有文件不会被覆盖，重做时使用新的目录，并同步修改配置中的校准目录。以下命令由用户在服务器执行，本轮交付没有启动这些任务。")
    heading(doc,"先运行收益生成工具")
    para(doc,"该步骤只读取声明的训练患者，拟合轻量教师，生成每折校准表。它是实际数据任务，需要配对 RNA 和已提取病理特征。")
    for command in [
        "python scripts/prepare_dct_v23_calibration.py \\",
        "  --config configs/dct_v23_blca_legacy_val_gain_calibrated.yaml \\",
        "  --output-dir /data1/DCT-Reg/实验档案/结果/E052_v23_20261014/calibration \\",
        "  --folds 0 1 2 3 4",
    ]:
        p=doc.add_paragraph();p.paragraph_format.space_after=Pt(2)
        set_font(p.add_run(command),9)
    heading(doc,"再运行校准候选")
    for command in [
        "python -m survot_rank.cli train \\",
        "  --config configs/dct_v23_blca_legacy_val_gain_calibrated.yaml",
    ]:
        p=doc.add_paragraph();p.paragraph_format.space_after=Pt(2)
        set_font(p.add_run(command),9)
    heading(doc,"对照与读取规则")
    para(doc,"control 配置关闭重构，masked_plain 配置启用普通严格遮盖重构；这两臂不要求收益表。gain_calibrated 臂必须先取得与当前训练患者、基因顺序和来源匹配的收益表，缺失或不符时拒绝训练。")
    para(doc,"三臂采用同一小容量、优化器设置、BLCA 划分、种子 3 和 30 轮预算。当前配方为 legacy val，最佳验证记录不是独立外部测试结论。患者实验完成后仍需按匹配协议比较，并另行锁定独立测试。")
    para(doc,"辅助审计入口为 model.reconstruction_audit，可返回相同隐藏位置下的原生、学习计划和同实际边际独立计划的重构。该入口需显式调用；普通生存评估不会读取隐藏答案、运行重构或查询收益。")
    # Simple page field without decorative rules.
    p=sec.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=p.add_run();fld=OxmlElement("w:fldSimple");fld.set(qn("w:instr"),"PAGE");r._r.addnext(fld)
    path.parent.mkdir(parents=True,exist_ok=True);doc.save(path)
    return str(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--spec",type=Path,default=DEFAULT_SPEC)
    ap.add_argument("--figure-only",action="store_true");ap.add_argument("--doc-only",action="store_true")
    args=ap.parse_args();spec=json.loads(args.spec.read_text(encoding="utf-8")) if args.spec.exists() else {}
    if not spec.get("implementation_frozen",False):
        raise SystemExit("Refusing final artifact authoring before implementation_frozen is recorded in the spec.")
    outputs={}
    if not args.doc_only:outputs.update(draw_architecture(REPO/"docs"/"figures",spec))
    if not args.figure_only:outputs["docx"]=build_document(REPO/"docs"/"DCT_v23_方法与模块说明.docx",spec)
    print(json.dumps(outputs,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
