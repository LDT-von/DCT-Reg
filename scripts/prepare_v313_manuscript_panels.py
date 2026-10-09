"""Render manuscript panels from saved v3.13 results; no model execution."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pypdfium2 as pdfium
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase.pdfmetrics import stringWidth

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "figures" / "v313_manuscript_integrated"
BLUE, ORANGE, GREEN = "#2874a6", "#d97732", "#438b54"
INK, GREY = "#202b36", "#dce2e6"
BLCA = np.array([
    [.6423,.6721,.6988,.7103,.7769], [.6296,.6918,.7044,.7340,.7889],
    [.6517,.6927,.7147,.7954,.7402], [.6576,.7030,.7399,.7524,.7496],
    [.6576,.7056,.6810,.7629,.7436], [.6576,.6910,.7203,.7805,.7444],
    [.6576,.6953,.7465,.7744,.7453],
])
KIRC = np.array([
    [.8073,.8640,.8379,.7837,.7671], [.7886,.8471,.7834,.7958,.7524],
    [.7800,.8268,.7752,.7943,.7841], [.8203,.8703,.7698,.7905,.7738],
    [.8239,.8527,.7732,.7890,.7841], [.7951,.8485,.7738,.7927,.8032],
    [.8253,.8485,.8202,.8094,.8084],
])
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def sha(path):
    data=path.read_bytes()
    # Git may normalize text line endings; checksums should survive OS changes.
    if path.suffix.lower() in {".json", ".svg"}:
        data=data.replace(b"\r\n",b"\n")
    return hashlib.sha256(data).hexdigest()
def text(d,x,y,value,size=9,color=INK,anchor="start",bold=False):
    d.add(String(x,y,str(value),fontName="Helvetica-Bold" if bold else "Helvetica",
                 fontSize=size,fillColor=HexColor(color),textAnchor=anchor))
def line(d,x1,y1,x2,y2,color=GREY,width=.6):
    d.add(Line(x1,y1,x2,y2,strokeColor=HexColor(color),strokeWidth=width))
def wrap(value,width=257,size=9):
    rows,current=[],""
    for word in value.replace("_"," ").split():
        trial=f"{current} {word}".strip()
        if current and stringWidth(trial,"Helvetica",size)>width:
            rows.append(current);current=word
        else:current=trial
    return rows+[current]
def palette(value,signed=False):
    value=float(np.clip(value,0,1))
    stops=["#2166ac","#f7f7f7","#b2182b"] if signed else ["#ffffcc","#feb24c","#e31a1c","#800026"]
    pos=value*(len(stops)-1);a=min(int(pos),len(stops)-2);fraction=pos-a
    ca,cb=HexColor(stops[a]),HexColor(stops[a+1])
    return Color(*(u+(v-u)*fraction for u,v in zip((ca.red,ca.green,ca.blue),(cb.red,cb.green,cb.blue))))
def save(d,name):
    pdf=OUT/f"{name}.pdf"
    renderPDF.drawToFile(d,str(pdf))
    renderSVG.drawToFile(d,str(OUT/f"{name}.svg"))
    with pdfium.PdfDocument(str(pdf)) as doc:
        doc[0].render(scale=3).to_pil().save(OUT/f"{name}.png")
def loss_panel():
    d=Drawing(540,305)
    text(d,270,287,"v3.13 loss ablation",13,anchor="middle",bold=True)
    text(d,270,271,"Five-fold best-validation C-index; one seed",9,anchor="middle")
    for x,arr,title,lo,hi,color in [(42,BLCA,"(a) BLCA",.60,.87,BLUE),(302,KIRC,"(b) KIRC",.72,.90,GREEN)]:
        y,w,h=66,220,168
        text(d,x+w/2,246,title,11,anchor="middle",bold=True)
        for tick in np.linspace(lo,hi,5):
            yy=y+(tick-lo)/(hi-lo)*h
            line(d,x,yy,x+w,yy);text(d,x-6,yy-3,f"{tick:.2f}",8,anchor="end")
        line(d,x,y,x,y+h,INK)
        for j,vals in enumerate(arr):
            xx=x+12+j*(w-24)/6;m,sd=float(vals.mean()),float(vals.std(ddof=1))
            mm=y+(m-lo)/(hi-lo)*h
            low,high=y+(m-sd-lo)/(hi-lo)*h,y+(m+sd-lo)/(hi-lo)*h
            c="#ad4937" if j==6 else color
            line(d,xx,low,xx,high,c,1.1)
            for yy in (low,high):line(d,xx-3,yy,xx+3,yy,c,1.1)
            for k,v in enumerate(vals):
                d.add(Circle(xx+(k-2)*2.6,y+(v-lo)/(hi-lo)*h,1.7,fillColor=HexColor("#9aabb7"),strokeColor=None))
            d.add(Circle(xx,mm,3.2,fillColor=HexColor(c),strokeColor=HexColor("#ffffff"),strokeWidth=.6))
            text(d,xx,high+7,f"{m:.4f}",7.4,c,"middle")
            text(d,xx,y-15,"Full" if j==6 else f"E{j}",8,anchor="middle")
        text(d,x+w/2,42,"E4: self only   E5: cross only",8,anchor="middle")
    text(d,270,22,"Dots: folds. Colored markers and bars: mean +/- sample SD.",8.5,anchor="middle")
    text(d,270,9,"E4 and E5 branch from E3; historical reconstruction weights differ.",8,anchor="middle")
    save(d,"fig2_loss_ablation")
def control_panel(status):
    audit_path=PAPER/"figures"/"v313_main_plots"/"audit.json"
    audit=read(audit_path)
    assert audit["passed"] and not audit["errors"] and len(audit["runs"])==20
    d=Drawing(540,470)
    text(d,270,450,"v3.13 Full and trained controls",13,anchor="middle",bold=True)
    colors=[BLUE,ORANGE,GREEN]
    for j,(name,c) in enumerate(zip(["Full","Direct","Independent"],colors)):
        xx=132+j*100;line(d,xx,429,xx+13,429,c,1.5);text(d,xx+17,426,name,9)
    for x,cancer in [(45,"blca"),(305,"kirc")]:
        y,w,h=250,216,130;lo,hi=(.62,.80) if cancer=="blca" else (.76,.88)
        full=np.array([r["cindex"][0] for r in sorted(status["ready_figures"]["4"]["records"],key=lambda a:a["fold"]) if r["cancer"]==cancer])
        arrays=[full]+[np.array([r["cindex"] for r in sorted(audit["runs"],key=lambda a:a["fold"]) if r["cancer"]==cancer and r["arm"]==arm]) for arm in ["direct","independent"]]
        assert all(a.shape==(5,) for a in arrays)
        text(d,x+w/2,400,f"({'a' if cancer=='blca' else 'b'}) {cancer.upper()}",11,anchor="middle",bold=True)
        for tick in np.linspace(lo,hi,4):
            yy=y+(tick-lo)/(hi-lo)*h;line(d,x,yy,x+w,yy);text(d,x-6,yy-3,f"{tick:.2f}",8,anchor="end")
        for a,c in zip(arrays,colors):
            pts=[(x+12+k*(w-24)/4,y+(v-lo)/(hi-lo)*h) for k,v in enumerate(a)]
            for aa,bb in zip(pts,pts[1:]):line(d,*aa,*bb,c,1)
            for xx,yy in pts:d.add(Circle(xx,yy,3,fillColor=HexColor(c),strokeColor=None))
        for k in range(5):text(d,x+12+k*(w-24)/4,y-14,str(k),8,anchor="middle")
        text(d,x+w/2,y-29,"Fold",9,anchor="middle")
        text(d,x+w/2,205,"Full minus control",10,anchor="middle",bold=True)
        by,bh=75,115
        def delta_y(v):return by+(v+.035)/.115*bh
        for tick in [-.02,0,.02,.04,.06]:
            yy=delta_y(tick);line(d,x,yy,x+w,yy,INK if tick==0 else GREY);text(d,x-6,yy-3,f"{tick:+.2f}" if tick else "0",8,anchor="end")
        for ci,(a,c,offset) in enumerate(zip(arrays[1:],colors[1:],[-5,5])):
            for k,v in enumerate(full-a):
                xx=x+12+k*(w-24)/4+offset;z,yy=delta_y(0),delta_y(v)
                d.add(Rect(xx-4,min(z,yy),8,abs(yy-z),fillColor=HexColor(c),strokeColor=None))
            text(d,x+w/2,36-13*ci,f"Mean Full - {'Direct' if ci==0 else 'Independent'}: {float((full-a).mean()):+.4f}",8,c,"middle")
        for k in range(5):text(d,x+12+k*(w-24)/4,55,str(k),8,anchor="middle")
    text(d,270,8,"Validation-selected checkpoints; final Full/control matching remains required.",8,anchor="middle")
    save(d,"fig3_trained_controls")
    return audit_path
def cohort_panel(cancer,source,raw=False):
    values=np.array(source["mean"],dtype=float);difference=values-values.mean(axis=1,keepdims=True)
    assert np.allclose(difference,source["difference"],rtol=0,atol=1e-9)
    assert values.shape==(len(source["names"]),4) and sum(source["group_counts"])==source["n"]
    rows=[wrap(n) for n in source["names"]];heights=[max(15,len(r)*9.8+5) for r in rows]
    height=sum(heights)+160;d=Drawing(500,height)
    text(d,250,height-21,f"{cancer.upper()} | v3.13 Full | N={source['n']}",12,anchor="middle",bold=True)
    text(d,250,height-40,"Raw mean pathway attention" if raw else "Difference from each pathway's four-group mean",10,anchor="middle")
    x,w,y=276,140,height-69;matrix=values if raw else difference;lim=float(np.abs(matrix).max())
    for i,(r,rh) in enumerate(zip(rows,heights)):
        yy=y-rh
        for j in range(4):
            val=matrix[i,j]/lim
            d.add(Rect(x+j*w/4,yy,w/4,rh,fillColor=palette(val if raw else (val+1)/2,not raw),strokeColor=None))
        text_y=yy+rh/2+(len(r)-1)*4.9-3
        for k,s in enumerate(r):text(d,x-8,text_y-k*9.8,s,9,anchor="end")
        y=yy
    d.add(Rect(x,y,w,sum(heights),fillColor=None,strokeColor=HexColor(INK),strokeWidth=.6))
    for j,n in enumerate(source["group_counts"]):
        xx=x+(j+.5)*w/4;text(d,xx,y-15,f"Q{j+1}",9,anchor="middle");text(d,xx,y-28,f"n={n}",8,anchor="middle")
    bx,by,bw,bh=434,y+max(0,(sum(heights)-180)/2),11,min(180,sum(heights))
    for k in range(100):d.add(Rect(bx,by+k*bh/100,bw,bh/100+.2,fillColor=palette(k/99,not raw),strokeColor=None))
    d.add(Rect(bx,by,bw,bh,fillColor=None,strokeColor=HexColor(INK),strokeWidth=.5))
    for fraction in [0,.25,.5,.75,1]:
        number=lim*fraction if raw else lim*(2*fraction-1)
        line(d,bx+bw,by+fraction*bh,bx+bw+3,by+fraction*bh,INK)
        text(d,bx+bw+6,by+fraction*bh-3,f"{number:.1e}",7.5)
    text(d,440,by+bh+14,"Attention" if raw else "Delta attention",8,anchor="middle")
    text(d,250,42,"Q1 to Q4: within-fold validation risk percentile, low to high",8.5,anchor="middle")
    text(d,250,27,"Absolute attention units; no row standardization or significance claim",8,anchor="middle")
    text(d,250,12,f"Raw range {values.min():.7f} to {values.max():.7f}; max |delta| {np.abs(difference).max():.2e}",8,anchor="middle")
    save(d,f"figS3_raw_attention_{cancer}" if raw else f"fig7_attention_difference_{cancer}")
    return dict(source_n=source["n"],group_counts=source["group_counts"],selected_pathways=len(rows),
                raw_min=float(values.min()),raw_max=float(values.max()),absolute_delta_limit=float(np.abs(difference).max()),
                delta_definition="group mean minus unweighted mean of four group means",
                grouping=source["grouping"],mean=values.tolist(),difference=difference.tolist())
def main():
    if OUT.exists():
        manifest_path=OUT/"manifest.json"
        if not manifest_path.is_file() or read(manifest_path).get("scope") != "Saved v3.13 development results; plotting only; zero model runs":
            raise FileExistsError(f"Unrecognized output directory: {OUT}")
    OUT.mkdir(parents=True,exist_ok=True);status_path=PAPER/"V313_MANUSCRIPT_STATUS.json";status=read(status_path)
    loss_panel();audit_path=control_panel(status);inputs=[audit_path];summary={}
    full_records=status["ready_figures"]["4"]["records"]
    for record in full_records:
        path=ROOT/record["path"]
        if "sha256_lf" in record:
            assert sha(path)==record["sha256_lf"]
        else:
            assert record["sha256"] in {sha(path), hashlib.sha256(path.read_bytes()).hexdigest()}
        inputs.append(path)
    snapshot_path=OUT/"source_values.json"
    snapshot_path.write_text(json.dumps({"loss_folds":{"blca":BLCA.tolist(),"kirc":KIRC.tolist()},
        "full_exact_folds":{c:[r["cindex"][0] for r in sorted(full_records,key=lambda a:a["fold"]) if r["cancer"]==c] for c in ["blca","kirc"]}},
        ensure_ascii=False,indent=2),encoding="utf-8")
    inputs.append(snapshot_path)
    for cancer in ["blca","kirc"]:
        path=PAPER/"figures"/"v313_slotspe_20261007_v2"/f"{cancer}_cohort"/"cohort_top_pathways.json"
        inputs.append(path);source=read(path);summary[cancer]=cohort_panel(cancer,source);cohort_panel(cancer,source,raw=True)
    outputs={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.json"}
    manifest=dict(scope="Saved v3.13 development results; plotting only; zero model runs",
                  source_files={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},outputs=outputs,cohort=summary,
                  hash_normalization="JSON/SVG CRLF normalized to LF; other files hashed as bytes",
                  loss_source="Four-decimal fold records retained in manuscript Appendix A",
                  controls_source="Audited control runs and exact factual sweep Full fold scores")
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(OUT),"figures":len(outputs)//3,"cohort":{k:{n:v for n,v in a.items() if n not in ["mean","difference"]} for k,a in summary.items()}},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
