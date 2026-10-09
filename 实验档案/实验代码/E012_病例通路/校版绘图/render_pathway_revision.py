from pathlib import Path
import hashlib,json,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
SRC=OUT.parent/'paper/figures/v313_reference_panels_real_20261008'
path=SRC/'raw/pathway_case_panel_raw.json'
source_bytes=path.read_bytes()
r=json.loads(source_bytes)
p=json.loads((SRC/'percentile/pathway_case_panel_percentile.json').read_bytes())
a=np.array(r['raw_attention'],dtype=float)
k,n=a.shape
assert (k,n)==(8,329) and np.isfinite(a).all()
assert np.allclose(a.sum(1),1,atol=1e-6,rtol=0)
order=np.lexsort((-a.max(0),a.argmax(0)))
names=np.empty(n,dtype=object)
names[order]=r['pathway_order']
assert len(set(names))==n
lookup={v:i for i,v in enumerate(names)}
def rank(x):
    _,inv,count=np.unique(x,return_inverse=True,return_counts=True)
    return (np.cumsum(count)[inv]-count[inv]/2)/len(x)
b=np.stack([rank(x) for x in a])
for panel in r['panels']:
    for v in panel['pathways']:
        assert abs(a[panel['slot'],lookup[v['pathway']]]-v['raw_attention'])<1e-12
for panel in p['panels']:
    for v in panel['pathways']:
        assert abs(b[panel['slot'],lookup[v['pathway']]]-v['displayed_value'])<1e-12
chosen={v['pathway'] for panel in r['panels'] for v in panel['pathways']}
selected=np.array([i for i in order if names[i] in chosen])
assert len(selected)==33
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none','font.size':9})
def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/(name+'.'+ext),dpi=220,facecolor='white')
    plt.close(fig)
def heat(rows,name,all_rows=False):
    fig=plt.figure(figsize=(12,28) if all_rows else (14,10.5))
    bottom,top=(.052,.944) if all_rows else (.125,.87)
    labels=fig.add_axes([.012,bottom,.51,top-bottom])
    labels.set(xlim=(0,1),ylim=(len(rows)-.5,-.5));labels.axis('off')
    for y,i in enumerate(rows):
        label=names[i].replace('_',' ')
        if not all_rows: label=textwrap.fill(label,66,break_long_words=False)
        labels.text(.99,y,label,ha='right',va='center',fontsize=3.9 if all_rows else 8.3,linespacing=1.05)
    for j,x in enumerate([.535,.78]):
        ax=fig.add_axes([x,bottom,.195,top-bottom])
        data=a[:,rows].T*1000 if j==0 else b[:,rows].T
        lo,hi=(a.min()*1000,a.max()*1000) if j==0 else (0,1)
        im=ax.imshow(data,aspect='auto',interpolation='nearest',cmap='YlOrRd',vmin=lo,vmax=hi)
        ax.set_xticks(range(k),range(k));ax.set_yticks([]);ax.set_xlabel('Omics slots',fontsize=10)
        ax.set_title('A  Raw attention' if j==0 else 'B  Within-slot percentile',loc='left',fontsize=11,pad=12)
        if not all_rows:
            for y in np.arange(.5,len(rows),1):ax.axhline(y,c='white',lw=.35,alpha=.65)
        for xx in np.arange(.5,k,1):ax.axvline(xx,c='white',lw=.4,alpha=.55)
        pos={i:y for y,i in enumerate(rows)}
        for panel in r['panels']:
            for v in panel['pathways']:
                i=lookup[v['pathway']]
                if i not in pos:continue
                is_top=v['selection']=='top'
                if is_top:ax.scatter(panel['slot'],pos[i],marker='o',s=5 if all_rows else 22,facecolors='none',edgecolors='black',linewidths=.45 if all_rows else .75)
                else:ax.scatter(panel['slot'],pos[i],marker='x',s=5 if all_rows else 22,c='black',linewidths=.45 if all_rows else .75)
        cax=fig.add_axes([x,.026 if all_rows else .065,.195,.009 if all_rows else .013])
        cb=fig.colorbar(im,cax=cax,orientation='horizontal')
        cb.ax.tick_params(labelsize=7 if all_rows else 8)
        cb.set_label('Raw attention × 10⁻³' if j==0 else 'Slot-local rank (0–1)',fontsize=8)
    fig.suptitle(f"{r['case_id']} | {len(rows)} pathways × {k} omics slots",y=.988,fontsize=15,fontweight='bold')
    desc='All 329 pathways' if all_rows else 'Union of each slot\'s Top-3 and Bottom-3: 33 distinct pathways'
    fig.text(.52,.975 if all_rows else .94,desc,ha='center',fontsize=9)
    fig.text(.52,.965 if all_rows else .918,'○ Top-3    × Bottom-3    |    Same recorded row order in both panels',ha='center',fontsize=8.5)
    fig.text(.52,.005 if all_rows else .012,'Raw color limits use the observed range (zero omitted); percentiles show order, not effect size.',ha='center',fontsize=8)
    save(fig,name)
def dots():
    fig=plt.figure(figsize=(18,14))
    grid=fig.add_gridspec(4,2,left=.012,right=.985,bottom=.058,top=.905,wspace=.12,hspace=.39)
    colors=['#D55E00','#0072B2','#009E73','#CC79A7','#E69F00','#56B4E9','#A6761D','#666666']
    for slot,spec in enumerate(grid):
        inner=spec.subgridspec(1,2,width_ratios=[1.75,1],wspace=.035)
        lab=fig.add_subplot(inner[0,0]);ax=fig.add_subplot(inner[0,1])
        entries=r['panels'][slot]['pathways']
        vals=np.array([v['raw_attention']*1000 for v in entries])
        ax.scatter(vals,range(6),s=33,c=[colors[slot]]*3+['#9CA6AF']*3,zorder=3)
        ax.axvline(1000/n,ls='--',lw=.8,c='#666666');ax.axhline(2.5,c='#D7DCE0',lw=.8)
        ax.set(xlim=(2.25,3.65),ylim=(5.7,-.7),yticks=[],xticks=[2.4,2.8,3.2,3.6])
        ax.grid(axis='x',c='#E4E7EB',lw=.7);ax.set_xlabel('Raw attention × 10⁻³',fontsize=9)
        ax.spines[['top','right','left']].set_visible(False)
        lab.set(xlim=(0,1),ylim=ax.get_ylim());lab.axis('off')
        lab.set_title(f'Slot {slot} | Top-3 / Bottom-3',loc='left',fontsize=11,fontweight='bold',color=colors[slot])
        for y,v in enumerate(entries):
            label=textwrap.fill(v['pathway'].replace('_',' '),56,break_long_words=False)
            lab.text(.99,y,label,ha='right',va='center',fontsize=9,linespacing=1.05)
            ax.annotate(f'{vals[y]:.3f}',(vals[y],y),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
    fig.suptitle(f"{r['case_id']} | recorded Top-3 / Bottom-3 pathway weights",fontsize=16,fontweight='bold',y=.982)
    fig.text(.5,.946,'Common raw scale in every slot; dashed line = uniform attention 1/329 = 0.003040',ha='center',fontsize=10)
    fig.text(.5,.017,'Dots show original weights; selection denotes rank, not pathway activation or causal risk contribution.',ha='center',fontsize=9)
    save(fig,'slot_top_bottom_raw_dotplots')
heat(selected,'pathway_selected_dense_comparison')
heat(order,'pathway_all_329_dense_comparison',True)
dots()
prob=a/a.sum(1,keepdims=True)
entropy=-(prob*np.log(prob)).sum(1)/np.log(n)
audit=dict(case_id=r['case_id'],source_commit='2b1c622',source_json_sha256=hashlib.sha256(source_bytes).hexdigest(),source_export=r['source'],shape=list(a.shape),uniform_attention=1/n,raw_min=float(a.min()),raw_max=float(a.max()),per_slot_sums=a.sum(1).tolist(),normalized_entropy=entropy.tolist(),top3_mass=np.sort(a,axis=1)[:,-3:].sum(1).tolist(),selected_pathways=len(selected),row_sort='preferred raw slot, descending raw maximum',raw_color_limits='global observed min/max; nonzero lower bound explicitly shown',percentile_formula='within-slot midrank / 329, matches source displayed values',identity_checks='all selected names, raw weights, and percentiles match both recorded panel JSONs',limitations='One patient final pooling/prototype-rollout weights; not a slot-collapse test or causal attribution')
assert path.read_bytes()==source_bytes
(OUT/'figure_revision_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(audit,indent=2,ensure_ascii=False))
