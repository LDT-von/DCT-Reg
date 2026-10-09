"""Redraw recorded v3.13 sweep JSONs for the manuscript; no model execution."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper'/'figures'

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
        'axes.spines.top':False, 'axes.spines.right':False,
        'pdf.fonttype':42, 'ps.fonttype':42})
    fig, axes=plt.subplots(1,2,figsize=(9.4,3.8),constrained_layout=True)
    for ax,cancer,color,panel in zip(axes,['blca','kirc'],['#2878A8','#278863'],['a','b']):
        records=[json.loads((OUT/f'fig3_sweep_{cancer}_fold{fold}.json').read_text()) for fold in range(5)]
        alphas=np.asarray(records[0]['alphas'])
        assert all(r['cancer']==cancer and r['fold']==f and r['alphas']==alphas.tolist()
                   for f,r in enumerate(records))
        values=np.asarray([r['cindex'] for r in records])
        assert values.shape==(5,5) and np.isfinite(values).all()
        means=values.mean(axis=0);sd=values.std(axis=0,ddof=1)
        for f,row in enumerate(values):
            ax.plot(alphas,row,'o--',color=color,alpha=.28,lw=.9,ms=3.2,
                label='Individual folds' if f==0 else None)
        ax.errorbar(alphas,means,yerr=sd,fmt='o-',color=color,lw=2,ms=5,
            capsize=3,label='Mean ± sample SD')
        ax.set(title=f'({panel}) {cancer.upper()}',xlabel=r'Replacement strength $\alpha$',ylabel='C-index',
            xlim=(-.05,1.05),ylim=(.60,.90),xticks=alphas)
        ax.grid(axis='y',alpha=.2,lw=.6);ax.set_axisbelow(True)
        delta=float(means[-1]-means[0])
        ax.text(.03,.96,f'Mean change at α = 1: {delta:+.5f}',transform=ax.transAxes,va='top',fontsize=9)
        ax.legend(loc='lower left',frameon=False,fontsize=8)
    for suffix in ['png','pdf']:
        fig.savefig(OUT/f'fig4_transport_sweep_manuscript.{suffix}',dpi=300,facecolor='white',bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    main()
