"""Reference-style v3.13 panels from recorded results, never model execution."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from .manifest import predictions, split_ids, save_json, sha256, revision

CANCERS = ("BRCA", "COADREAD", "KIRC", "UCEC", "LUAD", "LUSC", "HNSC", "SKCM", "BLCA", "STAD")
LABELS = ("Exp0: patient NLL only", "Exp1: + IPCW ranking", "Exp2: + per-slot NLL",
          "Exp3: + diversity, no reconstruction", "Exp4: self only (0.025)",
          "Exp5: cross only (0.025)", "Full: self + cross (0.05 each)")


def text_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def normalize_text_output(path):
    path=Path(path)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')


def recorded_ablation(root):
    """Read the manuscript appendix and audited controls; do not invent folds."""
    root = Path(root)
    manuscript = root / "paper/DCT_v313_初稿.md"
    comparison = root / "paper/V313_OT_UNI2H_COMPARISON_INPUT.json"
    controls = root / "results/v313_evidence_v2/controls_audit.json"
    source = manuscript.read_text(encoding="utf-8")
    values = {c: {} for c in ("BLCA", "KIRC")}
    for cancer in values:
        match = re.search(r"\| " + cancer + r" 配置 \|(?P<table>.*?)\*表 A", source, re.S)
        if match is None:
            raise ValueError(f"Missing recorded appendix: {cancer}")
        for line in match['table'].splitlines():
            fields = [x.strip() for x in line.strip().strip('|').split('|')]
            if len(fields) == 6 and re.fullmatch(r"Exp[0-6]", fields[0]):
                fold_values = [float(re.fullmatch(r"([0-9.]+)（\d+）", x).group(1)) for x in fields[1:]]
                if fields[0] in values[cancer]:
                    raise ValueError("Duplicate recorded loss arm")
                values[cancer][fields[0]] = fold_values
        if set(values[cancer]) != {f"Exp{i}" for i in range(7)}:
            raise ValueError("Require all seven recorded loss arms")
    published = json.loads(comparison.read_text(encoding="utf-8"))
    full = next(m['folds'] for m in published['models'] if m['id'] == 'dct_v313')
    for cancer in values:
        if values[cancer]['Exp6'] != full[cancer]:
            raise ValueError("Full appendix disagrees with the ten-cancer fold record")
    variants = [dict(id=f'exp{i}', label=LABELS[i], kind='loss_recipe',
                     folds=full if i == 6 else {c: v[f'Exp{i}'] for c, v in values.items()}) for i in range(7)]
    audit = json.loads(controls.read_text(encoding='utf-8'))
    if not audit.get('passed') or audit.get('errors'):
        raise ValueError("Controls audit did not pass")
    for arm, label in (('direct', 'Direct: cross reconstruction from WSI slots'),
                       ('independent', 'Independent: product coupling')):
        folds = {}
        for cancer in values:
            runs = sorted([r for r in audit['runs'] if r['arm'] == arm and r['cancer'] == cancer.lower()], key=lambda r:r['fold'])
            if [r['fold'] for r in runs] != list(range(5)) or any(r['errors'] or r['seed'] != 3 or r['protocol'] != 'legacy_val' for r in runs):
                raise ValueError("Incomplete audited mechanism control")
            folds[cancer] = [r['cindex'] for r in runs]
        variants.append(dict(id=arm, label=label, kind='trained_mechanism_control', folds=folds))
    return dict(schema_version=1, cancers=list(CANCERS), variants=variants,
                context=dict(endpoint='DSS', encoder='UNI2-h', seed=3, protocol='legacy_val', selection='best validation'),
                source_commit=revision(root),
                sources=[dict(path=p.relative_to(root).as_posix(), sha256_lf=text_sha(p)) for p in (manuscript, comparison, controls)],
                note='Historical single-branch weights are 0.025; Full weights are 0.05 each. Loss recipes are not uniformly one-component removals. Full/control patient-config pairing still requires final matching audit.')


def ablation_table(payload, cancers, *, dispersion=False):
    if payload.get('schema_version') != 1 or len(set(cancers)) != len(cancers) or not set(cancers) <= set(payload['cancers']):
        raise ValueError("Invalid ablation schema/cohort selection")
    ids = [v['id'] for v in payload['variants']]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate variant")
    values, text = [], []
    for v in payload['variants']:
        scores, cells = [], []
        for c in cancers:
            raw = v['folds'].get(c)
            if raw is None:
                scores.append(None);cells.append('NR');continue
            a = np.asarray(raw, dtype=float)
            if a.shape != (5,) or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
                raise ValueError("Require five finite C-index folds in [0,1]")
            mean = float(a.mean());scores.append(mean)
            cells.append(f'{mean:.4f} ± {a.std(ddof=1):.4f}' if dispersion else f'{mean:.4f}')
        macro = None if None in scores else float(np.mean(scores))
        values.append(scores+[macro]);text.append([v['label']]+cells+['NR' if macro is None else f'{macro:.4f}'])
    styles = {}
    for col in range(len(cancers)+1):
        available = [(i, round(row[col],4)) for i,row in enumerate(values) if row[col] is not None]
        if len(available) < 2:
            continue
        unique = sorted({x[1] for x in available}, reverse=True)
        for row, score in available:
            if score == unique[0]:styles[(row,col)] = 'first'
            elif len(unique) > 1 and score == unique[1]:styles[(row,col)] = 'second'
    return dict(text=text, values=values, styles=styles, cancers=list(cancers),
                complete=all(None not in row for row in values), dispersion=dispersion)


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    return plt


def fresh_output(output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    return output


def render_ablation(payload, output):
    plt = plotting()
    tables = [('ablation_two_cohorts', ablation_table(payload,['BLCA','KIRC'],dispersion=True)),
              ('ablation_ten_cohort_coverage', ablation_table(payload,list(CANCERS)))]
    out=fresh_output(output);reports=[]
    for name,data in tables:
        compact=len(data['cancers'])==2
        fig,ax=plt.subplots(figsize=(12 if compact else 17,4.9));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
        headers=['Variants']+data['cancers']+['Mean (2)' if compact else 'Mean (10)']
        widths=[.49,.18,.18,.15] if compact else [.32]+[.68/11]*11
        table=ax.table(cellText=data['text'],colLabels=headers,cellLoc='center',colWidths=widths,bbox=[0,.25,1,.65])
        table.auto_set_font_size(False);table.set_fontsize(8.7 if compact else 8)
        for (row,col),cell in table.get_celld().items():
            cell.set_linewidth(0)
            if col==0:cell.get_text().set_ha('left')
            if row==0:cell.get_text().set_fontweight('bold')
            if row>0 and payload['variants'][row-1]['id']=='exp6':
                cell.set_facecolor('#fff7d6');cell.get_text().set_fontweight('bold')
        for (row,col),style in data['styles'].items():
            if style=='first':
                text=table[row+1,col+1].get_text();text.set_color('#c62828');text.set_fontweight('bold')
        ax.axhline(.90,color='black',lw=1);ax.axhline(.835,color='black',lw=.6);ax.axhline(.25,color='black',lw=1)
        fig.canvas.draw()
        for (row,col),style in data['styles'].items():
            if style=='second':
                box=table[row+1,col+1].get_text().get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
                ax.plot([box.x0,box.x1],[box.y0-.004]*2,color='black',lw=.7,transform=ax.transAxes)
        fig.suptitle('DCT v3.13: recorded training variants'+('' if compact else ' — ten-cohort coverage'),fontsize=13)
        fig.text(.09,.17,'Five-fold best-validation C-index; UNI2-h; seed 3. Red: first; underline: second among available variants.',fontsize=9)
        fig.text(.09,.12,'Exp4/Exp5 retain weight 0.025; Full uses 0.05 each. These historical recipes are not pure matched-weight removals.',fontsize=9)
        fig.text(.09,.07, 'NR: missing variant results. Ten-cohort mean is shown only for a complete row; single-entry columns are not ranked.' if not compact else 'Two-cohort mean is descriptive BLCA/KIRC macro-average, not the ten-cancer Overall. ±: fold sample SD.',fontsize=9)
        for ext in ('png','pdf','svg'):
            path=out/f'{name}.{ext}'
            fig.savefig(path,dpi=220,bbox_inches='tight',facecolor='white')
            if ext=='svg':normalize_text_output(path)
        plt.close(fig)
        reports.append(dict(name=name,complete=data['complete'],cancers=data['cancers'],
                            rank_styles=[dict(row=r,column=c,style=s) for (r,c),s in data['styles'].items()],
                            files={ext:sha256(out/f'{name}.{ext}') for ext in ('png','pdf','svg')}))
    save_json(out/'ablation_tables.json',dict(input=payload,tables=reports))
    normalize_text_output(out/'ablation_tables.json')
    return reports


def validate_tradeoff(payload, base):
    """Recompute y from raw predictions and reject unmatched cost measurements."""
    if payload.get('schema_version')!=1 or len(payload.get('models',[]))<2:
        raise ValueError('At least two measured methods are required')
    context=payload['context'];cancers=context['cancers']
    if context['endpoint']!='DSS' or context['encoder']!='UNI2-h' or context['protocol']!='legacy_val' or not cancers or len(set(cancers))!=len(cancers) or not set(cancers)<=set(CANCERS):
        raise ValueError('Unsupported or ambiguous measurement context')
    expected={(c,f) for c in cancers for f in range(5)}
    common=None;references={};points=[];model_ids=set()
    fields=('hardware','torch_version','cuda_version','cudnn_version','precision','batch_size','patches','feature_dim','repeats','warmup','forward_mode','cudnn_benchmark','cudnn_deterministic','matmul_allow_tf32')
    for model in payload['models']:
        if model['id'] in model_ids:raise ValueError('Duplicate measured method')
        model_ids.add(model['id']);seen=set();rows=[];seen_patients={};universes={}
        for measurement in model['measurements']:
            key=(measurement['cancer'],measurement['fold'])
            if key not in expected or key in seen:raise ValueError('Duplicate or unexpected measurement fold')
            seen.add(key)
            pred_path=Path(base)/measurement['predictions'];profile_path=Path(base)/measurement['profile_json']
            if sha256(pred_path)!=measurement['predictions_sha256'] or text_sha(profile_path)!=measurement['profile_sha256_lf']:
                raise ValueError('Measurement source hash mismatch')
            pred=predictions(pred_path)
            meta=json.loads(profile_path.read_text(encoding='utf-8'));run=meta['run'];p=meta['profile']
            if (run['cancer'].upper(),run['fold'])!=key or run['seed']!=context['seed'] or run['protocol']!=context['protocol']:
                raise ValueError('Profile identity disagrees with requested prediction context')
            if meta['hashes']['predictions']!=measurement['predictions_sha256'] or not meta['hashes'].get('checkpoint'):
                raise ValueError('Profile is not linked to this prediction/checkpoint')
            split_path=Path(run['split_csv'])
            if not split_path.is_absolute():split_path=profile_path.parent/split_path
            if sha256(split_path)!=meta['hashes']['split_csv']:raise ValueError('Split hash mismatch')
            split=split_ids(split_path)
            if set(pred['case_ids'])!=set(split['val']):raise ValueError('Missing validation patients')
            cancer=key[0];patient_set=set(pred['case_ids']);universe=set(split['train'])|set(split['val'])
            if seen_patients.setdefault(cancer,set()) & patient_set:raise ValueError('Validation patients overlap across measurement folds')
            seen_patients[cancer].update(patient_set)
            if cancer in universes and universes[cancer]!=universe:raise ValueError('Patient universe differs across measurement folds')
            universes[cancer]=universe
            cases=p['benchmark_case_ids']
            if len(cases)!=p['batch_size'] or len(set(cases))!=len(cases) or not set(cases)<=set(pred['case_ids']):
                raise ValueError('Benchmark case identity is not a valid validation batch')
            settings=tuple(p[k] for k in fields)
            if common is None:common=settings
            elif settings!=common:raise ValueError('Hardware/input/precision/timing settings differ')
            if not p['device'].startswith('cuda') or p['batch_size']!=1 or p['repeats']<2 or p['warmup']<1 or p['forward_mode']!='eval_no_grad':
                raise ValueError('Require synchronized CUDA eval batch-1 measurements with warmup/repeats')
            if len(p['wsi_input_sha256'])!=64:raise ValueError('Missing measured WSI input fingerprint')
            identity=(pred['case_ids'],pred['time'].tolist(),pred['censor'].tolist(),sorted(split['train']),cases,p['wsi_input_sha256'])
            if key in references and references[key]!=identity:raise ValueError('Patients/outcomes/train split/benchmark WSI inputs differ')
            references[key]=identity
            cost=np.asarray([p['peak_allocated_mb'],p['latency_ms_median']],float)
            if not np.isfinite(cost).all() or (cost<=0).any():raise ValueError('Invalid measured GPU costs')
            rows.append(dict(cancer=key[0],fold=key[1],cindex=pred['cindex'],memory_mib=float(cost[0]),latency_ms=float(cost[1])))
        if seen!=expected:raise ValueError('Every method requires the same complete five folds per cohort')
        if any(seen_patients[c]!=universes[c] for c in cancers):raise ValueError('Incomplete validation cohort across measurement folds')
        points.append(dict(id=model['id'],label=model['label'],ours=bool(model.get('ours',False)),
                           cindex=float(np.mean([r['cindex'] for r in rows])),memory_mib=float(np.mean([r['memory_mib'] for r in rows])),
                           latency_ms=float(np.mean([r['latency_ms'] for r in rows])),folds=rows))
    if sum(p['ours'] for p in points)>1:raise ValueError('Select one DCT point as ours')
    return dict(context=context,settings=dict(zip(fields,common)),points=points,
                note='Macro-average of five-fold C-index and per-fold fixed-case costs; validation selection, not independent test. MiB is total peak allocated memory including resident weights and input.')


def render_tradeoff(payload, base, output):
    report=validate_tradeoff(payload,base)
    plt=plotting();out=fresh_output(output)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    colors=plt.get_cmap('tab10')(np.arange(len(report['points']))%10)
    for ax,metric,label in zip(axes,('memory_mib','latency_ms'),('Peak GPU allocated memory (MiB) ↓','Median forward wall-clock time (ms) ↓')):
        for i,p in enumerate(report['points']):
            ax.scatter(p[metric],p['cindex'],s=180 if p['ours'] else 55,marker='*' if p['ours'] else 'o',color='#e69f00' if p['ours'] else colors[i],label=p['label'],zorder=3)
            ax.annotate(p['label'],(p[metric],p['cindex']),xytext=(5,5 if i%2==0 else -12),textcoords='offset points',fontsize=8)
        ax.set_xlabel(label);ax.set_ylabel('Matched five-fold C-index ↑');ax.grid(ls='--',alpha=.35);ax.margins(.15)
    axes[1].legend(fontsize=7,loc='best')
    fig.suptitle('DCT v3.13: performance / measured inference cost',fontweight='bold')
    fig.tight_layout(rect=(0,.10,1,.93))
    fig.text(.5,.025,'Same validation patients, UNI2-h WSI input, hardware, precision and timing settings; standard eval forward.',ha='center',fontsize=8)
    for ext in ('png','pdf','svg'):
        path=out/f'efficiency_tradeoff.{ext}'
        fig.savefig(path,dpi=220,bbox_inches='tight')
        if ext=='svg':normalize_text_output(path)
    plt.close(fig)
    report['source_input']=payload
    report['files']={ext:sha256(out/f'efficiency_tradeoff.{ext}') for ext in ('png','pdf','svg')}
    save_json(out/'efficiency_tradeoff.json',report)
    normalize_text_output(out/'efficiency_tradeoff.json')
    return report
