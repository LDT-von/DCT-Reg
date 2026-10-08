"""Render source-reported OT/UNI2-h DSS results; never run a model.

Incomplete literature coverage is expected and is not a matched benchmark.
Only exact cohort names are aligned. No mixed-cohort Overall is calculated.
"""
from __future__ import annotations
import argparse
import json
import math
from decimal import Decimal
from pathlib import Path
from statistics import stdev
if __package__:
    from .build_v313_comparison_table import CANCERS, ROOT, dense_ranks, quantize, sha256
    from .build_v313_comparison_table import render_vector, render_docx, render_html
else:
    from build_v313_comparison_table import CANCERS, ROOT, dense_ranks, quantize, sha256
    from build_v313_comparison_table import render_vector, render_docx, render_html


def summarize_literature(payload, *, base=ROOT):
    if payload.get('comparison_type') != 'literature_reference' or payload.get('endpoint') != 'DSS':
        raise ValueError('Require a DSS literature_reference input; OS cannot be mixed')
    if tuple(payload.get('cancers', [])) != CANCERS:
        raise ValueError('Require the ten declared cohort columns')
    digits = payload.get('digits', 4)
    if digits not in (3, 4):
        raise ValueError('Display precision must be 3 or 4')
    sources = payload.get('sources', {})
    ids = [m['id'] for m in payload['models']]
    if len(set(ids)) != len(ids) or ids.count('dct_v313') != 1:
        raise ValueError('Require unique IDs and one DCT row')
    rows = []
    for model in payload['models']:
        if model['endpoint'] != 'DSS' or model.get('transcription_verified') is not True:
            raise ValueError(f"Unverified or different endpoint: {model['model']}")
        source = sources[model['source_id']]
        digest = source.get('sha256', '')
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError('Each source requires a saved SHA256')
        if source.get('path') and sha256(base/source['path']) != digest:
            raise ValueError('Local source hash mismatch')
        if not source.get('url') and not source.get('path'):
            raise ValueError('Each source requires a URL or saved source path')
        values = model.get('reported_means', {})
        folds = model.get('folds', {})
        if set(values) - set(model['original_cohorts']) or set(folds) - set(model['original_cohorts']):
            raise ValueError('Scores cannot be assigned to an unreported cohort')
        cells = {}
        for cancer in CANCERS:
            cell = None
            if cancer in model['original_cohorts']:
                if cancer in folds:
                    fs = folds[cancer]
                    if len(fs) != 5 or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in fs):
                        raise ValueError('Five valid real fold values are required')
                    mean = float(sum(Decimal(str(v)) for v in fs)/5)
                    cell = {'mean':mean,'folds':fs,'std':stdev(fs)}
                elif cancer in values:
                    v = values[cancer]
                    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0 <= v <= 1:
                        raise ValueError('Invalid published C-index')
                    cell = {'mean':v,'reported_dispersion':model.get('reported_dispersion',{}).get(cancer)}
            if cell:
                cell.update(text=str(quantize(cell['mean'],digits)),rank=None)
            cells[cancer] = cell
        suffix = 'U2' if model['encoder'] == 'UNI2-h' else 'U1'
        rows.append({'id':model['id'],'model':model['model'],'group':model['group'],
                     'modality':model['modality']+'/'+suffix,'encoder':model['encoder'],
                     'endpoint':model['endpoint'],'source_id':model['source_id'],
                     'source_locator':model['source_locator'],'cells':cells})
    ranked, counts = [], {}
    for cancer in CANCERS:
        available = [row for row in rows if row['cells'][cancer] is not None]
        counts[cancer] = sum(row['id'] != 'dct_v313' for row in available)
        if len(available) < 2:
            continue  # A model with no comparator must not be painted as a winner.
        for row, rank in zip(available,dense_ranks([r['cells'][cancer]['mean'] for r in available],digits)):
            row['cells'][cancer]['rank'] = rank
        ranked.append(cancer)
    ours = next(r for r in rows if r['id']=='dct_v313')
    return {'comparison_type':'literature_reference','title':'DCT v3.13 OT and UNI2-h Literature Reference',
            'missing_label':'NR','input_header':'Input / FM','columns':list(CANCERS),'digits':digits,'rows':rows,
            'sources':sources,'matching_status':'not_matched','complete':False,
            'published_transcription_complete':True,'ten_cancer_baseline_coverage_complete':all(counts.values()),
            'baseline_counts':counts,'ranked_columns':ranked,
            'ranking_scope':'available DSS report values only; cross-protocol, descriptive ranks',
            'dct_first':[c for c in ranked if ours['cells'][c]['rank']==1],
            'dct_second':[c for c in ranked if ours['cells'][c]['rank']==2],
            'notes':[
              'DSS report values only. Inputs, patient splits and checkpoint selection differ; this is not a matched benchmark.',
              'U1 = UNI; U2 = UNI2-h; g. = genomics; h. = histology. MOTCat [TTA] is a reproduction in TTA Table 1.',
              'NR = no separate source cohort. CRC, KIPAN, LUNG and STES are not renamed to DCT cohorts.',
              'Red bold = highest available report value; underline = second. LUSC has no comparator and is not ranked.',
              'DCT: author five-fold best-validation summary. No Overall across different cohort sets. Full source means/SD: JSON.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'paper/V313_OT_UNI2H_COMPARISON_INPUT.json')
    p.add_argument('--output',type=Path,default=ROOT/'paper/tables/v313_ot_uni2h_comparison')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();payload=json.loads(a.input.read_text(encoding='utf-8'))
    report=summarize_literature(payload)
    if not a.check_only:
        a.output.mkdir(parents=True,exist_ok=True);stem=a.output/'table_v313_ot_uni2h_reference'
        render_vector(report,stem);render_docx(report,stem);render_html(report,stem)
        report['input_sha256']=sha256(a.input)
        report['outputs']={s:sha256(stem.with_suffix('.'+s)) for s in ('pdf','svg','png','docx','html')}
        stem.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'rows':len(report['rows']),'published_cells':sum(r['cells'][c] is not None for r in report['rows'] if r['id']!='dct_v313' for c in CANCERS),'baseline_counts':report['baseline_counts'],'ranking_scope':report['ranking_scope']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
