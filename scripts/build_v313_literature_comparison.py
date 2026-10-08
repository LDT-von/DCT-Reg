"""Render source-reported OT/UNI2-h DSS results; never run a model.

Incomplete literature coverage is expected and is not a matched benchmark.
Only exact cohort names are aligned. Overall is restricted to complete ten-cohort rows.
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
    if any('slotspe' in (m['id']+' '+m['model']).lower() for m in payload['models']):
        raise ValueError('SlotSPE comparison rows are excluded by author request')
    columns = (*CANCERS, 'Overall')
    rows = []
    for model in payload['models']:
        if model['endpoint'] != 'DSS' or model.get('transcription_verified') is not True:
            raise ValueError(f"Unverified or different endpoint: {model['model']}")
        year = model.get('year')
        if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= int(payload['access_date'][:4]) or not model.get('year_verified') or not model.get('year_source'):
            raise ValueError('Every row requires a verified year and source')
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
        full_ten = len(model['original_cohorts']) == 10 and set(model['original_cohorts']) == set(CANCERS) and all(cells[c] is not None for c in CANCERS)
        overall = model.get('reported_overall')
        if overall is not None and (not full_ten or isinstance(overall, bool) or not isinstance(overall, (int,float)) or not math.isfinite(overall) or not 0 <= overall <= 1):
            raise ValueError('Overall requires exactly the same ten cohorts and a valid source value')
        if model['id'] == 'dct_v313' and full_ten:
            overall = float(sum(Decimal(str(v)) for c in CANCERS for v in folds[c])/50)
        cells['Overall'] = {'mean':overall,'text':str(quantize(overall,digits)),'rank':None} if overall is not None else None
        suffix = 'U2' if model['encoder'] == 'UNI2-h' else 'U1' if model['encoder'] == 'UNI' else '-'
        marker = '*' if model['year_kind'] == 'reference_year' else '**' if model['year_kind'] == 'current_work_year' else ''
        rows.append({'id':model['id'],'model':f"{model['model']} ({year}{marker})",'base_model':model['model'],'year':year,'year_kind':model['year_kind'],'year_source':model['year_source'],'group':model['group'],
                     'modality':model['modality']+'/'+suffix,'encoder':model['encoder'],
                     'endpoint':model['endpoint'],'source_id':model['source_id'],
                     'source_locator':model['source_locator'],'cells':cells})
    ranked, counts = [], {}
    for cancer in columns:
        available = [row for row in rows if row['cells'][cancer] is not None]
        counts[cancer] = sum(row['id'] != 'dct_v313' for row in available)
        if len(available) < 2:
            continue  # A model with no comparator must not be painted as a winner.
        for row, rank in zip(available,dense_ranks([r['cells'][cancer]['mean'] for r in available],digits)):
            row['cells'][cancer]['rank'] = rank
        ranked.append(cancer)
    ours = next(r for r in rows if r['id']=='dct_v313')
    return {'comparison_type':'literature_reference','title':'DCT v3.13 Ten Cancer Literature Comparison',
            'compact':True,'model_column_width':208,'model_column_cm':4.1,'model_header':'Model (year)','header_font_sizes':{'COADREAD':8.5},'missing_label':'NR','input_header':'Input / FM','columns':list(columns),'digits':digits,'rows':rows,
            'sources':sources,'matching_status':'not_matched','complete':False,
            'published_transcription_complete':True,'ten_cancer_baseline_coverage_complete':all(counts[c] for c in CANCERS),
            'baseline_counts':counts,'ranked_columns':ranked,
            'ranking_scope':'available DSS report values only; cross-protocol, descriptive ranks',
            'dct_first':[c for c in ranked if ours['cells'][c]['rank']==1],
            'dct_second':[c for c in ranked if ours['cells'][c]['rank']==2],
            'notes':[
              'Public DSS report values; inputs, splits and checkpoint selection differ. Ranks are descriptive, not matched results.',
              '15 baseline rows: SlotSPE Table 1; all SlotSPE model rows excluded. Extra rows: TTA, OTSurv, STEPH S2.',
              'U1 = UNI; U2 = UNI2-h; - = no histology input; g. = genomics; h. = histology. NR = no separate source cohort.',
              'Red bold = highest available value; underline = second. Overall only for rows reporting the same ten cohorts.',
              'Year: original publication; TTA: first preprint 2025 (scores from 2026 v2). *MLP: cited textbook; SNNTrans: component year.',
              '**DCT 2026 = current work year, not publication. DCT is a five-fold best-validation summary; source folds/SD in JSON.']}



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
