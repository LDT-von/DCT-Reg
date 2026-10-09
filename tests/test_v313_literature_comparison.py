import json
from pathlib import Path
import unittest
from scripts.build_v313_literature_comparison import summarize_literature

ROOT=Path(__file__).resolve().parents[1]
class LiteratureReferenceTests(unittest.TestCase):
    def setUp(self):
        self.p=json.loads((ROOT/'paper/V313_OT_UNI2H_COMPARISON_INPUT.json').read_text(encoding='utf-8'))
    def test_source_values_and_coverage(self):
        r=summarize_literature(self.p)
        self.assertEqual(len(r['rows']),22)
        self.assertEqual(r['baseline_counts']['LUSC'],15)
        self.assertIn('LUSC',r['ranked_columns'])
        self.assertFalse(r['complete'])  # Descriptive reference is never promoted to matched.
        self.assertTrue(r['ten_cancer_baseline_coverage_complete'])
        self.assertEqual(len(r['ranked_columns']),11)
        ours=next(x for x in r['rows'] if x['id']=='dct_v313')
        self.assertEqual(ours['cells']['LUSC']['rank'],1)
        self.assertAlmostEqual(ours['cells']['Overall']['mean'],.702956)
        steph=next(x for x in r['rows'] if x['id']=='steph')
        self.assertEqual(steph['cells']['COADREAD']['mean'],.7069)
        for c in ['KIRC','LUAD','STAD','Overall']:
            self.assertIsNone(steph['cells'][c])
    def test_original_baseline_scores_and_years_retained(self):
        old=json.loads((ROOT/'paper/V313_TEN_CANCER_REPORTED_COMPARISON_INPUT.json').read_text(encoding='utf-8'))
        byid={m['id']:m for m in self.p['models']}
        for m in old['models']:
            if m['id'].startswith('slotspe') or m['id']=='dct_v313':continue
            new=byid[m['id']]
            self.assertEqual(new['reported_means'],m['means'])
            self.assertEqual(new['reported_dispersion'],m['std'])
            self.assertEqual(new['reported_overall'],m['reported_overall'])
            self.assertIsInstance(new['year'],int)
            self.assertTrue(new['year_source'])
    def test_no_slotspe_model_rows(self):
        self.p['models'][0]['model']='SlotSPE'
        with self.assertRaises(ValueError):summarize_literature(self.p)
    def test_subset_overall_cannot_enter_ten_cohort_ranking(self):
        next(m for m in self.p['models'] if m['id']=='steph')['reported_overall']=.6949
        with self.assertRaises(ValueError):summarize_literature(self.p)
    def test_missing_year_rejected(self):
        self.p['models'][0].pop('year')
        with self.assertRaises(ValueError):summarize_literature(self.p)
    def test_os_cannot_enter_dss_table(self):
        self.p['models'][0]['endpoint']='OS'
        with self.assertRaises(ValueError):summarize_literature(self.p)
    def test_merged_cohort_cannot_be_relabelled(self):
        self.p['models'][-2]['reported_means']['KIRC']=.9
        with self.assertRaises(ValueError):summarize_literature(self.p)
    def test_local_source_hash_checked(self):
        self.p['sources']['dct_author']['sha256']='0'*64
        with self.assertRaises(ValueError):summarize_literature(self.p)
    def test_nonfinite_published_score_rejected(self):
        self.p['models'][0]['reported_means']['BRCA']=float('nan')
        with self.assertRaises(ValueError):summarize_literature(self.p)

if __name__=='__main__':unittest.main()
