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
        self.assertEqual(r['baseline_counts']['LUSC'],0)
        self.assertNotIn('LUSC',r['ranked_columns'])
        self.assertFalse(r['complete'])
        self.assertNotIn('Overall',r['columns'])
        ours=next(x for x in r['rows'] if x['id']=='dct_v313')
        self.assertIsNone(ours['cells']['LUSC']['rank'])
        steph=next(x for x in r['rows'] if x['id']=='steph')
        self.assertEqual(steph['cells']['COADREAD']['mean'],.7069)
        self.assertIsNone(steph['cells']['KIRC'])
        self.assertIsNone(steph['cells']['LUAD'])
        self.assertIsNone(steph['cells']['STAD'])
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
