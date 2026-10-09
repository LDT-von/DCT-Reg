"""Small numeric regression tests; no models, features, or patient data."""
import importlib.util
from pathlib import Path
import unittest
import json, pickle, tempfile
import numpy as np
SPEC=importlib.util.spec_from_file_location('e010_auditor',Path(__file__).with_name('audit_saved_exports.py'))
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def sample(risks=None):
    factual=np.array([4.,2.,3.,1.]) if risks is None else np.asarray(risks,dtype=float)
    return {'case_ids':np.array(['a','b','c','d']),'time':np.arange(1.,5.),'censor':np.zeros(4),'risk':factual,'sweep_risk':np.repeat(factual[:,None],5,axis=1),'alphas':M.ALPHAS.copy()}


class EvidenceTests(unittest.TestCase):
    def test_common_offset_changes_risk_not_order(self):
        d=sample();d['sweep_risk'][:,1:]+=1.;report,_=M.analyze_arrays(d);r=report['risk_pair_response'][1]
        self.assertEqual(r['risk_max_absolute_change'],1.)
        self.assertEqual(r['comparable_pair_state_changes'],0)
        self.assertEqual(r['risk_max_change_after_removing_common_offset'],0.)

    def test_equal_cindex_can_hide_pair_changes(self):
        d=sample();d['sweep_risk'][:,1:]=np.array([4.,3.,1.,2.])[:,None];report,_=M.analyze_arrays(d);r=report['risk_pair_response'][1]
        self.assertEqual(r['delta_cindex_from_alpha0'],0.)
        self.assertEqual(r['comparable_pair_state_changes'],2)

    def test_event_censor_time_tie_is_comparable(self):
        states,counts=M.pair_states([1,1,2],[0,1,0],[3,2,1])
        self.assertEqual(counts['comparable_pairs'],2)
        self.assertEqual(counts['tied_time_event_censor'],1)
        self.assertEqual(counts['cindex'],1.)

    def test_risk_ties_count_half(self):
        _,counts=M.pair_states([1,2,3],[0,0,0],[2,2,1])
        self.assertEqual(counts['tied_risk'],1)
        self.assertAlmostEqual(counts['cindex'],5/6)

    def test_nonzero_plan_change_preserves_margins(self):
        p=np.array([[.45,.05],[.05,.45]]).reshape(1,1,1,2,2)
        report=M.plan_diagnostics(p)['alphas'][-1]
        self.assertAlmostEqual(report['mean_normalized_plan_l1'],.8)
        self.assertEqual(report['max_row_change_from_factual'],0.)
        self.assertEqual(report['max_col_change_from_factual'],0.)

    def test_target_residual_is_not_mixed_vs_factual(self):
        p=np.array([[.45,.05],[.05,.45]]).reshape(1,1,1,2,2)
        report=M.plan_diagnostics(p,rows=np.array([[[.6,.4]]]),cols=np.array([[[.5,.5]]]))['alphas'][-1]
        self.assertAlmostEqual(report['max_row_residual_to_solver_target'],.1)
        self.assertEqual(report['max_row_change_from_factual'],0.)

    def test_missing_layers_are_marked_not_invented(self):
        report,rows=M.analyze_arrays(sample())
        self.assertEqual(len(report['missing_runtime_layers']),4)
        self.assertIsNone(report['plan_change'])
        self.assertEqual(len(rows),20)

    def test_captured_layer_delta_and_shape(self):
        d=sample();d['sweep_logits']=np.zeros((4,5,2));d['sweep_logits'][:,1,0]=.1
        report,_=M.analyze_arrays(d)
        self.assertAlmostEqual(report['layer_response']['sweep_logits'][1]['max_absolute'],.1)
        d['sweep_logits']=np.zeros((4,4,2))
        with self.assertRaisesRegex(ValueError,'sweep_logits'):M.analyze_arrays(d)

    def test_duplicate_ids_and_nonfinite_are_rejected(self):
        d=sample();d['case_ids'][1]='a'
        with self.assertRaises(ValueError):M.analyze_arrays(d)
        d=sample();d['sweep_risk'][0,2]=np.nan
        with self.assertRaises(ValueError):M.analyze_arrays(d)

    def test_alpha_zero_mismatch_and_no_pairs_are_rejected(self):
        d=sample();d['sweep_risk'][:,0]+=1
        with self.assertRaises(ValueError):M.analyze_arrays(d)
        d=sample();d['censor'][:]=1
        with self.assertRaisesRegex(ValueError,'No comparable'):M.analyze_arrays(d)

    def test_invalid_plan_mass_and_target_shape_rejected(self):
        with self.assertRaises(ValueError):M.plan_diagnostics(np.zeros((1,1,1,2,2)))
        with self.assertRaises(ValueError):M.plan_diagnostics(np.ones((1,1,1,2,2)),rows=np.ones((1,2,2)),cols=np.ones((1,1,2)))

    def test_statistics_dont_mutate_original_arrays(self):
        d=sample();p=np.array([[.45,.05],[.05,.45]]).reshape(1,1,1,2,2)
        original=d['sweep_risk'].copy();p0=p.copy();M.analyze_arrays(d);M.plan_diagnostics(p)
        np.testing.assert_array_equal(original,d['sweep_risk']);np.testing.assert_array_equal(p0,p)


    def test_runtime_plan_capture_exposes_failed_replacement(self):
        d=sample();d['plans']=np.tile(np.array([[.45,.05],[.05,.45]]),(4,1,1,1,1))
        d['sweep_plans']=np.repeat(d['plans'][:,None],5,axis=1)
        report,_=M.analyze_arrays(d)
        self.assertFalse(report['captured_plan_change']['alphas'][-1]['matches_expected_formula_with_float_tolerance'])
        self.assertEqual(report['captured_plan_change']['alphas'][-1]['max_element_difference_from_factual'],0.)
        self.assertGreater(report['plan_change']['alphas'][-1]['max_element_difference'],0.)

    def fixture_export(self, root):
        root=Path(root);directory=root/'export';directory.mkdir()
        d=sample();config=root/'config.yaml';config.write_text(json.dumps({'survot_method':'dct_v313_transport_reconstruction','rna_format':'Pathways'}),encoding='utf-8')
        checkpoint=root/'checkpoint.pth';checkpoint.write_bytes(b'SYNTHETIC PLACEHOLDER NEVER LOADED')
        predictions=root/'predictions.pkl'
        with predictions.open('wb') as stream:
            pickle.dump({str(cid):{key:float(d[key][i]) for key in ['risk','time','censor']} for i,cid in enumerate(d['case_ids'])},stream)
        score=M.cindex(d['time'],d['censor'],d['risk'])
        curve=root/'curve.csv';curve.write_text(f'epoch,val_cindex\n1,{score}\n',encoding='utf-8')
        split=root/'split.csv';split.write_text('train,val\nx,a\ny,b\nz,c\nw,d\n',encoding='utf-8')
        run={'id':'synthetic_blca_f0','arm':'exp6','cancer':'blca','fold':0,'seed':3,'protocol':'legacy_val','source_commit':'synthetic-not-a-model-run',
             'config':str(config),'checkpoint':str(checkpoint),'predictions':str(predictions),'curve':str(curve),'split_csv':str(split)}
        verified=M.audit({'runs':[run]});self.assertTrue(verified['passed'],verified['errors'])
        meta={'schema_version':1,'export_commit':'synthetic-export','run':run,'hashes':verified['runs'][0]['hashes'],'best_epoch':1,
              'sweep':[{'alpha':float(a),'cindex':score} for a in M.ALPHAS]}
        (directory/'export.json').write_text(json.dumps(meta),encoding='utf-8');np.savez(directory/'patients.npz',**d)
        spec=root/'input.json';spec.write_text(json.dumps({'schema_version':1,'exports':['export']}),encoding='utf-8')
        return directory,spec,d

    def test_source_hash_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            directory,_,_=self.fixture_export(root);M.read_export(directory)
            (Path(root)/'checkpoint.pth').write_bytes(b'CHANGED SYNTHETIC PLACEHOLDER')
            with self.assertRaisesRegex(ValueError,'Source hash changed'):M.read_export(directory)

    def test_export_patient_outcomes_must_match_source(self):
        with tempfile.TemporaryDirectory() as root:
            directory,_,d=self.fixture_export(root);d['time']+=10;np.savez(directory/'patients.npz',**d)
            with self.assertRaisesRegex(ValueError,'Export time differs'):M.read_export(directory)

    def test_non_v313_config_is_rejected_even_if_metadata_says_full(self):
        with tempfile.TemporaryDirectory() as root:
            directory,_,_=self.fixture_export(root)
            (Path(root)/'config.yaml').write_text(json.dumps({'survot_method':'other','rna_format':'Pathways'}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Effective config'):M.read_export(directory)

    def test_partial_coverage_is_explicit_and_outputs_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            _,spec,_=self.fixture_export(root);output=Path(root)/'new_results'
            with self.assertRaisesRegex(ValueError,'Missing folds'):M.main(['--input',str(spec),'--output',str(output),'--require-ten-folds'])
            self.assertFalse(output.exists());M.main(['--input',str(spec),'--output',str(output)])
            report=json.loads((output/'audit.json').read_text(encoding='utf-8'))
            self.assertFalse(report['complete_ten_folds']);self.assertEqual(len(report['missing_folds']),9)
            original=(output/'audit.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'Refuse to overwrite'):M.main(['--input',str(spec),'--output',str(output)])
            self.assertEqual(original,(output/'audit.json').read_bytes())


if __name__=='__main__':unittest.main()
