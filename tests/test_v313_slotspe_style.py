"""Synthetic invariants, not real-data interpretability or performance evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from survot_rank.evidence.manifest import sha256
from survot_rank.evidence.slotspe_style import (
    collect_attention, group_pathways, load_case, top_patch_rows,
    transport_maps, within_fold_percentile,
)


class TestSlotSPEFigures(unittest.TestCase):
    def test_transport_unequal_slot_counts_and_explicit_reference(self):
        plans = np.array([[[[1., 2., 0.], [3., 1., 4.]]],
                          [[[4., 1., 2.], [1., 3., 1.]]]])
        aw = np.array([[.7, .2, .1], [.1, .3, .6]])
        ao = np.array([[.8, .2], [.5, .5], [.1, .9]])
        values = transport_maps(plans, np.array([1., 3.]), aw, ao)
        reference = sum(q * (p[0] / p[0].sum(0)) for q, p in zip([.25, .75], plans))
        np.testing.assert_allclose(values['omics_spatial'], reference.T @ aw)
        np.testing.assert_allclose(values['patch_pathway'], aw.T @ reference @ ao)
        self.assertEqual(values['omics_spatial'].shape, (3, 3))

    def test_independent_reference_preserves_factual_mass_and_marginals(self):
        p = np.array([[[[1., 2.], [3., 5.]]]])
        a = np.eye(2)
        values = transport_maps(p, [1.], a, a)
        np.testing.assert_allclose(values['transport'].sum(0), values['independent'].sum(0))
        np.testing.assert_allclose(values['transport'].sum(1), values['independent'].sum(1))
        self.assertGreater(np.abs(values['residual']).sum(), 0)

    def test_patch_ranking_excludes_padding_duplicates_and_other_slide(self):
        got = top_patch_rows([.9, .8, 1., .7, .6], [0, 0, -1, 1, 0], [3, 3, -1, 5, 4], slide=0)
        np.testing.assert_array_equal(got, [0, 4])

    def test_equal_risks_have_equal_midrank_not_fake_quartile_differences(self):
        np.testing.assert_allclose(within_fold_percentile([-2., -2., -2.]), [.5, .5, .5])
        np.testing.assert_allclose(within_fold_percentile([-3., -1., -2., -2.]), [.125, .875, .5, .5])

    def test_uniform_attention_has_zero_pathway_group_contrast(self):
        c = dict(groups=np.array([0, 1, 2, 3]), attention=np.full((4, 6), 1/6), names=np.array(list('abcdef')))
        table = group_pathways(c, top=2)
        np.testing.assert_allclose(table['delta'], 0)
        self.assertEqual(table['counts'], [1]*4)

    def test_case_identity_uses_manifest_instead_of_patient_ordinal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez(root/'patients.npz', case_ids=np.array(['A', 'B']),
                     risk=[-1., -1.5], time=[7., 20.], censor=[0, 1], pathway_names=['P', 'Q'])
            np.savez(root/'case_000.npz', risk=[-1.5], survival=[[.8, .7]],
                     attention_wsi=[[[.4, .6]]], attention_omic=[[[.2, .8]]])
            (root/'export.json').write_text(json.dumps(dict(schema_version=1,
                run=dict(id='fixture'), cases=[dict(case_id='B', file='case_000.npz', features=[])])))
            case = load_case(root, 'B')
            self.assertEqual(case['time'], 20.)
            self.assertEqual(case['censor'], 1)

    def fixture_exports(self, root, *, reverse_names=False):
        universe = [f'P{i}' for i in range(20)]
        for fold in range(5):
            out = root/f'f{fold}'; out.mkdir()
            ids = universe[fold*4:(fold+1)*4]
            train = [p for p in universe if p not in ids]
            split = out/'split.csv'
            split.write_text('train,val\n' + '\n'.join(f'{p},{ids[i] if i < 4 else ""}' for i,p in enumerate(train)))
            names = ['A', 'B'] if not reverse_names or fold % 2 == 0 else ['B', 'A']
            attention = np.tile(np.array([[[.8, .2]]]), (4, 1, 1))
            if names[0] == 'B':
                attention = attention[:, :, ::-1]
            np.savez(out/'patients.npz', case_ids=ids, risk=[-4., -3., -2., -1.],
                     pathway_names=names, attention_omic=attention)
            (out/'export.json').write_text(json.dumps(dict(schema_version=1,
                run=dict(id=f'f{fold}', arm='exp6', cancer='fixture', seed=3, fold=fold,
                         protocol='legacy_val', split_csv=str(split)), hashes=dict(split_csv=sha256(split)))))

    def test_complete_partition_aligns_pathway_names_across_fold_orders(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture_exports(root, reverse_names=True)
            c = collect_attention(root, cancer='fixture')
            self.assertEqual(len(c['ids']), 20)
            np.testing.assert_allclose(c['attention'], np.tile([.8, .2], (20, 1)))
            np.testing.assert_array_equal(np.bincount(c['groups']), [5]*4)

    def test_incomplete_cohort_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture_exports(root)
            (root/'f4'/'export.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Require all five folds'):
                collect_attention(root, cancer='fixture')
            c = collect_attention(root, cancer='fixture', allow_partial=True)
            self.assertTrue(c['partial'])
            self.assertEqual(c['folds'], [0, 1, 2, 3])

    def test_invalid_projection_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'OT slot axes'):
            transport_maps(np.ones((1, 1, 2, 3)), [1], np.eye(2), np.eye(2))


if __name__ == '__main__':
    unittest.main()
