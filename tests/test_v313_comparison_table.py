"""Functional rank/provenance checks; fixtures do not measure any real model."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_v313_comparison_table import CANCERS, dense_ranks, main, sha256, summarize


class ComparisonTableChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.base=Path(self.tmp.name)
        protocol={"audit":"verified","split_sha256":{c:["a"*64]*5 for c in CANCERS}}
        self.payload={"schema_version":1,"cancers":list(CANCERS),"protocol":protocol,"models":[]}
        for ident,value in [("baseline_a",.8),("baseline_b",.75),("dct_v313",.7)]:
            sources={}
            for cancer in CANCERS:
                sources[cancer]=[]
                for fold in range(5):
                    record=self.base/f"{ident}_{cancer}_{fold}.txt"
                    record.write_text(f"Fixture value {value}",encoding="utf-8")
                    sources[cancer].append({"fold":fold,"path":str(record),
                                           "sha256":sha256(record),"audit":"verified"})
            self.payload["models"].append({"id":ident,"model":ident,"modality":"g.+h.",
                 "group":"multimodal","audit":"verified","protocol":copy.deepcopy(protocol),
                 "folds":{c:[value]*5 for c in CANCERS},"sources":sources})

    def tearDown(self):
        self.tmp.cleanup()

    def test_display_ties_share_dense_rank(self):
        self.assertEqual(dense_ranks([.72004,.72002,.71,None],4),[1,1,2,None])

    def test_dct_is_not_forced_to_be_first_or_second(self):
        result=summarize(self.payload)
        self.assertTrue(result["complete"])
        self.assertEqual(result["dct_first"],[])
        self.assertEqual(result["dct_second"],[])
        self.assertTrue(all(c["rank"]==3 for c in result["rows"][-1]["cells"].values()))

    def test_overall_uses_unrounded_cohort_means(self):
        model=self.payload["models"][-1]
        for i,cancer in enumerate(CANCERS):
            model["folds"][cancer]=[.700044 if i<5 else .700054]*5
        result=summarize(self.payload)
        self.assertAlmostEqual(result["rows"][-1]["cells"]["Overall"]["mean"],.700049,places=12)
        self.assertEqual(result["rows"][-1]["cells"]["Overall"]["text"],"0.7000")

    def test_missing_cohort_is_unranked_and_default_writes_nothing(self):
        self.payload["models"][0]["folds"]["BRCA"]=None
        result=summarize(self.payload)
        self.assertNotIn("BRCA",result["ranked_columns"])
        self.assertNotIn("Overall",result["ranked_columns"])
        self.assertIsNone(result["rows"][-1]["cells"]["BRCA"]["rank"])
        source=self.base/"input.json";source.write_text(json.dumps(self.payload),encoding="utf-8")
        output=self.base/"no_partial_output"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--input",str(source),"--output",str(output)])
        self.assertFalse(output.exists())

    def test_protocol_difference_prevents_ranking(self):
        self.payload["models"][0]["protocol"]["seed"]=999
        result=summarize(self.payload)
        self.assertFalse(result["complete"])
        self.assertEqual(result["ranked_columns"],[])

    def test_changed_source_hash_invalidates_cohort_ranking(self):
        item=self.payload["models"][0]["sources"]["KIRC"][0]
        Path(item["path"]).write_text("changed",encoding="utf-8")
        result=summarize(self.payload)
        self.assertNotIn("KIRC",result["ranked_columns"])
        self.assertNotIn("Overall",result["ranked_columns"])
        self.assertIn("BLCA",result["ranked_columns"])


if __name__=="__main__":
    unittest.main()
