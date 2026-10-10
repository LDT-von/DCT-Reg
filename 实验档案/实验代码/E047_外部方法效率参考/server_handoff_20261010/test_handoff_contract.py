"""Synthetic JSON integrity tests only. No torch, GPU or model execution."""
import unittest,tempfile,json,shutil,hashlib,copy,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
module_spec=importlib.util.spec_from_file_location('plot_contract',HERE/'plot_measured_tradeoff.py')
module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Contract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.profiles=self.root/'profiles';self.profiles.mkdir()
        self.spec=json.loads((HERE/'benchmark_spec.example.json').read_text(encoding='utf-8'))
        shutil.copy2(HERE/'blca_scores_frozen.csv',self.root/'blca_scores_frozen.csv')
        for c in self.spec['cases']:c.update(case_id='SYNTHETIC_QA_'+str(c['fold']),common_wsi_sha256=str(c['fold'])*64,common_omics_sha256='a'*64)
        for m in self.spec['methods']:
            if not m['source_commit']:m['source_commit']='b'*40
        self.path=self.root/'spec.json';self.path.write_text(json.dumps(self.spec),encoding='utf-8')
        for m in self.spec['methods']:
            for rep in range(3):
                cases=[]
                for c in self.spec['cases']:
                    cases.append({**c,'input_audit':{'same_wsi_values':True,'same_raw_omics':True},'latency_samples_ms':[10.]*30,'latency_ms_median':10.,'peak_allocated_samples_mib':[100.]*30,'peak_allocated_mib':100.,'baseline_allocated_mib':80.})
                r={'method_id':m['id'],'replicate':rep,'spec_sha256':sha(self.path),'adapter_sha256':'c'*64,'hardware':'SYNTHETIC_QA_ONLY','device_uuid':'synthetic','torch_version':'qa','cuda_version':'qa','cudnn_version':'qa','precision':'float32','matmul_allow_tf32':False,'cudnn_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True,'forward_mode':'eval_no_grad_native_full','metadata':{'source_commit':m['source_commit'],'native_full_prediction_forward':True,'weights_kind':'random_init'},'cases':cases}
                (self.profiles/(m['id']+f'_r{rep}.json')).write_text(json.dumps(r),encoding='utf-8')
    def tearDown(self):self.tmp.cleanup()
    def read(self):
        p=self.profiles/'mcat_r0.json';return p,json.loads(p.read_text(encoding='utf-8'))
    def reject(self,p,r):
        p.write_text(json.dumps(r),encoding='utf-8')
        with self.assertRaises(ValueError):module.collect(self.path,self.profiles)
    def test_complete_schema_only(self):
        s,points,sources,env=module.collect(self.path,self.profiles);self.assertEqual(len(points),9);self.assertEqual(len(sources),27)
    def test_missing_measurement(self):
        (self.profiles/'mcat_r0.json').unlink()
        with self.assertRaises(ValueError):module.collect(self.path,self.profiles)
    def test_hardware_mismatch(self):p,r=self.read();r['hardware']='OTHER';self.reject(p,r)
    def test_patient_mismatch(self):p,r=self.read();r['cases'][0]['case_id']='OTHER';self.reject(p,r)
    def test_summary_timing_fabrication(self):p,r=self.read();r['cases'][0]['latency_ms_median']=9.;self.reject(p,r)
    def test_summary_memory_fabrication(self):p,r=self.read();r['cases'][0]['peak_allocated_mib']=99.;self.reject(p,r)
    def test_unapproved_score_change(self):
        self.spec['methods'][0]['reported_blca_mean']=.99;self.path.write_text(json.dumps(self.spec),encoding='utf-8')
        with self.assertRaises(ValueError):module.collect(self.path,self.profiles)
    def test_duplicate_process(self):p,r=self.read();r['replicate']=1;self.reject(p,r)
if __name__=='__main__':unittest.main()
