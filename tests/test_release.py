import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation_utils import load_raw_targets
from ecg_qwen3_vl.output_space import LABEL_NAMES, exclude_sr

class ReleaseTests(unittest.TestCase):
    def test_preserve_original_decisions(self):
        probs=np.random.default_rng(42).random((100,14)).astype(np.float32)
        thresholds=np.asarray(json.loads((ROOT/'config/youden_thresholds.json').read_text())['thresholds'],dtype=np.float32)
        self.assertTrue(np.array_equal(exclude_sr(probs)>=thresholds,(probs>=np.r_[0.81,thresholds])[:,1:]))
        self.assertTrue(np.array_equal(exclude_sr(probs),probs[:,1:]))
        sr_only=np.zeros((1,14));sr_only[0,0]=1
        self.assertFalse((exclude_sr(sr_only)>=thresholds).any())

    def test_forward_preserves_fused_logits(self):
        import torch
        from torch import nn
        from ecg_qwen3_vl.model import ECGMultiTaskModel, ClearCLSGatedFusion
        torch.manual_seed(42)
        model=ECGMultiTaskModel.__new__(ECGMultiTaskModel)
        nn.Module.__init__(model)
        hidden=torch.randn(2,3,4)
        class Backbone(nn.Module):
            def forward(self,**kwargs): return SimpleNamespace(last_hidden_state=hidden)
        model.vlm=nn.Module();model.vlm.model=Backbone()
        model.classifier=nn.Linear(4,14)
        model.clear_fusion=ClearCLSGatedFusion(4,14)
        with torch.no_grad(): model.clear_fusion.residual_scale.fill_(0.7)
        model.lm_loss_weight=0.0
        clear=torch.rand(2,14)
        base=model.classifier(hidden[:,-1])
        original,gate=model.clear_fusion(base,hidden[:,-1],clear)
        out=model(attention_mask=torch.ones(2,3,dtype=torch.long),clear_probabilities=clear)
        self.assertEqual(out.classification_logits.shape,(2,13))
        self.assertTrue(torch.equal(out.classification_logits,original[:,1:]))
        self.assertTrue(torch.equal(out.logits,original[:,1:]))
        self.assertTrue(torch.equal(out.base_classification_logits,base[:,1:]))
        self.assertTrue(torch.equal(out.clear_fusion_gate,gate[:,1:]))

    def test_sr_ignored_in_targets(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'labels.csv'
            p.write_text('Filename,Diagnosis1\na.dat,"[\'sinus rhythm\']"\nb.dat,"[\'normal ECG\', \'sinus rhythm\']"\n')
            target=exclude_sr(load_raw_targets([{'id':'zs/a'},{'id':'zs/b'}],p))
            self.assertFalse(target[0].any())
            self.assertEqual(np.flatnonzero(target[1]).tolist(),[0])

    def test_evaluation_and_alignment(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'zs').mkdir()
            ids=['zs/a','zs/b']
            manifest=p/'manifest.jsonl'
            manifest.write_text(''.join(json.dumps({'id':i,'subset':'zs'})+'\n' for i in ids))
            labels=p/'labels.csv'
            labels.write_text('Filename,Diagnosis1\na.dat,"[\'normal ECG\']"\nb.dat,"[\'ST/T change\']"\n')
            targets=exclude_sr(load_raw_targets([{'id':i} for i in ids],labels))
            probs=np.where(targets,0.99,0.01)
            thresholds=np.asarray(json.loads((ROOT/'config/youden_thresholds.json').read_text())['thresholds'],dtype=np.float32)
            def artifact(order):
                np.savez(p/'zs/predictions.npz',sample_ids=order,labels=LABEL_NAMES,probabilities=probs,thresholds=thresholds,predictions=probs>=thresholds)
            cmd=[sys.executable,str(ROOT/'evaluate_zhongshan.py'),'--manifest',str(manifest),'--ground-truth',str(labels),'--prediction-root',str(p),'--output',str(p/'metrics.xlsx')]
            artifact(ids)
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            metrics=json.loads((p/'metrics.json').read_text())
            self.assertEqual(metrics['overall_micro'][0]['ROC-AUC'],1.0)
            self.assertEqual(metrics['overall_micro'][0]['Exact match'],1.0)
            self.assertEqual(metrics['overall_micro'][0]['num_classes'],13)
            self.assertEqual(len(metrics['per_class']),13)
            self.assertEqual(metrics['overall_micro'][0]['num_evaluable_classes'],2)
            artifact(ids[::-1])
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('manifest order',result.stderr)

    def test_unknown_label_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'labels.csv';p.write_text('Filename,Diagnosis1\nZS.dat,"[\'typo-label\']"\n')
            with self.assertRaisesRegex(ValueError,'Unknown'):
                load_raw_targets([{'id':'zs/ZS'}],p)

if __name__=='__main__': unittest.main()
