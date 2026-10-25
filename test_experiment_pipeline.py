"""Check actual failure boundaries and one-to-one metric matching; no model training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import yaml

from experiment_data import audit_dataset,freeze_dataset,verify_snapshot
from run_experiment import fixed_metrics,choose_validation_threshold,load_plan

class ListTensor:
    def __init__(self,values):self.values=values
    def cpu(self):return self
    def tolist(self):return self.values

class FakeModel:
    def predict(self,*args,**kwargs):
        class Boxes:
            xyxy=ListTensor([[25,25,75,75],[25,25,75,75]])
            cls=ListTensor([0.,0.]);conf=ListTensor([.9,.8])
        class Result:
            orig_shape=(100,100);boxes=Boxes()
        return [Result()]

class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.images={}
        for i,split in enumerate(['train','val','test']):
            (self.root/'images'/split).mkdir(parents=True)
            (self.root/'labels'/split).mkdir(parents=True)
            image=self.root/'images'/split/'sample.png'
            Image.fromarray(np.random.default_rng(180+i).integers(0,256,(32,32,3),dtype=np.uint8)).save(image)
            (self.root/'labels'/split/'sample.txt').write_text('0 0.5 0.5 0.5 0.5\n')
            self.images[split]=image
        self.data=self.root/'data.yaml'
        self.data.write_text(yaml.safe_dump({'path':str(self.root),'train':'images/train','val':'images/val','test':'images/test','names':['target']}))

    def test_clean_and_missing_label(self):
        self.assertTrue(audit_dataset(self.data)['passed'])
        (self.root/'labels'/'val'/'sample.txt').unlink()
        result=audit_dataset(self.data)
        self.assertFalse(result['passed'])
        self.assertTrue(any('Missing label' in r['error'] for r in result['errors']))

    def test_exact_cross_split_duplicate_blocks(self):
        self.images['test'].write_bytes(self.images['train'].read_bytes())
        result=audit_dataset(self.data)
        self.assertFalse(result['passed']);self.assertEqual(len(result['cross_split_exact_duplicates']),1)

    def test_scene_group_split_blocks(self):
        groups=self.root/'groups.json'
        groups.write_text(json.dumps({str(p):('shared_video' if split in ['train','test'] else 'val_video') for split,p in self.images.items()}))
        result=audit_dataset(self.data,groups_file=groups)
        self.assertFalse(result['passed']);self.assertEqual(result['cross_split_groups'][0]['group'],'shared_video')

    def test_near_duplicate_needs_specific_review(self):
        with Image.open(self.images['train']) as image:
            pixels=np.array(image)
        pixels[0,0,0]^=1
        Image.fromarray(pixels).save(self.images['test'],compress_level=0)
        self.assertNotEqual(self.images['train'].read_bytes(),self.images['test'].read_bytes())
        result=audit_dataset(self.data)
        self.assertFalse(result['passed']);self.assertEqual(len(result['cross_split_near_duplicates']),1)
        pair=result['cross_split_near_duplicates'][0]
        review=self.root/'review.json'
        review.write_text(json.dumps([{'sha256_a':pair['sha256_a'],'sha256_b':pair['sha256_b'],'verdict':'different_images',
            'reviewer':'software_test','reason':'Synthetic changed-pixel pair only exercises the explicit review mechanism; not a real data approval.'}]))
        self.assertTrue(audit_dataset(self.data,review_file=review)['passed'])

    def test_same_pixels_different_encoding_are_hard_duplicates(self):
        with Image.open(self.images['train']) as image:image.save(self.images['test'],compress_level=0)
        result=audit_dataset(self.data)
        self.assertFalse(result['passed'])
        self.assertEqual(result['cross_split_exact_duplicates'][0]['type'],'identical_decoded_pixels_different_file_bytes')

    def test_validation_precision_requirement_controls_threshold_choice(self):
        sweep=[{'confidence':.3,'micro':{'precision':.6,'recall':.9,'f1':.72}},
            {'confidence':.5,'micro':{'precision':.8,'recall':.4,'f1':.53}},
            {'confidence':.7,'micro':{'precision':.9,'recall':.2,'f1':.32}}]
        self.assertEqual(choose_validation_threshold(sweep,.7),.5)
        self.assertIsNone(choose_validation_threshold(sweep,.95))

    def test_misspelled_evaluation_options_are_not_silently_ignored(self):
        weights=self.root/'model.pt';weights.write_bytes(b'schema fixture, not model weights')
        plan={'experiment_name':'schema_check','data':str(self.data),'initial_weights':str(weights),
            'data_status':'synthetic_software_fixture','allow_training':False,'training':{},
            'evaluation':{'calibraton':{}},
            'acceptance':{'min_map50':.5,'min_map50_95':.3,'min_precision':.7,'min_recall':.4}}
        file=self.root/'plan.json';file.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'Unsupported evaluation'):load_plan(file)

    def test_snapshot_detects_label_edit(self):
        audit=audit_dataset(self.data);freeze_dataset(audit,self.root/'frozen')
        (self.root/'labels'/'train'/'sample.txt').write_text('0 0.5 0.5 0.4 0.4\n')
        with self.assertRaises(RuntimeError):verify_snapshot(audit)

    def test_invalid_box_blocks(self):
        (self.root/'labels'/'val'/'sample.txt').write_text('0 0.1 0.1 0.9 0.9\n')
        self.assertFalse(audit_dataset(self.data)['passed'])

    def test_duplicate_detections_do_not_double_count_true_positive(self):
        label=self.root/'labels'/'test'/'sample.txt'
        result=fixed_metrics(FakeModel(),[{'image':str(self.images['test']),'label':str(label)}],['target'],.5,640,0,.5)
        self.assertEqual(result['micro']['tp'],1)
        self.assertEqual(result['micro']['fp'],1)
        self.assertEqual(result['micro']['fn'],0)

if __name__=='__main__':unittest.main()
