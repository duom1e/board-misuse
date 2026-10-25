import json
import random
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from annotation_io import prepare_image, export_annotation
from prepare_manual_dataset import pool_audit, build_dataset

class ManualDatasetTests(unittest.TestCase):
    def fixture(self,root):
        rng=random.Random(24)
        for i in range(3):
            image=Image.frombytes('RGB',(64,48),rng.randbytes(64*48*3))
            source=root/f'source{i}.png';image.save(source)
            folder,_,meta=prepare_image(source,root/'working')
            export_annotation(folder/'image.png',meta,[[.1,.1,.8,.8]],f'video{i}','all_boards_boxed',root/'pool')
        audit=pool_audit(root/'pool')
        assignment={'pool_fingerprint':audit['fingerprint'],'groups':{'video0':'train','video1':'val','video2':'test'},'reviewer':'software_test','review_notes':'Synthetic fixtures only'}
        path=root/'assignment.json';path.write_text(json.dumps(assignment),encoding='utf-8')
        return audit,path,assignment
    def test_build_and_immutable_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audit,path,_=self.fixture(root)
            self.assertEqual(audit['errors'],[])
            result=build_dataset(root/'pool',path,root/'built')
            self.assertTrue(result['passed'])
            self.assertTrue((root/'built/READY').is_file())
            self.assertFalse(json.loads((root/'built/experiment_draft.json').read_text())['allow_training'])
            original=Path(audit['records'][0]['label']);original.write_text('')
            self.assertTrue(all(Path(r['label']).read_text().strip() for r in result['records']))
            with self.assertRaises(ValueError):build_dataset(root/'pool',path,root/'stale')
    def test_missing_assignments_and_positive_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,path,assignment=self.fixture(root)
            assignment['groups']['video2']=None;path.write_text(json.dumps(assignment))
            with self.assertRaises(ValueError):build_dataset(root/'pool',path,root/'built')
            assignment['groups']['video2']='exclude';path.write_text(json.dumps(assignment))
            with self.assertRaises(ValueError):build_dataset(root/'pool',path,root/'built2')
    def test_duplicate_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audit,_,_=self.fixture(root)
            record=audit['records'][0];meta=json.loads((Path(record['folder'])/'metadata.json').read_text())
            export_annotation(record['image'],meta,[[.1,.1,.8,.8]],'duplicate_group','all_boards_boxed',root/'pool')
            (root/'pool/incomplete').mkdir()
            report=pool_audit(root/'pool')
            self.assertEqual(len(report['exact_duplicates']),1)
            self.assertEqual(len(report['incomplete_ignored']),1)
            assignments={g:'train' for g in report['groups']};assignments['video1']='val';assignments['video2']='test'
            path=root/'new.json';path.write_text(json.dumps({'pool_fingerprint':report['fingerprint'],'groups':assignments,'reviewer':'test','review_notes':'test'}))
            with self.assertRaises(ValueError):build_dataset(root/'pool',path,root/'built')
    def test_label_tampering_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(pool_audit(root/'missing')['records'],[])
            audit,_,_=self.fixture(root)
            Path(audit['records'][0]['label']).write_text('0 .5 .5 .1 .1\n')
            report=pool_audit(root/'pool');self.assertEqual(len(report['errors']),1)

if __name__=='__main__':unittest.main(verbosity=2)
