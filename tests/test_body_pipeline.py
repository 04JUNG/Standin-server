"""Pipeline isolation and catalog-driven body IDs/counts; no real provider calls."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import CFG
from src.pipeline import Pipeline
from src.pose import MockPoseModel
from src.schema import dumps
from src.vlm.client import MockVLMClient
from src.experimental.body_matching.service import BodyMatchingService
from tests.test_body_matching import BodyMatchingFixture, FixtureClient, cut


class BodyPipelineTests(BodyMatchingFixture):
    def pipeline(self, service):
        return Pipeline([], vlm_client=MockVLMClient(), pose_model=MockPoseModel(), body_matcher=service)

    def test_off_skips_body_and_preserves_original_result(self):
        service=Mock();original=cut();before=dumps(original)
        pipe=self.pipeline(service)
        with patch.object(CFG,'body_matching_mode','off'), patch.object(pipe,'_process_cut',return_value=original):
            result=pipe.process_cut(self.image,200,200)
        service.analyze.assert_not_called()
        self.assertEqual(dumps(result),before)

    def test_auto_and_shadow_preserve_every_pose_field(self):
        for mode in ('auto','shadow'):
            with self.subTest(mode=mode):
                original=cut();baseline=json.loads(dumps(original));baseline.pop('body_matching')
                pipe=self.pipeline(BodyMatchingService(self.path,client=FixtureClient()))
                with patch.object(CFG,'body_matching_mode',mode), patch.object(pipe,'_process_cut',return_value=original):
                    result=pipe.process_cut(self.image,200,200)
                data=json.loads(dumps(result));body=data.pop('body_matching')
                self.assertEqual(data,baseline)
                self.assertEqual(body['people'][0]['auto_body_id'],'muscular')
                self.assertEqual(body['people'][0]['applied_body_id'],'muscular' if mode=='auto' else None)
                self.assertFalse(body['rendering_executed'])

    def test_body_stage_exception_keeps_successful_pose_result(self):
        service=Mock();service.analyze.side_effect=RuntimeError('fixture failure')
        original=cut();baseline=json.loads(dumps(original));baseline.pop('body_matching')
        pipe=self.pipeline(service)
        with patch.object(CFG,'body_matching_mode','auto'),patch.object(pipe,'_process_cut',return_value=original):
            result=pipe.process_cut(self.image,200,200)
        data=json.loads(dumps(result));body=data.pop('body_matching')
        self.assertEqual(data,baseline);self.assertEqual(body['status'],'unavailable')

    def test_model_count_and_default_are_catalog_driven(self):
        template=copy.deepcopy(self.raw['assets'][0])
        for count in (1,9,12):
            with self.subTest(count=count):
                assets=[]
                for i in range(count):
                    asset=copy.deepcopy(template);asset['body_id']=f'custom-avatar-{i:02d}'
                    qa=self.root/f'custom-{i}.qa.json'
                    identity={k:asset[k] for k in ('body_id','body_version','asset_sha256','rig_version','supported_pose_ids')}
                    qa.write_text(json.dumps(dict(status='passed',**identity)))
                    asset.update(qa_report=qa.name,qa_sha256=hashlib.sha256(qa.read_bytes()).hexdigest())
                    assets.append(asset)
                self.raw.update(assets=assets,default_body_id=assets[-1]['body_id']);self.write()
                service=BodyMatchingService(self.path)
                self.assertEqual(service.analyze(self.image,cut())['people'][0]['auto_body_id'],assets[-1]['body_id'])
                self.raw['default_body_id']=assets[0]['body_id'];self.write()
                self.assertEqual(service.analyze(self.image,cut())['people'][0]['auto_body_id'],assets[0]['body_id'])

    def test_new_named_body_is_selected_without_runtime_code_change(self):
        service=BodyMatchingService(self.path,client=FixtureClient())
        self.assertEqual(service.analyze(self.image,cut())['people'][0]['auto_body_id'],'muscular')
        asset=next(a for a in self.raw['assets'] if a['body_id']=='muscular')
        asset['body_id']='arbitrary-new-fbx-2027'
        qa=self.root/asset['qa_report'];report=json.loads(qa.read_text());report['body_id']=asset['body_id'];qa.write_text(json.dumps(report))
        asset['qa_sha256']=hashlib.sha256(qa.read_bytes()).hexdigest();self.write()
        self.assertEqual(service.analyze(self.image,cut())['people'][0]['auto_body_id'],asset['body_id'])


if __name__=='__main__':unittest.main(verbosity=2)
