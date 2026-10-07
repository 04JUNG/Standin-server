"""Body sidecar contracts. All generated FBX bytes are explicit test fixtures."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.config import CFG, Config
from src.schema import (BBox, CutResult, PersonDescriptor, PoseCandidate, Shot, Action,
                        View, Relationship, Skeleton, dumps)
from src.experimental.body_matching.catalog import load_catalog, file_sha256
from src.experimental.body_matching.schema import AXES, GEOMETRY_VERSION, unknown_attributes
from src.experimental.body_matching.observation import geometry_ratios, parse_response, GeminiBodyAttributeClient
from src.experimental.body_matching.service import BodyMatchingService


def attributes(build='regular'):
    return dict(head_ratio_class='h7', body_build=build, frame_width='regular',
                limb_proportion='regular', muscularity_visual=3 if build=='muscular' else 0 if build=='slim' else 1,
                soft_volume=3 if build=='chubby' else 1, volume_distribution='even')


def person_payload(person_id, build='regular'):
    return dict(person_id=person_id, attributes={k: dict(value=v, visibility='visible', evidence='fixture')
                for k, v in attributes(build).items()}, clothing_occlusion=False,
                foreshortening=False, ownership_ambiguous=False)


class FixtureClient:
    provider='recorded_test'
    model='fixture.v1'
    is_mock=False
    def __init__(self, build='muscular'):
        self.build=build
        self.calls=0
    def analyze(self, crops):
        self.calls+=1
        return {key: person_payload(key,self.build) for key,_ in crops}


def cut(count=1):
    xy=np.array([[50.,10.]]*17)
    xy[5:]=[[30,40],[70,40],[20,65],[80,65],[15,90],[85,90],
             [40,100],[60,100],[38,140],[62,140],[36,180],[64,180]]
    result=CutResult('core','high',count,count)
    for i in range(count):
        result.descriptors.append(PersonDescriptor(Shot.FULL_HALF,Action.STANDING,View.FRONT,
            Relationship.SOLO,Skeleton(xy.copy(),np.ones(17)),None,
            box=BBox(i*100,0,(i+1)*100,200),valid_joint_mask=np.ones(17,dtype=bool)))
        result.person_candidates.append([PoseCandidate('pose-'+str(p),View.FRONT,float(p),{}) for p in range(5)])
    return result


class BodyMatchingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.path=self.root/'catalog.json'
        self.raw=dict(schema_version='body-catalog.v1',catalog_version='test.v1',default_body_id='regular',assets=[])
        for build in ('regular','slim','muscular','chubby'):
            fbx=self.root/(build+'.fbx');fbx.write_bytes(b'TEST FIXTURE NOT A REAL FBX '+build.encode())
            qa=self.root/(build+'.qa.json')
            identity=dict(body_id=build,body_version='v1',asset_sha256=file_sha256(fbx),rig_version='fixture',
                          supported_pose_ids=['pose-'+str(i) for i in range(5)])
            qa.write_text(json.dumps(dict(status='passed',**identity)))
            self.raw['assets'].append(dict(**identity,availability='eligible',fbx_path=fbx.name,
                measurement_version='fixture.v1',qa_report=qa.name,qa_sha256=file_sha256(qa),
                attributes=attributes(build),projections=[]))
        self.write()
        self.image=Image.new('RGB',(200,200))
    def write(self):
        self.path.write_text(json.dumps(self.raw))
    def run_service(self,client=None,result=None,**kwargs):
        return BodyMatchingService(self.path,client=client or FixtureClient()).analyze(self.image,result or cut(),**kwargs)

    def test_visual_selection_and_pose_preservation(self):
        result=cut();before=dumps(result)
        output=self.run_service(result=result)
        person=output['people'][0]
        self.assertEqual(person['applied_body_id'],'muscular')
        self.assertEqual(person['selection_source'],'auto_best_effort')
        self.assertIsNone(person['acceptance_probability'])
        self.assertFalse(output['rendering_executed'])
        self.assertEqual([p['pose_id'] for p in person['pose_bindings']],['pose-'+str(i) for i in range(5)])
        self.assertEqual(dumps(result),before)

    def test_mock_default_is_not_visual_success(self):
        out=BodyMatchingService(self.path).analyze(self.image,cut(),pose_is_mock=True)
        p=out['people'][0]
        self.assertTrue(out['is_mock'])
        self.assertEqual(p['selection_source'],'auto_default')
        self.assertEqual(p['applied_body_id'],'regular')
        self.assertEqual(p['candidates'],[])

    def test_unknown_zero_and_occlusion(self):
        raw=person_payload('p','slim');raw['attributes']['soft_volume']['value']=0
        p=parse_response({'people':[raw]},['p'])['p']
        self.assertEqual(p['attributes']['soft_volume']['value'],0)
        raw['clothing_occlusion']=True
        p=parse_response({'people':[raw]},['p'])['p']
        self.assertIsNone(p['attributes']['soft_volume']['value'])
        self.assertEqual(p['attributes']['head_ratio_class']['value'],'h7')

    def test_invalid_vlm_values_ids_missing_duplicate(self):
        raw=person_payload('p')
        for value in (True, float('nan'), 9, '1'):
            broken=copy.deepcopy(raw);broken['attributes']['soft_volume']['value']=value
            with self.assertRaises(ValueError):parse_response({'people':[broken]},['p'])
        for people in ([],[raw,raw],[person_payload('wrong')]):
            with self.assertRaises(ValueError):parse_response({'people':people},['p'])
        raw['height_cm']=175
        with self.assertRaises(ValueError):parse_response({'people':[raw]},['p'])

    def test_real_lite_nullable_visibility_inconsistency_masks_only_axis(self):
        raw=person_payload('p')
        raw['attributes']['body_build'].update(value=None,visibility='uncertain')
        raw['attributes']['volume_distribution'].update(value='even',visibility='unknown')
        parsed=parse_response({'people':[raw]},['p'])['p']
        self.assertIsNone(parsed['attributes']['body_build']['value'])
        self.assertEqual(parsed['attributes']['body_build']['visibility'],'unknown')
        self.assertIsNone(parsed['attributes']['volume_distribution']['value'])
        self.assertEqual(parsed['attributes']['head_ratio_class']['value'],'h7')

    def test_timeout_and_invalid_batch_default_without_losing_pose(self):
        for error in (TimeoutError(), ValueError('invalid JSON')):
            client=FixtureClient()
            with patch.object(client,'analyze',side_effect=error):
                p=self.run_service(client)['people'][0]
                self.assertEqual(p['applied_body_id'],'regular')
                self.assertEqual(p['diagnostic'],'provider_error')
                self.assertEqual(len(p['pose_bindings']),5)
        client=FixtureClient()
        with patch.object(client,'analyze',return_value={'wrong':person_payload('wrong')}):
            self.assertEqual(self.run_service(client)['people'][0]['diagnostic'],'provider_error')

    def test_batch_once_cache_and_catalog_addition(self):
        client=FixtureClient();service=BodyMatchingService(self.path,client=client)
        first=service.analyze(self.image,cut(2))
        self.assertEqual(client.calls,1)
        self.assertEqual(len(first['people']),2)
        second=service.analyze(self.image,cut(2))
        self.assertEqual(client.calls,1)
        self.assertTrue(second['people'][0]['observations']['cache_hit'])
        self.raw['assets']=[a for a in self.raw['assets'] if a['body_id']!='muscular'];self.raw['catalog_version']='test.v2';self.write()
        third=service.analyze(self.image,cut(2))
        self.assertEqual(client.calls,1)
        self.assertEqual(third['catalog_version'],'test.v2')
        self.assertNotEqual(third['people'][0]['applied_body_id'],'muscular')

    def test_overlap_and_budget_no_wrong_person_observations(self):
        client=FixtureClient();result=cut(2);result.descriptors[1].box=result.descriptors[0].box
        out=self.run_service(client,result)
        self.assertEqual(client.calls,0)
        self.assertTrue(all(p['observations']['ownership_ambiguous'] for p in out['people']))
        service=BodyMatchingService(self.path,client=client,max_people=1)
        out=service.analyze(self.image,cut(2))
        self.assertEqual(out['people'][0]['applied_body_id'],'muscular')
        self.assertEqual(out['people'][1]['applied_body_id'],'regular')

    def test_qa_hash_pose_support_and_drafts(self):
        (self.root/'muscular.fbx').write_bytes(b'changed')
        out=self.run_service()
        self.assertNotEqual(out['people'][0]['applied_body_id'],'muscular')
        self.assertEqual(out['catalog_issues'][0]['reason'],'asset_or_qa_hash_mismatch')
        self.raw['assets'][1]['availability']='draft';self.write()
        self.assertEqual(len(load_catalog(self.path).eligible(['pose-0'])),2)
        result=cut();result.person_candidates[0][0].pose_id='unreviewed'
        self.assertEqual(self.run_service(result=result)['people'][0]['diagnostic'],'asset_incompatible')

    def test_empty_bad_catalog_and_no_candidates(self):
        self.raw['assets']=[];self.raw['default_body_id']=None;self.write()
        p=self.run_service()['people'][0]
        self.assertIsNone(p['applied_body_id']);self.assertEqual(p['diagnostic'],'catalog_empty')
        self.path.write_text('{')
        self.assertEqual(self.run_service()['people'][0]['diagnostic'],'catalog_error')
        self.write();result=cut();result.person_candidates[0]=[]
        self.assertEqual(self.run_service(result=result)['people'][0]['diagnostic'],'no_pose_candidates')

    def test_projection_shared_space_and_pose_hash(self):
        result=cut();posefile=self.root/'pose.bvh';posefile.write_text('fixture')
        for c in result.person_candidates[0]:c.bvh_path=str(posefile)
        ratios=geometry_ratios(result.descriptors[0].skeleton.keypoints,np.ones(17,bool))
        moved=geometry_ratios(result.descriptors[0].skeleton.keypoints*3+7,np.ones(17,bool))
        np.testing.assert_allclose(list(ratios.values()),list(moved.values()))
        for asset in self.raw['assets']:
            asset['projections']=[dict(pose_id='pose-0',view='front',pose_sha256=file_sha256(posefile),
                target_asset_sha256=asset['asset_sha256'],rig_version=asset['rig_version'],
                geometry_version=GEOMETRY_VERSION,ratios={k:v*(1 if asset['body_id']=='slim' else 2) for k,v in ratios.items()})]
        self.write()
        service=BodyMatchingService(self.path)
        out=service.analyze(self.image,result)
        self.assertEqual(out['people'][0]['applied_body_id'],'slim')
        self.assertEqual(len(out['people'][0]['geometry_hypotheses']),1)
        posefile.write_text('new pose version')
        p=service.analyze(self.image,result)['people'][0]
        self.assertEqual(p['applied_body_id'],'regular')
        self.assertIn('target_projection_missing_geometry_not_scored',p['reason_codes'])

    def test_missing_projection_cannot_advantage_candidate(self):
        result=cut();self.raw['assets'][0]['attributes']['body_build']=None;self.write()
        out=self.run_service(result=result)
        self.assertNotIn('body_build',out['people'][0]['evidence_axes'])
        self.assertIn('incomplete_catalog_attributes_excluded',out['people'][0]['reason_codes'])

    def test_shadow_and_noncore(self):
        p=self.run_service(mode='shadow')['people'][0]
        self.assertEqual(p['auto_body_id'],'muscular');self.assertIsNone(p['applied_body_id'])
        result=cut();result.route='skip';client=FixtureClient()
        self.assertEqual(self.run_service(client,result)['status'],'not_applicable')
        self.assertEqual(client.calls,0)

    def test_pipeline_integration_off_auto_and_rollback(self):
        from src.pipeline import Pipeline
        from src.library import build_synthetic_index
        from src.pose import MockPoseModel
        from src.vlm.client import MockVLMClient
        client=FixtureClient();service=BodyMatchingService(self.path,client=client)
        pipe=Pipeline(build_synthetic_index(),vlm_client=MockVLMClient(),pose_model=MockPoseModel(),body_matcher=service)
        with patch.object(CFG,'body_matching_mode','off'):
            before=pipe.process_cut(self.image,200,200)
        with patch.object(CFG,'body_matching_mode','auto'):
            after=pipe.process_cut(self.image,200,200)
        self.assertEqual(dumps(before.person_candidates),dumps(after.person_candidates))
        self.assertEqual(before.count_confidence,after.count_confidence)
        self.assertEqual(len(after.body_matching['people']),len(after.descriptors))
        self.assertEqual(before.body_matching,{})
        with patch.object(CFG,'body_matching_mode','off'):
            self.assertEqual(pipe.process_cut(self.image,200,200).body_matching,{})

    def test_config_validation(self):
        for changes in ({'body_matching_mode':'on'},{'body_timeout_seconds':float('nan')}, {'body_max_people':0}):
            with self.assertRaises(ValueError):Config(**changes)

    def test_api_response_exposes_sidecar(self):
        from api.models import CutResultOut,ImageInfoOut,InferenceMetadataOut
        sidecar=self.run_service()
        result=CutResultOut(route='core',count_confidence='high',detector_count=1,vlm_count=1,
            body_matching=sidecar,image=ImageInfoOut(width=200,height=200),
            inference_metadata=InferenceMetadataOut(deployment_version='test',vlm_provider='mock',
                vlm_model='test',pose_backend='mock',pose_model_version='test',pose_library_version='test',feature_version=1))
        self.assertEqual(json.loads(result.model_dump_json())['body_matching']['people'][0]['applied_body_id'],'muscular')

    def test_gemini_adapter_batches_labeled_images_and_uses_timeout(self):
        from types import SimpleNamespace
        try:
            from google import genai
        except ImportError:
            self.skipTest('optional Gemini SDK not installed')
        with patch.dict(os.environ,{'GEMINI_API_KEY':'test-placeholder'}), patch.object(genai,'Client') as factory:
            factory.return_value.models.generate_content.return_value=SimpleNamespace(
                text=json.dumps({'people':[person_payload('p1'),person_payload('p0')]}))
            client=GeminiBodyAttributeClient('fixture-model',2.5)
            output=client.analyze([('p0',self.image),('p1',self.image)])
            self.assertEqual(set(output),{'p0','p1'})
            self.assertEqual(factory.call_args.kwargs['http_options'].timeout,2500)
            request=factory.return_value.models.generate_content.call_args.kwargs
            self.assertEqual(request['contents'][1],'person_id=p0')
            self.assertEqual(request['contents'][3],'person_id=p1')
            self.assertEqual(request['config'].response_mime_type,'application/json')
            self.assertNotIn('body_id',request['config'].response_json_schema['properties'])

    def test_api_analyze_route_includes_body_decisions(self):
        import io
        from fastapi import UploadFile
        import api.app as app
        fixture=cut();fixture.body_matching=self.run_service(result=fixture)
        stream=io.BytesIO();self.image.save(stream,format='PNG');stream.seek(0)
        from types import SimpleNamespace
        with patch.dict(app.STATE,{'pipeline':SimpleNamespace(process_cut=lambda *a,**k: fixture)}):
            out=app.analyze(UploadFile(filename='rough.png',file=stream),hint='',rescue='')
        # HTTP compatibility is the serialized JSON; the response model is now typed.
        self.assertEqual(out.model_dump(mode='json')['body_matching']['people'][0]['applied_body_id'],'muscular')
        self.assertEqual([c.pose_id for c in out.people[0].candidates],['pose-'+str(i) for i in range(5)])

    def test_catalog_cli_and_projection_import(self):
        from scripts.body_catalog import main
        self.raw['assets'][0]['availability']='draft';self.write()
        with patch.object(sys,'argv',['body_catalog.py',str(self.path),'approve','--body-id','regular',
                                     '--qa',str(self.root/'regular.qa.json'),'--catalog-version','test.approved']):
            self.assertEqual(main(),0)
        cat=load_catalog(self.path)
        self.assertEqual(cat.assets[0]['availability'],'eligible')
        data={k:cat.assets[0][k] for k in ('body_id','body_version','rig_version','asset_sha256')}
        data.update(retarget_version='fixture',renderer_version='fixture',projections=[{
            'pose_id':'pose-0','view':'front','pose_sha256':'a'*64,
            'keypoints':cut().descriptors[0].skeleton.keypoints.tolist(),'valid_joint_mask':[True]*17}])
        path=self.root/'projections.json';path.write_text(json.dumps(data))
        from scripts.import_body_projections import main as import_main
        with patch.object(sys,'argv',['import_body_projections.py',str(self.path),str(path),'--catalog-version','test.projected']):
            import_main()
        self.assertEqual(len(load_catalog(self.path).assets[0]['projections']),1)

    def test_png_cli_without_models_reports_catalog_empty(self):
        self.raw.update(assets=[],default_body_id=None);self.write()
        image=self.root/'rough.png';self.image.save(image)
        output=self.root/'result.json'
        subprocess.run([sys.executable,'scripts/match_body.py',str(image),'--catalog',str(self.path),
                        '--synthetic','--provider','mock','--output',str(output)],check=True,capture_output=True,
                        cwd=Path(__file__).resolve().parents[1])
        raw=json.loads(output.read_text())
        self.assertTrue(raw['synthetic_pose'])
        self.assertTrue(raw['body_matching']['is_mock'])
        self.assertIsNone(raw['body_matching']['people'][0]['applied_body_id'])


if __name__=='__main__':unittest.main(verbosity=2)
