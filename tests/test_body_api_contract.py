"""HTTP and BFF boundary tests; all body/pose asset bytes are explicit fixtures."""
import copy,hashlib,io,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
import api.app as app_module
from api.models import CutResultOut, RefineResponse, ExportOrder
from api.body_models import BodyMatchingOut, EmptyBodyMatchingOut, validated_body_sidecar
from api.body_handoff import (BodyRenderRequest,BodyRenderResult,BodySelectionChange,BodyExportEnvelope,
    build_auto_render_request,apply_refine_result,render_result_matches)
from src.experimental.body_matching.service import BodyMatchingService
from tests.test_body_matching import BodyMatchingTests, FixtureClient, cut


class BodyAPIContractTests(BodyMatchingTests):
    def response(self,mode='auto',broken=None):
        result=cut()
        for i,c in enumerate(result.person_candidates[0]):
            path=self.root/f'pose-{i}.bvh';path.write_bytes(b'TEST BVH FIXTURE '+str(i).encode());c.bvh_path=str(path)
        result.body_matching={} if mode=='off' else BodyMatchingService(self.path,client=FixtureClient()).analyze(self.image,result,mode=mode)
        if broken:broken(result.body_matching)
        class Pipe:
            def process_cut(self,*args,**kwargs):return result
        encoded=io.BytesIO();self.image.save(encoded,format='PNG')
        with patch.dict(app_module.STATE,{'pipeline':Pipe(),'provider':'mock','pose_backend':'mock'}):
            response=TestClient(app_module.app).post('/analyze',files={'file':('rough.png',encoded.getvalue(),'image/png')})
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

    def test_http_off_remains_empty_and_legacy_people_unchanged(self):
        before=self.response('off');after=self.response()
        self.assertEqual(before['body_matching'],{})
        self.assertEqual(before['people'],after['people'])
        self.assertEqual(before['route'],after['route'])
        self.assertEqual(len(after['people'][0]['candidates']),5)

    def test_http_auto_typed_and_preserves_order(self):
        d=self.response();body=d['body_matching'];p=body['people'][0]
        self.assertEqual(p['applied_body_id'],p['auto_body_id']);self.assertTrue(p['render_required'])
        self.assertFalse(body['rendering_executed'])
        self.assertEqual([x['pose_id'] for x in p['pose_bindings']],[x['pose_id'] for x in d['people'][0]['candidates']])
        self.assertIsInstance(CutResultOut.model_validate(d).body_matching,BodyMatchingOut)

    def test_http_shadow_cannot_render(self):
        d=CutResultOut.model_validate(self.response('shadow'))
        self.assertIsNone(d.body_matching.people[0].applied_body_id)
        with self.assertRaises(ValueError):build_auto_render_request(d,0,request_id='job',retarget_version='fixture-r1',renderer_version='fixture-v1')

    def test_bad_plugin_person_or_pose_order_only_disables_body(self):
        for mutate in [lambda d:d['people'][0]['pose_bindings'].reverse(),
                       lambda d:d['people'][0].update(person_index=7),
                       lambda d:d['people'][0]['selected_asset'].update(body_id='wrong'),
                       lambda d:d['people'][0].update(applied_body_id='wrong')]:
            data=self.response(broken=mutate)
            self.assertEqual(data['body_matching']['reason'],'body_contract_invalid')
            self.assertEqual(len(data['people'][0]['candidates']),5)

    def test_missing_catalog_is_typed_unavailable(self):
        self.raw.update(assets=[],default_body_id=None);self.write()
        d=self.response()['body_matching'];self.assertEqual(d['status'],'partial')
        self.assertEqual(d['people'][0]['diagnostic'],'catalog_empty')
        self.assertIsNone(d['people'][0]['applied_body_id'])

    def test_top_level_exception_payload_is_valid(self):
        d=validated_body_sidecar(dict(schema_version='body-match.v1',mode='auto',status='unavailable',reason='body_execution_ValueError',people=[]),[])
        self.assertEqual(d.reason,'body_execution_ValueError')
        self.assertIsInstance(validated_body_sidecar({},[]),EmptyBodyMatchingOut)

    def test_unknown_attribute_is_null_and_zero_is_a_real_value(self):
        data=self.response()
        obs=data['body_matching']['people'][0]['observations']
        obs['attributes']['soft_volume']['value']=0
        parsed=CutResultOut.model_validate(data)
        self.assertEqual(parsed.body_matching.people[0].observations.attributes.soft_volume.value,0)
        obs['attributes']['soft_volume']['value']=True
        self.assertEqual(validated_body_sidecar(data['body_matching'],parsed.people).reason,'body_contract_invalid')

    def test_refine_fallback_references_returned_base_bytes(self):
        batch=self.batch()
        response=RefineResponse(pose_id=batch.poses[0].pose_id,view=batch.poses[0].view,refined=False,reason='no_gain',
            bvh_url=batch.poses[0].bvh_url,backend='none')
        updated=apply_refine_result(batch,0,response,request_id='fallback-job',resolved_pose_sha256=batch.poses[0].pose_sha256)
        self.assertEqual(updated.poses[0],batch.poses[0]);self.assertEqual(updated.body,batch.body)
        self.assertEqual(updated.request_id,'fallback-job')
        with self.assertRaisesRegex(ValueError,'new_request_id'):
            apply_refine_result(batch,0,response,request_id=batch.request_id,resolved_pose_sha256=batch.poses[0].pose_sha256)

    def test_openapi_has_body_schema_not_untyped_dict(self):
        schema=app_module.app.openapi();schemas=schema['components']['schemas']
        self.assertIn('BodyMatchingOut',schemas);self.assertIn('PresentationOut',schemas)
        variants=schemas['CutResultOut']['properties']['body_matching']['anyOf']
        self.assertEqual({v['$ref'].split('/')[-1] for v in variants},{'BodyMatchingOut','EmptyBodyMatchingOut'})

    def batch(self):
        return build_auto_render_request(CutResultOut.model_validate(self.response()),0,request_id='render-1',retarget_version='fixture-r1',renderer_version='fixture-v1')

    def render_result(self,batch):
        return BodyRenderResult(request_id=batch.request_id,person_id=batch.person_id,selection_revision=batch.selection_revision,
            body_id=batch.body.body_id,body_version=batch.body.body_version,asset_sha256=batch.body.asset_sha256,
            retarget_version='fixture-r1',renderer_version='fixture-v1',status='ready',
            artifacts=[dict(candidate_index=p.candidate_index,pose_id=p.pose_id,view=p.view,source_pose_sha256=p.pose_sha256,
                            preview_url=f'/fixture/{p.candidate_index}.png',fbx_url=f'/fixture/{p.candidate_index}.fbx',fbx_sha256='a'*64) for p in batch.poses])

    def test_auto_to_renderer_is_same_body_same_five_poses(self):
        batch=self.batch();self.assertEqual(len(batch.poses),5)
        self.assertEqual(batch.body.body_id,'muscular');self.assertEqual(batch.selection_source,'auto')
        self.assertTrue(render_result_matches(batch,self.render_result(batch)))

    def test_missing_pose_hash_must_be_resolved_before_render(self):
        d=CutResultOut.model_validate(self.response())
        for p in d.body_matching.people[0].pose_bindings:p.pose_sha256=None
        with self.assertRaisesRegex(ValueError,'pose_digest'):build_auto_render_request(d,0,request_id='job',retarget_version='fixture-r1',renderer_version='fixture-v1')
        resolved={c.bvh_url:'b'*64 for c in d.people[0].candidates}
        self.assertEqual(build_auto_render_request(d,0,request_id='job',retarget_version='fixture-r1',renderer_version='fixture-v1',resolved_pose_hashes=resolved).poses[0].pose_sha256,'b'*64)

    def test_stale_partial_wrong_body_or_pose_render_never_replaces_cards(self):
        batch=self.batch()
        for field,value in [('selection_revision',99),('body_id','other'),('asset_sha256','f'*64),('request_id','old'),('renderer_version','old'),('retarget_version','old')]:
            result=self.render_result(batch);setattr(result,field,value);self.assertFalse(render_result_matches(batch,result))
        result=self.render_result(batch);result.artifacts.pop();self.assertFalse(render_result_matches(batch,result))
        result=self.render_result(batch);result.artifacts.reverse();self.assertFalse(render_result_matches(batch,result))
        result=self.render_result(batch);result.artifacts[0].source_pose_sha256='e'*64;self.assertFalse(render_result_matches(batch,result))

    def test_manual_command_requires_version_and_same_person_identity(self):
        batch=self.batch()
        d=dict(analysis_input_sha256=batch.analysis_input_sha256,person_id=batch.person_id,person_index=0,body=batch.body.model_dump(),expected_revision=0)
        self.assertEqual(BodySelectionChange.model_validate(d).scope,'current_cut_person')
        del d['expected_revision']
        with self.assertRaises(ValidationError):BodySelectionChange.model_validate(d)

    def test_refine_changes_only_one_pose_keeps_body_and_invalidates_stale_render(self):
        batch=self.batch();old=self.render_result(batch)
        response=RefineResponse(pose_id=batch.poses[2].pose_id,view=batch.poses[2].view,refined=True,reason='ok',
                                bvh_url=batch.poses[2].bvh_url,bvh='refined fixture BVH',backend='numpy')
        updated=apply_refine_result(batch,2,response,request_id='refined-render-2',resolved_pose_sha256='d'*64)
        self.assertEqual(updated.body,batch.body);self.assertEqual(updated.poses[0],batch.poses[0])
        self.assertEqual(updated.poses[2].kind,'refined');self.assertEqual(updated.poses[2].pose_sha256,hashlib.sha256(b'refined fixture BVH').hexdigest())
        self.assertFalse(render_result_matches(updated,old));self.assertEqual(batch.selection_revision,0)

    def test_export_envelope_preserves_legacy_order_and_body_identity(self):
        batch=self.batch();pose=batch.poses[2]
        order=ExportOrder(cut_id='cut',created_at='2026-10-07T00:00:00Z',items=[dict(person_index=0,pose_id=pose.pose_id,
                            bvh_url=pose.bvh_url,view=pose.view,tags={})])
        assignment=dict(person_index=0,person_id=batch.person_id,selection_revision=0,body=batch.body,pose=pose)
        envelope=BodyExportEnvelope(analysis_input_sha256=batch.analysis_input_sha256,pose_order=order,body_assignments=[assignment])
        self.assertEqual(envelope.pose_order.model_dump(),order.model_dump())
        with self.assertRaises(ValidationError):BodyExportEnvelope(analysis_input_sha256=batch.analysis_input_sha256,pose_order=order,body_assignments=[])


if __name__=='__main__':
    suite=unittest.TestSuite(BodyAPIContractTests(name) for name in BodyAPIContractTests.__dict__ if name.startswith('test_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not result.wasSuccessful())
