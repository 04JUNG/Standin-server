"""Optional /analyze HTTP boundary tests; all body/pose asset bytes are explicit fixtures."""
import copy,hashlib,io,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
import api.app as app_module
from api.models import CutResultOut
from api.body_models import BodyMatchingOut, EmptyBodyMatchingOut, validated_body_sidecar
from src.experimental.body_matching.service import BodyMatchingService
from tests.test_body_matching import BodyMatchingFixture, FixtureClient, cut


class BodyAPIContractTests(BodyMatchingFixture):
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
        self.assertFalse(d.body_matching.people[0].render_required)

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

    def test_openapi_has_body_schema_not_untyped_dict(self):
        schema=app_module.app.openapi();schemas=schema['components']['schemas']
        self.assertIn('BodyMatchingOut',schemas);self.assertIn('PresentationOut',schemas)
        variants=schemas['CutResultOut']['properties']['body_matching']['anyOf']
        self.assertEqual({v['$ref'].split('/')[-1] for v in variants},{'BodyMatchingOut','EmptyBodyMatchingOut'})

    def test_no_render_endpoint_or_converter_contract_added(self):
        paths=app_module.app.openapi()['paths']
        self.assertNotIn('/body/render', paths)


if __name__ == '__main__':
    unittest.main()
