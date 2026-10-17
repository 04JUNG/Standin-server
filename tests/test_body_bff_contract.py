"""Inference HTTP -> frozen deployed BFF mapper/resolver; no rewritten Python BFF."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
import api.app as app_module
from src.experimental.body_matching.schema import unknown_attributes
from src.experimental.body_matching.service import BodyMatchingService
from tests import test_body_matching as fixtures


class BodyBffContractTests(fixtures.BodyMatchingFixture):
    def http_response(self, raw=None, mode='auto'):
        client=fixtures.FixtureClient()
        if raw is not None:
            client.analyze=lambda crops:{person:{**copy.deepcopy(raw),'person_id':person} for person,_ in crops}
        result=fixtures.cut()
        result.body_matching=BodyMatchingService(self.path,client=client).analyze(self.image,result,mode=mode)
        class Pipe:
            def process_cut(self,*args,**kwargs):return result
        encoded=io.BytesIO();self.image.save(encoded,format='PNG')
        with patch.dict(app_module.STATE,{'pipeline':Pipe(),'provider':'mock','pose_backend':'mock'}):
            response=TestClient(app_module.app).post('/analyze',files={'file':('rough.png',encoded.getvalue(),'image/png')})
        self.assertEqual(response.status_code,200,response.text)
        payload=response.json()
        self.assertNotEqual(payload['body_matching'].get('reason'),'body_contract_invalid')
        return payload

    def bff(self, cut, *, user_default=None, mode='auto', intent='inherit', manual=None, revision=0):
        catalog=dict(version='fixture',defaultCharacterId='char-'+self.raw['default_body_id'],assets=[dict(
            bodyId=a['body_id'],characterId='char-'+a['body_id'],bodyVersion=a['body_version'],
            assetSha256=a['asset_sha256'],rigVersion=a['rig_version'],measurementVersion=a['measurement_version'],
            supportedPoseIds=a['supported_pose_ids']) for a in self.raw['assets'] if a['availability']=='eligible'])
        payload=dict(cut=cut,catalog=catalog,preferences=dict(version='body-preferences.v1',scope='installation',
            mode=mode,defaultCharacterId='char-'+user_default if user_default else None,revision=4),
            intent=intent,manualCharacterId='char-'+manual if manual else None,revision=revision)
        script=Path(__file__).parent/'fixtures/body_bff_v1/resolve.mjs'
        result=subprocess.run(['node',str(script)],input=json.dumps(payload),capture_output=True,text=True,check=True)
        return json.loads(result.stdout)

    def weak_response(self):
        raw=fixtures.person_payload('p');raw['attributes']=unknown_attributes()
        raw['attributes']['body_build']=dict(value='slim',visibility='uncertain',evidence='weak single outline')
        return self.http_response(raw)

    def test_frozen_bff_fixtures_match_recorded_source_compilation(self):
        folder=Path(__file__).parent/'fixtures/body_bff_v1'
        provenance=json.loads((folder/'provenance.json').read_text())
        self.assertEqual(provenance['commit'],'ee0ba879cadb4e1bc71cedbb5640d1671ff049c4')
        for record in provenance['files']:
            self.assertEqual(hashlib.sha256((folder/record['compiled']).read_bytes()).hexdigest(),record['compiled_sha256'])

    def test_clear_body_maps_to_available_recommendation(self):
        resolved=self.bff(self.http_response())
        self.assertEqual(resolved['recommendations'][0]['status'],'available')
        self.assertEqual(resolved['people'][0]['resolvedSource'],'auto_recommendation')
        self.assertEqual(resolved['people'][0]['resolvedBody']['bodyId'],'muscular')

    def test_held_shape_uses_user_default_fallback(self):
        resolved=self.bff(self.weak_response(),user_default='slim')
        self.assertEqual(resolved['recommendations'][0]['status'],'unavailable')
        self.assertEqual(resolved['people'][0]['resolvedSource'],'user_default_fallback')
        self.assertEqual(resolved['people'][0]['resolvedBody']['bodyId'],'slim')

    def test_held_shape_uses_catalog_default_without_user_setting(self):
        person=self.bff(self.weak_response())['people'][0]
        self.assertEqual(person['resolvedSource'],'catalog_default_fallback')
        self.assertEqual(person['resolvedBody']['bodyId'],'regular')

    def test_manual_choice_and_revision_survive_unavailable_recommendation(self):
        person=self.bff(self.weak_response(),intent='manual',manual='chubby',revision=9)['people'][0]
        self.assertEqual(person['resolvedSource'],'manual')
        self.assertEqual(person['resolvedBody']['bodyId'],'chubby')
        self.assertEqual(person['selectionRevision'],9)

    def test_fixed_default_overrides_even_valid_auto_recommendation(self):
        person=self.bff(self.http_response(),user_default='slim',mode='fixed_default')['people'][0]
        self.assertEqual(person['resolvedSource'],'fixed_default')
        self.assertEqual(person['resolvedBody']['bodyId'],'slim')

    def test_explicit_auto_never_silently_becomes_default(self):
        person=self.bff(self.weak_response(),intent='auto',user_default='slim')['people'][0]
        self.assertEqual(person['resolutionStatus'],'unavailable')
        self.assertIsNone(person['resolvedBody'])

    def test_strong_feminine_hidden_shape_preserves_feminine_default(self):
        for asset in self.raw['assets']:
            asset['metadata']={'presentation_style':'feminine' if asset['body_id']=='regular' else 'masculine'}
        self.raw.update(default_body_id='muscular',presentation_defaults={'feminine':'regular'})
        self.write()
        raw=fixtures.person_payload('p');raw['attributes']=unknown_attributes()
        raw['presentation']=dict(value='feminine',visibility='visible',cues=['face_design'],evidence='clear fictional face design')
        cut=self.http_response(raw)
        self.assertEqual(cut['body_matching']['people'][0]['selection_source'],'auto_presentation_default')
        resolved=self.bff(cut)
        self.assertEqual(resolved['recommendations'][0]['selectionSource'],'auto_presentation_default')
        self.assertEqual(resolved['recommendations'][0]['reasonCodes'],['presentation_supported_shape_default'])
        self.assertEqual(resolved['people'][0]['resolvedBody']['bodyId'],'regular')

    def test_hair_only_does_not_activate_presentation_default(self):
        raw=fixtures.person_payload('p');raw['attributes']=unknown_attributes()
        raw['presentation']=dict(value='feminine',visibility='visible',cues=['hair_design'],evidence='long hair only')
        resolved=self.bff(self.http_response(raw))
        self.assertEqual(resolved['recommendations'][0]['status'],'unavailable')
        self.assertEqual(resolved['people'][0]['resolvedSource'],'catalog_default_fallback')

    def test_shadow_recommendation_keeps_default_policy(self):
        resolved=self.bff(self.http_response(mode='shadow'),user_default='slim')
        self.assertEqual(resolved['recommendations'][0]['status'],'disabled')
        self.assertEqual(resolved['people'][0]['resolvedSource'],'user_default_fallback')


if __name__=='__main__':unittest.main(verbosity=2)
