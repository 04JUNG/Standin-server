import unittest
from unittest.mock import patch
from fastapi import HTTPException
from scripts.prepare_refine_release_staging import prepare
from api import refine_release
class StagingContracts(unittest.TestCase):
    def test_production_definition_rejected(self):
        with self.assertRaises(ValueError):prepare({'family':'StandinAppInferenceTask'},'repo@sha256:abc')
    def test_moving_tag_rejected(self):
        with self.assertRaises(ValueError):prepare({'family':'StandinStagingAppInferenceTask'},'repo:develop')
    def test_bff_overlay_preserves_other_settings(self):
        source={'family':'StandinStagingAppBffTask','taskDefinitionArn':'old','containerDefinitions':[{'name':'bff','image':'before','environment':[{'name':'EXISTING','value':'unchanged'}]},{'name':'init','image':'original'}]}
        result=prepare(source,'repo@sha256:abc')
        self.assertEqual(source['containerDefinitions'][0]['image'],'before')
        self.assertNotIn('taskDefinitionArn',result)
        self.assertEqual(result['containerDefinitions'][1],source['containerDefinitions'][1])
        self.assertEqual({e['name']:e['value'] for e in result['containerDefinitions'][0]['environment']},{'EXISTING':'unchanged','REFINE_RELEASE_ENABLED':'1'})
    def test_proxy_off_never_calls_worker(self):
        with patch.dict('os.environ',{'REFINE_RELEASE_ENABLED':'0','REFINE_RELEASE_WORKER_URL':'http://127.0.0.1:8002'}),patch.object(refine_release.httpx,'request') as request:
            with self.assertRaises(HTTPException) as caught:refine_release.ready()
            self.assertEqual(caught.exception.status_code,503);request.assert_not_called()
