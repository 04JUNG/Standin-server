import os, unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.app import app

class APIContracts(unittest.TestCase):
    def test_disabled_endpoint_is_fail_closed(self):
        with patch.dict(os.environ,{},clear=True):
            client=TestClient(app)
            self.assertEqual(client.get('/refine/release/ready').status_code,503)
            response=client.post('/refine/release',data={'pose_id':'unknown'},files={'image':('image.png',b'input','image/png')})
            self.assertEqual(response.status_code,503)
    def test_legacy_route_remains_registered(self):
        paths=app.openapi()['paths']
        self.assertIn('/refine',paths)
        self.assertIn('/refine/release',paths)
        self.assertIn('pose_id',paths['/refine']['post']['requestBody']['content']['application/json']['schema'].get('properties',{}) if '$ref' not in paths['/refine']['post']['requestBody']['content']['application/json']['schema'] else app.openapi()['components']['schemas']['RefineRequest']['properties'])
