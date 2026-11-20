import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts.character_upgrade.check_sampled_runtime import catalog_controls, published_value, verify_geometry


class PublishedDomainTests(unittest.TestCase):
    def test_exact_source_preset_coverage_and_unchanged_bounds(self):
        header = dict(rules='new-model', source=dict(blendSha256='a'*64, controllerSha256='b'*64),
                      controls=dict(height_cm=dict(min=150, max=195)), presets=dict(BASE={'height_cm':170}, SOFT={'height_cm':170}))
        def asset(name, low):
            return dict(shapeSource=dict(rules=header['rules'], **header['source']),
                        shapeRuntime=dict(preset=name,format='standin-shape-runtime.v2',rules=header['rules'],sha256='d'*64,bytes=100),
                        shape=dict(controls=[dict(key='height', min=low, max=195, step=1, default=170)]))
        assets = [asset('BASE', 160), asset('SOFT', 150)]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'catalog.json'
            def check(rows, pin=None, identity=None):
                raw = json.dumps(dict(assets=rows)).encode(); p.write_bytes(raw)
                return catalog_controls(p, pin or hashlib.sha256(raw).hexdigest(), header, identity)
            self.assertEqual(check(assets)['SOFT']['height_cm'], dict(min=150, max=195, step=1, default=170))
            for rows in (assets[:1], [assets[0], assets[0]], [asset('BASE', 149), assets[1]]):
                with self.assertRaises(ValueError): check(rows)
            with self.assertRaises(ValueError): check(assets, '0'*64)
            altered = asset('BASE', 160); altered['shapeSource']['blendSha256'] = 'c'*64
            with self.assertRaises(ValueError): check([altered, assets[1]])
            altered = asset('BASE', 160); altered['shape']['controls'][0]['default'] = 171
            with self.assertRaises(ValueError): check([altered, assets[1]])
            check(assets, identity=dict(sha256='d'*64,bytes=100))
            for identity in (dict(sha256='e'*64,bytes=100),dict(sha256='d'*64,bytes=101)):
                with self.assertRaises(ValueError): check(assets, identity=identity)
            for key,value in (('format','standin-shape-runtime.v1'),('rules','another-model')):
                altered=asset('BASE',160);altered['shapeRuntime'][key]=value
                with self.assertRaises(ValueError):check([altered,assets[1]])

    def test_fractional_height_uses_the_public_grid_without_rounding_below_minimum(self):
        bounds=dict(min=160.684899, max=195, step=1)
        self.assertEqual(published_value(bounds,160.684899),160.684899)
        self.assertEqual(published_value(bounds,195),194.684899)
        self.assertEqual(published_value(bounds,170.1),169.684899)

    def test_runtime_faces_and_array_sizes_are_checked_against_final_source(self):
        preview=SimpleNamespace(data=SimpleNamespace(vertices=[None]*3, polygons=[SimpleNamespace(vertices=(0,1,2))]))
        header=dict(vertexCount=3,faceCount=1,basisCount=1,coefCount=1)
        arrays=dict(faces=np.array([0,1,2],dtype=np.uint16),mean=np.zeros(9),basisScale=np.ones(1),basis=np.zeros(9),coef=np.zeros(1))
        self.assertTrue(verify_geometry(header,arrays,preview)['indicesMatchFinalSource'])
        arrays['faces'][2]=3
        with self.assertRaises(ValueError):verify_geometry(header,arrays,preview)
        arrays['faces'][2]=2;arrays['basis']=np.zeros(8)
        with self.assertRaises(ValueError):verify_geometry(header,arrays,preview)


if __name__ == '__main__': unittest.main()
