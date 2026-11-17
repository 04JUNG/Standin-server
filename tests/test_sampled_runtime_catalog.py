import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.character_upgrade.check_sampled_runtime import catalog_controls


class PublishedDomainTests(unittest.TestCase):
    def test_exact_source_preset_coverage_and_unchanged_bounds(self):
        header = dict(rules='new-model', source=dict(blendSha256='a'*64, controllerSha256='b'*64),
                      controls=dict(height_cm=dict(min=150, max=195)), presets=dict(BASE={}, SOFT={}))
        def asset(name, low):
            return dict(shapeSource=dict(rules=header['rules'], **header['source']),
                        shapeRuntime=dict(preset=name), shape=dict(controls=[dict(key='height', min=low, max=195)]))
        assets = [asset('BASE', 160), asset('SOFT', 150)]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'catalog.json'
            def check(rows, pin=None):
                raw = json.dumps(dict(assets=rows)).encode(); p.write_bytes(raw)
                return catalog_controls(p, pin or hashlib.sha256(raw).hexdigest(), header)
            self.assertEqual(check(assets)['SOFT']['height_cm'], dict(min=150, max=195))
            for rows in (assets[:1], [assets[0], assets[0]], [asset('BASE', 149), assets[1]]):
                with self.assertRaises(ValueError): check(rows)
            with self.assertRaises(ValueError): check(assets, '0'*64)
            altered = asset('BASE', 160); altered['shapeSource']['blendSha256'] = 'c'*64
            with self.assertRaises(ValueError): check([altered, assets[1]])


if __name__ == '__main__': unittest.main()
