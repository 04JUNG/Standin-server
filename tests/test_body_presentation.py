"""Fictional presentation signals cannot be replaced by body-build stereotypes."""
import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests.test_body_matching import person_payload, attributes, BodyMatchingFixture, FixtureClient, cut
from src.experimental.body_matching.presentation import parse_presentation, presentation_candidates
from src.experimental.body_matching.schema import parse_person,unknown_attributes
from src.experimental.body_matching.selection import compare_body_shapes
from src.experimental.body_matching.catalog import load_catalog


def presentation(value='feminine',visibility='visible',cues=None):
    return dict(value=value,visibility=visibility,cues=cues or ['face_design'],evidence='Authored character design cue')


def assets():
    return [dict(body_id=style+'-'+build,body_version='1',asset_sha256='a'*64,rig_version='r',measurement_version='m',
                 metadata={'presentation_style':style},attributes=attributes(build),projections=[])
            for style in ['masculine','feminine'] for build in ['regular','muscular']]


def observe(style='feminine',build='regular'):
    p=person_payload('p',build);p['presentation']=presentation(style)
    return dict(parse_person(p,'p'),geometry={})


def select(obs,candidates=None):
    return compare_body_shapes(candidates or assets(),obs,default_body_id='masculine-regular',
                presentation_defaults={'feminine':'feminine-regular','masculine':'masculine-regular'})


class PresentationTests(unittest.TestCase):
    def test_female_style_survives_hidden_shape_and_uses_female_default(self):
        o=observe();o['attributes']=unknown_attributes();d=select(o)
        self.assertEqual(d['auto_body_id'],'feminine-regular');self.assertEqual(d['selection_source'],'auto_default')
        self.assertEqual(d['presentation_selection']['mode'],'compatible_candidates')

    def test_muscular_female_keeps_muscularity(self):
        d=select(observe('feminine','muscular'));self.assertEqual(d['auto_body_id'],'feminine-muscular')

    def test_muscular_male_does_not_pick_alphabetic_female(self):
        self.assertEqual(select(observe('masculine','muscular'))['auto_body_id'],'masculine-muscular')

    def test_no_gender_inference_from_slimness(self):
        o=observe('masculine','slim');d=select(o)
        self.assertTrue(d['auto_body_id'].startswith('masculine'))

    def test_hair_only_is_uncertain_and_cannot_filter(self):
        o=observe();o['presentation']=parse_presentation(presentation(cues=['hair_design']))
        self.assertEqual(o['presentation']['visibility'],'uncertain')
        pool,_,trace=presentation_candidates(assets(),o,'masculine-regular',{})
        self.assertEqual(len(pool),4);self.assertEqual(trace['mode'],'tie_or_default')

    def test_weak_hint_only_breaks_ties_not_shape_scores(self):
        a=assets();a=[a[0],a[3]] # regular male vs muscular female
        o=observe();o['presentation']=presentation(visibility='uncertain')
        self.assertEqual(select(o,a)['auto_body_id'],'masculine-regular')
        o['attributes']=unknown_attributes()
        self.assertEqual(select(o,a)['auto_body_id'],'feminine-muscular')

    def test_unknown_does_not_fabricate_female_or_male_detection(self):
        o=observe();o['presentation']=presentation(value=None,visibility='unknown');o['attributes']=unknown_attributes()
        d=select(o);self.assertEqual(d['auto_body_id'],'masculine-regular')
        self.assertEqual(d['presentation_selection']['mode'],'unresolved')

    def test_ownership_ambiguous_masks_presentation(self):
        o=person_payload('p');o.update(presentation=presentation(),ownership_ambiguous=True)
        self.assertIsNone(parse_person(o,'p')['presentation']['value'])

    def test_invalid_cues_and_values_rejected(self):
        for key,value in [('value','female'),('cues',['muscularity']),('cues',['face_design','face_design'])]:
            p=presentation();p[key]=value
            with self.assertRaises(ValueError):parse_presentation(p)

    def test_missing_compatible_asset_is_explicit(self):
        d=select(observe(),[assets()[0]])
        self.assertEqual(d['presentation_selection']['mode'],'matching_style_unavailable')
        self.assertIn('presentation_matching_asset_unavailable',d['reason_codes'])

    def test_legacy_observation_has_identical_choice(self):
        old=dict(person_payload('p'),geometry={});d=select(old)
        self.assertEqual(d['auto_body_id'],'masculine-regular')


class Fixture(BodyMatchingFixture):
    def test_catalog_rejects_mislabeled_presentation_default(self):
        self.raw['presentation_defaults']={'feminine':'regular'}
        self.raw['assets'][0]['metadata']={'presentation_style':'masculine'};self.write()
        with self.assertRaises(ValueError):load_catalog(self.path)

    def test_unapproved_matching_asset_cannot_be_selected(self):
        from src.experimental.body_matching.service import BodyMatchingService
        for a in self.raw['assets']:a['metadata']={'presentation_style':'masculine'}
        self.raw['assets'][1].update(metadata={'presentation_style':'feminine'},availability='draft')
        self.raw['presentation_defaults']={'feminine':'slim'};self.write()
        class Client(FixtureClient):
            def analyze(self,crops):return {k:dict(person_payload(k),presentation=presentation()) for k,_ in crops}
        result=BodyMatchingService(self.path,client=Client()).analyze(self.image,cut())
        self.assertNotEqual(result['people'][0]['auto_body_id'],'slim')
        self.assertEqual(result['people'][0]['presentation_selection']['mode'],'matching_style_unavailable')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PresentationTests)
    suite.addTests(Fixture(name) for name in Fixture.__dict__ if name.startswith('test_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not result.wasSuccessful())
