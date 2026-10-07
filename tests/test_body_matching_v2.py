"""Coverage regressions and candidate comparison gates; no live API required."""
import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests.test_body_matching import person_payload, BodyMatchingTests, FixtureClient, cut
from src.experimental.body_matching.schema import parse_person
from src.experimental.body_matching.visual_selection import parse_comparison, apply_agreement, GeminiVisualBodySelector


def observed():
    p=person_payload('p')
    p.update(full_body_visible=True,coverage=dict(head='visible',torso='visible',arms='visible',legs='visible'))
    return p


def comparison(best=('A',)):
    return dict(best_ids=list(best),evidence_regions=['arms'],evidence_axes=['limb_thickness'],distinguishing_evidence='Visible arm contour',
                candidates=[dict(candidate_id=k,fit='close' if k in best else 'possible',evidence='Contour comparison') for k in ['A','B']])


class CoverageTests(unittest.TestCase):
    def test_cropped_person_cannot_supply_head_count(self):
        p=observed();p['full_body_visible']=False;p['attributes']['head_ratio_class']['value']='h3'
        q=parse_person(p,'p');self.assertIsNone(q['attributes']['head_ratio_class']['value'])
        self.assertIsNotNone(q['attributes']['frame_width']['value'])
        self.assertEqual(parse_person(q,'p'),q)

    def test_covered_torso_keeps_exposed_arm_muscularity(self):
        p=observed();p['clothing_occlusion']=True;p['coverage']['torso']='unknown'
        q=parse_person(p,'p')['attributes']
        self.assertIsNotNone(q['muscularity_visual']['value'])
        for k in ['frame_width','soft_volume','volume_distribution']:self.assertIsNone(q[k]['value'])

    def test_foreshortening_and_missing_feet_exclude_head_count(self):
        for change in ['foreshortening','coverage']:
            p=observed()
            if change=='foreshortening':p[change]=True
            else:p['coverage']['legs']='unknown'
            self.assertIsNone(parse_person(p,'p')['attributes']['head_ratio_class']['value'])

    def test_ownership_unknown_masks_everything(self):
        p=observed();p['ownership_ambiguous']=True
        self.assertTrue(all(x['value'] is None for x in parse_person(p,'p')['attributes'].values()))

    def test_invalid_or_partial_coverage_rejected(self):
        for change in ['partial','invalid','flag']:
            p=observed()
            if change=='partial':del p['coverage']['arms']
            elif change=='invalid':p['coverage']['arms']='clear'
            else:p['full_body_visible']='yes'
            with self.assertRaises(ValueError):parse_person(p,'p')

    def test_unknown_and_duplicate_candidate_ids_rejected(self):
        for bad_id in ['A','X']:
            raw=comparison();raw['candidates'][1]['candidate_id']=bad_id
            with self.assertRaises(ValueError):parse_comparison(raw,['A','B'],observed())

    def test_partial_head_reason_rejected_even_if_axis_is_mislabeled(self):
        p=observed();p['full_body_visible']=False
        raw=comparison();raw['distinguishing_evidence']='Oversized head-to-body proportion is the difference'
        with self.assertRaises(ValueError):parse_comparison(raw,['A','B'],p)
        raw=comparison();raw['evidence_axes']=['head_body_ratio']
        with self.assertRaises(ValueError):parse_comparison(raw,['A','B'],p)

    def test_verbose_valid_rationale_does_not_discard_comparison(self):
        raw=comparison();raw['distinguishing_evidence']='Visible arm contour differs. '*30
        self.assertEqual(parse_comparison(raw,['A','B'],observed())['best_ids'],['A'])
        raw['distinguishing_evidence']='x'*4097
        with self.assertRaises(ValueError):parse_comparison(raw,['A','B'],observed())

    def test_hidden_evidence_rejected(self):
        p=observed();p['coverage']['arms']='unknown'
        with self.assertRaises(ValueError):parse_comparison(comparison(),['A','B'],p)

    def test_agreement_never_resolves_disagreement_or_tie(self):
        base={'auto_body_id':'original','reason_codes':[]}
        for second in [('B',),('A','B'),()]:
            q=apply_agreement(base,comparison(),comparison(second),{},[])
            self.assertEqual(q['auto_body_id'],'original');self.assertFalse(q['visual_comparison']['accepted'])
        self.assertNotIn('visual_comparison',base)

    def test_agreement_never_creates_asset_or_render_permission(self):
        a={k:'x' for k in ['body_id','body_version','asset_sha256','rig_version','measurement_version']}
        base={'auto_body_id':'original','render_required':False,'evaluation_scope':'shape_only','reason_codes':[]}
        q=apply_agreement(base,comparison(),comparison(),{'A':'x'},[a])
        self.assertEqual(q['auto_body_id'],'x');self.assertFalse(q['render_required']);self.assertIsNone(q['acceptance_probability'])
        with self.assertRaises(StopIteration):apply_agreement(base,comparison(),comparison(),{'A':'x'},[])

    def test_alias_rotation_is_remapped_before_agreement(self):
        selector=object.__new__(GeminiVisualBodySelector)
        assets=[{k:x for k in ['body_id','body_version','asset_sha256','rig_version','measurement_version']} for x in ['x','y','z']]
        orders=[]
        def compare(crop,obs,ordered,aliases):
            orders.append(ordered)
            winner=next(k for k,v in aliases.items() if v=='y')
            return dict(best_ids=[winner],evidence_regions=['arms'],evidence_axes=['limb_thickness'],distinguishing_evidence='arms',
                candidates=[dict(candidate_id=k,fit='close',evidence='arms') for k in aliases])
        selector._compare=compare
        q=selector.select(None,observed(),assets,{'reason_codes':[]})
        self.assertEqual(q['auto_body_id'],'y');self.assertNotEqual(orders[0][1][0],orders[1][1][0])


class IntegrationTests(BodyMatchingTests):
    # Inherit fixture setup only; avoid rerunning inherited base tests below.
    def test_optional_visual_failure_preserves_selection_and_poses(self):
        from src.experimental.body_matching.service import BodyMatchingService
        class Failed:
            def select(self,*args):raise TimeoutError()
        baseline=self.run_service()
        result=BodyMatchingService(self.path,client=FixtureClient(),visual_selector=Failed()).analyze(self.image,cut())
        self.assertEqual(result['people'][0]['auto_body_id'],baseline['people'][0]['auto_body_id'])
        self.assertEqual(result['people'][0]['pose_bindings'],baseline['people'][0]['pose_bindings'])
        self.assertEqual(result['people'][0]['visual_comparison']['reason'],'visual_provider_TimeoutError')

    def test_visual_only_receives_pose_eligible_assets(self):
        from src.experimental.body_matching.service import BodyMatchingService
        self.raw['assets'][1]['availability']='draft';self.write()
        seen=[]
        class Capture:
            def select(self,crop,observation,assets,base):seen.extend(a['body_id'] for a in assets);return base
        BodyMatchingService(self.path,client=FixtureClient(),visual_selector=Capture()).analyze(self.image,cut())
        self.assertNotIn('slim',seen);self.assertEqual(set(seen),{'regular','muscular','chubby'})


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CoverageTests)
    suite.addTests(IntegrationTests(name) for name in IntegrationTests.__dict__ if name.startswith('test_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not result.wasSuccessful())
