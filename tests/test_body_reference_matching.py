import copy
import unittest
from src.experimental.body_matching.reference_matching import AXES, parse_comparison

class ReferenceMatchingTests(unittest.TestCase):
    def response(self):
        return dict(criteria={a:dict(visibility="unknown",evidence="not observed",regions=[]) for a in AXES},
            ranking=[dict(ref="r1",match="broad shoulders and thick upper arm",difference="waist not visible")],
            separation="distinct",tied_refs=[])
    def observe(self):
        return dict(coverage={r:"visible" for r in ("head","torso","arms","legs")},
            representation={r:"outline" for r in ("head","torso","arms","legs")},
            ownership_ambiguous=False,full_body_visible=True,foreshortening=False)
    def run_case(self,raw,obs=None):
        return parse_comparison(raw,{"r1":"muscular","r2":"bulk"},obs or self.observe())
    def clear(self,raw,axis,regions):
        raw['criteria'][axis]=dict(visibility="visible",evidence="owned visible contour",regions=regions)
    def test_clear_half_body_has_reviewable_separate_torso_and_arm_evidence(self):
        r=self.response();self.clear(r,'shoulder_hip',['torso']);self.clear(r,'arm_bulk',['arms'])
        o=self.observe();o['full_body_visible']=False;o['coverage']['legs']='unknown'
        x=self.run_case(r,o)
        self.assertEqual(x['decision'],'review_recommendation')
        self.assertEqual(x['evidence_groups'],['arms','torso'])
        self.assertIsNone(x['auto_body_id']);self.assertIsNone(x['applied_body_id'])
        self.assertFalse(x['application_policy']['allowed'])
    def test_correlated_torso_labels_never_manufacture_multiple_groups(self):
        r=self.response()
        for a in ('shoulder_hip','waist_chest','abdominal_profile'):self.clear(r,a,['torso'])
        x=self.run_case(r)
        self.assertEqual(x['evidence_groups'],['torso']);self.assertEqual(x['decision'],'ambiguous')
    def test_stick_lines_cannot_become_thin_body_even_if_provider_claims_visible(self):
        r=self.response();self.clear(r,'arm_bulk',['arms']);self.clear(r,'leg_bulk',['legs'])
        o=self.observe();o['representation']={k:'skeletal' for k in o['representation']}
        x=self.run_case(r,o);self.assertEqual(x['candidates'],[]);self.assertEqual(x['decision'],'insufficient_evidence')
    def test_ownership_failure_masks_all_axes(self):
        r=self.response();self.clear(r,'arm_bulk',['arms']);self.clear(r,'waist_chest',['torso'])
        o=self.observe();o['ownership_ambiguous']=True
        self.assertEqual(self.run_case(r,o)['candidates'],[])
    def test_cropped_head_and_hidden_torso_are_not_inferred(self):
        r=self.response();self.clear(r,'head_body',['head','legs']);self.clear(r,'waist_chest',['torso'])
        o=self.observe();o['full_body_visible']=False;o['coverage']['torso']='unknown'
        self.assertEqual(self.run_case(r,o)['candidates'],[])
    def test_uncertain_axes_do_not_qualify_recommendation(self):
        r=self.response()
        for a,regions in [('arm_bulk',['arms']),('waist_chest',['torso'])]:
            self.clear(r,a,regions);r['criteria'][a]['visibility']='uncertain'
        x=self.run_case(r);self.assertEqual(x['evidence_groups'],[]);self.assertEqual(x['decision'],'ambiguous')
    def test_tied_candidates_survive_without_automatic_tiebreak(self):
        r=self.response();self.clear(r,'arm_bulk',['arms']);self.clear(r,'leg_bulk',['legs'])
        r.update(separation='close',tied_refs=['r1','r2'])
        x=self.run_case(r);self.assertEqual(x['tied_body_ids'],['muscular','bulk']);self.assertEqual(x['decision'],'ambiguous')
    def test_surface_definition_alone_cannot_qualify(self):
        r=self.response();self.clear(r,'surface_definition',['arms','legs','torso'])
        self.assertEqual(self.run_case(r)['evidence_groups'],[])
    def test_foreign_or_duplicate_candidate_ids_rejected(self):
        for change in ('unknown','duplicate'):
            r=self.response()
            if change=='unknown':r['ranking'][0]['ref']='not-in-catalog'
            else:r['ranking']*=2
            with self.assertRaises(ValueError):self.run_case(r)
    def test_contradictory_tie_and_empty_reasons_rejected(self):
        for change in ('tie','reason','close'):
            r=self.response()
            if change=='tie':r['tied_refs']=['r1','r2']
            elif change=='close':r['separation']='close'
            else:r['ranking'][0]['difference']=' '
            with self.assertRaises(ValueError):self.run_case(r)
    def test_unreadable_input_preserves_no_rank(self):
        r=self.response();r.update(ranking=[],separation='unreadable')
        self.assertEqual(self.run_case(r)['decision'],'insufficient_evidence')

    def test_legacy_null_coverage_is_masked_not_crashed(self):
        r=self.response();self.clear(r,'arm_bulk',['arms'])
        o=self.observe();o.update(coverage=None,representation=None)
        self.assertEqual(self.run_case(r,o)['candidates'],[])
    def test_ownership_guard_prevents_provider_request(self):
        from src.experimental.body_matching.reference_matching import GeminiReferenceClient
        client=GeminiReferenceClient.__new__(GeminiReferenceClient)
        result=client.compare(None,[],{'r1':{'body_id':'a'}},{'ownership_ambiguous':True})
        self.assertEqual(result['decision'],'insufficient_evidence')
