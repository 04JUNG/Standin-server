"""Regression cases for weak, symbolic, and independently supported body evidence."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import test_body_matching as fixtures
from src.experimental.body_matching.schema import parse_person, unknown_attributes, response_schema
from src.experimental.body_matching.selection import compare_body_shapes


class BodyPolicyTests(unittest.TestCase):
    setUp = fixtures.BodyMatchingTests.setUp
    write = fixtures.BodyMatchingTests.write
    run_service = fixtures.BodyMatchingTests.run_service

    def observe(self, attributes=None):
        raw = fixtures.person_payload('p')
        raw['attributes'] = attributes or unknown_attributes()
        return raw

    def analyze_raw(self, raw, **kwargs):
        client = fixtures.FixtureClient()
        def analyze(crops):
            return {person: {**copy.deepcopy(raw), 'person_id': person} for person, _ in crops}
        client.analyze = analyze
        return self.run_service(client, **kwargs)['people'][0]

    def assert_held(self, person):
        # This is also the existing BFF's explicit unavailable-sidecar branch.
        self.assertIsNone(person['auto_body_id'])
        self.assertIsNone(person['selected_asset'])
        self.assertIsNone(person['applied_body_id'])
        self.assertFalse(person['render_required'])
        self.assertEqual(person['selection_state'], 'unavailable')
        self.assertEqual(len(person['pose_bindings']), 5)

    def test_uncertain_only_never_auto_applies_even_with_score_zero(self):
        raw = self.observe()
        raw['attributes']['body_build'] = dict(value='slim', visibility='uncertain', evidence='thin outline')
        person = self.analyze_raw(raw)
        self.assert_held(person)
        self.assertEqual(person['suggested_body_id'], 'slim')
        self.assertEqual(person['candidates'][0]['rank_score'], 0)
        self.assertEqual(person['decision'], 'suggest_only')

    def test_single_clear_attribute_does_not_auto_apply(self):
        raw = self.observe()
        raw['attributes']['body_build'] = dict(value='slim', visibility='visible', evidence='thin outline')
        person = self.analyze_raw(raw)
        self.assert_held(person)
        self.assertEqual(person['application_policy']['evidence_groups'], ['body_contour'])

    def test_correlated_outline_labels_are_one_group(self):
        raw = fixtures.person_payload('p', 'muscular')
        for axis in ('head_ratio_class', 'limb_proportion'):
            raw['attributes'][axis] = unknown_attributes()[axis]
        person = self.analyze_raw(raw)
        self.assert_held(person)
        self.assertEqual(person['application_policy']['evidence_groups'], ['body_contour'])

    def test_weak_axis_cannot_break_tie_in_otherwise_clear_evidence(self):
        raw = self.observe()
        for axis in ('head_ratio_class', 'limb_proportion'):
            raw['attributes'][axis] = fixtures.person_payload('p')['attributes'][axis]
        raw['attributes']['body_build'] = dict(value='slim', visibility='uncertain', evidence='weak outline')
        person = self.analyze_raw(raw)
        self.assert_held(person)
        self.assertIn('clear_evidence_does_not_resolve_winner', person['reason_codes'])

    def test_tied_clear_candidates_do_not_become_confident_via_id_sort(self):
        self.raw['assets'][1]['attributes'] = fixtures.attributes()
        self.write()
        person = self.analyze_raw(fixtures.person_payload('p'))
        self.assert_held(person)
        self.assertEqual(set(person['tied_body_ids']), {'regular', 'slim'})

    def test_no_shape_evidence_defaults_are_suggestions_only(self):
        person = self.analyze_raw(self.observe())
        self.assert_held(person)
        self.assertEqual(person['decision'], 'insufficient_evidence')
        self.assertEqual(person['suggested_body_id'], 'regular')

    def test_clear_independent_shape_still_applies_without_changing_poses(self):
        person = self.analyze_raw(fixtures.person_payload('p', 'muscular'))
        self.assertEqual(person['applied_body_id'], 'muscular')
        self.assertEqual(person['decision'], 'auto_apply')
        self.assertTrue(person['application_policy']['allowed'])
        self.assertFalse(person['application_policy']['calibrated'])
        self.assertIsNone(person['acceptance_probability'])

    def test_stick_figure_masks_visual_attributes_and_joint_ratios(self):
        raw = fixtures.person_payload('p', 'slim')
        raw['representation'] = {r: 'skeletal' for r in raw['representation']}
        person = self.analyze_raw(raw)
        self.assert_held(person)
        self.assertEqual(person['observations']['geometry'], {})
        self.assertTrue(all(v['value'] is None for v in person['observations']['attributes'].values()))

    def test_smooth_surface_is_not_zero_muscle_volume(self):
        raw = fixtures.person_payload('p', 'slim')
        raw['surface_detail'] = 'omitted'
        parsed = parse_person(raw, 'p')
        self.assertIsNone(parsed['attributes']['muscularity_visual']['value'])
        # A visibly large outer muscle contour survives a lack of internal lines.
        raw['attributes']['muscularity_visual']['value'] = 3
        self.assertEqual(parse_person(raw, 'p')['attributes']['muscularity_visual']['value'], 3)

    def test_cropped_or_foreshortened_head_count_stays_masked(self):
        for changes in ({'full_body_visible': False}, {'foreshortening': True}):
            raw = fixtures.person_payload('p')
            raw['attributes']['head_ratio_class']['value'] = 'h4'
            raw.update(changes)
            self.assertIsNone(parse_person(raw, 'p')['attributes']['head_ratio_class']['value'])

    def test_covered_torso_cannot_be_filled_in_from_visible_arms(self):
        raw = fixtures.person_payload('p', 'muscular')
        raw['coverage']['torso'] = 'unknown'
        raw['representation']['torso'] = 'unknown'
        raw['clothing_occlusion'] = True
        parsed = parse_person(raw, 'p')
        self.assertEqual(parsed['attributes']['muscularity_visual']['value'], 3)
        for axis in ('frame_width', 'soft_volume', 'volume_distribution'):
            self.assertIsNone(parsed['attributes'][axis]['value'])
        self.assertEqual(self.analyze_raw(raw)['observations']['geometry'], {})

    def test_legacy_payload_is_readable_but_cannot_bypass_gate(self):
        raw = fixtures.person_payload('p', 'muscular')
        for key in ('coverage', 'full_body_visible', 'representation', 'surface_detail'):
            del raw[key]
        person = self.analyze_raw(raw)
        self.assert_held(person)
        self.assertIn('shape_coverage_or_representation_missing', person['reason_codes'])

    def test_strong_presentation_preserves_explicit_default_without_claiming_shape_match(self):
        self.raw['assets'][0]['metadata'] = {'presentation_style': 'masculine'}
        self.raw['presentation_defaults'] = {'masculine': 'regular'}
        self.write()
        raw = self.observe()
        raw['presentation'] = dict(value='masculine', visibility='visible', cues=['face_design'], evidence='drawn face')
        person = self.analyze_raw(raw)
        self.assertEqual(person['applied_body_id'], 'regular')
        self.assertEqual(person['selection_source'], 'auto_presentation_default')
        self.assertEqual(person['decision'], 'presentation_default')
        self.assertFalse(person['application_policy']['allowed'])
        self.assertEqual(person['suggested_body_id'], 'regular')

    def test_clear_upper_body_frame_and_arm_volume_are_independent(self):
        raw = self.observe()
        raw.update(full_body_visible=False)
        raw['coverage']['legs'] = 'unknown'
        raw['representation']['legs'] = 'unknown'
        raw['attributes']['frame_width'] = dict(value='broad',visibility='visible',evidence='wide shoulder frame')
        raw['attributes']['muscularity_visual'] = dict(value=3,visibility='visible',evidence='exposed upper-arm volume')
        raw['evidence_regions'] = {axis:[] for axis in raw['attributes']}
        raw['evidence_regions'].update(frame_width=['torso'],muscularity_visual=['arms'])
        self.raw['assets'][2]['attributes']['frame_width'] = 'broad'
        self.write()
        person = self.analyze_raw(raw)
        self.assertEqual(person['applied_body_id'],'muscular')
        self.assertEqual(person['application_policy']['evidence_groups'],['frame_structure','regional_volume'])
        # The same two labels from the same torso outline remain one evidence group.
        raw['evidence_regions']['muscularity_visual'] = ['torso']
        self.assert_held(self.analyze_raw(raw))

    def test_visibility_and_representation_cannot_be_borrowed_across_regions(self):
        raw = fixtures.person_payload('p','muscular')
        raw['coverage'].update(torso='unknown',legs='unknown',arms='visible')
        raw['representation'].update(torso='outline',legs='unknown',arms='skeletal')
        raw['full_body_visible'] = False
        parsed = parse_person(raw,'p')
        self.assertIsNone(parsed['attributes']['muscularity_visual']['value'])
        self.assertIsNone(parsed['attributes']['body_build']['value'])
        self.assert_held(self.analyze_raw(raw))

    def test_missing_arm_outline_does_not_mask_full_height_head_proportion(self):
        raw = fixtures.person_payload('p')
        raw['coverage']['arms'] = 'unknown'
        raw['representation']['arms'] = 'unknown'
        self.assertEqual(parse_person(raw,'p')['attributes']['head_ratio_class']['value'],'h7')

    def test_evidence_regions_must_be_supported_unique_and_complete(self):
        for regions in ({}, {a:['arms','arms'] for a in unknown_attributes()},
                        {a:['head'] for a in unknown_attributes()}):
            raw=self.observe();raw['evidence_regions']=regions
            with self.assertRaises(ValueError):parse_person(raw,'p')

    def test_provider_failure_does_not_apply_and_preserves_pose_bindings(self):
        client = fixtures.FixtureClient()
        with patch.object(client, 'analyze', side_effect=TimeoutError()):
            person = self.run_service(client)['people'][0]
        self.assert_held(person)
        self.assertEqual(person['diagnostic'], 'provider_error')

    def test_invalid_representation_and_detail_rejected(self):
        for field, value in (('representation', {}), ('surface_detail', 'child'),
                             ('representation', {r: 'thin' for r in ('head', 'torso', 'arms', 'legs')})):
            raw = self.observe()
            raw[field] = value
            with self.assertRaises(ValueError):
                parse_person(raw, 'p')
        person_schema = response_schema()['properties']['people']['items']
        self.assertIn('representation', person_schema['required'])
        self.assertIn('surface_detail', person_schema['required'])

    def test_claimed_visible_values_without_evidence_are_not_applied(self):
        raw = fixtures.person_payload('p','muscular')
        for item in raw['attributes'].values():
            item['evidence'] = '   '
        self.assert_held(self.analyze_raw(raw))

    def test_original_child_failure_is_held_with_actual_selection_path(self):
        assets = copy.deepcopy(self.raw['assets'])
        child = copy.deepcopy(assets[0])
        child.update(body_id='child', attributes={**fixtures.attributes(), 'head_ratio_class': 'h3', 'body_build': 'small_frame'})
        assets.append(child)
        for axis, value in (('head_ratio_class', 'h4'), ('body_build', 'small_frame')):
            raw = self.observe()
            raw['attributes'][axis] = dict(value=value, visibility='uncertain', evidence='isolated hypothesis')
            observation = {**parse_person(raw, 'p'), 'geometry': {}}
            decision = compare_body_shapes(assets, observation, default_body_id='regular')
            self.assertIsNone(decision['auto_body_id'])
            self.assertEqual(decision['suggested_body_id'], 'child')
            self.assertEqual(decision['decision'], 'suggest_only')

    def test_projection_hypotheses_cannot_disagree_on_applied_body(self):
        from src.experimental.body_matching.selection import choose_body
        from src.experimental.body_matching.catalog import Catalog
        from src.experimental.body_matching.schema import GEOMETRY_VERSION
        observation = {**parse_person(self.observe(), 'p'), 'geometry': {
            'shoulder_torso': 1.0, 'left_leg_torso': 2.0}}
        assets = copy.deepcopy(self.raw['assets'][:2])
        poses = [dict(pose_id='pose-'+str(i), view='front', pose_sha256='a'*64) for i in range(2)]
        for index, asset in enumerate(assets):
            asset['projections'] = [{**pose, 'geometry_version':GEOMETRY_VERSION,
                'ratios':{'shoulder_torso':1.0 if i==index else 1.8,
                          'left_leg_torso':2.0 if i==index else 3.6}}
                for i,pose in enumerate(poses)]
        catalog = Catalog('fixture','a'*64,'regular',tuple(assets),())
        result = choose_body(catalog,observation,poses)
        self.assertIsNone(result['auto_body_id'])
        self.assertIn('projection_hypotheses_disagree',result['reason_codes'])

    def test_shadow_never_applies_an_otherwise_accepted_body(self):
        person = self.analyze_raw(fixtures.person_payload('p','muscular'),mode='shadow')
        self.assertEqual(person['auto_body_id'],'muscular')
        self.assertIsNone(person['applied_body_id'])
        self.assertFalse(person['render_required'])

    def test_noncontradictory_tied_projection_does_not_veto_clear_hypothesis(self):
        from src.experimental.body_matching.selection import choose_body
        from src.experimental.body_matching.catalog import Catalog
        observation={**parse_person(self.observe(),'p'),'geometry':{'shoulder_torso':1.,'left_leg_torso':2.}}
        assets=copy.deepcopy(self.raw['assets'][:2])
        poses=[dict(pose_id='pose-'+str(i),view='front',pose_sha256='a'*64) for i in range(2)]
        for index,asset in enumerate(assets):
            asset['projections']=[{**pose,'ratios':{'shoulder_torso':1. if i==1 or index==0 else 1.8,
                                                 'left_leg_torso':2. if i==1 or index==0 else 3.6}}
                for i,pose in enumerate(poses)]
        result=choose_body(Catalog('fixture','a'*64,'regular',tuple(assets),()),observation,poses)
        self.assertEqual(result['auto_body_id'],'regular')
        self.assertNotIn('projection_hypotheses_disagree',result['reason_codes'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
