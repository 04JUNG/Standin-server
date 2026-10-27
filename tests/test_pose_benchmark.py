"""Synthetic regression contracts; no source image/GT or provider is bundled."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from src.schema import COCO17
from standin_eval.pose_dataset import import_pose_suite, load_pose_dataset
from standin_eval.pose_benchmark import POLICY, evaluate_pose_predictions, run_pose_benchmark
from standin_eval.util import read_json, read_jsonl, sha256_file, write_json, write_jsonl


def source_fixture(root):
    suite = root/'suite'
    selected = suite/'set-01'
    (selected/'images').mkdir(parents=True)
    (selected/'gt').mkdir()
    cuts = []
    for ci, n in enumerate((2, 1)):
        cid = 'cut_' + str(ci)
        image = selected/'images'/(cid+'.png')
        Image.new('RGB',(200,100),(100+ci,200,255)).save(image)
        people = []
        for i in range(n):
            kp = [[None,None] for _ in range(17)]
            vis = [0]*17
            if ci == 0:
                kp[5], kp[6] = [20+i*100,20],[40+i*100,20]
                vis[5] = vis[6] = 2
            kp[7],vis[7] = [20+i*100,50],1
            people.append(dict(person_id=i,keypoints=kp,visibility=vis,review_status='draft'))
        gt = dict(schema_version='standin-coco17-gt-draft-v1',keypoint_names=COCO17,
            image=dict(relative_path='../images/'+image.name,sha256=sha256_file(image),width=200,height=100),
            coordinate_system=dict(unit='pixel',origin='top_left',x_direction='right',y_direction='down',left_right='subject_anatomical'),
            people=people,human_verified=False,evaluation_ready=False)
        write_json(selected/'gt'/(cid+'.json'),gt)
        cuts.append(dict(id=cid,image_path='images/'+image.name,image_sha256=sha256_file(image),width=200,height=100,
            person_count=n,gt_path='gt/'+cid+'.json',gt_status='draft_unverified',
            metadata_path='../../must-not-be-opened',source_paths={'image_path':'/never/open/historical/path'}))
    # A physically retained excluded image must never reenter the active set.
    Image.new('RGB',(10,10),'black').save(selected/'images'/'removed.png')
    write_json(suite/'index.json',dict(schema_version='standin-vlm-pose-suite-v1',sets=[dict(set=1,manifest='set-01/dataset.json',active_cut_count=2)],excluded_cut_ids=['removed']))
    write_json(selected/'dataset.json',dict(schema_version='standin-vlm-pose-dataset-v1',dataset_id='synthetic-pose',
        selection_policy=dict(cuts=2,people=3),cuts=cuts,ground_truth=dict(human_verified=False,evaluation_ready=False),revision_history=[]))
    return suite


def observations(cuts,persons,delta=0):
    return [dict(cut_id=c['cut_id'],image_sha256=c['image_sha256'],status='ok',execution_kind='gt_contract_fixture',
        requested_backend={'pose':'fixture'},actual_backend={'pose':'fixture'},vlm_count=c['person_count'],elapsed_ms=0,
        people=[dict(keypoints=[[x+delta,y] if x is not None and y is not None else [None,None] for x,y in p['keypoints']])
                for p in persons if p['cut_id']==c['cut_id']]) for c in cuts]


class PoseDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = source_fixture(self.root)
        self.output = self.root/'dataset'

    def do_import(self):
        return import_pose_suite(self.source,self.output,source_commit='a'*40,usage_authorized=True)

    def test_manifest_only_selection_and_permissions_without_fake_human_approval(self):
        frozen,cuts,persons = self.do_import()
        self.assertEqual(frozen['counts'],dict(cuts=2,persons=3,visible_joints=4,occluded_estimated_joints=3))
        self.assertNotIn('removed',{c['cut_id'] for c in cuts})
        self.assertEqual(frozen['usage_authorization']['evaluation'],'allow')
        self.assertTrue(all(not c['human_verified'] and not c['evaluation_ready'] for c in cuts))
        self.assertEqual(persons[0]['keypoints'][0],[None,None])
        self.assertFalse(any('bbox_xyxy' in p or 'eligible' in p for p in persons))

    def test_authorization_is_required_but_does_not_change_source_flags(self):
        with self.assertRaisesRegex(ValueError,'authorization'):
            import_pose_suite(self.source,self.output,source_commit='a'*40)

    def test_source_or_normalized_gt_mutation_breaks_seal(self):
        self.do_import()
        p = self.source/'set-01/gt/cut_0.json'
        p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'source changed'):
            load_pose_dataset(self.output)

    def test_excluded_duplicate_and_path_escape_rejected(self):
        p = self.source/'set-01/dataset.json'
        data = read_json(p); data['cuts'][0]['id']='removed';write_json(p,data)
        with self.assertRaisesRegex(ValueError,'excluded'):
            self.do_import()
        data['cuts'][0]['id']='cut_0';data['cuts'][0]['image_path']='../../outside.png';write_json(p,data)
        with self.assertRaisesRegex(ValueError,'escapes'):
            self.do_import()

    def test_keypoint_order_and_fixed_person_denominator(self):
        p=self.source/'set-01/gt/cut_0.json'
        data=read_json(p);data['keypoint_names']=list(reversed(COCO17));write_json(p,data)
        with self.assertRaisesRegex(ValueError,'COCO17'):
            self.do_import()
        data['keypoint_names']=COCO17;data['people'].pop();write_json(p,data)
        with self.assertRaisesRegex(ValueError,'denominator'):
            self.do_import()


class PoseScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=source_fixture(self.root)
        self.dataset=self.root/'dataset'
        self.frozen,self.cuts,self.persons=import_pose_suite(self.source,self.dataset,source_commit='a'*40,usage_authorized=True)

    def score(self,rows,policy=None):
        return evaluate_pose_predictions(self.frozen,self.cuts,self.persons,rows,policy=policy)

    def test_oracle_is_only_fixture_and_occlusion_is_separate(self):
        report,cuts,persons,joints=self.score(observations(self.cuts,self.persons))
        self.assertEqual(report['visible_primary']['correct'],4)
        self.assertEqual(report['occluded_estimates']['denominator'],3)
        self.assertEqual(report['score_status'],'PROVISIONAL_DRAFT_GT')
        self.assertEqual(report['decision'],'INCONCLUSIVE')
        self.assertEqual(report['actual_model_cuts'],0)
        self.assertEqual(report['latency']['measured'],0)
        self.assertEqual(report['person_association']['unknown'],1)

    def test_no_predictions_keep_every_cut_person_and_joint_unknown(self):
        report,cuts,persons,joints=self.score([])
        self.assertEqual((len(cuts),len(persons),len(joints)),(2,3,7))
        self.assertEqual(report['visible_primary']['unknown'],4)
        self.assertIsNone(report['visible_primary']['correct_rate'])
        self.assertIsNone(report['visible_primary']['mean_error_px_on_comparable_joints'])

    def test_swapped_person_order_is_matched_one_to_one(self):
        obs=observations(self.cuts,self.persons);obs[0]['people'].reverse()
        report,_,people,_=self.score(obs)
        self.assertEqual(report['visible_primary']['correct'],4)
        self.assertEqual([p['prediction_index'] for p in people[:2]],[1,0])

    def test_duplicate_prediction_cannot_satisfy_two_gt_people(self):
        obs=observations(self.cuts,self.persons);obs[0]['people'][1]=copy.deepcopy(obs[0]['people'][0])
        report,cuts,_,_=self.score(obs)
        self.assertEqual(report['visible_primary']['correct'],2)
        self.assertEqual(report['visible_primary']['incorrect'],2)
        self.assertEqual(cuts[0]['false_positives'],1)

    def test_null_is_not_zero_and_failure_does_not_shrink_denominator(self):
        obs=observations(self.cuts,self.persons)
        for person in obs[0]['people']:person['keypoints']=[[None,None] for _ in range(17)]
        report,*_=self.score(obs)
        self.assertEqual(report['visible_primary']['incorrect'],4)
        obs[0]['status']='timeout'
        report,*_=self.score(obs)
        self.assertEqual(report['visible_primary']['denominator'],4)
        self.assertEqual(report['visible_primary']['incorrect'],4)

    def test_explicit_diagonal_error_and_score_threshold(self):
        report,*_=self.score(observations(self.cuts,self.persons,delta=22))
        self.assertEqual(report['visible_primary']['incorrect'],4)
        self.assertAlmostEqual(report['visible_primary']['mean_error_px_on_comparable_joints'],22)
        obs=observations(self.cuts,self.persons)
        for row in obs:
            for person in row['people']:person['scores']=[0]*17
        report,*_=self.score(obs)
        self.assertEqual(report['visible_primary']['correct'],4)
        report,*_=self.score(obs,{**POLICY,'prediction_score_threshold':.5})
        self.assertEqual(report['visible_primary']['incorrect'],4)

    def test_labeled_null_gt_joint_is_unknown_without_denominator_drop(self):
        self.persons[0]['keypoints'][5]=[None,None]
        report,*_=self.score(observations(self.cuts,self.persons))
        self.assertEqual(report['visible_primary']['denominator'],4)
        self.assertEqual(report['visible_primary']['unknown'],1)

    def test_image_mismatch_mock_fallback_and_invalid_predictions(self):
        obs=observations(self.cuts,self.persons);obs[0]['image_sha256']='b'*64
        report,*_=self.score(obs);self.assertEqual(report['decision'],'FAIL')
        obs=observations(self.cuts,self.persons);obs[0]['requested_backend']={'pose':'rtmlib'};obs[0]['actual_backend']={'pose':'mock'}
        report,*_=self.score(obs);self.assertEqual(report['decision'],'FAIL');self.assertEqual(report['actual_model_cuts'],0)
        obs=observations(self.cuts,self.persons);obs[0]['people'][0]['keypoints']=[[float('nan'),0]]*17
        report,cuts,_,_=self.score(obs);self.assertEqual(cuts[0]['status'],'error');self.assertEqual(report['visible_primary']['incorrect'],4)

    def test_excluded_duplicate_predictions_and_bad_policy_rejected(self):
        obs=observations(self.cuts,self.persons)
        with self.assertRaises(ValueError):self.score(obs+obs)
        with self.assertRaises(ValueError):self.score([dict(obs[0],cut_id='removed')])
        with self.assertRaises(ValueError):self.score(obs,{**POLICY,'pck_image_diagonal_fraction':float('nan')})

    def test_explicit_model_identity_is_bound_and_mismatch_is_not_success(self):
        obs=observations(self.cuts,self.persons)
        for row in obs:row['inference_metadata']={'pose_model_version':'model-A'}
        policy={**POLICY,'expected_inference_identity':{'pose_model_version':'model-A'}}
        report,*_=self.score(obs,policy)
        self.assertEqual(report['model_identity_states'],{'pass':2})
        obs[0]['inference_metadata']['pose_model_version']='model-B'
        report,*_=self.score(obs,policy)
        self.assertEqual(report['decision'],'FAIL')
        self.assertIn('inference_identity_mismatch:cut_0',report['hard_failures'])
        with self.assertRaises(ValueError):self.score(obs,{**POLICY,'expected_inference_identity':{'pose_model_version':None}})

    def test_time_budget_has_fixed_cut_denominator_and_excludes_fixture(self):
        obs=observations(self.cuts,self.persons)
        obs[0].update(execution_kind='recorded_model',requested_backend={'pose':'rtmlib'},actual_backend={'pose':'rtmlib'},elapsed_ms=20)
        report,*_=self.score(obs,{**POLICY,'latency_budget_ms':10})
        timing=report['latency']['within_budget']
        self.assertEqual((timing['denominator'],timing['incorrect'],timing['unknown']),(2,1,1))
        self.assertEqual(report['latency']['measured'],1)

    def test_http_default_is_off_and_plan_exists_before_any_request(self):
        with patch('standin_eval.pose_benchmark._request_json') as transport:
            with self.assertRaises(ValueError):
                run_pose_benchmark(self.dataset,self.root/'blocked',target='http://fixture')
            transport.assert_not_called()
        output=self.root/'http-run'
        def fake_request(url,**kwargs):
            self.assertTrue((output/'plan.json').exists())
            return 200,b'{}',dict(people=[],vlm_count=0,inference_metadata=dict(vlm_provider='mock',pose_backend='mock'))
        with patch('standin_eval.pose_benchmark._request_json',fake_request):
            report=run_pose_benchmark(self.dataset,output,target='http://fixture',allow_http=True,timeout_seconds=1,
                requested_backend={'vlm':'mock','pose':'mock'})
        self.assertEqual(report['cut_statuses'],{'ok':2})
        self.assertEqual(report['visible_primary']['incorrect'],4)
        self.assertEqual(report['actual_model_cuts'],0)
        self.assertEqual(len(read_jsonl(output/'persons.jsonl')),3)

    def test_midrun_source_change_fails_without_losing_rows(self):
        output=self.root/'changed'
        def fake_request(url,**kwargs):
            if 'cut_0' not in str(getattr(fake_request,'done','')):
                p=self.source/'set-01/gt/cut_0.json';p.write_text(p.read_text()+' ');fake_request.done='cut_0'
            return 200,b'{}',dict(people=[],inference_metadata={})
        with patch('standin_eval.pose_benchmark._request_json',fake_request):
            report=run_pose_benchmark(self.dataset,output,target='http://fixture',allow_http=True,timeout_seconds=1,
                requested_backend={'vlm':'mock','pose':'mock'})
        self.assertEqual(report['decision'],'FAIL')
        self.assertEqual(len(read_jsonl(output/'cuts.jsonl')),2)
        self.assertEqual(len(read_jsonl(output/'persons.jsonl')),3)
        self.assertIn('source_changed_during_execution',report['hard_failures'])


if __name__=='__main__': unittest.main()
