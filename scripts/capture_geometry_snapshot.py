"""Offline current-X recapture using archived final slot boxes (not original VLM)."""
import argparse
import json
import os
from pathlib import Path
import sys

# Set before importing config/model. This CLI never constructs a Gemini client.
os.environ.update(POSE_BACKEND='rtmlib',POSE_MODEL_VARIANT='current-x',POSE_STRICT='1',
    SEARCH_ROUTING='0',VLM_PERSON_TAGS='0',OMP_NUM_THREADS='1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.pipeline import Pipeline
from src.pose import build_pose_model
from src.repo import load_entries
from src.config import CFG
from src.schema import VLMAnalysis,BBox,Shot,Action,View,Relationship
from src.body_scope import BodyScope
from src.vlm.client import BaseVLMClient


class FrozenAnalysis(BaseVLMClient):
    current=None
    def analyze(self,image,width,height):
        c=self.current;people=c['people'];boxes=[];scopes=[];visible=[];known=[]
        for p in people:
            boxes.append(BBox(*p['box'],source='saved_final_box') if p.get('box') else None)
            m=(p.get('routing') or {}).get('joint_evidence',{}).get('metrics',{})
            scope=m.get('output_scope');scopes.append(BodyScope(scope) if scope in BodyScope._value2member_map_ else None)
            visible.append(m.get('lower_body_observed') is True);known.append(m.get('lower_body_visibility_known') is True)
        return VLMAnalysis(num_people=len(people),shot=Shot.FULL_HALF,action=Action.OTHER,
            view=View.FRONT,relationship=Relationship.SOLO if len(people)<=1 else Relationship.TALKING,
            approx_boxes=boxes,body_scopes=scopes,lower_body_visible=visible,lower_body_visibility_known=known,
            raw={'body_scopes':True,'source':'archived_final_boxes_not_original_vlm'})


class Capture(Pipeline):
    slots=[]
    def _route_slots(self,image,width,height,vlm,slots):
        self.slots=slots
        return {'decisions':{},'summary':{'describe':{'status':'not_requested','requested_people':0}}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','images','db','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();source=args.reference;raw=json.loads(source.read_text())
    images=args.images;entries=load_entries(str(args.db))
    vlm=FrozenAnalysis();model=build_pose_model();pipeline=Capture(entries,vlm_client=vlm,pose_model=model)
    CFG.search_routing_enabled=True
    output={'evaluation_kind':'actual-current-X-recapture-with-archived-final-boxes',
        'metadata_source':str(source),'extra_vlm_calls':0,'pose_variant':'current-x',
        'limitations':['final boxes are NOT original VLM boxes; not an exact replay of previous live pipeline',
            'search before/after must share these same recaptured coordinates/masks',
            'neutral action/view and SOLO/TALKING relationship are offline fixtures, not original VLM metadata',
            'no new Gemini or Human-Art cascade'],'cases':[]}
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    for c in raw['cases']:
        vlm.current=c
        path=images/c['image']
        with Image.open(path) as im:width,height=im.size
        result=pipeline.process_cut(str(path),width,height)
        slots=sorted(pipeline.slots,key=lambda s:(s.result_box.x1 if s.result_box else float('inf'),s.slot_id))
        case={'test_id':c['test_id'],'image':c['image'],'size':[width,height],'people':[]}
        for i,(s,descriptor) in enumerate(zip(slots,result.descriptors)):
            e=s.evidence;sk=s.skeleton
            record={'index':i,'slot_id':s.slot_id,'state':s.state,'source':s.skeleton_source,
                'owner_box':s.vlm_box.as_list() if s.vlm_box else None,
                'box':s.result_box.as_list() if s.result_box else None,
                'peer_boxes':[p.vlm_box.as_list() for p in slots if p is not s and p.vlm_box],
                'keypoints':sk.keypoints.tolist() if sk else None,
                'scores':sk.scores.tolist() if sk else None,
                'joint_mask':e.valid_joint_mask.tolist() if e else [False]*17,
                'bone_mask':e.valid_bone_mask.tolist() if e else [False]*12,
                'suspect_limbs':list(e.suspect_limbs) if e else [],
                'coverage':e.coverage_class if e else 'insufficient','reasons':s.reasons,
                'confidence':result.person_confidence[i],
                'confidence_threshold':s.confidence_threshold,
                'baseline':[{'pose_id':p.pose_id,'view':p.view.value,'distance':p.distance,'bvh_path':p.bvh_path} for p in result.person_candidates[i]],
                'refine_allowed':descriptor.refine_allowed,'policy_block':descriptor.output_scope.detected==BodyScope.HEAD or vlm.analyze(None,width,height).relationship.is_entangled}
            case['people'].append(record)
        output['cases'].append(case)
        (out/'capture.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
        print(c['test_id'],[(p['state'],sum(p['bone_mask'][:4]),sum(p['bone_mask'][4:8]),len(p['baseline'])) for p in case['people']],flush=True)


if __name__=='__main__':main()
