"""Full-scene relations and owned limbs, freshly inferred locally, no feedback labels."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,sys,hashlib
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)) and any(s in str(args[0]) for s in ('feedback-eval','evaluation-only','/gt/','visual-review')):raise RuntimeError('Feedback is evaluation-only')
sys.addaudithook(guard)
os.environ['HF_HOME']=str(WORK/'outputs/04j-289-auto-multicut-experiment/model-cache');os.environ['HF_HUB_OFFLINE']='1'
from mlx_vlm import load,generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
MODEL='mlx-community/Qwen3-VL-8B-Instruct-4bit';model,processor=load(MODEL);config=load_config(MODEL)
SCENE='''Describe the physical interaction among the labeled people in this storyboard viewport. P labels mark heads, not body boundaries. Ignore speech text. First explain in one sentence what the people are physically doing. Then return JSON: {"scene_action":"one sentence","people":[{"person_index":0,"action":"visible action including arms","role":"physical role in scene","target_person_indices":[]}],"relations":[{"actor":0,"target":1,"type":"looking_at/reaching_toward/restraining/opposing/leaving/none","evidence":"what drawn body or arm establishes this"}]}. Do not leave descriptions empty. Use only actual visible actions; a face direction alone is not a role.'''
ACTOR='''Analyze only P{index}, identified by its red HEAD marker, using the ENTIRE viewport. Do not restrict the actor to a head crop. Trace shoulder, sleeve and hand ownership through occlusion. Large detached foreground hands may belong to the same foreground head. Body direction comes from the actual chest/back/shoulder surface, independently of nose direction. A profile head at the image edge with only a narrow shoulder is not evidence of a frontal chest. Find the lowest visible human anatomy from the whole scene, including small background legs. Do not substitute furniture for legs.
Return JSON only: {{"person_index":{index},"torso_heading":"front/front_left/front_right/left/right/back_left/back_right/back/unknown","torso_surface":"front/back/left_side/right_side/oblique_front/oblique_back/not_visible","torso_evidence":"visible shoulder/chest/back cue","torso_confidence":"high/medium/low","head_heading":"same direction choices","head_pitch":"level/down/up/unknown","head_evidence":"visible nose/eyes/hair cue","head_confidence":"high/medium/low","posture":"standing/sitting/kneeling/crouching/unknown","torso_lean":"upright/forward_bent/side_bent/unknown","lowest_visible_anatomy":"head/chest/waist/hips/thighs/knees/shins/feet/unknown","visible_regions":[],"body_scope":"full/half/bust/head/unknown","scope_evidence":"specific lowest visible garment/anatomy","body_bbox":[x0,y0,x1,y1],"arms":[{{"shoulder_side_in_image":"left/right/unknown","shoulder":[x,y],"elbow":[x,y],"hand":[x,y],"visible":true,"hand_state":"open/closed/holding/unknown","action":"reaching/restraining/raised/phone/relaxed/unknown","target_person_index":null,"confidence":"high/medium/low"}}],"search_query_ko":"short physical pose action"}}.
All positions are viewport normalized 0..1000. Use null for any genuinely invisible joint; do not invent it. Include only arms/legs belonging to P{index}. Full scope when human thighs/knees/shins/feet are visibly present. Half scope when any owned elbow/forearm/hand needs to be shown even when separated from the head. A clipped head does not mean its visible distant hand should be dropped.'''
for stage in ('scene','actor'):
 for item in json.loads((ROOT/'inputs.json').read_text()):
  folder=ROOT/item['cut_id'];assert hashlib.sha256(Path(item['image_path']).read_bytes()).hexdigest()==item['image_sha256'];ident=json.loads((folder/'identities.json').read_text());targets=[None] if stage=='scene' else ident['people']
  for person in targets:
   i=None if person is None else person['person_index'];path=folder/('scene-analysis.json' if i is None else f'person_{i}-analysis.json')
   if path.exists():continue
   prompt=SCENE if i is None else ACTOR.format(index=i);begin=time.monotonic();raw=generate(model,processor,apply_chat_template(processor,config,prompt,num_images=1),[str(folder/'scene-context.png')],max_tokens=1900 if i is None else 1300,temperature=0.,repetition_penalty=1.10,verbose=False).text
   try:a=json.loads(raw[raw.index('{'):raw.rindex('}')+1]);error=None
   except Exception as exc:a=None;error=str(exc)
   path.write_text(json.dumps(dict(image_sha256=item['image_sha256'],started_at=datetime.now(timezone.utc).isoformat(),model=MODEL,prompt=prompt,raw=raw,analysis=a,error=error,elapsed_s=time.monotonic()-begin,analysis_viewport_bbox=ident['analysis_viewport_bbox'],inference_image_sha256=hashlib.sha256((folder/'scene-context.png').read_bytes()).hexdigest(),manual_inputs=False),ensure_ascii=False,indent=2));print(stage,item['cut_id'],i,a,'ERROR',error,flush=True)
