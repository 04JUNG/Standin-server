"""Fresh crop/context analysis, independent torso/head and scope/posture passes.
No user feedback, previous headings, pose IDs or requested angles enter prompts.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys,os,time
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1];BASE=ROOT.parent/'04j-289-set34-refine-scenes';LAYOUT=ROOT.parent/'04j-289-set3-layout-v2'
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  p=str(args[0])
  if any(x in p for x in ('user-feedback-eval.json','evaluation-only-draft.json','visual-review.json','/gt/','/gt-overlays/')):raise RuntimeError('Evaluation data prohibited as inference input')
sys.addaudithook(guard)
os.environ['HF_HOME']=str(WORK/'outputs/04j-289-auto-multicut-experiment/model-cache');os.environ['HF_HUB_OFFLINE']='1'
from mlx_vlm import load,generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
MODEL='mlx-community/Qwen3-VL-8B-Instruct-4bit'
COMMON='''Image 1 shows the complete storyboard with ONE target outlined in RED and labeled. Image 2 is a larger crop of that SAME target. Analyze that actor only; other people, speech bubbles and the red annotations are not body parts. Use actual visible anatomy and clothing surfaces; do not assume a person's face and torso point the same way. Left/right mean IMAGE left/right. Return short JSON only.\n'''
DIRECTION=COMMON+'''Judge the TORSO independently of the face: front chest/lapels/sternum/belly versus back shoulder blades/spine/back clothing panel/rear pockets. A visible profile face can coexist with a back-facing torso. Long hair falling down a back is not a face. Do not use the direction of a hand as torso direction.
Return {"torso_surface":"front/back/left_side/right_side/oblique_back/oblique_front/not_visible","torso_heading":"front/front_left/front_right/left/right/back_left/back_right/back/unknown","torso_evidence":"under 18 words naming the visible chest/back/shoulder cue","torso_confidence":"high/medium/low","head_heading":"front/front_left/front_right/left/right/back_left/back_right/back/unknown","head_evidence":"under 15 words","head_confidence":"high/medium/low","head_pitch":"level/down/up/unknown","head_relative_to_torso":"same/turned_left/turned_right/unknown"}.
Front means chest toward CAMERA; back means chest away from CAMERA. front_left means chest partly toward camera and left. back_left means chest mostly away from camera and left; the camera sees the back/side. Only use unknown if evidence is insufficient. Never answer by copying the face direction into torso_heading.'''
STRUCTURE=COMMON+'''Classify the actual BODY SUPPORT and VISIBLE RANGE. Trace hips, thighs and knees, rather than guessing posture from a face or vertical torso. Bent legs drawn forward beside a chair/table can mean sitting. A seated torso can be upright. Furniture legs and chair backs are not human limbs. Do not infer hidden knees or feet. Hair is not a leg. Cropped legs do not prove standing.
Return {"posture":"standing/sitting/kneeling/crouching/lying/bent_over_surface/unknown","support_evidence":"under 18 words","visible_regions":[choose from head,neck,chest,shoulders,upper_arms,forearms,hands,hips,thighs,knees,shins,feet],"body_scope":"full/half/bust/head/unknown","scope_evidence":"under 18 words describing lowest visible human region","torso_lean":"upright/forward_bent/side_bent/unknown","leg_action":"both_grounded/one_leg_raised/crossed/seated_bent/not_visible/unknown","arm_action":"under 18 English words describing visible arms only","head_support":"one_palm/both_palms/forearms_on_surface/none/unknown","search_query_ko":"short Korean posture and visible arm action, no story or emotion"}.
Scope full: human legs/thighs/knees/shins are visible, including legs partly clipped by frame. half: torso/arms near waist/hips, no clear knees. bust: shoulders/chest to head, arms mostly absent. head: head/face/neck only. A person's legs seen from behind still require full. Do not label full merely because you imagine hidden legs.'''
inputs=json.loads((LAYOUT/'inputs.json').read_text());(ROOT/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2));model,processor=load(MODEL);config=load_config(MODEL);font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',24)
# Each pass finishes all actors before the next one; outputs are independently auditable.
for stage,prompt_text in [('direction',DIRECTION),('structure',STRUCTURE)]:
 if stage=='structure':prompt_text=prompt_text.replace(COMMON,'This ONE image is an isolated crop of ONE storyboard actor. Analyze ONLY visible anatomy inside this crop. Never infer legs from a sleeve, hair, furniture or a seated context. A seated upper body with arms and waist but NO human thigh/knee/shin pixels is HALF, not FULL. Return short JSON only.\n')
 for item in inputs:
  cut=item['cut_id'];folder=ROOT/cut;folder.mkdir(exist_ok=True);source=json.loads((BASE/cut/'people.json').read_text());path=Path(item['image_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==item['image_sha256'];image=Image.open(path).convert('RGB');w,h=image.size
  for p in source['people']:
   i=p['person_index'];output=folder/f'person_{i}-{stage}.json'
   if output.exists() and json.loads(output.read_text()).get('image_sha256')==item['image_sha256'] and (stage=='direction' or json.loads(output.read_text()).get('isolated_crop_only')):continue
   x0,y0,x1,y1=p['bbox'];bw,bh=x1-x0,y1-y0;cropbox=[max(0,int(x0-.10*bw)),max(0,int(y0-.10*bh)),min(w,int(x1+.10*bw)),min(h,int(y1+.10*bh))]
   context=image.copy();d=ImageDraw.Draw(context);d.rectangle(p['bbox'],outline='#ee213c',width=max(2,w//250));d.text((max(0,x0),max(0,y0-28)),'TARGET P'+str(i),font=font,fill='#bc1029');scale=max(1,1024/max(context.size));context=context.resize(tuple(round(v*scale) for v in context.size),Image.Resampling.LANCZOS);cp=folder/f'person_{i}-context.png';context.save(cp)
   crop=image.crop(cropbox);scale=1024/max(crop.size);crop=crop.resize(tuple(round(v*scale) for v in crop.size),Image.Resampling.LANCZOS);tp=folder/f'person_{i}-crop.png';crop.save(tp)
   paths=[str(tp)] if stage=='structure' else [str(cp),str(tp)];started=datetime.now(timezone.utc).isoformat();begin=time.monotonic();raw=generate(model,processor,apply_chat_template(processor,config,prompt_text,num_images=len(paths)),paths,max_tokens=650,temperature=0.,repetition_penalty=1.12,verbose=False).text;(folder/f'person_{i}-{stage}-raw.txt').write_text(raw)
   try:analysis=json.loads(raw[raw.index('{'):raw.rindex('}')+1]);error=None
   except Exception as exc:analysis=None;error=str(exc)
   record=dict(cut_id=cut,person_index=i,image_sha256=item['image_sha256'],model=MODEL,stage=stage,prompt=prompt_text,crop_bbox=cropbox,crop_sha256=hashlib.sha256(tp.read_bytes()).hexdigest(),started_at=started,elapsed_s=time.monotonic()-begin,analysis=analysis,error=error,manual_direction=False,manual_scope=False)
   record['isolated_crop_only']=stage=='structure';output.write_text(json.dumps(record,ensure_ascii=False,indent=2));print(stage,cut,i,analysis,'error',error,flush=True)
