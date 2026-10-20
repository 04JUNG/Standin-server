"""Neutral isolated anatomical inventory; independent of earlier categorical answers."""
from pathlib import Path
import sys,os,json,time
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)) and any(s in str(args[0]) for s in ('user-feedback-eval','evaluation-only','/gt/','visual-review')):raise RuntimeError('Feedback blocked')
sys.addaudithook(guard)
os.environ['HF_HOME']=str(WORK/'outputs/04j-289-auto-multicut-experiment/model-cache');os.environ['HF_HUB_OFFLINE']='1'
from mlx_vlm import load,generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
MODEL='mlx-community/Qwen3-VL-8B-Instruct-4bit';model,processor=load(MODEL);cfg=load_config(MODEL)
PROMPT='''Describe the single drawn human in this crop. Identify visible clothes/anatomy, especially in the LOWER HALF of the image, before classifying. A trouser waistband, rear pockets and two trouser legs are human pelvis/legs even when the feet are clipped. Sleeves ending in hands are arms. Do not infer hidden anatomy. For a profile face, use the direction the NOSE points in image coordinates, independently from the torso. Return JSON only: {"lower_half_description":"one short sentence naming actual visible shapes","lowest_anatomy":"head/chest/waist/hips/thighs/knees/shins/feet","leg_evidence":"specific visible trouser or leg cue, or none","posture":"standing/sitting/kneeling/crouching/lying/unknown","head_heading":"front/front_left/front_right/left/right/back_left/back_right/back/unknown","head_evidence":"nose/eyes/back of head cue"}.'''
for item in json.loads((ROOT/'inputs.json').read_text()):
 folder=ROOT/item['cut_id']
 for p in sorted(folder.glob('person_*-structure.json')):
  i=json.loads(p.read_text())['person_index'];out=folder/f'person_{i}-anatomy.json'
  if out.exists():continue
  begin=time.monotonic();raw=generate(model,processor,apply_chat_template(processor,cfg,PROMPT,num_images=1),[str(folder/f'person_{i}-crop.png')],max_tokens=350,temperature=0.,verbose=False).text
  try:a=json.loads(raw[raw.index('{'):raw.rindex('}')+1])
  except Exception:a={}
  out.write_text(json.dumps(dict(image_sha256=item['image_sha256'],prompt=PROMPT,analysis=a,raw=raw,elapsed_s=time.monotonic()-begin,manual_inputs=False),ensure_ascii=False,indent=2));print(item['cut_id'],i,a,flush=True)
