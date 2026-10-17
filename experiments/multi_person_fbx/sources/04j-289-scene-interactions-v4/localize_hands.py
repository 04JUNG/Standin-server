from pathlib import Path
import os,json,time,hashlib
from PIL import Image
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1];os.environ['HF_HOME']=str(WORK/'outputs/04j-289-auto-multicut-experiment/model-cache');os.environ['HF_HUB_OFFLINE']='1'
from mlx_vlm import load,generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
model,processor=load('mlx-community/Qwen3-VL-8B-Instruct-4bit');cfg=load_config('mlx-community/Qwen3-VL-8B-Instruct-4bit')
prompt='''Find the largest OPEN HUMAN HAND in this crop. Only locate a hand with fingers and palm, not a face, sleeve, bag or rectangle. Return one JSON object with visible (boolean), open_palm (boolean), x and y (separate numbers for the palm center in THIS crop normalized 0..1000), bbox (four numbers for fingers+palm, excluding most sleeve), wrist_x and wrist_y (separate normalized numbers), description (describe the visible fingers and palm). Do not include coordinates for any non-hand object. If no open hand is drawn, visible=false.'''
for item in json.loads((ROOT/'inputs.json').read_text()):
 folder=ROOT/item['cut_id'];proposals=json.loads((folder/'open-hand-proposals.json').read_text())['proposals'];image=Image.open(item['image_path']).convert('RGB');rows=[]
 for j,p in enumerate(proposals):
  if p['margin']<.02:continue
  b=p['bbox'];pad=.8*max(b[2]-b[0],b[3]-b[1]);cropbox=[max(0,round(b[0]-pad)),max(0,round(b[1]-pad)),min(image.width,round(b[2]+pad)),min(image.height,round(b[3]+pad))];im=image.crop(cropbox);im=im.resize(tuple(round(v*1000/max(im.size)) for v in im.size));path=folder/f'hand-proposal-{j}.png';im.save(path);raw=generate(model,processor,apply_chat_template(processor,cfg,prompt,num_images=1),[str(path)],max_tokens=700,temperature=0.,verbose=False).text
  try:a,_=json.JSONDecoder().raw_decode(raw[raw.index('{'):])
  except Exception:a={}
  rows.append(dict(proposal_index=j,crop_bbox=cropbox,analysis=a,raw=raw,prompt=prompt));print(item['cut_id'],j,a,flush=True)
 (folder/'localized-hands.json').write_text(json.dumps(dict(image_sha256=item['image_sha256'],hands=rows,manual_inputs=False),ensure_ascii=False,indent=2))
