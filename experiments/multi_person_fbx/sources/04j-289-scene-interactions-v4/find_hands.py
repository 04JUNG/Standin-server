"""Uniform sliding-window contrastive proposals for detached open hands."""
from pathlib import Path
import os,json
os.environ['HF_HOME']=str(Path(__file__).resolve().parent.parent/'04j-289-auto-refinement-v2/clip-cache');os.environ['HF_HUB_OFFLINE']='1'
import numpy as np,torch
from PIL import Image
from transformers import CLIPModel,CLIPProcessor
ROOT=Path(__file__).resolve().parent
model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32',use_safetensors=False,local_files_only=True).eval();processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32',local_files_only=True)
captions=['a close-up of a human hand with open fingers and a wrist','a human head or face or hair','a sleeve or arm with no hand','a human torso with clothing','a mobile phone held in both hands','a door or a bag or furniture','printed Korean words and a speech balloon'];texts=['A black and white line drawing of '+t+'.' for t in captions]
def overlap(a,b):
 lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);inter=np.maximum(hi-lo,0).prod();return inter/max(1,min(np.prod(np.subtract(a[2:],a[:2])),np.prod(np.subtract(b[2:],b[:2]))))
with torch.inference_mode():
 z=model.get_text_features(**processor(text=texts,return_tensors='pt',padding=True));z=z if isinstance(z,torch.Tensor) else z.pooler_output;text=z/z.norm(dim=-1,keepdim=True)
 for item in json.loads((ROOT/'inputs.json').read_text()):
  folder=ROOT/item['cut_id'];original=Image.open(item['image_path']).convert('RGB');vp=json.loads((folder/'identities.json').read_text())['analysis_viewport_bbox'];scene=original.crop(vp);w,h=scene.size;boxes=[]
  for fraction in (.22,.33,.45):
   side=round(w*fraction)
   for y in np.linspace(0,max(0,h-side),7):
    for x in np.linspace(0,max(0,w-side),7):boxes.append([int(x),int(y),min(w,int(x)+side),min(h,int(y)+side)])
  proposals=[]
  for offset in range(0,len(boxes),24):
   bb=boxes[offset:offset+24];enc=processor(images=[scene.crop(b) for b in bb],return_tensors='pt');z=model.get_image_features(**enc);z=z if isinstance(z,torch.Tensor) else z.pooler_output;z=z/z.norm(dim=-1,keepdim=True);scores=(z@text.T).numpy()
   for b,ss in zip(bb,scores):
    margin=float(ss[0]-max(ss[1:]));
    if margin>0:proposals.append(dict(bbox=(np.array(b)+np.tile(vp[:2],2)).tolist(),score=float(ss[0]),margin=margin,class_scores=ss.tolist()))
  picked=[]
  for p in sorted(proposals,key=lambda x:-x['margin']):
   if all(overlap(p['bbox'],q['bbox'])<.45 for q in picked):picked.append(p)
   if len(picked)==4:break
  (folder/'open-hand-proposals.json').write_text(json.dumps(dict(image_sha256=item['image_sha256'],captions=captions,windows=len(boxes),proposals=picked),indent=2));print(item['cut_id'],picked,flush=True)
