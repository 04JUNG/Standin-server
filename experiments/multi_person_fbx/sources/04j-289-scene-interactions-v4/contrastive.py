"""Independent local contrastive evidence on original actor crops."""
from pathlib import Path
import os,json
os.environ['HF_HOME']=str(Path(__file__).resolve().parent.parent/'04j-289-auto-refinement-v2/clip-cache');os.environ['HF_HUB_OFFLINE']='1'
from PIL import Image,ImageOps
import torch
from transformers import CLIPModel,CLIPProcessor
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1];BASE=ROOT.parent/'04j-289-set34-refine-scenes'
os.environ['HF_HOME']=str(ROOT.parent/'04j-289-auto-refinement-v2/clip-cache');os.environ['HF_HUB_OFFLINE']='1'
model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32',use_safetensors=False,local_files_only=True).eval();processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32',local_files_only=True)
CLASSES={
 'surface':{'front':'the front of a person, chest and front of shoulders facing the viewer','back':'the back of a person, rear shoulders and back of shirt facing the viewer','left':'a person seen from the side, facing left','right':'a person seen from the side, facing right'},
 'action':{'phone':'a person looking down and holding a mobile phone with both hands','reach':'a person extending an open hand toward someone walking away','restrain':'a person standing between two other people with arms spread to stop both of them','hip':'a person standing with hands on hips facing someone','neutral':'a person standing upright with arms resting down','away':'a person walking away viewed from behind holding a bag'},
 'range':{'full':'a whole person whose hips and two legs and feet are drawn','half':'a partial person whose head torso and arms are drawn without visible legs','bust':'a cropped head neck and shoulder without arms or hands'}
}
with torch.inference_mode():
 texts={}
 for group,labels in CLASSES.items():
  enc=processor(text=['A black and white storyboard sketch of '+v+'.' for v in labels.values()],return_tensors='pt',padding=True);z=model.get_text_features(**enc);z=z if isinstance(z,torch.Tensor) else z.pooler_output;texts[group]=z/z.norm(dim=-1,keepdim=True)
 for item in json.loads((ROOT/'inputs.json').read_text()):
  image=Image.open(item['image_path']).convert('RGB');people=json.loads((BASE/item['cut_id']/'people.json').read_text())['people'];rows=[]
  for p in people:
   bb=p['bbox'];crop=image.crop(tuple(round(x) for x in bb));views=[]
   for im in (crop,ImageOps.grayscale(crop).convert('RGB')):
    canvas=Image.new('RGB',(max(im.size),)*2,'white');canvas.paste(im,((canvas.width-im.width)//2,(canvas.height-im.height)//2));views.append(canvas)
   z=model.get_image_features(**processor(images=views,return_tensors='pt'));z=z if isinstance(z,torch.Tensor) else z.pooler_output;z=z/z.norm(dim=-1,keepdim=True);rankings={}
   for group,labels in CLASSES.items():
    scores=(z@texts[group].T).mean(0).tolist();rankings[group]=sorted([dict(label=k,similarity=v) for k,v in zip(labels,scores)],key=lambda x:-x['similarity'])
   rows.append(dict(person_index=p['person_index'],rankings=rankings));print(item['cut_id'],p['person_index'],{k:v[:2] for k,v in rankings.items()},flush=True)
  (ROOT/item['cut_id']/'contrastive.json').write_text(json.dumps(dict(image_sha256=item['image_sha256'],classes=CLASSES,people=rows),indent=2))
