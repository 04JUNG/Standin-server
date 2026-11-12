"""Compose the authored muscle distance mask over flesh, preserving identical ink coverage."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--preview',action='store_true');a=p.parse_args()
style=json.loads(Path(__file__).with_name('skin_style.json').read_text())
for gender in ['male','female']:
 mask=np.array(Image.open(a.assets/f'{gender}-masks.png'))[:,:,2].astype(np.float32)/255
 coverage=np.clip((mask-(1-.0008/.006))/.13+.5,0,1)*.78
 base=np.array(style['flesh']['reference_sample_rgb']);line=np.array([133,139,145])
 color=base*(1-coverage[:,:,None])+line*coverage[:,:,None]
 image=Image.fromarray(np.clip(color,0,255).astype(np.uint8));image.save(a.assets/f'{gender}-flesh-muscle.png')
 if a.preview:image.resize((2048,2048),Image.Resampling.LANCZOS).save(a.assets/f'{gender}-flesh-muscle-preview.png',optimize=True)
