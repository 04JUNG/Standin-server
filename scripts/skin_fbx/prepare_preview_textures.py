"""One-time bounded preview derivatives; final FBXs retain original 4096 PNGs."""
from pathlib import Path
from PIL import Image
root=Path(__file__).resolve().parents[2]/'out/skin-fbx-csp'
for gender in ('male','female'):
 for skin in ('gray-guide','flesh','flesh-guide','muscle'):
  with Image.open(root/'assets'/f'{gender}-{skin}.png') as image:
   assert image.mode=='RGB' and image.size==(4096,4096)
   image.resize((2048,2048),Image.Resampling.LANCZOS).save(root/'release'/f'{gender}-{skin}-preview.png',optimize=True)
