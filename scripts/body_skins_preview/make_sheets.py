"""Assemble actual WebGL render captures into labeled review sheets."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();root=a.output;r=root/'renders';font='/System/Library/Fonts/AppleSDGothicNeo.ttc'
def f(n):return ImageFont.truetype(font,n)
bg='#ebede6';ink='#24473b';muted='#71816e'
def title(im,main,sub):
 d=ImageDraw.Draw(im);d.text((44,26),main,fill=ink,font=f(36));d.text((46,75),sub,fill=muted,font=f(19))
skins=['기본','회색 가이드선','살색','살색 가이드선','근육선 강조']
im=Image.new('RGB',(1800,2050),bg);title(im,'STANDIN / 교체형 스킨 5종','실제 남·여 기본형 · 같은 자세 / 카메라 / 조명 · 2026.10.28 초안')
for i,(model,label) in enumerate([('male','남성 기본형 0.7.2'),('female','여성 기본형 0.3.4')]):
 y=125+i*950;d=ImageDraw.Draw(im);d.text((45,y),label,fill=ink,font=f(24));tile=Image.open(r/(model+'-comparison.png')).convert('RGB').resize((1800,900),Image.Resampling.LANCZOS);im.paste(tile,(0,y+35))
 d=ImageDraw.Draw(im)
 for j,s in enumerate(skins):d.text((273+j*313.4,y+878),s,fill=ink,font=f(23),anchor='mm')
im.save(root/'comparison.png')
for model in ['male','female']:
 im=Image.new('RGB',(1800,1230),bg);title(im,'STANDIN / 표면 선 확대',('남성 0.7.2' if model=='male' else '여성 0.3.4')+' · 가이드선 / 주요 근육 경계 · 화면 오버레이가 아닌 UV 재질')
 for j,skin in enumerate(['gray-guide','muscle']):im.paste(Image.open(r/f'{model}-{skin}-detail.png').convert('RGB'),(j*900,120))
 im.save(root/(model+'-line-details.png'))
im=Image.new('RGB',(1800,1680),bg);title(im,'STANDIN / 포즈·표면 추종 검수','원본 전체 가중치로 계산한 Blender 표면 · 굽힘 / 앉기 / 진단용 상체 폭 변형')
for i,model in enumerate(['male','female']):
 for j,(pose,skin,label) in enumerate([('bent','gray-guide','팔 굽힘 · 가이드선'),('seated','muscle','앉은 자세 · 근육선'),('broader','gray-guide','상체 폭 변형 · 가이드선')]):
  tile=Image.open(r/f'{model}-{pose}-{skin}.png').convert('RGB').resize((600,733),Image.Resampling.LANCZOS);y=125+i*770;im.paste(tile,(j*600,y));ImageDraw.Draw(im).text((j*600+300,y+726),('남성' if i==0 else '여성')+' / '+label,fill=ink,font=f(20),anchor='mm')
im.save(root/'pose-validation.png')
for model in ['male','female']:
 for skin in ['gray-guide','muscle']:
  im=Image.new('RGB',(2000,800),bg);title(im,f'STANDIN / {model} · '+('가이드선' if skin=='gray-guide' else '근육선'),'정면 / 45도 / 측면 / 후면')
  for j,view in enumerate(['front','quarter','side','back']):im.paste(Image.open(r/f'{model}-{skin}-{view}.png').convert('RGB').resize((500,611),Image.Resampling.LANCZOS),(j*500,135))
  im.save(root/f'{model}-{skin}-four-views.png')
print('Review sheets saved')
