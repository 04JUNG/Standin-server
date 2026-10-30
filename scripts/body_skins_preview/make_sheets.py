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
im=Image.new('RGB',(1800,2050),bg);title(im,'STANDIN / 교체형 스킨 5종','실제 남·여 기본형 · 같은 자세 / 카메라 / 조명 · 클튜 기준 선 배치 · 2026.10.29')
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

# Final authored guide layout, with the same unmodified source surface in three views.
for model in ['male','female']:
 im=Image.new('RGB',(2400,1430),bg);title(im,'STANDIN / 클튜 기준 가이드선 0.1.3',('남성' if model=='male' else '여성')+' · T 포즈 정면 / 기본 자세 측면 / T 포즈 후면')
 for view,box in [('front',(0,180,970,1230)),('side',(970,180,460,1230)),('back',(1430,180,970,1230))]:
  file=r/(f'{model}-gray-guide-side.png' if view=='side' else f'{model}-tpose-gray-guide-{view}.png')
  tile=Image.open(file).convert('RGB');crop=(330,55,530,1015) if view=='side' else (125,70,1275,1195)
  tile=tile.crop(crop);tile.thumbnail((box[2],box[3]),Image.Resampling.LANCZOS);im.paste(tile,(box[0]+(box[2]-tile.width)//2,box[1]))
 im.save(root/f'{model}-csp-guide-layout.png')
# Source references remain separate from the newly authored render.
refs=root/'references'
if (refs/'csp-tpose-front.png').exists():
 board=Image.new('RGB',(2100,2290),'white');d=ImageDraw.Draw(board)
 d.text((45,25),'CSP 참고 화면 / STANDIN 적용 결과',font=f(38),fill=ink)
 d.text((45,80),'위: 실제 클튜 정면 관찰 + 사용자 제공 측면/후면 · 아래: 같은 배치를 기존 Standin 형상에 대응',font=f(21),fill=muted)
 columns=[(0,850,'front'),(850,400,'side'),(1250,850,'back')]
 for x,w,view in columns:
  if view=='front':tile=Image.open(refs/'csp-tpose-front.png').crop((739,327,1693,1361))
  elif view=='side':tile=Image.open(refs/'csp-profile-user.png').crop((135,88,341,1109))
  else:tile=Image.open(refs/'csp-back-user.png').crop((50,59,997,1085))
  tile=tile.convert('RGB');tile.thumbnail((w-24,1000),Image.Resampling.LANCZOS);board.paste(tile,(x+(w-tile.width)//2,150))
  file=r/('male-gray-guide-side.png' if view=='side' else f'male-tpose-gray-guide-{view}.png');tile=Image.open(file).convert('RGB').crop((330,55,530,1015) if view=='side' else (125,70,1275,1195));tile.thumbnail((w-24,1000),Image.Resampling.LANCZOS);board.paste(tile,(x+(w-tile.width)//2,1220))
 d.text((45,1160),'STANDIN 0.1.3 · 기존 메시/체형 유지 · 참고 인형과 형상 차이 있음',font=f(25),fill=ink)
 board.save(root/'csp-reference-comparison.png')
