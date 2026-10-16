"""Short independent visual questions; no example answers to copy."""
from pathlib import Path
import os,json,sys,time,hashlib
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)) and any(s in str(args[0]) for s in ('feedback-eval','evaluation-only','/gt/','visual-review')):raise RuntimeError('Feedback blocked')
sys.addaudithook(guard);os.environ['HF_HOME']=str(WORK/'outputs/04j-289-auto-multicut-experiment/model-cache');os.environ['HF_HUB_OFFLINE']='1'
from mlx_vlm import load,generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
MODEL='mlx-community/Qwen3-VL-8B-Instruct-4bit';model,processor=load(MODEL);cfg=load_config(MODEL)
QUESTIONS={
 'hands':'''이 콘티에서 실제로 보이는 손을 모두 찾아 주세요. 빨간 P 표시는 머리의 인물 ID입니다. 손이 머리에서 멀리 떨어져 있어도 소매와 팔의 연결을 따라 소유 인물을 판단하세요. 인물의 머리만 보고 손이 없다고 하지 마세요. 말풍선·문·가방은 손이 아닙니다. JSON 객체의 hands 배열로 답하세요. 각 항목에는 owner(인물 번호 또는 null), center(현재 이미지 0~1000 좌표), elbow(보이면 같은 좌표, 안 보이면 null), open_palm(boolean), touching(실제 만지거나 쥔 대상), action(물리적 동작), evidence(소유 근거)를 넣으세요. 장면을 실제로 관찰해서 쓰고 예시 문구를 반복하지 마세요.''',
 'surfaces':'''각 인물 P의 몸통과 머리를 따로 관찰하세요. 원본 그림에서 실제로 보이는 가슴·등·어깨·옷의 단서를 설명하세요. 가슴 정면이 보이는지, 옆면인지, 등인지 먼저 판단하고 그 다음 방향을 쓰세요. 얼굴 옆모습과 몸통 정면을 혼동하지 마세요. JSON people 배열로 답하세요. 각 항목에는 person_index, torso_surface(front/back/side), torso_heading(front/front_left/front_right/left/right/back_left/back_right/back/unknown), torso_evidence(실제 관찰한 문장), head_heading(같은 방향 종류), head_pitch(level/down/up/unknown), head_bbox(머리카락~턱의 보이는 전체 머리 사각형, 이미지 0~1000 좌표), lowest_anatomy(그 인물의 가장 아래 보이는 신체 부위), scope_evidence(실제 옷·다리·신체 근거)를 넣으세요. 신체가 잘렸으면 정직하게 설명하세요.''',
 'roles':'''말풍선은 읽지 말고 세부 동작을 보세요. 각 사람의 몸·시선·팔이 누구를 향하는지 설명하세요. 가운데 사람이 두 사람 사이에 서서 팔로 막는 동작이 있는지도 실제로 확인하세요. 팔을 뻗은 사람이 물건을 쥐고 있다면 제지 동작으로 바꾸지 마세요. JSON으로 scene_action(관찰한 장면 한 문장), people 배열(person_index, action, target_person_indices), relations 배열(actor, target, type, evidence)을 반환하세요. type은 opposing/restraining/reaching_toward/looking_at/leaving/unknown 중 근거가 맞는 것을 고르세요. 실제로 안 보이는 관계는 만들지 마세요.'''
}
for item in json.loads((ROOT/'inputs.json').read_text()):
 folder=ROOT/item['cut_id'];vp=json.loads((folder/'identities.json').read_text())['analysis_viewport_bbox']
 for stage,prompt in QUESTIONS.items():
  out=folder/(stage+'-review.json')
  if out.exists():continue
  begin=time.monotonic();raw=generate(model,processor,apply_chat_template(processor,cfg,prompt,num_images=1),[str(folder/'scene-context.png')],max_tokens=1500,temperature=0.,verbose=False).text
  try:a=json.loads(raw[raw.index('{'):raw.rindex('}')+1]);error=None
  except Exception as e:a=None;error=str(e)
  out.write_text(json.dumps(dict(image_sha256=item['image_sha256'],analysis_viewport_bbox=vp,prompt=prompt,raw=raw,analysis=a,error=error,elapsed_s=time.monotonic()-begin,manual_inputs=False),ensure_ascii=False,indent=2));print(item['cut_id'],stage,a,'ERROR',error,flush=True)
