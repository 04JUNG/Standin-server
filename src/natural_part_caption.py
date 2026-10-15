"""Separate upper/lower Gemini descriptions, preserving owned frozen boxes."""
import hashlib
import json
from pathlib import Path

from .natural_parts import GROUPS,validate_part

PROMPT_VERSION='rough-natural-parts-v2-visibility-and-subject'


def prompt_for(boxes):
    return ('웹툰 러프에서 아래 사람 박스의 인물만 순서대로 분석하라. 사람 수를 다시 세거나 '
        '옆 인물 팔/다리를 가져오지 마라. 각 대상 subject_kind=human/nonhuman/uncertain를 먼저 표시한다. '
        '박스가 동물/물체를 가리키면 nonhuman으로 쓰고 사람으로 바꾸거나 새 사람을 찾지 마라. '
        '좌우는 캐릭터 자신 기준이다. 좌표/관절 각도는 생성하지 않는다. '
        '상체(몸통 기울기/회전, 양팔/손 위치, 하는 동작, 보이는 물건/지지 관계)와 '
        '하체(골반/다리/무릎 배치, 서기/앉기/눕기/공중)를 각각 독립적으로 설명하라. '
        '동작 이름만 말하지 말고 그림에서 관찰한 배치를 구체적으로 유지하라. '
        '상체 문장에 무관한 하체를 섞지 않는다. 장면 맥락은 참고하되 보이지 않는 부위를 추측하지 마라. '
        '그림에 해당 부위가 일부라도 식별되면 visible=true, 부분 가림은 관찰 가능한 것만 적는다. '
        'visible과 별도로 searchable=bool을 쓴다. 부위 자세를 실제로 구분할 관측 근거가 있을 때만 true. '
        '얼굴/목/어깨 윤곽만 보이고 몸통 기울기나 팔 배치가 안 읽히면 상체 searchable=false. '
        '골반/바지 상단만 보이고 다리 자세는 안 보이면 하체 searchable=false. '
        '다리 없이 상체만으로 매달림/서기를 추정해 하체를 검색하지 마라. '
        '컷 밖/완전 가림/판단 불가능이면 visible=false, group=unknown, observations=[], '
        'description="", confidence=unknown, searchable=false로 기권한다. 안 보임을 기본 서기/팔 내림으로 쓰지 마라. '
        'group은 거친 분류일 뿐 설명을 지우지 않는다. 서로 다른 자세를 특정 사례에 맞춰 억지로 분류하지 마라. '
        '관찰 사실과 추정 동작이 다르면 observations에 관찰, description에 추정임을 명시한다. '
        '각 부위 JSON {visible:bool,searchable:bool,group:허용값,observations:짧은 한국어 2~5개,description:한국어 1~2문장,'
        'confidence:high/medium/low/unknown}. 응답은 {"people":[{"index":0,"subject_kind":"human","upper":{...},"lower":{...}}]} '
        'JSON만. 허용 그룹='+json.dumps(GROUPS,ensure_ascii=False)+
        ' 정규화 사람 박스(x1,y1,x2,y2)='+json.dumps(boxes))


def parse_caption(text,count):
    parsed=json.loads(text);people=parsed.get('people')
    if not isinstance(people,list) or len(people)!=count:raise ValueError('caption person count mismatch')
    if {p.get('index') for p in people}!=set(range(count)):raise ValueError('caption indices mismatch')
    result=[]
    for row in sorted(people,key=lambda p:p['index']):
        if row.get('subject_kind') not in ('human','nonhuman','uncertain'):raise ValueError('subject kind required')
        parts={p:validate_part(row[p],p,rough=True) for p in GROUPS}
        if any(type(p.get('searchable')) is not bool or (p['searchable'] and not p['visible'])for p in parts.values()):
            raise ValueError('part searchability must be explicit and visible')
        result.append(dict(index=row['index'],subject_kind=row['subject_kind'],**parts))
    return result


def caption_image(path,boxes,client,*,cache_dir):
    from google.genai import types
    image=Path(path).read_bytes();prompt=prompt_for(boxes)
    identity=dict(prompt_version=PROMPT_VERSION,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        image_sha256=hashlib.sha256(image).hexdigest(),model='gemini-3.8-flash',boxes=boxes)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    target=Path(cache_dir)/(fingerprint+'.json');target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        old=json.loads(target.read_text());parse_caption(old['raw_text'],len(boxes))
        if old['identity']!=identity:raise ValueError('caption cache identity mismatch')
        return dict(old,cache_reused=True)
    attempts=[]
    for _ in range(3):
        try:
            response=client.models.generate_content(model=identity['model'],contents=[prompt,
                types.Part.from_bytes(data=image,mime_type='image/png')],
                config=types.GenerateContentConfig(response_mime_type='application/json',temperature=0,
                    max_output_tokens=5000,thinking_config=types.ThinkingConfig(thinking_budget=0)))
            text=response.text;attempt=dict(raw_text=text)
            attempts.append(attempt);people=parse_caption(text,len(boxes))
            result=dict(status='ok',identity=identity,people=people,raw_text=text,
                attempts=attempts,cache_reused=False,uploaded=False)
            target.write_text(json.dumps(result,ensure_ascii=False,indent=2));return result
        except Exception as error:
            if not attempts or 'error' in attempts[-1]:attempts.append({})
            attempts[-1]['error']=type(error).__name__+':'+str(error)[:160]
    result=dict(status='error',identity=identity,people=[],attempts=attempts,uploaded=False)
    (target.parent/(fingerprint+'-failed.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result
