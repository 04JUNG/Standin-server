#!/usr/bin/env python3
"""One explicit live VLM review of rough and actual FBX renders; no safety authority."""
from pathlib import Path
import argparse,hashlib,json,os,time

def main():
    from google import genai
    from google.genai import types
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rough',type=Path,required=True);p.add_argument('--review-dir',type=Path,required=True)
    p.add_argument('--timeout-seconds',type=int,default=45)
    p.add_argument('--palm-elbow',action='store_true',help='Explicitly compare elbow depth and broad palm/side-waist contact')
    p.add_argument('--env-file',type=Path);p.add_argument('--model',default='gemini-3.8-flash')
    a=p.parse_args()
    if not 1<=a.timeout_seconds<=120:raise ValueError('invalid_timeout')
    key=os.environ.get('GEMINI_API_KEY')
    if not key and a.env_file:
        from dotenv import dotenv_values
        key=dotenv_values(a.env_file).get('GEMINI_API_KEY')
    if not key:raise ValueError('missing_gemini_key')
    names=['before-front.png','after-front.png','before-side.png','after-side.png','before-contact-three_quarter.png','after-contact-three_quarter.png']
    paths=[a.rough,*[a.review_dir/n for n in names]]
    prompt='실제 러프와 실제 FBX 렌더를 비교하라. 이미지 순서: 1 원본 러프, 2 이전 정면, 3 밀착 실험 정면, 4 이전 측면, 5 실험 측면, 6 이전 손/골반 확대 사선, 7 실험 확대 사선. 주황색은 검사하는 손의 일부 표면을 표시한 재질일 뿐 형상 변형이 아니다. 개선되었다고 가정하지 말라. 손 전체/손바닥/엄지 부위의 접촉을 구분하라. 보이는 간격, 접촉의 자연스러움, 러프 포즈와의 일치, 팔과 다리 원근을 각각 비교하라. 단일 러프의 실제 3D 깊이는 알 수 없고 카메라 차이가 있다. 이미지로 메시 관통이나 관절 안전을 인증하지 말라. JSON만 반환: preferred(before/after/uncertain), hand_body_contact_change(improved/same/worse/uncertain), rough_pose_match_change(improved/same/worse/uncertain), contacted_hand_part, reasons 배열, remaining_mismatches 배열, confidence(high/medium/low), camera_limitations. 한국어로 답하라.'
    if a.palm_elbow:
        prompt='첨부 러프 및 실제 FBX 렌더를 독립적으로 평가하라. 순서: 1 러프, 2 기존 정면, 3 후보 정면, 4 기존 측면, 5 후보 측면, 6 기존 손바닥/허리 확대 사선, 7 후보 확대 사선. 주황색은 손바닥 전체의 검사 영역을 표시한 재질이다. 개선을 가정하지 말라. 팔꿈치가 몸통 기준 뒤쪽으로 이동했는지, 러프의 팔꿈치 관절 위치와 팔 굽힘에 더 가까운지, 손바닥 전체가 허리 옆에 자연스럽게 밀착됐는지를 각각 평가하라. 엄지 한 점 접촉이나 하복부 앞의 손을 손바닥/옆허리 밀착 성공으로 인정하지 말라. 이미지로 3D 안전이나 정확한 깊이를 인증하지 말라. JSON만 반환: preferred(before/after/uncertain), elbow_rearward_change(improved/same/worse/uncertain), elbow_rough_match_change(improved/same/worse/uncertain), full_palm_side_waist_contact(achieved/not_achieved/uncertain), rough_pose_match_change(improved/same/worse/uncertain), reasons 배열, remaining_mismatches 배열, confidence(high/medium/low), camera_limitations. 한국어로 답하라.'
    client=genai.Client(api_key=key,http_options=types.HttpOptions(timeout=a.timeout_seconds*1000,retry_options=types.HttpRetryOptions(attempts=1)))
    start=time.monotonic()
    try:
        r=client.models.generate_content(model=a.model,contents=[prompt,*[types.Part.from_bytes(data=p.read_bytes(),mime_type='image/png') for p in paths]],config=types.GenerateContentConfig(response_mime_type='application/json',temperature=0,max_output_tokens=2500))
        report=dict(model_requested=a.model,model_actual=r.model_version,provider_requests=1,elapsed_seconds=time.monotonic()-start,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),images=[dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
            response=json.loads(r.text),usage=r.usage_metadata.model_dump(),safety_authority=False,depth_ground_truth=False)
        (a.review_dir/'gemini-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report['response'],ensure_ascii=False,indent=2))
    finally:client.close()
if __name__=='__main__':main()
