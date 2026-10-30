"""Real SDK request construction, target ownership, isolation and shared budget."""
from contextvars import ContextVar
import io
import json
from threading import Event, Lock
from time import monotonic, sleep
from types import SimpleNamespace

import pytest
from PIL import Image
from google.genai.errors import ServerError

from src.analysis_budget import analysis_budget
from src.config import CFG
from src.hand_skin_parallel import request_cancelled
from src.search_routing import DescribeOutcome, PoseDescription
from src.vlm.client import GeminiVLMClient
from src.vlm.person_crops import person_crop
from src.vlm import person_crops

BOXES = [(0,.1,.4,.9),(.6,.1,1,.9)]

@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setattr(CFG,'routing_describe_person_crops',True)
    monkeypatch.setattr(CFG,'routing_describe_person_workers',2)
    monkeypatch.setattr(CFG,'routing_describe_budget_ms',45000)

@pytest.fixture
def image():
    im=Image.new('RGB',(100,100),'red')
    im.paste('blue',(50,0,100,100))
    yield im
    im.close()


def test_crop_pixel_bounds_target_remap_and_no_mirroring(image):
    crop,target,report=person_crop(image,BOXES[1])
    try:
        assert report['crop_rect']==[52,0,100,100]
        assert target==pytest.approx((8/48,.1,1,.9))
        assert crop.size==(48,100) and crop.getpixel((0,0))==(0,0,255)
        assert image.size==(100,100)
    finally:crop.close()


@pytest.mark.parametrize('box',[None,(-.1,0,.3,1),(.7,0,.2,1),(0,0,float('nan'),1)])
def test_bad_target_never_becomes_whole_image(box,image):
    with pytest.raises(ValueError):person_crop(image,box)


def test_two_real_one_person_requests_keep_original_indices_after_out_of_order_completion(image):
    calls=[]
    def generate(**kw):
        config=kw['config']
        crop=Image.open(io.BytesIO(kw['contents'][1].inline_data.data))
        name='red' if crop.getpixel((crop.width//2,crop.height//2))[0]>100 else 'blue'
        calls.append((name,config,kw['contents'][0],crop.size))
        if name=='red':sleep(.03)
        return SimpleNamespace(text=json.dumps({'people':[dict(visible=True,body_details=name+' arm',search_text=name+' pose',certainty='high')]}))
    client=GeminiVLMClient(client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    result=client.describe_people(image,BOXES)
    assert result.status=='ok' and len(calls)==2 and not result.person_errors
    assert result.descriptions[0].body_details=='red arm'
    assert result.descriptions[1].body_details=='blue arm'
    assert set(result.person_reports)=={0,1}
    for _,config,prompt,size in calls:
        assert size==(48,100) and '대상 한 명만' in prompt
        assert config.response_json_schema['properties']['people']['maxItems']==1


def test_one_failed_person_does_not_remove_other_description(monkeypatch,image):
    client=GeminiVLMClient(client=object(),sleep=lambda _:None,jitter=lambda *_:.5)
    seen={0:0,1:0}
    def once(crop,boxes,timeout):
        index=0 if crop.getpixel((crop.width//2,50))[0]>100 else 1
        seen[index]+=1
        if index==0:return DescribeOutcome('error',error_reason='upstream_error',retryable=True)
        return DescribeOutcome('ok',descriptions={0:PoseDescription(body_details='success blue')})
    monkeypatch.setattr(client,'_describe_people_once',once)
    result=client.describe_people(image,BOXES)
    assert result.status=='partial_error'
    assert set(result.descriptions)=={1} and set(result.person_errors)=={0}
    assert result.person_errors[0]['error_reason']=='upstream_error'
    assert seen=={0:2,1:1}


def test_crop_workers_share_context_and_original_deadline(monkeypatch,image):
    marker=ContextVar('marker',default=None)
    client=GeminiVLMClient(client=object())
    seen=[]
    def bounded(crop,boxes,*,deadline,stop):
        seen.append((deadline,marker.get()))
        return DescribeOutcome('ok',descriptions={0:PoseDescription()})
    monkeypatch.setattr(client,'_describe_people_bounded',bounded)
    token=marker.set('owned-request')
    try:
        with analysis_budget(.3) as deadline:
            # Keep a short shared budget without the production 15s reserve.
            monkeypatch.setattr(CFG,'routing_describe_reserve_ms',0)
            client.describe_people(image,BOXES)
    finally:marker.reset(token)
    assert seen==[(deadline,'owned-request')]*2


def test_slow_person_cannot_block_completed_person_past_budget(monkeypatch,image):
    monkeypatch.setattr(CFG,'routing_describe_budget_ms',50)
    client=GeminiVLMClient(client=object())
    release=Event()
    def bounded(crop,boxes,*,deadline,stop):
        if crop.getpixel((crop.width//2,50))[0]>100:
            release.wait(.5)
        return DescribeOutcome('ok',descriptions={0:PoseDescription()})
    monkeypatch.setattr(client,'_describe_people_bounded',bounded)
    started=monotonic()
    try:
        result=client.describe_people(image,BOXES)
        assert monotonic()-started<.3
        assert set(result.descriptions)=={1} and 0 in result.person_errors
    finally:release.set()


def test_global_concurrency_is_at_most_two_across_requests(monkeypatch,image):
    from concurrent.futures import ThreadPoolExecutor
    client=GeminiVLMClient(client=object())
    lock=Lock();active=0;peak=0
    def bounded(*args,**kw):
        nonlocal active,peak
        with lock:active+=1;peak=max(active,peak)
        sleep(.02)
        with lock:active-=1
        return DescribeOutcome('ok',descriptions={0:PoseDescription()})
    monkeypatch.setattr(client,'_describe_people_bounded',bounded)
    with ThreadPoolExecutor(3) as pool:
        results=list(pool.map(lambda _:client.describe_people(image,BOXES),range(3)))
    assert peak==2 and all(r.status=='ok' for r in results)


def test_cancelled_or_invalid_person_never_calls_provider(monkeypatch,image):
    client=GeminiVLMClient(client=object())
    monkeypatch.setattr(client,'_describe_people_bounded',lambda *a,**kw:pytest.fail('must not call'))
    result=client.describe_people(image,[None,None])
    assert len(result.person_errors)==2 and result.attempts==0
    event=Event();event.set();token=request_cancelled.set(event)
    try:result=client.describe_people(image,BOXES)
    finally:request_cancelled.reset(token)
    assert len(result.person_errors)==2 and result.attempts==0


def test_feature_rollback_preserves_existing_batch(monkeypatch,image):
    monkeypatch.setattr(CFG,'routing_describe_person_crops',False)
    client=GeminiVLMClient(client=object())
    seen=[]
    def bounded(im,boxes,**kw):
        seen.append((im,boxes))
        return DescribeOutcome('ok')
    monkeypatch.setattr(client,'_describe_people_bounded',bounded)
    client.describe_people(image,BOXES)
    assert seen==[(image,BOXES)]
