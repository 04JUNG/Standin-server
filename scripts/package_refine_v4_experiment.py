"""Package frozen-rubric partial reviews and generated native previews for a PR.

Original roughs/photos, local FBX models and weights are deliberately excluded
under the repository's artifact policy. Exact source hashes remain in receipts.
"""
import copy,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
root,dest=map(lambda p:Path(p).resolve(),sys.argv[1:]);dest.mkdir(parents=True,exist_ok=True)
base=root/'outputs/standin-refine-v4-20261029';gallery=base/'leg-corrections-run05';frozen=root/'outputs/standin-rescore-v3.2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):return json.loads(p.read_text())
def module(p,n):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
contract=dest/'scoring-contract';contract.mkdir(exist_ok=True)
for name in ['rubric.json','review-record.schema.json','score_reviews.py','schema_check.py']:shutil.copy2(frozen/name,contract/name)
scorer=module(contract/'score_reviews.py','parts_frozen_scorer');schema_check=module(contract/'schema_check.py','parts_schema')
rubric=load(contract/'rubric.json');schema=load(contract/'review-record.schema.json')
original={r['case_id']:r for r in map(json.loads,(frozen/'reviews.jsonl').read_text().splitlines())}
requests=load(gallery/'requests.json');rows=[];baseline_records=[];after_records=[];catalog=[]
judgments={
 'T024':{'right_leg':('match','무릎을 거의 편 채 책상 위로 길게 뻗고 오른발목을 위에 겹친 배치를 확인했다.'),'left_leg':('match','왼다리도 길게 뻗고 오른발목 아래에 둔 배치를 확인했다. 실제 책상 접촉력은 측정하지 않았다.')},
 'T041':{'right_leg':('match','뒤로 깊게 접힌 오른무릎과 바깥쪽으로 빠진 발의 큰 배치를 확인했다. 이전 사용자 확정본 그대로다.')},
 'T042':{'left_leg':('match','왼무릎을 거의 펴고 몸 바깥쪽으로 길게 뻗은 배치가 원래 러프와 일치한다.'),'right_leg':('uncertain','오른다리의 아래쪽 배치는 확인했으나 검증된 바닥선·체중 지지 자료가 없어 실제 지지발 접촉은 보류한다.')},
 'T043':{'left_leg':('match','같은 러프의 T042와 동일한 하체 파츠를 적용하여 왼다리의 바깥쪽 뻗기를 확인했다.'),'right_leg':('uncertain','같은 파츠를 옮기며 오른다리도 바뀌었다. 검증된 접지 자료가 없어 기존 match를 자동 계승하지 않고 보류한다.')},
 'T049':{'right_leg':('match','골반을 낮추고 오른무릎을 바깥으로 벌린 깊은 쪼그림을 확인했다.'),'left_leg':('match','왼무릎 역시 바깥으로 넓게 벌려 깊게 굽힌 배치를 확인했다. 피부 triage 경고 해소를 뜻하지 않는다.')},
 'T050':{'left_leg':('match','서 있는 교차에서 왼다리가 반대 다리 앞으로 오고 왼발이 더 앞쪽으로 나온 배치를 확인했다.')},
}
for req in requests:
    t=req['id'];p=gallery/t;old=copy.deepcopy(original[req['case_id']]);new=copy.deepcopy(old)
    ev=load(p/'intent-evidence.json');latest=t=='T022'
    media=base/'leg-corrections-run06/T022' if latest else p
    generated=[]
    for variant in ['before','after']:
        for view in ['reference','front','oblique','side']:
            source=media/f'{variant}_legs_{view}.png';target=dest/'images'/t/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            generated.append({'file':str(target.relative_to(dest)),'sha256':sha(target),'native_sha256':sha(media/(variant+'.fbx'))})
    record={'id':t,'case_id':req['case_id'],'intent':req['sentence'],
      'anatomical_side_reference':'subject_own_left_right','input_sha256':ev['initial_vlm']['input_sha256'],
      'initial_vlm_body_details':ev['initial_vlm']['description']['body_details'],
      'initial_vlm_search_text':ev['initial_vlm']['description']['search_text'],
      'initial_vlm_source_sha256':ev['initial_vlm']['sha256'],
      'authored_fbx_sha256':sha(media/'after.fbx'),'generated_previews':generated,
      'skin_triage':load(media/'verification.json')['skin'] if latest else load(p/'correction-verification.json')['skin'],
      'automatic_image_diagnosis':False,'global_skin_certification':False}
    before=scorer.score_case(old,rubric)
    if latest:
        record['intent']='오른다리가 위인 관계를 유지하고 왼무릎을 조금 더 펴며 오른발끝을 화면 왼쪽으로 더 감는다.'
        record['latest_authored_adjustment']=load(media/'verification.json')
        row={'test_id':t,'case_id':req['case_id'],'baseline':before,'after':None,'status':'changed_reference_excluded',
             'reason':'추가 사진과 이후 사용자 지시로 하체 목표가 바뀌어 동결 원래 러프와 직접 점수 비교하지 않는다. 오른다리 위·무릎 펴기·발끝 방향은 별도 형상 검증이다.'}
    else:
        criteria=judgments[t]
        for key,(status,reason) in criteria.items():new['people'][0]['criteria'][key]={'status':status,'reason':reason}
        new['run_id']='refine-v4-parts-authored-partial-review-20261029';new['people'][0]['reviewer']='ai_assisted_unreviewed'
        new['people'][0]['final_image']=f'images/{t}/after_legs_reference.png';new['people'][0]['final_image_sha256']=sha(media/'after_legs_reference.png')
        for key,value in old['people'][0]['criteria'].items():
            if key not in criteria:assert new['people'][0]['criteria'][key]==value
        schema_check.validate(new,schema);baseline_records.append(old);after_records.append(new)
        after=scorer.score_case(new,rubric);low=high=0
        # Carry the same unmodified-body unknowns together. Both T042/T043
        # support legs changed, so their support uncertainty is independent.
        for key,(status,_) in criteria.items():
            weight=rubric['weights'][key];bounds={'match':(0,0),'mismatch':(weight,weight),'uncertain':(0,weight)}
            a,b=bounds[old['people'][0]['criteria'][key]['status']];c,d=bounds[status];low+=a-d;high+=b-c
        row={'test_id':t,'case_id':req['case_id'],'baseline':before,'after':after,'status':'authored_partial_visual_review',
             'paired_gain':[low,high],'changed_criteria':criteria,'human_rubric_reviewed':False,
             'output_sha256':sha(media/'after.fbx'),'evidence':generated}
    rows.append(row);catalog.append(record)
eligible=[r for r in rows if r['after'] is not None];n=len(eligible)
mean=lambda label:[sum(r[label][k] for r in eligible)/n for k in ['score_low','score_high']]
result={'mode':'authored_partial_visual_review','rubric_weights_version':'3.2','frozen_protocol_conformant':False,
 'manual_authoring_allowed_in_this_experiment':True,'automatic_benchmark':False,'human_rubric_reviewed':False,
 'selected_cases':7,'comparable_scored_cases':n,'reference_changed_cases_excluded':['T022'],
 'mean_before':mean('baseline'),'mean_after':mean('after'),
 'mean_paired_gain':[sum(r['paired_gain'][i] for r in eligible)/n for i in range(2)],'scale':100,'rows':rows,
 'notes':['수정하지 않은 항목의 기존 판정은 유지했다.',
          '카메라·가림·접지 불확실성을 일치로 올리지 않았다.','피부 triage 경고는 별도 공개하며 점수 상승이 전체 피부 검증 통과를 의미하지 않는다.',
          '50장 재실행 성능이나 자동 검색 성공률이 아니다. 상반신 /45와 전신 /100을 합산하지 않는다.']}
save(dest/'scores.json',result);save(dest/'part-catalog.json',catalog)
for name,records in [('baseline-reviews.jsonl',baseline_records),('after-reviews.jsonl',after_records)]:
    (dest/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
for name,source in [('saved-geometry-verification.json',gallery/'saved-geometry-verification.json'),
 ('geometry-negative-tests.json',base/'pr-geometry-tests.json'),
 ('transaction-tests.json',base/'pr-transaction-tests02/native-tests.json'),
 ('native-runtime.json',base/'pr-native-validation-run03/summary.json')]:
    value=load(source);save(dest/name,value)
old_scores=load(base/'score-run04/scores.json')
save(dest/'previous-automatic-score-summary.json',{'cases_executed':old_scores['cases_executed'],
 'native_saved_edits':old_scores['native_saved_edits'],'selected_group_means':old_scores['selected_group_means'],
 'fresh_fifty_case_execution':False,'source_sha256':sha(base/'score-run04/scores.json'),
 'rows':[{k:r[k] for k in ['test_id','case_id','status','before','after','paired_gain_low','paired_gain_high']} for r in old_scores['rows']]})
save(dest/'contract-hashes.json',{p.name:sha(p) for p in contract.iterdir() if p.is_file()})
print('Comparable authored cases',n,'before',result['mean_before'],'after',result['mean_after'],'paired',result['mean_paired_gain'])
