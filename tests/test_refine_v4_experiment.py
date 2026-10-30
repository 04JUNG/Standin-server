"""Recompute the published partial scores and verify their native image receipts."""
from pathlib import Path
import hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[1]/'experiments/refine_v4/2026-10-29'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_frozen_scoring_contract_fingerprints():
    for name,digest in load(ROOT/'contract-hashes.json').items():assert sha(ROOT/'scoring-contract'/name)==digest
def test_scores_are_replayable_and_scales_stay_separate():
    path=ROOT/'scoring-contract/score_reviews.py';spec=importlib.util.spec_from_file_location('v4_test_scorer',path)
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    rubric=load(ROOT/'scoring-contract/rubric.json');scores=load(ROOT/'scores.json')
    before=[json.loads(s) for s in (ROOT/'baseline-reviews.jsonl').read_text().splitlines()]
    after=[json.loads(s) for s in (ROOT/'after-reviews.jsonl').read_text().splitlines()]
    rows=[r for r in scores['rows'] if r['after'] is not None];assert len(rows)==len(before)==len(after)==6
    for r,a,b in zip(rows,before,after):
        assert scorer.score_case(a,rubric)==r['baseline'];assert scorer.score_case(b,rubric)==r['after']
        assert a['evaluation_scope']==b['evaluation_scope']=='full_body'
        for k,v in a['people'][0]['criteria'].items():
            if k not in r['changed_criteria']:assert b['people'][0]['criteria'][k]==v
    for label,out in [('baseline','mean_before'),('after','mean_after')]:
        assert scores[out]==[sum(r[label][k] for r in rows)/6 for k in ['score_low','score_high']]
    excluded=next(r for r in scores['rows'] if r['test_id']=='T022');assert excluded['after'] is None
    assert scores['automatic_benchmark'] is False and scores['frozen_protocol_conformant'] is False
def test_generated_images_match_the_published_receipts():
    for item in load(ROOT/'part-catalog.json'):
        assert len(item['generated_previews'])==8
        for e in item['generated_previews']:assert sha(ROOT/e['file'])==e['sha256']
def test_actual_native_replay_and_current_asset_fallback():
    rows=load(ROOT/'native-runtime.json')
    assert {r['id'] for r in rows if r['status']=='applied'}=={'T002','T018','T043','T050'}
    for r in rows:
        if r['status']=='applied':
            assert r['saved_verification']['matrix_error']<=2e-5
            assert r['saved_verification']['surface_error_m']<=2e-5
            assert r['saved_verification']['final_check']['accepted'] is True
        else:assert r['source_sha256']==r['after_sha256']
