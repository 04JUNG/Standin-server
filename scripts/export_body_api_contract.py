#!/usr/bin/env python3
"""Export live inference OpenAPI + transport-neutral handoff JSON Schemas.
Examples use synthetic HTTP test fixtures, never real roughs/models or API calls.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from api.app import app
from api.body_models import BodyMatchingOut
from api.body_handoff import BodyRenderRequest,BodyRenderResult,BodySelectionChange,BodyExportEnvelope
from api.models import CutResultOut,ExportOrder
from tests.test_body_api_contract import BodyAPIContractTests


from api.body_render import BodyPreviewResult

def main():
    out=ROOT/'docs/contracts/body';out.mkdir(parents=True,exist_ok=True)
    def save(name,data):
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    api=app.openapi()
    save('analyze.openapi.json',dict(openapi=api['openapi'],info=api['info'],paths={'/analyze':api['paths']['/analyze']},components=api['components']))
    for cls in [BodyMatchingOut,BodyRenderRequest,BodyRenderResult,BodySelectionChange,BodyExportEnvelope]:
        save(cls.__name__+'.schema.json',cls.model_json_schema())
    case=BodyAPIContractTests('test_http_auto_typed_and_preserves_order');case.setUp()
    try:
        auto=case.response();shadow=case.response('shadow');off=case.response('off')
        unavailable=case.response(broken=lambda d:d['people'][0]['pose_bindings'].reverse())
        for name,example in [('analyze.auto',auto),('analyze.shadow',shadow),('analyze.off',off),('analyze.body-error',unavailable)]:
            CutResultOut.model_validate(example);save(name+'.example.json',example)
        batch=case.batch();save('render.request.example.json',batch.model_dump(mode='json'))
        ready=case.render_result(batch);save('render.ready.example.json',ready.model_dump(mode='json'))
        change=BodySelectionChange(analysis_input_sha256=batch.analysis_input_sha256,person_id=batch.person_id,
            person_index=0,expected_revision=0,body=batch.body)
        save('selection.change.example.json',change.model_dump(mode='json'))
        pose=batch.poses[0]
        order=ExportOrder(cut_id='example-cut',created_at='2026-10-07T00:00:00Z',items=[dict(person_index=0,
            pose_id=pose.pose_id,view=pose.view,bvh_url=pose.bvh_url,tags={})])
        envelope=BodyExportEnvelope(analysis_input_sha256=batch.analysis_input_sha256,pose_order=order,
            body_assignments=[dict(person_index=0,person_id=batch.person_id,selection_revision=0,body=batch.body,pose=pose)])
        save('export.envelope.example.json',envelope.model_dump(mode='json'))
    finally:case.doCleanups()
    save('BodyPreviewResult.schema.json',BodyPreviewResult.model_json_schema())
    save('manifest.json',dict(description='Synthetic contract fixtures only. No real roughs, usable FBX, network calls or renderer execution.',
        implemented_http=['POST /analyze additive typed body_matching', 'POST /body/render selected-body PNG previews'],
        proposed_external_transport=['BodySelectionChange','BodyRenderResult (future FBX artifact jobs)','BodyExportEnvelope'],
        generation_command='python scripts/export_body_api_contract.py'))
    print(out)
if __name__=='__main__':main()
