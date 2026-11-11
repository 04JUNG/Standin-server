import copy
import json
import sys
import pytest
from scripts.verify_production_parity import verify, PROFILE
from scripts import prepare_staging_recall_deployment as recall

def task(role='inference'):
    pin=PROFILE['bodyManifests'][role]
    env={'BODY_MATCHING_MODE':'off','POSE_LIBRARY_URI':'s3://production-assets/'+PROFILE['poseLibraryKey'],
        'POSE_THUMBNAIL_OVERLAY_URI':'s3://production-assets/'+PROFILE['thumbnailKey'],
        'POSE_THUMBNAIL_OVERLAY_SHA256':PROFILE['thumbnailSha256'],
        'CONVERTER_INTERACTIVE_ENABLED':'true','REVIEWED_LIBRARY_SHA256':PROFILE['reviewedManifestSha256']}
    return {'family':'StandinApp'+role.title()+'Task','containerDefinitions':[
        {'name':role,'environment':[{'name':k,'value':v} for k,v in env.items()],
         'secrets':[{'name':'CONVERTER_INTERACTIVE_TOKEN','valueFrom':'arn:aws:secretsmanager:ap-northeast-2:123:secret:standin/production/converter-interactive-token-example'}]},
        {'name':'body-assets-init','environment':[{'name':'BODY_ASSETS_MANIFEST_KEY','value':pin['manifestKey']},{'name':'BODY_ASSETS_MANIFEST_SHA256','value':pin['manifestSha256']}]}]}

@pytest.mark.parametrize('role',['inference','converter'])
def test_current_approved_production_task_passes(role):assert verify(task(role),role)

def test_old_body_pin_cannot_silently_ship_partial_features():
    value=task();value['containerDefinitions'][1]['environment'][1]['value']='a'*64
    with pytest.raises(ValueError,match='infrastructure first'):verify(value,'inference')

def test_staging_assets_and_private_credential_are_not_reused_in_production():
    value=task();value['containerDefinitions'][0]['environment'][1]['value']='s3://standinstagingapp-assets/'+PROFILE['poseLibraryKey']
    with pytest.raises(ValueError,match='staging asset'):verify(value,'inference')
    value=task('converter');value['containerDefinitions'][0]['secrets'][0]['valueFrom']='arn:aws:secretsmanager:region:123:secret:standin/staging/token'
    with pytest.raises(ValueError,match='credential'):verify(value,'converter')

def test_production_recall_derives_own_bucket_with_no_github_overrides(tmp_path,monkeypatch):
    value=task();env=value['containerDefinitions'][0]['environment']
    env.extend({'name':k,'value':v} for k,v in {'ROUTING_FULLBODY_ENABLED':'1','VLM_PROVIDER':'openai','OPENAI_MODEL':'gpt-6-luna','OPENAI_REASONING_EFFORT':'low'}.items())
    path=tmp_path/'task.json';path.write_text(json.dumps(value))
    monkeypatch.delenv('ROUTING_RECALL_ASSETS_URI',raising=False);monkeypatch.delenv('ROUTING_RECALL_ASSETS_SHA256',raising=False)
    monkeypatch.setattr(sys,'argv',['prepare',str(path),'--enabled','1','--production']);recall.main()
    result={e['name']:e['value'] for e in json.loads(path.read_text())['containerDefinitions'][0]['environment']}
    assert result['ROUTING_RECALL_ASSETS_URI']=='s3://production-assets/'+PROFILE['recallKey']
    assert result['ROUTING_LUNA_SIX_FIELD_ENABLED']=='1'
