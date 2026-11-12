"""Refuse partial production promotion before the matching infrastructure is ready."""
import argparse
import json
from pathlib import Path
PROFILE=json.loads((Path(__file__).resolve().parents[1]/'config/production-runtime-parity.json').read_text())
def verify(task,role):
    if role not in ('inference','converter') or not task.get('family','').startswith('StandinApp'):
        raise ValueError('expected production inference or Converter task')
    matches=[c for c in task['containerDefinitions'] if c['name']==role]
    if len(matches)!=1:raise ValueError('ambiguous serving container')
    main=matches[0];env={x['name']:x['value'] for x in main.get('environment',[])}
    if 'standinstagingapp-' in json.dumps(env):raise ValueError('staging asset reference in production')
    init=[c for c in task['containerDefinitions'] if c['name']=='body-assets-init']
    if len(init)!=1:raise ValueError('production body initializer must be provisioned first')
    body={e['name']:e['value'] for e in init[0]['environment']}
    for key,value in [('BODY_ASSETS_MANIFEST_KEY',PROFILE['bodyManifests'][role]['manifestKey']),('BODY_ASSETS_MANIFEST_SHA256',PROFILE['bodyManifests'][role]['manifestSha256'])]:
        if body.get(key)!=value:raise ValueError('deploy the approved production body infrastructure first: '+key)
    if role=='inference':
        for key,tail in [('POSE_LIBRARY_URI',PROFILE['poseLibraryKey']),('POSE_THUMBNAIL_OVERLAY_URI',PROFILE['thumbnailKey'])]:
            if not env.get(key,'').endswith('/'+tail):raise ValueError('production asset parity required: '+key)
        if env.get('POSE_THUMBNAIL_OVERLAY_SHA256')!=PROFILE['thumbnailSha256']:raise ValueError('thumbnail checksum differs')
        if env.get('BODY_MATCHING_MODE')!='off':raise ValueError('automatic body detection must remain off')
    else:
        if env.get('CONVERTER_INTERACTIVE_ENABLED')!='true':raise ValueError('production private transport is not provisioned')
        if not any(s['name']=='CONVERTER_INTERACTIVE_TOKEN' and ':secret:standin/production/' in s['valueFrom'] for s in main.get('secrets',[])):
            raise ValueError('production private transport credential required')
        if env.get('REVIEWED_LIBRARY_SHA256')!=PROFILE['reviewedManifestSha256']:raise ValueError('reviewed FBX pin differs')
    return True
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',type=Path);p.add_argument('--role',required=True);a=p.parse_args();verify(json.loads(a.task.read_text()),a.role)
    print('Production feature and approved asset parity verified')
