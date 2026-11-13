"""Real original-model registration/pose/export test through a packaged local host.

Keep originals, runtime and raw output FBX local. A nonzero exit means any model
was rejected or any requested pose did not export. No BFF/inference is called.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.local_converter.verify_runtime import Session, sha


def terminal(session, action, **fields):
    rid = session.send(action, **fields)
    while True:
        event = session.events.get(timeout=150)
        assert event is not None, 'Runtime exited before a terminal result'
        session.transcript.append({'event': event})
        assert event['requestId'] == rid
        if event['event'] != 'progress': return event


def verify(launcher, sources, poses, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {'packageManifestSha256': sha(launcher.parent/'runtime-manifest.json'),
              'liveBff': False, 'cspImportTested': False, 'models': []}
    session = Session(launcher, output/'store')
    try:
        start = time.monotonic()
        hello = terminal(session, 'hello')
        assert hello['event'] == 'result'
        report['startupSeconds'] = time.monotonic()-start
        report['hello'] = hello['result']
        assert report['hello']['materialContractVersion'] == 'personal-solid-specular/1'
        for index, source in enumerate(sources):
            model = f'actual-model-{index}'
            digest = sha(source)
            start = time.monotonic()
            event = terminal(session, 'import', localModelId=model, sourcePath=str(source), sourceHash=digest)
            row = {'inputName':source.name,'sourceHash':digest,'registrationSeconds':time.monotonic()-start,
                   'registrationEvent':event['event'], 'poses':[]}
            if event['event'] == 'error': row['error'] = event['error']
            else:
                reg = event['result']
                row['registration'] = {k:reg[k] for k in ['normalizedFormat','normalizedHash','archiveHash','validation','versions','capabilities']}
                assert sha(reg['sourcePath']) == digest and sha(reg['normalizedPath']) == reg['normalizedHash']
                assert sha(reg['archivePath']) == reg['archiveHash'] and reg['normalizedFormat'] == 'fbx'
                row['sourceGeometry'] = reg['character'].get('sourceFormatDetails', {})
                row['sourceMaterials'] = row['sourceGeometry'].get('materials', [])
                shutil.copyfile(reg['normalizedPath'], output/f'model-{index}-registered.fbx')
                for pi, pose in enumerate(poses):
                    fields = dict(localModelId=model, sourceHash=digest, posePath=str(pose), poseHash=sha(pose),
                                  selectionId=f'pose-{pi}', hands={'left':None,'right':None},
                                  camera={'version':'personal-camera/1','sourceBvhSha256':sha(pose),'rotation':None})
                    start = time.monotonic()
                    applied = terminal(session, 'apply', **fields)
                    result = {'poseName':pose.name,'poseHash':sha(pose),'event':applied['event'],'seconds':time.monotonic()-start}
                    if applied['event'] == 'error': result['error'] = applied['error']
                    else:
                        value = applied['result']
                        dest = output/f'model-{index}-pose-{pi}.fbx'
                        exported = terminal(session, 'export', localModelId=model, snapshotHash=value['snapshotHash'], destinationPath=str(dest))
                        assert exported['event'] == 'result' and sha(dest) == value['artifactHash']
                        result.update({k:value[k] for k in ['snapshotHash','artifactHash','validation','retarget']})
                        result['file'] = dest.name
                        start = time.monotonic()
                        cached = terminal(session, 'apply', **fields)
                        assert cached['event'] == 'result' and cached['result']['cacheHit']
                        assert cached['result']['artifactHash'] == value['artifactHash']
                        result['cachedSeconds'] = time.monotonic()-start
                    row['poses'].append(result)
                if reg.get('previewPath'):
                    assert sha(reg['previewPath']) == reg['previewHash']
                    shutil.copyfile(reg['previewPath'],output/f'model-{index}-registration.glb')
            assert sha(source) == digest
            report['models'].append(row)
            (output/'report.json').write_text(json.dumps(report,indent=2))
    finally:
        session.close()
        # The transcript contains local file paths. Never publish it as portable evidence.
        (output/'local-transcript.json').write_text(json.dumps(session.transcript,indent=2))
    report['allPassed'] = all(m['registrationEvent']=='result' and len(m['poses'])==len(poses)
                             and all(p['event']=='result' for p in m['poses']) for m in report['models'])
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report['allPassed']


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher',type=Path,required=True)
    parser.add_argument('--source',type=Path,action='append',required=True)
    parser.add_argument('--pose',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    passed=verify(args.launcher.resolve(),[p.resolve() for p in args.source],[p.resolve() for p in args.pose],args.output.resolve())
    print(json.dumps({'allPassed':passed,'models':len(args.source),'posesPerModel':len(args.pose)}))
    raise SystemExit(0 if passed else 1)
