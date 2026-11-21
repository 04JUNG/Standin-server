"""Real packaged-host proof: rejected poses cannot become exportable artifacts."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.local_converter.verify_runtime import Session, sha
from scripts.local_converter.verify_actual_models import terminal
from converter.local.contract import digest, snapshot


def verify(launcher, source, neutral, unsafe, output):
    output.mkdir(parents=True, exist_ok=False)
    session = Session(launcher, output/'store')
    report = {'sourceSha256':sha(source),'neutralSha256':sha(neutral),'unsafeSha256':sha(unsafe),
              'manifestSha256':sha(launcher.parent/'runtime-manifest.json'),'checks':[]}
    def check(name, value):
        assert value, name
        report['checks'].append({'name':name,'passed':True})
    try:
        model = terminal(session,'import',localModelId='quality-boundary',sourcePath=str(source),sourceHash=sha(source))
        check('fresh import',model['event']=='result')
        reg=model['result']
        fields={'localModelId':'quality-boundary','sourceHash':sha(source),'selectionId':'neutral',
                'hands':{'left':None,'right':None},'camera':{'version':'personal-camera/1','sourceBvhSha256':None,'rotation':None}}
        start=time.monotonic()
        accepted=terminal(session,'apply',**fields,posePath=str(neutral),poseHash=sha(neutral))
        check('neutral accepted',accepted['event']=='result')
        report['neutralApplySeconds']=time.monotonic()-start
        preserved=output/'preserved-neutral.fbx'
        result=terminal(session,'export',localModelId='quality-boundary',snapshotHash=accepted['result']['snapshotHash'],destinationPath=str(preserved))
        check('verified neutral saved',result['event']=='result' and sha(preserved)==accepted['result']['artifactHash'])
        previous_hash=sha(preserved)
        start=time.monotonic()
        rejected=terminal(session,'apply',**{**fields,'selectionId':'unsafe'},posePath=str(unsafe),poseHash=sha(unsafe))
        report['unsafeApplySeconds']=time.monotonic()-start
        check('unsafe explicitly rejected',rejected['event']=='error' and rejected['error']['code']=='POSE_QUALITY_REJECTED')
        report['rejection']=rejected['error']
        rejected_hash=digest(snapshot(reg,sha(unsafe),fields['hands'],fields['camera']))
        forbidden=output/'must-not-exist.fbx'
        exported=terminal(session,'export',localModelId='quality-boundary',snapshotHash=rejected_hash,destinationPath=str(forbidden))
        check('rejected snapshot not exportable',exported['event']=='error' and exported['error']['code']=='SNAPSHOT_MISMATCH' and not forbidden.exists())
        cached=terminal(session,'apply',**fields,posePath=str(neutral),poseHash=sha(neutral))
        check('previous verified neutral retained',cached['event']=='result' and cached['result']['cacheHit'] and cached['result']['artifactHash']==previous_hash)
        terminal(session,'delete',localModelId='quality-boundary')
        check('delete preserves saved file and source',sha(preserved)==previous_hash and sha(source)==report['sourceSha256'])
        report['preservedOutputSha256']=previous_hash
        report['allPassed']=True
    finally:
        session.close()
        (output/'report.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launcher',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--neutral',type=Path,required=True);p.add_argument('--unsafe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.launcher.resolve(),a.source.resolve(),a.neutral.resolve(),a.unsafe.resolve(),a.output.resolve());print(json.dumps({'allPassed':r['allPassed'],'checks':len(r['checks'])}))
