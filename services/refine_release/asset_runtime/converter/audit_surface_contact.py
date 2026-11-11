"""Measure contact on delivered FBX bytes, gated by the existing full receipt."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def verified_snapshot(path,qa_path):
    from converter import retarget as rt
    from converter.shape_bake import import_full_character
    from converter.refine_v3_safe_snapshot_qa import snapshot,sha
    from converter.artifact_receipt import issue_receipt,verify_receipt
    q=json.loads(qa_path.read_text());raw=path.read_bytes()
    if q['artifact_sha256']!=sha(path) or q['neutral_qa_source']:raise ValueError('contact_artifact_binding')
    receipt=issue_receipt(raw,source_sha256=q['source_sha256'],context_sha256=q['receipt']['context_sha256'],
        profile_sha256=q['profile_sha256'],revision=q['receipt']['revision'],scope='full',
        mesh_audit=q['mesh_audit'],integrity_audit=q['artifact_integrity'])
    if receipt.digest!=q['receipt_sha256'] or not verify_receipt(receipt,raw,revision=receipt.revision,context_sha256=receipt.context_sha256,profile_sha256=receipt.profile_sha256):
        raise ValueError('contact_invalid_mesh_receipt')
    if q.get('pose_safety',{}).get('final_body',{}).get('status')!='pose_validated':raise ValueError('contact_missing_pose_safety')
    rt.import_character=import_full_character
    return snapshot(path),q

def main():
    from converter.surface_contact import define_contact,audit_contact
    from converter.refine_v3_safe_snapshot_qa import sha
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('baseline','baseline-qa','fbx','qa','contract','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);contract=json.loads(a.contract.read_text())
    base,bq=verified_snapshot(a.baseline,a.baseline_qa);actual,q=verified_snapshot(a.fbx,a.qa)
    if bq['source_sha256']!=q['source_sha256'] or contract['source_sha256']!=q['source_sha256'] or contract['asset_sha256']!=q['original_asset_sha256']:
        raise ValueError('contact_source_binding')
    regenerated=define_contact(base['vertices'],base['faces'],base['weights'],base['names'],contract['side'],base['bones'])
    for k in regenerated:
        if regenerated[k]!=contract[k]:raise ValueError('contact_frozen_patch_mismatch:'+k)
    before=audit_contact(base['vertices'],base['faces'],contract,mesh_validated=True)
    after=audit_contact(actual['vertices'],actual['faces'],contract,mesh_validated=True)
    result=dict(version=contract['version'],before=before,after=after,baseline_sha256=sha(a.baseline),artifact_sha256=sha(a.fbx),
        mesh_receipt_sha256=q['receipt_sha256'],contract_sha256=sha(a.contract),deployed=False,
        contact_applied=after['status']=='contact_validated')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['contact_applied']:raise ValueError('final_surface_contact_blocked')
if __name__=='__main__':main()
