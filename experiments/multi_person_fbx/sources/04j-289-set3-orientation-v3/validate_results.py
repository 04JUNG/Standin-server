"""Artifact/provenance checks on actual output files, not image-quality grading."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent;checks=[];actors=0;upright=0;headmax=0.;library={r['pose_id']:r for r in json.loads((ROOT.parent/'04j-289-set34-refine-scenes/contact-index.json').read_text())['poses']}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for item in json.loads((ROOT/'inputs.json').read_text()):
 folder=ROOT/item['cut_id'];layout=json.loads((folder/'layout.json').read_text());selection=json.loads((folder/'selection.json').read_text());geometry=json.loads((folder/'geometry.json').read_text());scene=json.loads((folder/'scene-result.json').read_text());assert sha(item['image_path'])==item['image_sha256'];assert scene['layout_sha256']==sha(folder/'layout.json');assert len(layout['people'])==len(selection['people'])==len(geometry['people']);assert layout['zero_visible_silhouette_overlap'];assert not any(p['reversed'] for p in layout['depth_order'])
 for p,s,g in zip(layout['people'],selection['people'],geometry['people']):
  i=p['person_index'];actors+=1;assert i==s['person_index']==g['person_index'];assert sha(p['source_fbx'])==p['fbx_sha256']==g['fbx_sha256'];assert s['pose_id']==g['pose_id'];assert g['bone_count']==g['scope_execution']['bones_preserved']==52;assert p['scale']==1.;assert abs(p['depth_m']-layout['constrained_depths'][i])<1e-7;headmax=max(headmax,g['head_aim']['angular_error_degrees']);assert headmax<.05;assert g['head_aim']['bone_length_max_error']<1e-4
  if g['output_scope']=='full':assert g['scope_execution']['vertices_after']==g['scope_execution']['vertices_before']
  else:assert 0<g['scope_execution']['vertices_after']<g['scope_execution']['vertices_before']
  if s['structure'].get('torso_lean')=='upright' and 'upright_torso_geometry_gate' in s['notes']:upright+=1;assert library[s['pose_id']]['measures']['torso_lean_degrees']<=25
  for stage in ('direction','structure','anatomy'):
   evidence=json.loads((folder/f'person_{i}-{stage}.json').read_text());assert evidence['image_sha256']==item['image_sha256'];assert evidence['analysis'];assert 'user-feedback-eval' not in evidence['prompt'];assert 'evaluation-only' not in evidence['prompt']
 for variant,name in [('raw','scene-raw.fbx'),('final','scene.fbx')]:assert sha(folder/name)==scene['results'][variant]['fbx_sha256'];assert scene['results'][variant]['verification']['embedded_camera']
 checks.append(dict(cut_id=item['cut_id'],people=len(layout['people']),source_hash_and_fresh_receipts=True,fbx_hashes_and_camera=True,scope_skin_contract=True,head_and_bone_geometry=True,clearance_and_depth_order=True))
three=json.loads((ROOT/'three-verification.json').read_text());assert three['ok'] and len(three['cases'])==26;assert actors==34
report=dict(ok=True,cuts=13,people=actors,independent_image_evidence_receipts=actors*3,verified_upright_pose_gates=upright,head_aim_max_angular_error_degrees=headmax,cases=checks,meaning='Proves reproducible input/output consistency and rig geometry; does not prove original-image fidelity or correct person census.')
(ROOT/'artifact-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
