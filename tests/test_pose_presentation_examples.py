"""Private pack generator: authored test rig only, no licensed library inputs."""
import hashlib
import json
import zipfile

import pytest
from scripts.build_pose_presentation_examples import make_examples, ROOT
from api.models import CutResultOut


def test_private_pack_roundtrips_real_schema_and_bytes(tmp_path):
    from tests.test_part_composition import rig
    from tests.test_text_pose_generation import serialize_rig
    joints,frame=rig()
    raw=serialize_rig(joints,frame)
    library=tmp_path/'library-pose.bvh';library.write_text(raw)
    generated=tmp_path/'generated.bvh';generated.write_text(raw)
    output=tmp_path/'private-examples'
    output.mkdir();(output/'unrelated-secret.txt').write_text('not part of artifact')
    receipt=make_examples(library,generated,output)
    manifest=json.loads((output/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==expected
    with zipfile.ZipFile(receipt['archive']) as z:
        assert 'unrelated-secret.txt' not in z.namelist()
        assert z.read('assets/library.bvh')==library.read_bytes()
        assert len([n for n in z.namelist() if n[:2].isdigit() and n.endswith('.json')])==11
    for path in output.glob('[0-9][0-9]-*.json'):
        CutResultOut.model_validate_json(path.read_text())
    normal=json.loads((output/'01-library.json').read_text())['people'][0]['pose_presentation']
    new=json.loads((output/'03-generated.json').read_text())['people'][0]['pose_presentation']
    assert normal['library_candidates'][0]['pose_id']=='library-pose'
    assert new['generated_candidates'][0]['pose_id'] is None
    assert new['generated_candidates'][0]['asset']['bvh'].encode('utf-8')==generated.read_bytes()
    assert receipt['cases']==11 and manifest['fixture_kind']=='offline_contract_not_live_capture'


def test_pack_refuses_library_files_in_public_repository(tmp_path):
    with pytest.raises(ValueError,match='outside_repository'):
        make_examples(tmp_path/'missing.bvh',tmp_path/'missing.bvh',ROOT/'docs'/'bad-assets')
