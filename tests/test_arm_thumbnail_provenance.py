"""Never attach a wrong-pose, wrong-side or modified thumbnail to a part."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('arm_thumb_publisher',Path(__file__).resolve().parents[1]/'scripts/publish_arm_thumbnails.py')
publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)


def fixture(tmp_path):
    package=tmp_path/'parts';native=tmp_path/'renders';package.mkdir();native.mkdir()
    part=package/'part.json';part.write_text('part payload')
    item={'id':'pose--left','selection':'left','source':{'sha256':'bvh-hash'},'json':'part.json','sha256':publisher.sha(part)}
    record={'part_id':'pose--left','selection':'left','source_bvh_sha256':'bvh-hash','style':publisher.STYLE,'images':{},'library_references':{}}
    for group in ['images','library_references']:
        for view in publisher.VIEWS:
            path=native/(group+'-'+view+'.png');path.write_bytes((group+view).encode())
            record[group][view]={'path':path.name,'sha256':publisher.sha(path),'width':512,'height':512,'border_pixels':20}
    return item,record,package,native


def test_verified_source_accepted(tmp_path):
    publisher.verify(*fixture(tmp_path))


@pytest.mark.parametrize('bad',['source','side','part','image','view','framing'])
def test_mixed_or_modified_asset_rejected(tmp_path,bad):
    item,record,package,native=fixture(tmp_path)
    if bad=='source':record['source_bvh_sha256']='another-pose'
    elif bad=='side':record['selection']='right'
    elif bad=='part':(package/'part.json').write_text('changed pose')
    elif bad=='image':(native/record['images']['front']['path']).write_bytes(b'other image')
    elif bad=='view':del record['images']['side']
    else:record['images']['front']['border_pixels']=0
    with pytest.raises(ValueError):publisher.verify(item,record,package,native)
