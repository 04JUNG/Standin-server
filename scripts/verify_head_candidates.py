"""Render selected real face suggestions and reimport the actual delivered FBX."""
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from PIL import Image, ImageDraw
from pose_curation.head.candidates import HeadCandidates
from pose_curation.head.queries import HeadQueries
from pose_curation.orientation import Orientation
from pose_curation.review.framing import FramedPreviews, NativeReference
from pose_curation.storage import read_json, sha256, write_json


def main():
    curation = PROJECT / 'data/curation'
    queries = HeadQueries(curation)
    candidates = HeadCandidates(curation, queries)
    service = FramedPreviews(curation)
    reference = NativeReference(service.character, sha256(service.character))
    output = PROJECT / 'data/dev/head-candidate-verification'
    output.mkdir(parents=True, exist_ok=True)
    results, tiles = [], []
    try:
        for key in ('rough_026','rough_084','rough_114','rough_279','rough_313'):
            query = queries.get(key)
            candidate = next(c for c in candidates.get(key)['candidates'] if c['status']=='suggested')
            suggestion = candidates.selected(key, query['content_hash'], candidate['id'])
            angles = Orientation(**{k:suggestion['orientation'][k] for k in ('yaw','pitch','roll')})
            state = service.wait_ready(reference,'head',orientation=angles)
            paths = {kind:service.artifact(reference,'head',angles,kind,state['version']) for kind in ('preview','fbx','settings')}
            for kind, path in paths.items():
                shutil.copyfile(path,output/(key+path.suffix))
            metadata = read_json(paths['settings'])
            results.append({'query':key,'suggestion':suggestion,'fbx_validation':metadata['fbx_validation'],
                            'fbx_sha256':sha256(paths['fbx']),'csp_verified':False})
            x0,y0,x1,y1=suggestion['bbox']
            side=max(x1-x0,y1-y0)*1.4
            with Image.open(query['path']) as image:
                crop=image.convert('RGB').crop(((x0+x1-side)/2,(y0+y1-side)/2,(x0+x1+side)/2,(y0+y1+side)/2))
            crop.thumbnail((300,300))
            tile=Image.new('RGB',(600,340),'white')
            tile.paste(crop,((300-crop.width)//2,(300-crop.height)//2))
            with Image.open(paths['preview']) as image:tile.paste(image.resize((300,300)),(300,0))
            ImageDraw.Draw(tile).text((8,312),f'{key}    y={angles.yaw} p={angles.pitch} r={angles.roll}',fill='black')
            tiles.append(tile)
            print({'query':key,'angles':angles.public(),'validation':metadata['fbx_validation']},flush=True)
    finally:
        service.close()
    sheet=Image.new('RGB',(600,340*len(tiles)),'white')
    for i,tile in enumerate(tiles):sheet.paste(tile,(0,340*i))
    sheet.save(output/'comparison.jpg')
    write_json(output/'verification.json',{'cases':results,'kind':'visual review candidates, not 3D ground truth'})


if __name__=='__main__':main()
