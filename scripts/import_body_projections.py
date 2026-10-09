#!/usr/bin/env python3
"""Import converter-exported target-FBX COCO17 projections for body comparison.
This does not render or retarget an FBX. Its input must come from that exact target.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.body_catalog import atomic_write
from src.experimental.body_matching.catalog import file_sha256, load_catalog
from src.experimental.body_matching.observation import geometry_ratios
from src.experimental.body_matching.schema import GEOMETRY_VERSION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalog',type=Path)
    parser.add_argument('projections',type=Path)
    parser.add_argument('--catalog-version',required=True)
    args=parser.parse_args()
    load_catalog(args.catalog)
    raw=json.loads(args.catalog.read_text())
    data=json.loads(args.projections.read_text())
    asset=next(a for a in raw['assets'] if a['body_id']==data['body_id'])
    for key in ('body_version','rig_version','asset_sha256'):
        if data[key]!=asset[key]:raise ValueError('projection asset identity mismatch: '+key)
    if file_sha256(args.catalog.resolve().parent/asset['fbx_path'])!=data['asset_sha256']:
        raise ValueError('projection FBX hash mismatch')
    for key in ('retarget_version','renderer_version'):
        if not isinstance(data.get(key),str) or not data[key]:raise ValueError('missing '+key)
    records=[]
    for p in data['projections']:
        if p['pose_id'] not in asset['supported_pose_ids']:raise ValueError('pose has not passed asset QA')
        if p['view'] not in {'front','three_quarter','side','back','side_opposite','three_quarter_opposite'}:
            raise ValueError('invalid projection view')
        if len(p['valid_joint_mask'])!=17 or any(type(x) is not bool for x in p['valid_joint_mask']):
            raise ValueError('valid_joint_mask must contain 17 booleans')
        ratios=geometry_ratios(p['keypoints'],p['valid_joint_mask'])
        if not ratios:raise ValueError('projection has no usable torso/ratios')
        records.append({key:p[key] for key in ('pose_id','view','pose_sha256')} | {
            'geometry_version':GEOMETRY_VERSION,'ratios':ratios,
            'target_asset_sha256':asset['asset_sha256'],'rig_version':asset['rig_version'],
            'retarget_version':data['retarget_version'],'renderer_version':data['renderer_version']})
    asset['projections']=records
    raw['catalog_version']=args.catalog_version
    atomic_write(args.catalog,raw)
    print(json.dumps({'body_id':asset['body_id'],'imported':len(records),'catalog_version':args.catalog_version}))


if __name__=='__main__':main()
