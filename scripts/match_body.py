#!/usr/bin/env python3
"""PNG -> existing person/pose pipeline -> body observations and automatic choice."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.config import CFG
from src.pipeline import Pipeline
from src.repo import load_entries
from src.experimental.body_matching.service import BodyMatchingService


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--catalog',type=Path,default=Path(CFG.body_catalog_path))
    parser.add_argument('--db',default='data/poses.db')
    parser.add_argument('--provider',choices=['mock','gemini'],default=CFG.body_vlm_provider)
    parser.add_argument('--model',default=CFG.body_vlm_model)
    parser.add_argument('--shadow',action='store_true')
    parser.add_argument('--synthetic',action='store_true',help='explicit mock person/pose pipeline; no real pose inference')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    service=BodyMatchingService(args.catalog,provider=args.provider,model=args.model,
                               timeout_seconds=CFG.body_timeout_seconds,max_people=CFG.body_max_people)
    CFG.body_matching_mode='shadow' if args.shadow else 'auto'
    if args.synthetic:
        from src.library import build_synthetic_index
        from src.pose import MockPoseModel
        from src.vlm.client import MockVLMClient
        pipe=Pipeline(build_synthetic_index(),vlm_client=MockVLMClient(),pose_model=MockPoseModel(),body_matcher=service)
    else:
        from src.runtime_guard import ensure_production_backends
        pipe=Pipeline(load_entries(args.db),body_matcher=service)
        ensure_production_backends(pipe,is_production=CFG.app_env=='production',
                                   requested_vlm=CFG.vlm_provider,requested_pose=CFG.pose_backend,
                                   requested_pose_variant=CFG.pose_model_variant)
    with Image.open(args.image) as image:
        image=image.convert('RGB')
        result=pipe.process_cut(image,*image.size)
    output=dict(route=result.route,count_confidence=result.count_confidence,
                synthetic_pose=args.synthetic,body_matching=result.body_matching)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(str(args.output.resolve()))


if __name__=='__main__':main()
