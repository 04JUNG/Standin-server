"""Real image/model/library preflight. No VLM request, ECS write or export.

Uses the same provisioning and resource builder as API lifespan. VLM secret
references are validated separately in prepare_fullbody_deployment, not fetched
or replaced with dummy keys here. This is not an end-to-end VLM accuracy test.
"""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    if os.getenv('ROUTING_FULLBODY_ENABLED') != '1':
        print('fullbody preflight skipped: explicit legacy deployment')
        return
    from api.app import _ensure_db, _resolve_library_identity, _ensure_pose_model_bundle, DB_PATH
    from src.fullbody_routing import _build_fullbody_resources
    entries = _ensure_db()
    library = _resolve_library_identity()
    _ensure_pose_model_bundle()
    from src.search_recall_assets import ensure_recall_assets
    assets_root = ensure_recall_assets()
    pose, semantic, engine = _build_fullbody_resources(entries, db_path=DB_PATH)
    if assets_root is not None:
        if engine.search_evidence is None or os.getenv('ROUTING_LUNA_SIX_FIELD_ENABLED') != '1':
            raise ValueError('staging recall requires search evidence and Luna six-field descriptions')
        from src.config import CFG
        from src.experimental.search_recall.engine import RecallAssets, RecallSearch
        assets = RecallAssets.from_root(assets_root)
        recall = RecallSearch(entries, Path(DB_PATH), Path(CFG.data_dir), assets)
        recall.close()
    print(json.dumps(dict(status='passed', pose_model=pose.model_id,
                          library_version=library['version'],
                          searchable_poses=len(engine.camera.pose_ids), camera_count=len(engine.camera.cameras),
                          semantic_backend=semantic.backend,
                          ranking=engine.ranking, semantic_weight=engine.semantic_weight,
                          geometry_weight=engine.geometry_weight,
                          generation_files_checked=engine.generator is not None,
                          recall_assets_checked=assets_root is not None,
                          vlm_request_tested=False, deployed=False)))


if __name__ == '__main__':
    main()
