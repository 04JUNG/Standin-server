"""Fresh-input recall profile for the inference server."""
from copy import copy, deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Lock
from time import sleep

from src.camera_search import CameraIndex
from src.fullbody_assets import prepare_assets
from src.fullbody_routing import API_SEMANTIC_WEIGHT, FullBodySearch
from src.hand_skin_screen import build_hand_skin_gate
from src.semantic_adapter_285 import build_semantic285_searcher, excluded_pose_ids

from . import service_context as service
from .crop_joints import RotatedCropJoints
from .cross_chest_batch import replay as replay_cross_chest
from .display import prepare_engine as prepare_baseline
from .fitted_geometry import AlternatePoseObserver, DepthObserver, FittedGeometry
from .hand_geometry import HandContactIndex, VERSION as HAND_INDEX_VERSION
from .learned_fusion import FOLDS, fold
from .overnight_search import BoneEvidence, COMPOUND_CONTACT_EXPERIMENT, ScreeningUnavailable, replay
from .structured_search import StructuredIndex
from .verified_recall import VerifiedRecall
from .visible_relations import VisibleRelations


@dataclass(frozen=True)
class RecallAssets:
    pose_manifest: Path
    body_model: Path
    depth_model: Path
    embedding_model: Path
    prop_model: Path
    prop_python: str
    cache_root: Path

    @classmethod
    def from_root(cls, root):
        root = Path(root).resolve()
        assets = cls(root / 'pose-model/manifest.json', root / 'body.onnx', root / 'depth.onnx',
                     root / 'multilingual-e5-small', root / 'grounding-dino',
                     os.getenv('ROUTING_RECALL_PROP_PYTHON') or sys.executable,
                     root / 'cache')
        required = (assets.pose_manifest, assets.body_model, assets.depth_model,
                    assets.embedding_model, assets.prop_model)
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise ValueError('Missing 04J-344 recall assets: ' + ', '.join(missing))
        return assets


def write_cache(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


class RecallSearch:
    def __init__(self, entries, db_path, gallery, assets):
        allowed = {entry.pose_id for entry in entries} - excluded_pose_ids()
        paths = prepare_assets(db_path=db_path, root=gallery, allowed=allowed)
        semantic = build_semantic285_searcher(
            tags_path=paths['tags'], taxonomy_path=paths['taxonomy'],
            db_path=db_path, library_pose_ids={entry.pose_id for entry in entries},
            body_index_path=paths['body'], shape_index_path=paths['shapes'])
        camera = CameraIndex.from_library(allowed, paths['bvh'], count=117)
        gate = build_hand_skin_gate(camera, paths['bvh'])
        if gate is None:
            raise ValueError('04J-344 recall requires the hand mesh screening service')
        self.engine = service.ArmPenaltySearch(entries, semantic, camera,
                                               semantic_weight=API_SEMANTIC_WEIGHT, hand_skin_gate=gate)
        self.cross_chest = FullBodySearch(
            entries, copy(semantic), camera, semantic_weight=API_SEMANTIC_WEIGHT, hand_skin_gate=gate,
            enable_waist_down_pair=True, enable_cross_chest_hint=True)
        self.bone = BoneEvidence(camera)
        self.fitted = FittedGeometry(camera)
        self.structured = StructuredIndex(self.engine)
        self.crop = RotatedCropJoints(assets.pose_manifest, AlternatePoseObserver(assets.depth_model))
        self.crop.rotation_fallback = True
        self.depth = DepthObserver(assets.depth_model)
        self.relations = VisibleRelations(self.engine, pointing=True, hand_placement=True, kneeling=True)
        self.hands = HandContactIndex(camera, semantic.bvh_root, assets.cache_root / f'{HAND_INDEX_VERSION}.json',
                                      write_cache)
        weights = json.loads(Path(__file__).with_name('compound-weights.json').read_text())['weights']
        if len(weights) != FOLDS:
            raise ValueError('Unexpected compound ranking weights')
        self.weights = weights
        self.verified_recall = VerifiedRecall(self.engine, assets, paths['tags'])
        self.lock = Lock()

    def _baseline(self, person):
        previous = person['dynamic_search']
        if previous['route'] not in ('hybrid', 'geometric', 'semantic'):
            return
        joint = service.JointEvidence(**previous['geometry_evidence'])
        semantic = service.SemanticEvidence(**previous['description_evidence'])
        engine = prepare_baseline(self.engine, 'combined_geometry_weight', joint.sufficient)
        scope = person['output_scope']
        evidence = dict(keypoints=person['keypoints'], coverage_class=person['coverage_class'],
                        joint_mask=person['quality_trace']['search_valid_joint_mask'],
                        output_scope=scope['detected'], output_scope_source=scope['source'],
                        hand_presets={side: person['hands'][side]['preset'] for side in ('left', 'right')})
        report = engine.search(service.decide_route(joint, semantic), evidence,
                               review_geometry=service.integrated_enabled(service.CFG))
        if report['status'] == previous['status'] == 'no_result' and previous.get('generation'):
            report['generation'] = deepcopy(previous['generation'])
        person['dynamic_search'] = report
        person['pose_presentation'] = service.build_pose_presentation(
            report, person_index=person['index'], output_scope=scope['mesh_scope'],
            resolve_library=service._library_preview_asset).model_dump(mode='json')

    def _search(self, cut, image, weights):
        people = cut['people']
        for person in people:
            if not person.get('dynamic_search'):
                continue
            self._baseline(person)
            result = replay(self.engine, self.bone, person, COMPOUND_CONTACT_EXPERIMENT,
                            self.crop, image, self.fitted, self.depth, self.structured,
                            rank_weights=weights, visible_relations=self.relations,
                            person_count=len(people), hand_index=self.hands, allow_recognition_error=True)
            if result is None:
                replay_cross_chest(self.cross_chest, person)

    def rerank(self, data, cut):
        digest = hashlib.sha256(data).hexdigest()
        weights = self.weights[fold(digest)]
        with TemporaryDirectory(prefix='standin-search-recall-') as directory, self.lock:
            image = Path(directory) / 'rough.png'
            image.write_bytes(data)
            observations = self.verified_recall.observe(cut, image)
            for attempt in range(2):
                try:
                    candidate = deepcopy(cut)
                    self._search(candidate, image, weights)
                    return self.verified_recall.rerank(candidate, digest, observations)
                except ScreeningUnavailable:
                    if attempt:
                        raise
                    sleep(1)

    def close(self):
        self.verified_recall.close()
