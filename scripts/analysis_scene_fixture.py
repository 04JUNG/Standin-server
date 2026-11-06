"""Offline original-VLM cut metadata through the production pipeline. No remote calls.

python -m scripts.analysis_scene_fixture /tmp/scene-producer.json
"""
import json
from pathlib import Path
import sys

from src.analysis_progress import capture_progress
from src.config import CFG
from src.vlm.client import _coerce
from tests.test_description_prefetch import Client
from tests.test_search_routing import _pipe, _Img


class FixtureClient(Client):
    def __init__(self, *, people):
        super().__init__(people=people)
        self.analyze_calls = 0

    def analyze(self, *args):
        self.analyze_calls += 1
        return super().analyze(*args)


def main(target):
    CFG.body_matching_mode = "off"
    CFG.vlm_person_tags = False
    CFG.vlm_hand_shapes = False
    CFG.search_routing_enabled = True
    boxes = [dict(x1=.1,y1=.1,x2=.4,y2=.9), dict(x1=.55,y1=.1,x2=.85,y2=.9)]
    inputs = [
        ("full-half-global-view-2p", dict(num_people=2,shot="full_half",view="three_quarter",
            person_views=["front","back"],approx_boxes=boxes)),
        ("bust-side-1p", dict(num_people=1,shot="bust",view="side",approx_boxes=boxes[:1])),
        ("face-back-1p", dict(num_people=1,shot="face",view="back",approx_boxes=boxes[:1])),
        ("explicit-zero-people", dict(num_people=0,shot="face",view="front",approx_boxes=[])),
        ("missing-count-invalid-labels-no-defaults", dict(shot="unknown",view="top",approx_boxes=boxes)),
        ("independent-valid-shot-only", dict(num_people="2",shot="face",view="unknown",approx_boxes=boxes)),
        ("independent-valid-view-only", dict(num_people=2.5,shot="unknown",view="back",approx_boxes=boxes)),
    ]
    cases = []
    for name, raw in inputs:
        raw = {**raw, "dialogue":"private fixture sentinel", "prompt":"never public"}
        analysis = _coerce(raw,100,200)
        client = FixtureClient(people=analysis.num_people)
        client.analysis = analysis
        pipe = _pipe(vlm=client)
        snapshots = []
        with capture_progress(snapshots.append):
            pipe.process_cut(_Img("full_half standing front 2p rough"),100,200)
        cases.append(dict(name=name,providerCalls=client.analyze_calls+len(client.calls),snapshots=snapshots))
    Path(target).write_text(json.dumps(dict(fixtureOnly=True,
        description="OFFLINE mock original-VLM inputs; production pipeline emits only validated scene observations",
        cases=cases),ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main(sys.argv[1])
