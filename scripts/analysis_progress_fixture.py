"""Generate OFFLINE mock-pipeline milestones; never calls a provider or renderer.

Run from repository root: python -m scripts.analysis_progress_fixture /tmp/producer.json
These are test fixtures, not evidence of a real uploaded artwork analysis.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile

from PIL import Image
from src.analysis_progress import capture_progress
from src.config import CFG
from tests.test_description_prefetch import Client, pipeline


def main(target):
    CFG.body_matching_mode = "off"
    CFG.vlm_person_tags = False
    CFG.vlm_hand_shapes = False
    with tempfile.TemporaryDirectory(prefix="pose-progress-fixture-") as directory:
        old_data_dir = CFG.data_dir
        CFG.data_dir = directory
        try:
            cases=[]
            for people, rough in ((1,False),(1,True),(2,False),(2,True)):
                image=Image.new("RGB",(100,200),"white")
                image.hint="full_half standing front rough" if rough else "full_half standing front"
                client=Client(people=people)
                for description in client.value.descriptions.values():
                    description.body_observations=dict(arms="raised",torso="forward")
                pipe=pipeline(client,rough=rough)
                snapshots=[]
                with capture_progress(snapshots.append):
                    pipe.process_cut(image,100,200)
                cases.append(dict(name=f"{'description' if rough else 'combined'}-{people}p",
                    providerCalls=len(client.calls), snapshots=snapshots))
            from tests.test_search_routing import _pipe, _Img
            CFG.search_routing_enabled = True
            pipe=_pipe()
            snapshots=[]
            with capture_progress(snapshots.append):
                pipe.process_cut(_Img("full_half standing front 2p rough"))
            cases.append(dict(name="legacy-mixed-2p",providerCalls=pipe.vlm.describe_calls,snapshots=snapshots))
            Path(target).write_text(json.dumps(dict(fixtureOnly=True,
                description="OFFLINE mock input/model; production pipeline emits milestones without extra calls/renders",
                cases=cases),ensure_ascii=False,indent=2)+"\n")
        finally:
            CFG.data_dir=old_data_dir


if __name__ == "__main__":
    main(sys.argv[1])
