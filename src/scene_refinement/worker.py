"""Isolated Blender entry for scene preparation stages."""

import json, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.scene_refinement import people, arms, scenes, baseline

p = Path(sys.argv[sys.argv.index("--") + 1]).resolve(strict=True)
v = json.loads(p.read_text())
ctx = SimpleNamespace(**{k: Path(x) for k, x in v.items()})
for stage in sys.argv[sys.argv.index("--") + 2 :]:
    {
        "baseline": baseline.run,
        "people": people.run,
        "arms": arms.run,
        "scenes": scenes.run,
    }[stage](ctx)
