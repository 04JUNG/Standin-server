"""Executed by trusted Blender; verifies actual version, imports and rig."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpy
from converter.protocol import EXPECTED_BLENDER_VERSION, EXPECTED_BLENDER_BUILD_HASH
from converter.scene import _import
from src.scene_refinement.rig import validate_rig

# Import every worker stage before announcing readiness.
from src.scene_refinement import baseline, people, arms, scenes
from src.refine_v4 import runner as parts_runner
from src.arm_refine import v4_adapter as parts_arms
from converter.scene_camera_transport import camera_binding

assert bpy.app.version[:3] == tuple(map(int, EXPECTED_BLENDER_VERSION.split(".")))
assert bpy.app.build_hash.decode() == EXPECTED_BLENDER_BUILD_HASH
bpy.ops.wm.read_factory_settings(use_empty=True)
objects = _import(Path(sys.argv[sys.argv.index("--") + 1]))
rigs = [o for o in objects if o.type == "ARMATURE"]
meshes = [o for o in objects if o.type == "MESH"]
assert len(rigs) == 1 and meshes
names = validate_rig(rigs[0], meshes)
assert all(v.co.length < 1e6 for m in meshes for v in m.data.vertices)
print("SCENE_PREFLIGHT_OK " + json.dumps({"bones": len(names)}))
