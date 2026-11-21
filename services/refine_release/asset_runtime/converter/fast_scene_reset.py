"""Lab-only reset to a SHA-bound Blender factory-empty scene; no solver/math changes."""
import hashlib,os
from pathlib import Path
PATH=Path(__file__).with_name('empty_factory.blend')
EXPECTED_SHA='5c51c07d9b0263e0997ab27f49d65f8bcc5cab6211fed8935f7383132904654c'
EXPECTED_SCENE={'objects': 0, 'scenes': 1, 'worlds': 0, 'collections': 0, 'frame': [1, 250, 1], 'unit': 'METRIC', 'scale': 1.0, 'fps': 24, 'engine': 'BLENDER_EEVEE', 'render': [1920, 1080, 100]}
CALLS=0

def template_path():
    return Path(os.getenv('CONVERTER_EMPTY_SCENE_TEMPLATE', str(PATH)))

def identity():
    path=template_path()
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED_SHA:
        raise RuntimeError('factory-empty template integrity mismatch')
    return dict(strategy='factory-empty-blend.lab.v1',template_sha256=EXPECTED_SHA)

def fingerprint():
    import bpy
    s=bpy.context.scene
    return dict(objects=len(bpy.data.objects),scenes=len(bpy.data.scenes),worlds=len(bpy.data.worlds),collections=len(bpy.data.collections),
        frame=[s.frame_start,s.frame_end,s.frame_current],unit=s.unit_settings.system,scale=s.unit_settings.scale_length,
        fps=s.render.fps,engine=s.render.engine,render=[s.render.resolution_x,s.render.resolution_y,s.render.resolution_percentage])

def reset_scene():
    import bpy
    global CALLS
    result=bpy.ops.wm.open_mainfile(filepath=str(template_path()),load_ui=False,use_scripts=False)
    if result!={'FINISHED'} or fingerprint()!=EXPECTED_SCENE:
        raise RuntimeError('factory-empty scene state mismatch')
    CALLS+=1

def install():
    from converter import retarget as rt
    identity()
    rt.reset_scene=reset_scene

def receipt():
    return dict(**identity(),calls=CALLS)
