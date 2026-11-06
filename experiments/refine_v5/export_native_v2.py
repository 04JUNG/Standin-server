"""V5 offline export adapter; preserve already imported terminal bones.
Callers import with ignore_leaf_bones=False. Frozen converter settings stay intact.
"""

def export_preserving_bones(path):
    import bpy
    bpy.ops.export_scene.fbx(filepath=str(path), use_selection=False, apply_unit_scale=True, global_scale=1.0, axis_forward='-Z', axis_up='Y', object_types={'ARMATURE', 'MESH'}, use_mesh_modifiers=True, add_leaf_bones=False, primary_bone_axis='Y', secondary_bone_axis='X', bake_anim=False, bake_anim_use_all_bones=False, bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False, path_mode='AUTO', embed_textures=False, mesh_smooth_type='FACE')
