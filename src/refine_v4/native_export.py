"""Bake only edited native chains before FBX round-trip verification."""


def bake_selected_rest(arm, meshes, selected):
    import bpy
    from mathutils import Matrix
    selected=set(selected)
    if not selected or not selected<={p.name for p in arm.pose.bones}:
        raise ValueError('explicit existing rest-bake bones required')
    if arm.animation_data:raise ValueError('animation must be baked first')
    for p in arm.pose.bones:
        if p.name not in selected and max(abs(p.matrix_basis[r][c]-Matrix.Identity(4)[r][c]) for r in range(4) for c in range(4))>2e-5:
            raise ValueError('nonselected pose must be baked before selected rest export')
    if any(m.data.shape_keys for m in meshes):raise ValueError('shape-key rest export unsupported')
    if any(len(m.modifiers)!=1 or m.modifiers[0].type!='ARMATURE' or m.modifiers[0].object!=arm for m in meshes):
        raise ValueError('one matching armature modifier per mesh required')
    baked=[]
    for m in meshes:
        bpy.ops.object.select_all(action='DESELECT');m.select_set(True);bpy.context.view_layer.objects.active=m
        mod=m.modifiers[0];baked.append((m,mod.name));bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.ops.object.select_all(action='DESELECT');arm.select_set(True);bpy.context.view_layer.objects.active=arm
    bpy.ops.object.mode_set(mode='POSE');bpy.ops.pose.select_all(action='DESELECT')
    for p in arm.pose.bones:p.select=p.name in selected
    bpy.ops.pose.armature_apply(selected=True);bpy.ops.object.mode_set(mode='OBJECT')
    for m,name in baked:
        mod=m.modifiers.new(name,type='ARMATURE');mod.object=arm;mod.use_vertex_groups=True
