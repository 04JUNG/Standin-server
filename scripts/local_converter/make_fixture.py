"""Blender-only, authored geometry fixture; no third-party models in Git.

Creates one skinned humanoid with two meshes, a transformed accessory, preserved
helper joints and left-only finger bones. Use -- <output.fbx> in Blender.
This is a structural regression fixture, not an actual-artist compatibility gate.
"""
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from converter.bone_map import CANONICAL_PARENT


def build(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    heads = {"hips": (0, 0, 1), "spine": (0, 0, 1.1), "spine1": (0, 0, 1.25),
             "spine2": (0, 0, 1.4), "neck": (0, 0, 1.55), "head": (0, 0, 1.65)}
    for side, sign in (("L", 1), ("R", -1)):
        for name, x, z in (("shoulder", .08, 1.45), ("upperarm", .2, 1.45),
                           ("forearm", .5, 1.45), ("hand", .8, 1.45),
                           ("upleg", .12, .95), ("leg", .12, .55),
                           ("foot", .12, .12), ("toe", .12, .05)):
            heads[name + "." + side] = (sign*x, -.1 if name == "toe" else 0, z)
    data = bpy.data.armatures.new("FixtureSkeleton")
    arm = bpy.data.objects.new("FixtureCharacter", data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for name, head in heads.items():
        bone = data.edit_bones.new(name)
        bone.head = head
        children = [n for n, p in CANONICAL_PARENT.items() if p == name]
        bone.tail = heads[children[0]] if children else Vector(head) + Vector((0, 0, .08))
        if CANONICAL_PARENT[name]:
            bone.parent = data.edit_bones[CANONICAL_PARENT[name]]
    helper = data.edit_bones.new("ForearmHelper.L")
    helper.head, helper.tail = (.65, 0, 1.45), (.72, 0, 1.45)
    helper.parent = data.edit_bones["forearm.L"]
    data.edit_bones["hand.L"].parent = helper
    for finger, offset in (("Thumb", -.04), ("Index", -.02), ("Middle", 0), ("Ring", .02), ("Pinky", .04)):
        parent = data.edit_bones["hand.L"]
        for joint in range(1, 4):
            bone = data.edit_bones.new(f"LeftHand{finger}{joint}")
            bone.head = (.82 + .035*joint, offset, 1.45)
            bone.tail = (.855 + .035*joint, offset, 1.45)
            bone.parent = parent
            parent = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    vertices, faces, groups = [], [], {}
    for name in heads:
        bone = data.bones[name]
        center = (bone.head_local + bone.tail_local) * .5
        indices = []
        for x, y, z in ((-.035,-.035,-.035),(.035,-.035,-.035),(.035,.035,-.035),(-.035,.035,-.035),
                         (-.035,-.035,.035),(.035,-.035,.035),(.035,.035,.035),(-.035,.035,.035)):
            indices.append(len(vertices))
            vertices.append(center + Vector((x,y,z)))
        start = indices[0]
        faces.extend(tuple(start+n for n in quad) for quad in ((0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)))
        groups[name] = indices
    mesh = bpy.data.meshes.new("OriginalVertices")
    mesh.from_pydata(vertices, [], faces)
    body = bpy.data.objects.new("FixtureBody", mesh)
    bpy.context.collection.objects.link(body)
    for name, indices in groups.items():
        body.vertex_groups.new(name=name).add(indices, 1., "REPLACE")
    modifier = body.modifiers.new("OriginalSkin", "ARMATURE")
    modifier.object = arm
    bpy.ops.mesh.primitive_cube_add(size=.08, location=(.04, -.08, 1.2), rotation=(0, 0, .3))
    accessory = bpy.context.object
    accessory.name = "TransformedClothing"
    accessory.vertex_groups.new(name="spine").add(list(range(8)), 1., "REPLACE")
    accessory.modifiers.new("ClothingSkin", "ARMATURE").object = arm
    material = bpy.data.materials.new("FixtureSolid")
    material.diffuse_color = (.4, .5, .6, 1)
    body.data.materials.append(material)
    accessory.data.materials.append(material)
    bpy.ops.export_scene.fbx(filepath=str(path), object_types={"ARMATURE", "MESH"}, add_leaf_bones=False,
                             bake_anim=False, axis_forward="-Z", axis_up="Y")


if __name__ == "__main__":
    build(Path(sys.argv[sys.argv.index("--") + 1]).resolve())
