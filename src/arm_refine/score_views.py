"""Native paired-evaluation cameras. Shared by before, part-only and contact views."""
import math
import bpy
from mathutils import Vector
from converter.preview_style import configure_workbench

def points(meshes,upper=False):
 graph=bpy.context.evaluated_depsgraph_get();result=[]
 for obj in meshes:
  names={g.index:g.name.split(':')[-1] for g in obj.vertex_groups}
  ev=obj.evaluated_get(graph);mesh=ev.to_mesh()
  try:
   for v in mesh.vertices:
    if upper and sum(g.weight for g in v.groups if names.get(g.group,'').startswith(('LeftHand','RightHand')) or names.get(g.group) in ['Spine','Spine1','Spine2','Neck','Head','LeftShoulder','RightShoulder','LeftArm','RightArm','LeftForeArm','RightForeArm'])<.5:continue
    result.append(ev.matrix_world@v.co)
  finally:ev.to_mesh_clear()
 return result
def camera_spec(pts,yaw):
 low=Vector([min(p[i] for p in pts) for i in range(3)]);high=Vector([max(p[i] for p in pts) for i in range(3)])
 center=(low+high)/2;direction=Vector((math.sin(math.radians(yaw)),-math.cos(math.radians(yaw)),.015)).normalized()
 rot=(-direction).to_track_quat('-Z','Y');view=rot.to_matrix().transposed()
 projected=[view@(p-center) for p in pts]
 size=max(max(p.x for p in projected)-min(p.x for p in projected),max(p.y for p in projected)-min(p.y for p in projected))*1.16
 return {'location':list(center+direction*5),'rotation':list(rot),'ortho_scale':size}
def render(path,spec):
 scene=bpy.context.scene;configure_workbench(scene)
 for o in list(bpy.data.objects):
  if o.type in ['CAMERA','LIGHT']:bpy.data.objects.remove(o,do_unlink=True)
 bpy.ops.object.camera_add(location=spec['location']);c=bpy.context.object;c.rotation_mode='QUATERNION';c.rotation_quaternion=spec['rotation']
 c.data.type='ORTHO';c.data.ortho_scale=spec['ortho_scale'];c.data.clip_end=100;scene.camera=c
 sh=scene.display.shading;sh.single_color=(.82,.84,.88);sh.show_cavity=True;sh.cavity_type='BOTH';sh.background_type='VIEWPORT';sh.background_color=(.045,.055,.075)
 scene.display.render_aa='16';scene.render.resolution_x=scene.render.resolution_y=640;scene.render.resolution_percentage=100
 scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.filepath=str(path)
 bpy.ops.render.render(write_still=True)
