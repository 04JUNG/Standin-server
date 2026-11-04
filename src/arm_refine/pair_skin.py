"""Blender-only palm-to-opposite-upperarm skin proximity measurements."""
import numpy as np
import bpy
from mathutils.bvhtree import BVHTree
from converter.hand_geometry import hand_frame

def skin_metrics(arm,meshes,*,authoring_steps=False):
    vertices=[];faces={side:[] for side in ['left','right']};palms={side:[] for side in faces}
    graph=bpy.context.evaluated_depsgraph_get();frames={};origins={}
    for side in faces:
        def point(suffix):return np.array(arm.matrix_world @ arm.pose.bones['mixamorig:'+side.title()+suffix].head)
        origins[side]=point('Hand');bases={f:point('Hand'+f.title()+'1') for f in ['thumb','index','middle','ring','pinky']}
        frames[side]=(hand_frame(side,origins[side],bases),float(np.linalg.norm(bases['middle']-origins[side])))
    for obj in meshes:
        ev=obj.evaluated_get(graph);mesh=ev.to_mesh()
        try:
            offset=len(vertices);world=[ev.matrix_world@v.co for v in mesh.vertices];vertices.extend(world)
            labels={g.index:g.name.split(':')[-1] for g in obj.vertex_groups};upper=[]
            for v,p in zip(mesh.vertices,world):
                weights={s:sum(g.weight for g in v.groups if labels.get(g.group,'') in [s.title()+'Arm',s.title()+'Shoulder']) for s in faces}
                upper.append(max(weights,key=weights.get) if max(weights.values())>=.45 else None)
                for side in faces:
                    palm_weight=sum(g.weight for g in v.groups if labels.get(g.group,'')==side.title()+'Hand')
                    if palm_weight<.45:continue
                    frame,length=frames[side];delta=np.array(p)-origins[side]
                    along=np.dot(delta,frame.length_axis)/length
                    if .15<=along<=1.05 and np.dot(delta,frame.palm_normal)>=0:palms[side].append(p.copy())
            mesh.calc_loop_triangles()
            for tri in mesh.loop_triangles:
                labels3=[upper[i] for i in tri.vertices]
                for side in faces:
                    if labels3.count(side)>=2:faces[side].append(tuple(offset+i for i in tri.vertices))
        finally:ev.to_mesh_clear()
    # Authoring may target only the surface facing away from the palm. The
    # ordinary quality measurement below always uses the entire upperarm.
    trees={s:BVHTree.FromPolygons(vertices,faces[s],all_triangles=True) for s in faces}
    results={}
    for side in faces:
        target='right' if side=='left' else 'left';ds=[];opposing=[]
        for p in palms[side]:
            hit=trees[target].find_nearest(p)
            ds.append(float(hit[3])*1000)
            opposing.append(float(np.dot(np.array(hit[1]),frames[side][0].palm_normal)))
        if not ds:raise ValueError('palm samples unavailable')
        results[side]={'samples':len(ds),'distance_mm_p10':float(np.percentile(ds,10)),
            'distance_mm_median':float(np.median(ds)),'distance_mm_min':float(min(ds)),
            'within_12mm_fraction':float(np.mean(np.array(ds)<=12)),
            'near_and_opposing_fraction':float(np.mean((np.array(ds)<=12)&(np.array(opposing)<-.2)))}
        if authoring_steps:
            normal=frames[side][0].palm_normal
            facing=[]
            for tri in faces[target]:
                a,b,c=[vertices[i] for i in tri]
                n=(b-a).cross(c-a).normalized()
                if np.dot(np.array(n),normal)<-.25:facing.append(tri)
            tree=BVHTree.FromPolygons(vertices,facing,all_triangles=True)
            shifts=[]
            for p in palms[side]:
                hit=tree.find_nearest(p)
                shifts.append(np.array(hit[0]+hit[1]*.003-p))
            results[side]['authoring_surface_step_m']=np.mean(shifts,axis=0).tolist()
    return results
