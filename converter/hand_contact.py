"""Conservative bilateral-hand screening of the delivered, deformed mesh.

This is not a complete body/prop collision certificate. It detects crossing
triangles on opposite hands; containment and same-hand contacts need other gates.
"""
import numpy as np


def triangle_crossing(a,b,eps=1e-7):
    """Non-coplanar segment/triangle crossing, excluding near-touch endpoints."""
    a=np.asarray(a,float);b=np.asarray(b,float)
    for edges,face in ((a,b),(b,a)):
        e1,e2=face[1]-face[0],face[2]-face[0]
        for i,j in ((0,1),(1,2),(2,0)):
            origin=edges[i];direction=edges[j]-origin
            h=np.cross(direction,e2);det=np.dot(e1,h)
            if abs(det)<eps:continue
            v=origin-face[0];u=np.dot(v,h)/det
            if u<-eps or u>1+eps:continue
            q=np.cross(v,e1);w=np.dot(direction,q)/det
            if w<-eps or u+w>1+eps:continue
            t=np.dot(e2,q)/det
            if eps<t<1-eps:return True
    return False


def screen_hands(meshes):
    import bpy
    from mathutils.bvhtree import BVHTree
    vertices=[];triangles={'left':[],'right':[]}
    for obj in meshes:
        if obj.type!='MESH':continue
        groups={g.index:g.name.lower().replace('_','').replace(':','') for g in obj.vertex_groups}
        labels=[]
        for vertex in obj.data.vertices:
            weights={side:sum(g.weight for g in vertex.groups if side+'hand' in groups.get(g.group,''))
                     for side in ('left','right')}
            labels.append(max(weights,key=weights.get) if max(weights.values())>=.45 else None)
        evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
        try:
            if len(mesh.vertices)!=len(labels):raise ValueError('hand_screen_topology_changed')
            offset=len(vertices);vertices.extend([obj.matrix_world@v.co for v in mesh.vertices]);mesh.calc_loop_triangles()
            for face in mesh.loop_triangles:
                ids=list(face.vertices);side=labels[ids[0]]
                if side and all(labels[i]==side for i in ids):triangles[side].append(tuple(i+offset for i in ids))
        finally:evaluated.to_mesh_clear()
    if not all(triangles.values()):return dict(status='unavailable',reason='opposite_hand_skin_regions_missing')
    trees={side:BVHTree.FromPolygons(vertices,faces,all_triangles=True,epsilon=0.) for side,faces in triangles.items()}
    broad=trees['left'].overlap(trees['right']);crossings=[]
    for a,b in broad:
        if triangle_crossing([vertices[i] for i in triangles['left'][a]],[vertices[i] for i in triangles['right'][b]]):
            crossings.append((a,b))
    return dict(status='intersection' if crossings else 'clear_screen',
        method='evaluated_skin_opposite_hand_triangle_crossing_v1',
        triangle_counts={k:len(v) for k,v in triangles.items()},intersection_pairs=len(crossings),
        examples=crossings[:8],scope='opposite_hands_only',complete_collision_certificate=False)
