"""Actual deformed-skin self-penetration triage, including head and fingers.

No skeleton-only clearance claims. Partition all weighted triangles into body
regions, cap their cut boundaries, test non-adjacent surfaces AND containment.
Connected joints have local seam exemptions, not whole-region exemptions.
"""
from collections import Counter,defaultdict
import math
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

VERSION='full-weighted-skin-v2-whole-hands-20261015'
DEPTH=.002  # 2 mm skin-intersection triage tolerance, not clinical clearance.


def region(name):
    name=name.removeprefix('mixamorig:')
    if name in {'Hips','Spine','Spine1','Spine2','Neck'}:return 'Torso'
    if name=='Head' or name.startswith('Head_'):return 'Head'
    for side in ('Left','Right'):
        # Preserve every finger skin triangle, but do not cut one continuous
        # hand into artificial, sometimes non-manifold webbing volumes.
        if name.startswith(side+'Hand'):return side+'Hand'
        for bone,part in (('Shoulder','UpperArm'),('ForeArm','Forearm'),('Arm','UpperArm'),
                          ('Hand','Palm'),('UpLeg','Thigh'),('Leg','Shin'),('ToeBase','Foot'),('Foot','Foot')):
            if name==side+bone or name.startswith(side+bone+'_'):return side+part
    return None


def cap_loops(vertices,triangles):
    """Close region cuts so points completely inside another part are tested."""
    edges=Counter(tuple(sorted((a,b))) for face in triangles for a,b in zip(face,(face[1],face[2],face[0])))
    adj=defaultdict(set)
    for (a,b),count in edges.items():
        if count==1:adj[a].add(b);adj[b].add(a)
    remaining=set(adj);caps=[];unclosed=0
    while remaining:
        first=next(iter(remaining));component=set();stack=[first]
        while stack:
            v=stack.pop()
            if v in component:continue
            component.add(v);stack.extend(adj[v]-component)
        remaining-=component
        if any(len(adj[v])!=2 for v in component):unclosed+=1;continue
        loop=[first];previous=None;current=first
        for _ in range(len(component)+1):
            options=adj[current]-({previous} if previous is not None else set())
            nxt=next(iter(options))
            if nxt==first:break
            loop.append(nxt);previous,current=current,nxt
        if len(loop)!=len(component):unclosed+=1;continue
        center=sum((vertices[v] for v in loop),Vector())/len(loop)
        idx=len(vertices);vertices.append(center)
        for i,a in enumerate(loop):caps.append((a,loop[(i+1)%len(loop)],idx))
    return triangles+caps,unclosed,len(caps)


def inspect_self(arm):
    graph=bpy.context.evaluated_depsgraph_get();vertices=[];groups=defaultdict(list)
    for obj in bpy.context.scene.objects:
        if obj.type!='MESH' or not any(m.type=='ARMATURE' and m.object==arm for m in obj.modifiers):continue
        meshobj=obj.evaluated_get(graph);mesh=meshobj.to_mesh()
        try:
            offset=len(vertices);vertices.extend(meshobj.matrix_world@v.co for v in mesh.vertices)
            labels={g.index:region(g.name) for g in obj.vertex_groups}
            scores=[]
            for v in mesh.vertices:
                sums=defaultdict(float)
                for g in v.groups:
                    if labels.get(g.group):sums[labels[g.group]]+=g.weight
                scores.append(sums)
            mesh.calc_loop_triangles()
            for tri in mesh.loop_triangles:
                sums=defaultdict(float)
                for vi in tri.vertices:
                    for label,weight in scores[vi].items():sums[label]+=weight
                if sums:
                    winner=max(sums,key=sums.get)
                    groups[winner].append(tuple(offset+vi for vi in tri.vertices))
        finally:meshobj.to_mesh_clear()
    if not vertices or 'Head' not in groups:raise ValueError('full-body weighted skin regions missing')
    parts={}
    for name,triangles in groups.items():
        ids=sorted({v for tri in triangles for v in tri});local=[vertices[i].copy() for i in ids]
        remap={v:i for i,v in enumerate(ids)};faces=[tuple(remap[v] for v in tri) for tri in triangles]
        closed,unclosed,caps=cap_loops(local,faces)
        low=Vector(tuple(min(v[i] for v in local) for i in range(3)));high=Vector(tuple(max(v[i] for v in local) for i in range(3)))
        parts[name]=dict(vertices=local,original_count=len(ids),faces=faces,tree=BVHTree.FromPolygons(local,closed,all_triangles=True),
                         surface=BVHTree.FromPolygons(local,faces,all_triangles=True),low=low,high=high,unclosed=unclosed,caps=caps)
    def joint(name):
        b=arm.pose.bones.get('mixamorig:'+name)
        return arm.matrix_world@b.head if b else None
    seams={}
    def seam(a,b,bone,radius):
        center=joint(bone)
        if center is not None:seams[frozenset((a,b))]=(center,radius)
    seam('Head','Torso','Head',.085)
    for side in ('Left','Right'):
        seam('Torso',side+'UpperArm',side+'Arm',.155)
        seam('Torso',side+'Thigh',side+'UpLeg',.13)
        seam(side+'UpperArm',side+'Forearm',side+'ForeArm',.085)
        seam(side+'Forearm',side+'Hand',side+'Hand',.07)
        seam(side+'Thigh',side+'Shin',side+'Leg',.10)
        seam(side+'Shin',side+'Foot',side+'Foot',.08)
    collisions=[];checked=0;surface_intersections=0
    names=sorted(parts)
    ray=Vector((.823,.411,.391)).normalized()
    for n,a in enumerate(names):
        pa=parts[a]
        for b in names[n+1:]:
            pb=parts[b]
            if any(pa['high'][i]<pb['low'][i]-DEPTH or pb['high'][i]<pa['low'][i]-DEPTH for i in range(3)):continue
            checked+=1;seam_info=seams.get(frozenset((a,b)))
            overlaps=pa['surface'].overlap(pb['surface']);surface_intersections+=len(overlaps)
            max_depth=0.;inside=0;witness=None
            for source,target in ((pa,pb),(pb,pa)):
                for point in source['vertices'][:source['original_count']]:
                    if any(point[i]<target['low'][i]+1e-6 or point[i]>target['high'][i]-1e-6 for i in range(3)):continue
                    if seam_info and (point-seam_info[0]).length<seam_info[1]:continue
                    nearest=target['surface'].find_nearest(point)
                    if nearest[0] is None or nearest[3]<=DEPTH:continue
                    # Closed-region ray parity also catches full containment;
                    # signed nearest normals alone are unsafe on open skin cuts.
                    origin=point.copy();hits=0
                    for _ in range(64):
                        hit=target['tree'].ray_cast(origin,ray,10.)
                        if hit[0] is None:break
                        hits+=1;origin=hit[0]+ray*1e-5
                    if hits%2:
                        inside+=1
                        if nearest[3]>max_depth:max_depth=nearest[3];witness=list(point)
            if inside:collisions.append({'regions':[a,b],'penetrating_vertices':inside,
                'max_depth_mm':round(max_depth*1000,3),'surface_crossing_triangle_pairs':len(overlaps),'witness_world':witness})
    return {'version':VERSION,'method':'deformed weighted triangle regions + capped-region ray-parity containment; local connected-joint seam exemptions',
        'depth_tolerance_mm':DEPTH*1000,'regions':len(parts),'finger_skin_included':True,'tested_overlapping_region_pairs':checked,
        'raw_surface_crossings':surface_intersections,'collisions':collisions,
        'flags':[f"{' / '.join(c['regions'])}: {c['max_depth_mm']:.1f} mm skin penetration" for c in collisions],
        'region_closure_failures':{n:p['unclosed'] for n,p in parts.items() if p['unclosed']},
        'limitations':['Automated triage, not human visual approval.','Within one anatomical region is not tested against itself, including finger-to-finger/palm collisions within the same hand.','Connected-joint seams have small local exemptions.','Cut capping can overapproximate concave regions; flagged contacts require inspection.']}
