"""Blender-only authoring surrogate. Never replaces the full skin safety screen."""
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


class SkinFitCost:
    """Deterministic deformed-surface samples for local coordinate descent.

    Signed nearest normals are deliberately only an optimization heuristic.
    Final acceptance still requires inspect_self's containment/intersection test.
    """
    def __init__(self, arm, meshes):
        self.arm=arm;self.meshes=meshes;self.spec=[]
        for obj in meshes:
            names={g.index:g.name.split(':')[-1] for g in obj.vertex_groups}
            def region(name):
                if name in ['Hips','Spine','Spine1','Spine2','Neck']:return 'Torso'
                for s in ['Left','Right']:
                    if name.startswith(s+'Hand'):return s+'Hand'
                    if name in [s+'Arm',s+'Shoulder']:return s+'UpperArm'
                    if name==s+'ForeArm':return s+'Forearm'
                return None
            regions={i:region(n) for i,n in names.items()};scores=[]
            for v in obj.data.vertices:
                weights={}
                for g in v.groups:
                    r=regions.get(g.group)
                    if r:weights[r]=weights.get(r,0)+g.weight
                scores.append(weights)
            obj.data.calc_loop_triangles();faces={}
            for tri in obj.data.loop_triangles:
                w={}
                for i in tri.vertices:
                    for r,value in scores[i].items():w[r]=w.get(r,0)+value
                if w:faces.setdefault(max(w,key=w.get),[]).append(tuple(tri.vertices))
            self.spec.append((obj,faces))
        self.pairs=[('LeftForearm','RightForearm'),('LeftForearm','RightUpperArm'),('LeftUpperArm','RightForearm'),
            ('LeftHand','RightHand'),('LeftHand','RightUpperArm'),('RightHand','LeftUpperArm'),
            ('LeftHand','RightForearm'),('RightHand','LeftForearm'),('LeftHand','Torso'),('RightHand','Torso'),
            ('LeftUpperArm','Torso'),('RightUpperArm','Torso'),('LeftForearm','Torso'),('RightForearm','Torso')]
        self.seams={frozenset((s+'UpperArm','Torso')):s+'Arm' for s in ['Left','Right']}

    def __call__(self):
        graph=bpy.context.evaluated_depsgraph_get();vertices=[];faces={}
        for obj,spec in self.spec:
            ev=obj.evaluated_get(graph);mesh=ev.to_mesh()
            try:
                offset=len(vertices);vertices.extend(ev.matrix_world@v.co for v in mesh.vertices)
                for region,triangles in spec.items():faces.setdefault(region,[]).extend(tuple(offset+i for i in t) for t in triangles)
            finally:ev.to_mesh_clear()
        regions={}
        for name,triangles in faces.items():
            ids=sorted({i for t in triangles for i in t});sample=ids[::max(1,len(ids)//180)]
            points=np.array([vertices[i] for i in sample]);all_points=np.array([vertices[i] for i in ids])
            regions[name]={'points':points,'low':all_points.min(axis=0),'high':all_points.max(axis=0),
                'tree':BVHTree.FromPolygons(vertices,triangles,all_triangles=True)}
        cost=0.;details={}
        self.last_regions=regions
        for a,b in self.pairs:
            if a not in regions or b not in regions:continue
            penalty=0.
            for source,target in [(a,b),(b,a)]:
                points=regions[source]['points'];dest=regions[target]
                possible=np.all((points>dest['low'])&(points<dest['high']),axis=1)
                seam=self.seams.get(frozenset((source,target)))
                if seam:
                    center=np.array(self.arm.matrix_world@self.arm.pose.bones['mixamorig:'+seam].head)
                    possible&=np.linalg.norm(points-center,axis=1)>=.155
                for p in points[possible]:
                    hit=dest['tree'].find_nearest(Vector(p))
                    if hit[0] is not None and np.dot(p-np.array(hit[0]),np.array(hit[1]))<0:
                        penalty+=max(0,hit[3]*1000-.7)**2/len(points)
            details[a+'/'+b]=penalty;cost+=penalty
        return cost,details

    def insertion_slot(self,bones):
        """Is the rising palm between torso and opposite upperarm surfaces?"""
        point=lambda name:self.arm.matrix_world@bones[name].head
        wrist=point('RightHand');middle=point('RightHandMiddle1')
        centers=[(wrist+middle)/2,middle,(point('RightHandIndex1')+point('RightHandPinky1'))/2]
        rows=[]
        for p in centers:
            body=self.last_regions['Torso']['tree'].find_nearest(p)
            arm=self.last_regions['LeftUpperArm']['tree'].find_nearest(p)
            a=body[0]-p;b=arm[0]-p
            cosine=a.normalized().dot(b.normalized()) if a.length>1e-7 and b.length>1e-7 else 1.
            rows.append({'opposing_surface_cosine':cosine,'torso_distance_mm':body[3]*1000,'upperarm_distance_mm':arm[3]*1000,
                'between_surfaces':cosine<-.15 and body[3]<.06 and arm[3]<.06})
        return {'samples':rows,'between_fraction':sum(r['between_surfaces'] for r in rows)/len(rows),
            'method':'opposing_nearest_torso_and_upperarm_surfaces_at_palm_core','contact_certified':False}


def role_contact_metrics(arm,meshes):
    """Whole tucked hand and rising distal fingers, without assuming two palms."""
    graph=bpy.context.evaluated_depsgraph_get();vertices=[];faces={};samples={}
    for obj in meshes:
        ev=obj.evaluated_get(graph);mesh=ev.to_mesh()
        try:
            offset=len(vertices);world=[ev.matrix_world@v.co for v in mesh.vertices];vertices.extend(world)
            names={g.index:g.name.split(':')[-1] for g in obj.vertex_groups};weights=[]
            for v,p in zip(mesh.vertices,world):
                row={}
                for side in ['Left','Right']:
                    row[side+'UpperArm']=sum(g.weight for g in v.groups if names.get(g.group) in [side+'Arm',side+'Shoulder'])
                    row[side+'Forearm']=sum(g.weight for g in v.groups if names.get(g.group)==side+'ForeArm')
                    hand=sum(g.weight for g in v.groups if names.get(g.group,'').startswith(side+'Hand'))
                    finger=sum(g.weight for g in v.groups if names.get(g.group) in [side+'Hand'+f+str(i) for f in ['Index','Middle','Ring','Pinky'] for i in [2,3]])
                    if hand>=.45:samples.setdefault(side+'Hand',[]).append(p)
                    if finger>=.45:samples.setdefault(side+'DistalFingers',[]).append(p)
                weights.append(row)
            mesh.calc_loop_triangles()
            for tri in mesh.loop_triangles:
                for region in weights[0]:
                    if sum(weights[i][region]>=.45 for i in tri.vertices)>=2:faces.setdefault(region,[]).append(tuple(offset+i for i in tri.vertices))
        finally:ev.to_mesh_clear()
    trees={n:BVHTree.FromPolygons(vertices,f,all_triangles=True) for n,f in faces.items()}
    result={}
    for name,source,target in [('tucked_hand_to_opposite_upperarm','LeftHand','RightUpperArm'),
                               ('rising_fingers_to_opposite_upperarm','RightDistalFingers','LeftUpperArm'),
                               ('rising_hand_to_front_forearm','RightHand','LeftForearm')]:
        ds=np.array([trees[target].find_nearest(p)[3]*1000 for p in samples[source]])
        result[name]={'samples':len(ds),'distance_mm_min':float(ds.min()),'distance_mm_p10':float(np.percentile(ds,10)),
            'within_12mm_fraction':float(np.mean(ds<=12)),'method':'actual_deformed_skin_nearest_surface',
            'contact_certified':False}
    return result
