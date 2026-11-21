"""Offline Blender QA for registered versus posed FBX skin.

Diagnostic thresholds are review triggers, not a release certificate. Collision
checks detect strict edge-through-triangle crossings; coplanar overlap and fully
contained shells are not certified. Uses all FBX vertices and skin influences.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree


def read(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False, use_image_search=False,
                            ignore_leaf_bones=False, automatic_bone_orientation=False)
    objects = sorted((o for o in bpy.context.scene.objects if o.type == 'MESH'), key=lambda o:o.name)
    assert len(objects) == 1, 'This QA adapter expects the actual single-mesh fixture'
    obj = objects[0]
    mesh = obj.data
    mesh.calc_loop_triangles()
    vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices], dtype=float)
    faces = np.array([list(t.vertices) for t in mesh.loop_triangles], dtype=int)
    polygons = [list(p.vertices) for p in mesh.polygons]
    weights = [sorted((obj.vertex_groups[g.group].name, float(g.weight)) for g in v.groups) for v in mesh.vertices]
    labels = [max(w, key=lambda x:x[1])[0] if w else 'unweighted' for w in weights]
    arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    bones = {}
    for bone in arm.data.bones:
        matrix=np.asarray(arm.matrix_world @ bone.matrix_local,dtype=float).copy()
        # Registered FBX carries a 0.01 object unit scale; posed FBX applies
        # it to bone coordinates. Compare world-space rigid frames, not those
        # arbitrary basis-vector lengths. Reject nonuniform/scaled rigs here.
        u,s,vt=np.linalg.svd(matrix[:3,:3])
        assert np.allclose(s,s[0],rtol=1e-4), 'Nonuniform bone frame unsupported by this QA adapter'
        rotation=u@vt
        assert np.linalg.det(rotation)>0, 'Reflected bone frame unsupported'
        matrix[:3,:3]=rotation
        bones[bone.name]=matrix
    return vertices, faces, polygons, weights, labels, bones


def topology(vertices, faces):
    raw = np.concatenate((faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]))
    edges, counts = np.unique(np.sort(raw, axis=1), axis=0, return_counts=True)
    parent = np.arange(len(vertices))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a,b in edges:
        a,b=root(a),root(b)
        if a != b: parent[a]=b
    used = np.unique(faces)
    components = Counter(root(int(i)) for i in used)
    result = {'vertices':len(vertices), 'triangles':len(faces), 'edges':len(edges),
              'boundaryEdges':int((counts == 1).sum()), 'nonmanifoldEdges':int((counts > 2).sum()),
              'looseVertices':len(vertices)-len(used), 'components':len(components),
              'componentVertexCounts':sorted(components.values(), reverse=True),
              'triangulationSha256':hashlib.sha256(faces.astype('<i8').tobytes()).hexdigest()}
    return result, edges


def seam_pairs(vertices, tolerance=1e-7):
    tree = KDTree(len(vertices))
    for i,p in enumerate(vertices): tree.insert(Vector(p), i)
    tree.balance()
    return np.array([(i,j) for i,p in enumerate(vertices) for _,j,_ in tree.find_range(Vector(p), tolerance) if j>i], dtype=int).reshape(-1,2)


def crossing_mask(a, b):
    # A strict transverse intersection requires an edge to pass through the
    # interior of the other face; mere shared boundaries/contact do not qualify.
    found = np.zeros(len(a), dtype=bool)
    for source,target in ((a,b),(b,a)):
        e1,e2 = target[:,1]-target[:,0],target[:,2]-target[:,0]
        for k in range(3):
            start = source[:,k]
            direction = source[:,(k+1)%3]-start
            h = np.cross(direction,e2)
            determinant = np.einsum('ij,ij->i',e1,h)
            scale = np.linalg.norm(direction,axis=1)*np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)
            usable = abs(determinant) > np.maximum(scale*1e-10, 1e-25)
            inv = np.divide(1.,determinant,out=np.zeros_like(determinant),where=usable)
            s=start-target[:,0]
            u=inv*np.einsum('ij,ij->i',s,h)
            q=np.cross(s,e1)
            v=inv*np.einsum('ij,ij->i',direction,q)
            t=inv*np.einsum('ij,ij->i',e2,q)
            eps=1e-7
            found |= usable & (u>eps) & (v>eps) & (u+v<1-eps) & (t>eps) & (t<1-eps)
    return found


def collisions(vertices, faces):
    bvh=BVHTree.FromPolygons([Vector(v) for v in vertices], faces.tolist(), all_triangles=True, epsilon=0.)
    pairs=np.asarray([(a,b) for a,b in bvh.overlap(bvh) if a<b], dtype=int).reshape(-1,2)
    hits=[]
    triangles=vertices[faces]
    for offset in range(0,len(pairs),4096):
        chunk=pairs[offset:offset+4096]
        hits.extend(map(tuple,chunk[crossing_mask(triangles[chunk[:,0]],triangles[chunk[:,1]])].tolist()))
    return set(hits),len(pairs)


def areas(vertices, faces):
    triangles=vertices[faces]
    return np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2


def quantiles(values):
    return {name:float(x) for name,x in zip(['min','p01','p50','p95','p99','max'],np.percentile(values,[0,1,50,95,99,100]))}


def region_counts(indices, faces, labels):
    return dict(Counter(Counter(labels[i] for i in faces[index]).most_common(1)[0][0] for index in indices))


def inspect(base, posed, cache):
    rv,rf,rp,rw,labels,rb=base
    v,f,p,w,_,pb=posed
    assert rp==p and np.array_equal(rf,f), 'Polygon topology/index correspondence changed'
    assert rw==w, 'FBX skin influences changed'
    digest=hashlib.sha256(rv.tobytes()+v.tobytes()+f.tobytes()).hexdigest()
    if digest in cache: return dict(cache[digest], reusedGeometryAnalysis=True)
    topo,edges=topology(rv,rf)
    current_topo,_=topology(v,f)
    seams=seam_pairs(rv)
    gaps=np.linalg.norm(v[seams[:,0]]-v[seams[:,1]],axis=1) if len(seams) else np.array([])
    rest_area,pose_area=areas(rv,rf),areas(v,f)
    lengths=np.linalg.norm(rv[edges[:,0]]-rv[edges[:,1]],axis=1)
    new_lengths=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
    valid=lengths>1e-9
    ratios=new_lengths[valid]/lengths[valid]
    area_ratios=pose_area[rest_area>1e-14]/rest_area[rest_area>1e-14]
    rest_hits,rest_candidates=collisions(rv,rf)
    pose_hits,pose_candidates=collisions(v,f)
    new_hits=pose_hits-rest_hits
    near_collapse=np.flatnonzero((rest_area>1e-14)&(pose_area/np.maximum(rest_area,1e-30)<.01))
    extended=edges[valid][ratios>2]
    compressed=edges[valid][ratios<.2]
    worst=np.argsort(ratios)[-20:][::-1]
    collision_faces=sorted(set(i for pair in new_hits for i in pair))
    # Independently reconstruct ordinary linear-blend skinning from the two
    # exported bone rest matrices and every unchanged weight. This diagnoses
    # export/bake corruption versus deformation already implied by the rig.
    transforms={name:pb[name]@np.linalg.inv(matrix) for name,matrix in rb.items()}
    predicted=np.zeros_like(rv)
    weight_sum=np.zeros(len(rv))
    homogeneous=np.column_stack((rv,np.ones(len(rv))))
    by_bone={}
    for index,weights in enumerate(rw):
        for name,weight in weights:
            if name in transforms:
                by_bone.setdefault(name,[]).append((index,weight))
    for name,rows in by_bone.items():
        ids=np.array([r[0] for r in rows]);amount=np.array([r[1] for r in rows])
        predicted[ids]+=(homogeneous[ids]@transforms[name].T)[:,:3]*amount[:,None]
        weight_sum[ids]+=amount
    weighted=weight_sum>1e-12
    predicted[weighted]/=weight_sum[weighted,None]
    predicted[~weighted]=rv[~weighted]
    reconstruction_error=np.linalg.norm(predicted-v,axis=1)
    result={'geometryAnalysisSha256':digest, 'reusedGeometryAnalysis':False,
            'registeredTopology':topo, 'posedTopology':current_topo, 'samePolygonsAndSkinWeights':True,
            'finiteCoordinates':bool(np.isfinite(v).all()),
            'independentAllVertexLbsErrorM':quantiles(reconstruction_error),
            'matchesIndependentLbsWithin20Micrometers':bool((reconstruction_error<=2e-5).all()),
            'coincidentSeamPairsAtRegistration':len(seams), 'seamToleranceM':1e-7,
            'seamGapReviewThresholdM':1e-5, 'seamPairsOpeningAboveThreshold':int((gaps>1e-5).sum()),
            'maxSeamGapM':float(gaps.max()) if len(gaps) else 0.,
            'registeredDegenerateTriangles':int((rest_area<=1e-14).sum()),
            'posedDegenerateTriangles':int((pose_area<=1e-14).sum()),
            'nearCollapsedBelowOnePercentArea':len(near_collapse),
            'nearCollapsedRegions':region_counts(near_collapse,f,labels),
            'edgeLengthRatio':quantiles(ratios), 'triangleAreaRatio':quantiles(area_ratios),
            'edgesOverTwoTimesLength':len(extended), 'edgesBelowOneFifthLength':len(compressed),
            'stretchedEdgeRegions':dict(Counter(labels[int(i)] for i in extended.flatten())),
            'worstStretchEdges':[{'vertices':edges[valid][i].tolist(),'ratio':float(ratios[i]),
                                  'restLengthM':float(lengths[valid][i]),'posedLengthM':float(new_lengths[valid][i]),
                                  'midpoint':v[edges[valid][i]].mean(0).tolist()} for i in worst],
            'registeredStrictCrossingPairs':len(rest_hits), 'posedStrictCrossingPairs':len(pose_hits),
            'newStrictCrossingPairs':len(new_hits), 'resolvedStrictCrossingPairs':len(rest_hits-pose_hits),
            'newCrossingFaceRegions':region_counts(collision_faces,f,labels),
            'newCrossingFaces':collision_faces,
            'newCrossingPairs':[list(pair) for pair in sorted(new_hits)],
            'nearCollapsedFaces':near_collapse.tolist(),
            'collisionCandidatePairs':{'registered':rest_candidates,'posed':pose_candidates},
            'newCrossingExamples':[{'faces':[int(a),int(b)],'center':v[f[[a,b]]].mean((0,1)).tolist(),
                                    'labels':[labels[int(f[a,0])],labels[int(f[b,0])]]} for a,b in sorted(new_hits)[:50]]}
    cache[digest]=result
    return result


def self_check():
    a=np.array([[[0.,0,0],[1,0,0],[0,1,0]]])
    crossing=np.array([[[.2,.2,-1],[.2,.2,1],[.8,.2,1]]])
    separated=crossing+np.array([3.,0,0])
    adjacent=np.array([[[0.,0,0],[1,0,0],[0,-1,0]]])
    assert crossing_mask(a,crossing).tolist()==[True]
    assert crossing_mask(a,separated).tolist()==[False]
    assert crossing_mask(a,adjacent).tolist()==[False]
    assert crossing_mask(a,crossing[:,::-1]).tolist()==[True]
    # A vertex touching the face is contact, not a strict transverse crossing.
    touching=np.array([[[.2,.2,0],[.2,.2,1],[.8,.2,1]]])
    assert crossing_mask(a,touching).tolist()==[False]
    vertices=np.concatenate((a[0],crossing[0]))
    hits,_=collisions(vertices,np.array([[0,1,2],[3,4,5]]))
    assert hits=={(0,1)}
    # Rigidly rotating/translating both triangles must not create a failure.
    rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    for scale in (.001,1.,1000.):
        transform=lambda x:(x@rotation.T)*scale+np.array([7.,-2,3])
        assert crossing_mask(transform(a),transform(crossing)).tolist()==[True]
        assert crossing_mask(transform(a),transform(adjacent)).tolist()==[False]
    print('MESH_AUDIT_DETECTOR_SELF_CHECK_PASS',flush=True)


def main(directory, destination):
    self_check()
    run=json.loads((directory/'report.json').read_text())
    rows=[];cache={}
    for i,model in enumerate(run['models']):
        base=read(directory/f'model-{i}-registered.fbx')
        for j in range(2):
            path=directory/f'model-{i}-pose-{j}.fbx'
            row={'source':model['inputName'],'file':path.name,
                 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 **inspect(base,read(path),cache)}
            rows.append(row)
            print('MESH_AUDIT',path.name,'gaps',row['seamPairsOpeningAboveThreshold'],
                  'new_crossings',row['newStrictCrossingPairs'],'max_stretch',row['edgeLengthRatio']['max'],flush=True)
    destination.write_text(json.dumps({'version':'actual-fbx-mesh-diagnostic/1',
        'surfaceDeformationIssuesDetected':any(r['newStrictCrossingPairs'] or r['edgesOverTwoTimesLength'] for r in rows),
        'scope':'four native variants of the same Standin male 0.7.6, two poses each',
        'completeCollisionOrFoldCertificate':False,
        'limitations':['Strict transverse crossings only; coplanar overlap and full containment are not certified.',
                       'No certified transported normal reference; reversed/folded-face certification is not claimed.',
                       'Ratio thresholds are diagnostic triggers, not approved release criteria.'],
        'models':rows},indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);main(args.run_dir,args.output)
