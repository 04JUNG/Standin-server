"""Offline local surface regularization with tolerated interpenetration.

ARAP-inspired uniform edge energy + positional tether, bounded local solve.
No collision term, bone/weight edits, remeshing, or product/runtime integration.
Diagnostic thresholds and mask anatomy are specific to the actual male fixture.
Reference: https://igl.ethz.ch/projects/ARAP/ (Sorkine and Alexa, 2007).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_actual_mesh import read, topology, areas, quantiles


def face_neighbors(faces):
    raw=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    edges,inverse,counts=np.unique(np.sort(raw,axis=1),axis=0,return_inverse=True,return_counts=True)
    assert (counts==2).all(),'This diagnostic requires a closed triangular manifold'
    pair=np.tile(np.arange(len(faces)),3)[np.argsort(inverse,kind='stable')].reshape(-1,2)
    return edges,pair


def dihedral(vertices,faces,pairs):
    t=vertices[faces];n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
    return np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',n[pairs[:,0]],n[pairs[:,1]]),-1,1)))


def measurements(vertices,rest,faces,edges,pairs):
    lengths=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
    ratio=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)/lengths
    ar=areas(vertices,faces)/areas(rest,faces)
    rd=dihedral(rest,faces,pairs);pd=dihedral(vertices,faces,pairs)
    return {'maxEdgeRatio':float(ratio.max()),'p99EdgeRatio':float(np.percentile(ratio,99)),
            'edgesOverTwoTimes':int((ratio>2).sum()),'edgesBelowOneFifth':int((ratio<.2).sum()),
            'facesBelowOnePercentArea':int((ar<.01).sum()),'degenerateFaces':int((areas(vertices,faces)<=1e-14).sum()),
            'newCreasesOver60DegFromRestUnder30Deg':int(((pd>60)&(rd<30)).sum()),
            'newCreasesOver90DegFromRestUnder30Deg':int(((pd>90)&(rd<30)).sum()),
            'newCreasesOver120DegFromRestUnder30Deg':int(((pd>120)&(rd<30)).sum())}


def active_mask(rest,posed,faces,edges,pairs,labels):
    ratio=np.linalg.norm(posed[edges[:,0]]-posed[edges[:,1]],axis=1)/np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
    seeds=(ratio>1.6)|(ratio<.3)
    rd=dihedral(rest,faces,pairs);pd=dihedral(posed,faces,pairs)
    seeds|=(pd>45)&(rd<25)
    mask=np.zeros(len(rest));mask[np.unique(edges[seeds])]=1.
    for level in (.75,.5,.25):
        touching=np.any(mask[edges]>0,axis=1)
        ids=np.unique(edges[touching]);mask[ids]=np.maximum(mask[ids],level)
    protected=np.array([any(token in name for token in ('Head','Neck','Hand','Thumb','Index','Middle','Ring','Pinky','Foot','Toe')) for name in labels])
    mask[protected]=0
    return mask,protected


def fit_rotations(rest,posed,edges):
    i,j=edges.T
    a=rest[i]-rest[j];b=posed[i]-posed[j]
    terms=b[:,:,None]*a[:,None,:]
    cov=np.zeros((len(rest),3,3))
    np.add.at(cov,i,terms);np.add.at(cov,j,terms)
    u,_,vt=np.linalg.svd(cov)
    sign=np.linalg.det(u@vt)
    u[:,:,2]*=np.where(sign<0,-1,1)[:,None]
    return u@vt


def relax(rest,posed,edges,mask,tether,cap=.02,local_steps=24):
    i,j=edges.T;degree=np.bincount(edges.flatten(),minlength=len(rest)).astype(float)
    # Smoothly tether the transition ring, hard pin everything outside it.
    anchor=degree*tether/np.maximum(mask,.02)**2
    x=posed.copy();rest_edge=rest[i]-rest[j]
    for _ in range(local_steps):
        rotations=fit_rotations(rest,x,edges)
        desired=np.einsum('eij,ej->ei',(rotations[i]+rotations[j])*.5,rest_edge)
        rhs=np.zeros_like(x);np.add.at(rhs,i,desired);np.add.at(rhs,j,-desired)
        for _ in range(12):
            neighbor=np.zeros_like(x);np.add.at(neighbor,i,x[j]);np.add.at(neighbor,j,x[i])
            proposal=(neighbor+rhs+anchor[:,None]*posed)/(degree+anchor)[:,None]
            delta=proposal-posed
            delta*=np.minimum(1.,cap/np.maximum(np.linalg.norm(delta,axis=1),1e-20))[:,None]
            proposal=posed+delta
            x=.5*x+.5*proposal
            x[mask==0]=posed[mask==0]
    return x


def skin_rotation_field(weights,base_frames,pose_frames):
    field=np.zeros((len(weights),3,3))
    for i,rows in enumerate(weights):
        for name,weight in rows:
            field[i]+=weight*(pose_frames[name][:3,:3]@base_frames[name][:3,:3].T)
    u,_,vt=np.linalg.svd(field);sign=np.linalg.det(u@vt)
    u[:,:,2]*=np.where(sign<0,-1,1)[:,None]
    return u@vt


def preserve_local_detail(rest,posed,edges,mask,rotations,tether,cap=.02):
    # Tethered differential coordinates preserve the neutral surface's local
    # detail after rotation, while relaxing new high-frequency irregularities.
    i,j=edges.T;degree=np.bincount(edges.flatten(),minlength=len(rest)).astype(float)
    neighbors=np.zeros_like(rest);np.add.at(neighbors,i,rest[j]);np.add.at(neighbors,j,rest[i])
    detail=rest-neighbors/degree[:,None]
    target_detail=np.einsum('nij,nj->ni',rotations,detail)
    anchor=degree*tether/np.maximum(mask,.02)**2
    x=posed.copy()
    for _ in range(240):
        neighbor=np.zeros_like(x);np.add.at(neighbor,i,x[j]);np.add.at(neighbor,j,x[i])
        proposal=(neighbor+degree[:,None]*target_detail+anchor[:,None]*posed)/(degree+anchor)[:,None]
        delta=proposal-posed
        delta*=np.minimum(1.,cap/np.maximum(np.linalg.norm(delta,axis=1),1e-20))[:,None]
        x=.5*x+.5*(posed+delta);x[mask==0]=posed[mask==0]
    return x


def fair_surface(posed,edges,mask,iterations,cap=.02):
    # Paired positive/negative Laplacian passes reduce high-frequency surface
    # noise while limiting shrinkage. All displacements remain pose-tethered.
    i,j=edges.T;degree=np.bincount(edges.flatten(),minlength=len(posed)).astype(float)
    x=posed.copy()
    for _ in range(iterations):
        for step in (.45,-.47):
            accum=np.zeros_like(x);np.add.at(accum,i,x[j]);np.add.at(accum,j,x[i])
            x+=step*mask[:,None]*(accum/degree[:,None]-x)
            delta=x-posed
            delta*=np.minimum(1.,cap/np.maximum(np.linalg.norm(delta,axis=1),1e-20))[:,None]
            x=posed+delta;x[mask==0]=posed[mask==0]
    return x


def bound_edge_strain(candidate,posed,rest,edges,mask,cap=.02):
    i,j=edges.T;reference=np.linalg.norm(rest[j]-rest[i],axis=1)
    x=candidate.copy();mobility=mask[i]+mask[j]
    for _ in range(100):
        vector=x[j]-x[i];length=np.linalg.norm(vector,axis=1)
        target=np.clip(length,.25*reference,1.9*reference)
        active=(abs(length-target)>1e-9)&(mobility>0)
        a,b=i[active],j[active]
        correction=vector[active]*((length[active]-target[active])/np.maximum(length[active],1e-20))[:,None]
        delta=np.zeros_like(x)
        np.add.at(delta,a,correction*(mask[a]/mobility[active])[:,None])
        np.add.at(delta,b,-correction*(mask[b]/mobility[active])[:,None])
        count=np.bincount(np.concatenate((a,b)),minlength=len(x))
        x+=.7*delta/np.maximum(count,1)[:,None]
        offset=x-posed
        offset*=np.minimum(1.,cap/np.maximum(np.linalg.norm(offset,axis=1),1e-20))[:,None]
        x=posed+offset;x[mask==0]=posed[mask==0]
    return x


def refresh_local_normals(obj,changed):
    mesh=obj.data;positions=np.array([v.co[:] for v in mesh.vertices])
    previous=np.array([normal.vector[:] for normal in mesh.corner_normals])
    support=np.zeros(len(mesh.vertices),dtype=bool)
    for face in mesh.polygons:
        if any(changed[i] for i in face.vertices):support[list(face.vertices)]=True
    mesh.normals_split_custom_set([(0.,0.,0.)]*len(mesh.loops));mesh.update()
    automatic=np.array([normal.vector[:] for normal in mesh.corner_normals])
    loop_ids=np.array([loop.vertex_index for loop in mesh.loops]);active=support[loop_ids]
    result=previous.copy();result[active]=automatic[active]
    angles=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',previous,automatic),-1,1)))
    mesh.normals_split_custom_set(result.tolist());mesh.update()
    assert np.array_equal(positions,np.array([v.co[:] for v in mesh.vertices]))
    return {'changedCorners':int(active.sum()),'positionCoordinatesUnchanged':True,
            'outsideSupportOriginalNormalsSupplied':True,'oldVsGeometricNormalAngleDeg':quantiles(angles[active]),
            'activeCornersOver20DegMismatch':int((angles[active]>20).sum())}


def self_check():
    # A rigidly moved triangulated tetrahedron must remain unchanged even if
    # all its vertices are eligible. A single isolated spike must be reduced.
    rest=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    edges=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    posed=rest@rotation.T+np.array([2.,3,4])
    out=relax(rest,posed,edges,np.ones(4),.25,cap=2,local_steps=5)
    assert np.max(abs(out-posed))<1e-10
    broken=posed.copy();broken[0]+=[.6,.3,.2]
    mask=np.array([1.,0,0,0]);out=relax(rest,broken,edges,mask,.25,cap=.2,local_steps=10)
    assert np.array_equal(out[1:],broken[1:])
    assert np.linalg.norm(out[0]-posed[0])<np.linalg.norm(broken[0]-posed[0])
    assert np.linalg.norm(out[0]-broken[0])<=.2+1e-10
    field=np.broadcast_to(rotation,(4,3,3))
    out=preserve_local_detail(rest,posed,edges,np.ones(4),field,.1)
    assert np.max(abs(out-posed))<1e-10
    grid=np.array([[x,y,0.] for y in range(3) for x in range(3)])
    grid_edges=np.array([[0,1],[1,2],[3,4],[4,5],[6,7],[7,8],[0,3],[3,6],[1,4],[4,7],[2,5],[5,8]])
    spike=grid.copy();spike[4,2]=.5;grid_mask=np.zeros(9);grid_mask[4]=1
    out=fair_surface(spike,grid_edges,grid_mask,10,cap=.2)
    assert np.array_equal(out[grid_mask==0],spike[grid_mask==0])
    assert 0<out[4,2]<.5 and np.linalg.norm(out[4]-spike[4])<=.2+1e-10
    assert np.array_equal(fair_surface(posed,edges,np.zeros(4),10),posed)
    out=bound_edge_strain(broken,broken,rest,edges,mask,cap=.2)
    assert np.array_equal(out[1:],broken[1:]) and np.linalg.norm(out[0]-broken[0])<=.2+1e-10
    line=np.array([[0.,0,0],[1,0,0]]);long=line.copy();long[1,0]=3
    out=bound_edge_strain(long,long,line,np.array([[0,1]]),np.array([0.,1.]),cap=2.)
    assert np.array_equal(out[0],long[0]) and np.linalg.norm(out[1]-out[0])<=1.90001
    print('SURFACE_ONLY_SELF_CHECK_PASS',flush=True)


def main(directory,runtime):
    self_check();sys.path.insert(0,str(runtime))
    from converter.local import inspection
    from converter import bone_map
    rest,faces,_,_,labels,base_frames=read(directory/'model-0-registered.fbx')
    edges,pairs=face_neighbors(faces)
    rows=[]
    for p in range(2):
        source=directory/f'model-0-pose-{p}.fbx'
        destination=directory/f'surface-only-pose-{p}-experimental.fbx'
        destination.unlink(missing_ok=True)  # Own previous disposable QA output only.
        original=read(source);posed=original[0]
        mask,protected=active_mask(rest,posed,faces,edges,pairs,labels)
        before=measurements(posed,rest,faces,edges,pairs)
        candidates=[];coordinates={}
        rotations=skin_rotation_field(original[3],base_frames,original[5])
        for method,tether in [('arap',.5),('arap',.15),('arap',.04),('detail',.04),('detail',.01),('detail',.003),('fairing',6),('fairing',15),('fairing',30),('bounded_fairing',2),('bounded_fairing',5),('bounded_fairing',10)]:
            started=time.perf_counter()
            if method=='arap':corrected=relax(rest,posed,edges,mask,tether)
            elif method=='detail':corrected=preserve_local_detail(rest,posed,edges,mask,rotations,tether)
            elif method=='fairing':corrected=fair_surface(posed,edges,mask,tether)
            else:
                seed=coordinates[('arap',.04)]
                fair=fair_surface(seed,edges,mask,tether)
                corrected=bound_edge_strain(fair,posed,rest,edges,mask)
            elapsed=time.perf_counter()-started
            after=measurements(corrected,rest,faces,edges,pairs)
            delta=np.linalg.norm(corrected-posed,axis=1)
            assert np.array_equal(corrected[mask==0],posed[mask==0])
            assert np.array_equal(corrected[protected],posed[protected])
            assert delta.max()<=.02+1e-9 and np.isfinite(corrected).all()
            gates={'stretchNotWorse':after['maxEdgeRatio']<=before['maxEdgeRatio'],
                   'p99StretchNotWorse':after['p99EdgeRatio']<=before['p99EdgeRatio']+1e-6,
                   'severeEdgeCountNotWorse':after['edgesOverTwoTimes']<=before['edgesOverTwoTimes'],
                   'compressedEdgeCountNotWorse':after['edgesBelowOneFifth']<=before['edgesBelowOneFifth'],
                   'nearCollapseNotWorse':after['facesBelowOnePercentArea']<=before['facesBelowOnePercentArea'],
                   'sharpCreasesNotWorse':after['newCreasesOver60DegFromRestUnder30Deg']<=before['newCreasesOver60DegFromRestUnder30Deg'],
                   'verySharpCreasesNotWorse':after['newCreasesOver90DegFromRestUnder30Deg']<=before['newCreasesOver90DegFromRestUnder30Deg'],
                   'extremeCreasesNotWorse':after['newCreasesOver120DegFromRestUnder30Deg']<=before['newCreasesOver120DegFromRestUnder30Deg'],
                   'noDegenerateTriangles':after['degenerateFaces']==0}
            candidate={'method':method,'tether':tether,'elapsedSeconds':elapsed,'metrics':after,
                       'vertexDisplacementM':quantiles(delta),'verticesMovedOver1mm':int((delta>.001).sum()),
                       'fixedOutsideMask':True,'protectedVerticesUnchanged':True,'diagnosticGates':gates,
                       'eligibleUnderDiagnosticGates':all(gates.values())}
            if method=='bounded_fairing':
                seed=next(c for c in candidates if c['method']=='arap' and c['tether']==.04)
                candidate['seed']={'method':'arap','tether':.04}
                candidate['computeSecondsIncludingSeed']=elapsed+seed['elapsedSeconds']
            candidates.append(candidate);coordinates[(method,tether)]=corrected
            print('SURFACE_CANDIDATE',p,json.dumps(candidate),flush=True)
        eligible=[c for c in candidates if c['eligibleUnderDiagnosticGates']]
        selected=min(eligible,key=lambda c:(c['metrics']['newCreasesOver60DegFromRestUnder30Deg'],c['metrics']['maxEdgeRatio'])) if eligible else None
        row={'sourceFile':source.name,'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(),
             'before':before,'activeVertices':int((mask>0).sum()),'protectedVertices':int(protected.sum()),
             'candidates':candidates,'selectedTether':selected['tether'] if selected else None}
        row['selectedMethod']=selected['method'] if selected else None
        if selected:
            corrected=coordinates[(selected['method'],selected['tether'])]
            bpy.ops.wm.read_factory_settings(use_empty=True)
            arm,meshes=inspection.import_fbx(source);obj=meshes[0];inv=obj.matrix_world.inverted()
            bones_before={b.name:np.array(arm.matrix_world@b.matrix_local) for b in arm.data.bones}
            for vertex,point in zip(obj.data.vertices,corrected):vertex.co=inv@Vector(point)
            obj.data.update();bpy.context.view_layer.update()
            row['localNormalRefresh']=refresh_local_normals(obj,np.linalg.norm(corrected-posed,axis=1)>1e-9)
            dest=destination
            validation=inspection.roundtrip(arm,meshes,dest,bone_map.MIXAMO)
            exported=read(dest)
            assert np.array_equal(exported[1],faces) and exported[3]==original[3]
            bone_error=max(float(abs(exported[5][name][:3,3]-frame[:3,3]).max()) for name,frame in bones_before.items())
            surface_error=float(np.linalg.norm(exported[0]-corrected,axis=1).max())
            assert bone_error<2e-5 and surface_error<2e-5
            row.update(experimentalFile=dest.name,experimentalSha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                       exportValidation=validation,exactOriginalWeights=True,exactFaceIndices=True,
                       maxExportBoneHeadErrorM=bone_error,maxExportSurfaceErrorM=surface_error)
            row['exportedMetrics']=measurements(exported[0],rest,faces,edges,pairs)
        # Separate shading from geometry: same input positions/weights/bones,
        # only recompute geometric normals in the defect support region.
        bpy.ops.wm.read_factory_settings(use_empty=True)
        arm,meshes=inspection.import_fbx(source)
        normals=refresh_local_normals(meshes[0],mask>0)
        control=directory/f'surface-only-pose-{p}-normals-control.fbx'
        normal_validation=inspection.roundtrip(arm,meshes,control,bone_map.MIXAMO)
        normal_result=read(control)
        error=float(np.linalg.norm(normal_result[0]-posed,axis=1).max())
        assert error<2e-5 and normal_result[3]==original[3]
        row['normalOnlyControl']={'file':control.name,'sha256':hashlib.sha256(control.read_bytes()).hexdigest(),
                                  'maxPositionRoundtripErrorM':error,'normalRefresh':normals,'validation':normal_validation}
        rows.append(row)
    result={'version':'surface-only-diagnostic/1','scope':'two poses of the same actual male0.7.6',
            'collisionObjective':False,'collisionGate':False,'productionIntegrated':False,
            'maxCorrectionM':.02,'reference':'https://igl.ethz.ch/projects/ARAP/',
            'limitations':['Diagnostic thresholds are not release criteria.','No new hole filling or remeshing; existing connectivity is required.',
                           'Untested models and thin/detail anatomy need separate qualification.','No complete fold/collision/CSP certificate.'],
            'poses':rows}
    (directory/'surface-only-experiment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);main(args.run_dir,args.runtime)
