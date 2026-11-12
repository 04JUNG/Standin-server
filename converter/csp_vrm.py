"""Neutral VRM0 export for approved body-shape sources (Blender process only)."""
import inspect
import json
import math
import struct
import textwrap
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def mapping():
    names = dict(hips='Hips', spine='Spine', chest='Spine1', upperChest='Spine2', neck='Neck', head='Head')
    for side in ('Left', 'Right'):
        for human, bone in [('Shoulder','Shoulder'),('UpperArm','Arm'),('LowerArm','ForeArm'),('Hand','Hand'),('UpperLeg','UpLeg'),('LowerLeg','Leg'),('Foot','Foot'),('Toes','ToeBase')]:
            names[side.lower()+human] = side+bone
        for finger, label in [('Thumb','Thumb'),('Index','Index'),('Middle','Middle'),('Ring','Ring'),('Pinky','Little')]:
            for i, joint in enumerate(('Proximal','Intermediate','Distal'), 1):
                names[side.lower()+label+joint] = side+'Hand'+finger+str(i)
    return {k: 'mixamorig:'+v for k,v in names.items()}


def read(path):
    data = Path(path).read_bytes()
    assert struct.unpack_from('<4sII',data) == (b'glTF',2,len(data))
    size = struct.unpack_from('<I',data,12)[0]; d=json.loads(data[20:20+size])
    pos=20+size; n,kind=struct.unpack_from('<I4s',data,pos); assert kind==b'BIN\x00'
    return d, bytearray(data[pos+8:pos+8+n])


def write(path,d,b):
    raw=json.dumps(d,ensure_ascii=False,separators=(',',':')).encode();raw+=b' '*((-len(raw))%4)
    body=struct.pack('<I4s',len(raw),b'JSON')+raw+struct.pack('<I4s',len(b),b'BIN\x00')+b
    Path(path).write_bytes(struct.pack('<4sII',b'glTF',2,12+len(body))+body)


def accessor(d,b,i):
    a=d['accessors'][i];v=d['bufferViews'][a['bufferView']]
    n={'SCALAR':1,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
    dtype=np.dtype({5126:'<f4',5123:'<u2',5121:'u1',5125:'<u4'}[a['componentType']])
    return np.ndarray((a['count'],n),dtype=dtype,buffer=b,offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',n*dtype.itemsize),dtype.itemsize))


def worlds(d):
    parents={c:i for i,n in enumerate(d['nodes']) for c in n.get('children',[])};cache={}
    def get(i):
        if i not in cache:
            n=d['nodes'][i]
            if 'matrix' in n:m=np.array(n['matrix']).reshape(4,4).T
            else:
                x,y,z,w=n.get('rotation',[0,0,0,1]);m=np.eye(4)
                m[:3,:3]=np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],[2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],[2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])@np.diag(n.get('scale',[1,1,1]));m[:3,3]=n.get('translation',[0,0,0])
            cache[i]=get(parents[i])@m if i in parents else m
        return cache[i]
    return [get(i) for i in range(len(d['nodes']))],parents


def export(neutral_fbx, output, lineage):
    from converter.shape_bake import import_full_character
    from io_scene_gltf2.blender.exp import primitive_extract
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm,meshes=import_full_character(neutral_fbx)
    assert len(meshes)==1 and len(arm.data.bones)==52
    for obj in bpy.context.scene.objects:obj.select_set(False)
    for obj in [arm,*meshes]:obj.select_set(True)
    rotate=Matrix.Rotation(math.pi,4,'Z')
    for obj in bpy.context.scene.objects:
        if obj.parent is None:obj.matrix_world=rotate@obj.matrix_world
    bpy.context.view_layer.update()
    expected=np.array([meshes[0].matrix_world@v.co for v in meshes[0].data.vertices])
    original=primitive_extract.PrimitiveCreator._PrimitiveCreator__get_bone_data
    source=textwrap.dedent(inspect.getsource(original));assert source.count('min_influence = 0.0001')==1
    scope={};exec(compile(source.replace('min_influence = 0.0001','min_influence = 0.0'),'<CSP weight export>','exec'),primitive_extract.__dict__,scope)
    primitive_extract.PrimitiveCreator._PrimitiveCreator__get_bone_data=scope['__get_bone_data']
    try:
        bpy.ops.export_scene.gltf(filepath=str(Path(output).with_suffix('.glb')),export_format='GLB',use_selection=True,export_yup=True,export_animations=False,export_skins=True,export_all_influences=True,export_def_bones=False,export_apply=False,export_morph=False,export_materials='EXPORT',export_normals=True)
    finally:primitive_extract.PrimitiveCreator._PrimitiveCreator__get_bone_data=original
    d,b=read(Path(output).with_suffix('.glb'));old,parent=worlds(d);index={n.get('name'):i for i,n in enumerate(d['nodes'])}
    human=[{'bone':h,'node':index[n],'useDefaultValues':True} for h,n in mapping().items()]
    assert len({x['node'] for x in human})==52
    for s in d['skins']:
        ibm=accessor(d,b,s['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
        assert max(np.max(abs(old[j]@m-np.eye(4))) for j,m in zip(s['joints'],ibm))<3e-6
    for i,n in enumerate(d['nodes']):
        for key in ('matrix','rotation','scale'):n.pop(key,None)
        n['translation']=(old[i][:3,3]-(old[parent[i]][:3,3] if i in parent else 0)).tolist()
    new,_=worlds(d)
    for s in d['skins']:
        ibm=accessor(d,b,s['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
        for k,j in enumerate(s['joints']):ibm[k]=np.linalg.inv(new[j])
    max_error=0.;changed=0
    for node in d['nodes']:
        if 'mesh' not in node:continue
        s=d['skins'][node['skin']];names=[d['nodes'][j]['name'] for j in s['joints']]
        hips=names.index('mixamorig:Hips')
        for p in d['meshes'][node['mesh']]['primitives']:
            a=p['attributes'];v=accessor(d,b,a['POSITION']);ws=[];js=[];k=0
            while f'WEIGHTS_{k}' in a:
                ws.append(accessor(d,b,a[f'WEIGHTS_{k}']).copy());js.append(accessor(d,b,a[f'JOINTS_{k}']).copy());k+=1
            w=np.concatenate(ws,1);j=np.concatenate(js,1);order=np.argsort(-w,axis=1,kind='stable')[:,:4]
            w=np.take_along_axis(w,order,axis=1);j=np.take_along_axis(j,order,axis=1);w/=w.sum(1)[:,None]
            # Same local hip-support policy as the reviewed candidate, scaled
            # with stature; supports arbitrary approved body slider values.
            total=np.zeros(len(v));span=.06*(float(np.ptp(v[:,1]))/1.78)
            for side in ('Left','Right'):
                thigh=names.index('mixamorig:'+side+'UpLeg');pivot=new[s['joints'][thigh]][1,3]
                t=np.clip((v[:,1]-pivot)/span,0,1);factor=.6*t*t*(3-2*t)
                for c in range(4):
                    delta=np.where(j[:,c]==thigh,w[:,c]*factor,0)
                    assert np.all(np.any(j[delta>0]==hips,axis=1))
                    w[:,c]-=delta;total+=delta
            rows=np.flatnonzero(total>0);slots=np.argmax(np.where(j[rows]==hips,w[rows]+1,0),axis=1);w[rows,slots]+=total[rows];changed+=len(rows)
            assert np.isfinite(w).all() and np.all(w>=0) and np.max(abs(w.sum(1)-1))<3e-7
            accessor(d,b,a['JOINTS_0'])[:]=j;accessor(d,b,a['WEIGHTS_0'])[:]=w
            for c in range(1,k):del a[f'JOINTS_{c}'];del a[f'WEIGHTS_{c}']
            # glTF Y-up converts Blender (x,y,z) to (x,z,-y).
            from mathutils.kdtree import KDTree
            tree=KDTree(len(expected))
            for ii,(x,y,z) in enumerate(expected):tree.insert((x,z,-y),ii)
            tree.balance();max_error=max(max_error,max(tree.find(co)[2] for co in v)*1000)
    assert max_error<.015, ('neutral mesh mismatch mm',max_error)
    d.setdefault('extensionsUsed',[]).append('VRM')
    d.setdefault('extensions',{})['VRM']={'specVersion':'0.0','exporterVersion':'Standin CSP shape v1',
        'meta':{'title':'Standin shaped CSP review','version':lineage['rules'],'author':'Standin','allowedUserName':'OnlyAuthor','violentUssageName':'Disallow','sexualUssageName':'Disallow','commercialUssageName':'Disallow','licenseName':'Redistribution_Prohibited'},
        'humanoid':{'humanBones':human,'armStretch':0.,'legStretch':0.,'upperArmTwist':.5,'lowerArmTwist':.5,'upperLegTwist':.5,'lowerLegTwist':.5,'feetSpacing':0.,'hasTranslationDoF':False},
        'firstPerson':{'firstPersonBone':index['mixamorig:Head'],'firstPersonBoneOffset':{'x':0,'y':0,'z':0},'meshAnnotations':[{'mesh':i,'firstPersonFlag':'Both'} for i in range(len(d['meshes']))],'lookAtTypeName':'Bone'},
        'blendShapeMaster':{'blendShapeGroups':[]},'secondaryAnimation':{'boneGroups':[],'colliderGroups':[]},
        'materialProperties':[{'name':m.get('name','material'),'shader':'VRM_USE_GLTFSHADER','renderQueue':-1,'floatProperties':{},'vectorProperties':{},'textureProperties':{},'keywordMap':{},'tagMap':{}} for m in d.get('materials',[])]}
    write(output,d,b)
    return {'human_bones':52,'max_neutral_mesh_error_mm':max_error,'hip_support_vertices':changed,'four_influences':True,'all_rest_rotations_identity':True,'locks_embedded':False}
