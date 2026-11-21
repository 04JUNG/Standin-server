"""Build the explicit female-v4 SSR2 reference codec from immutable source data.

Run in Blender Python. The public catalog supplies preset definitions; it never
supplies geometry. This builder does not approve the generated runtime.
"""
import argparse,base64,hashlib,importlib.util,json,struct,sys,zlib
from pathlib import Path
import bpy
import numpy as np
ap=argparse.ArgumentParser();
for key in ('blend','controller','catalog','catalog-sha256','out','report'):ap.add_argument('--'+key,required=True)
args=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
source={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in [('blendSha256',args.blend),('controllerSha256',args.controller)]}
assert source==dict(blendSha256='aa10276a7c63a2b0f2336b6f4569e0fb541c6f2aa2dcf689ab96ec04573159d3',controllerSha256='3a3ccd4e1fe09967b86dde7823aad9d0886d47b86b47021a3fcea58e61145052')
catalog_raw=Path(args.catalog).read_bytes();assert hashlib.sha256(catalog_raw).hexdigest()==args.catalog_sha256
bpy.ops.wm.open_mainfile(filepath=args.blend);spec=importlib.util.spec_from_file_location('reference_codec_source',args.controller);sb=importlib.util.module_from_spec(spec);spec.loader.exec_module(sb);sb.register()
mesh=sb.objects()[0];controls={k:dict(min=float(v[0]),max=float(v[1]),default=float(sb.defaults(mesh)[k])) for k,v in sb.FIELDS.items()};presets={}
for asset in json.loads(catalog_raw)['assets']:
 if asset.get('shapeSource')==dict(rules=sb.VERSION,**source):
  name=asset['shapeRuntime']['preset'];assert name not in presets
  presets[name]={('height_cm' if c['key']=='height' else c['key']):c['default'] for c in asset['shape']['controls']};assert set(presets[name])==set(controls)
assert len(presets)==11 and 'BASE' in presets
ref,outer_ids,outer_bary,outer_offsets=sb._lod_reference();s=ref.source();props=ref.objects()[0];N=len(s['co']);base_controls=presets['BASE'];preview=sb.preview_mesh();nv=len(preview.data.vertices);preview_faces=np.asarray([poly.vertices[:] for poly in preview.data.polygons],dtype='<u2')
assert N==105430 and nv==50000 and preview_faces.shape==(99996,3)
assert outer_ids.shape==(94901,1) and np.array_equal(outer_bary,np.ones_like(outer_bary)) and not np.any(outer_offsets)
packed=zlib.decompress(base64.b64decode(preview['projection']));preview_ids=np.frombuffer(packed[:nv*12],dtype='<i4').reshape(nv,3);preview_weights=np.frombuffer(packed[nv*12:],dtype='<f4').reshape(nv,3)
preview_ref_ids=outer_ids[preview_ids,0];ref.laplacian_smooth(s,np.zeros_like(s['co']),0);dst,src,degree=s['adjacency'];names=s['names'];W=s['W'];co=s['co'];P=ref.P;arrays=[]
def add(name,value,dtype='<f8'):
 a=np.ascontiguousarray(value,dtype=dtype);arrays.append((name,a));return name
add('base',co);add('referenceFaces',s['faces'],'<u4');add('adjDst',dst,'<u4');add('adjSrc',src,'<u4');add('degree',degree);add('normals',ref.vertex_normals(s))
protected=W[:,[i for i,n in enumerate(names) if any(k in n for k in ['Head','Neck','Hand','Foot','Toe'])]].sum(1)
add('fatProtect',(1-ref.smooth(.02,.35,protected))*ref.smooth(.09,.17,co[:,2]));add('pin',protected>.02,'u1');add('pelvisMask',ref.pelvis_refine_mask(s))
delta_names=list(ref.mappings(base_controls));offset_names=[]
for name in delta_names:
 raw=s['deltas'][name];add('raw:'+name,raw);offset=ref.anatomy_delta(s,name)-raw
 if np.any(offset!=0):add('offset:'+name,offset);offset_names.append(name)
limbs=[]
for i,(name,amount,mask) in enumerate(ref.muscle_refine_limb_fields(s,base_controls,props)):
 add('limbMask:'+str(i),mask);limbs.append(dict(name=name,amount=amount,head=s['rest'][name]['head'],tail=s['rest'][name]['tail']))
forearms=[]
for side in ['Left','Right']:
 name=P+side+'ForeArm';h,t=[np.array(s['rest'][name][k]) for k in ['head','tail']];axis=t-h;u=(co-h)@axis/(axis@axis);radial=co-(h+np.clip(u,0,1)[:,None]*axis);taper=ref.smooth(.08,.25,u)*(1-ref.smooth(.68,.94,u))
 add('forearmRadial:'+side,radial);add('forearmTaper:'+side,taper);add('forearmWeight:'+side,W[:,names.index(name)]);add('armOwn:'+side,W[:,names.index(P+side+'Arm')]+W[:,names.index(name)])
 forearms.append(dict(side=side,head=h.tolist()))
add('previewIndices',preview_ref_ids,'<u4');add('previewWeights',preview_weights);add('faces',preview_faces,'<u2')
parts=[];layout={};offset=0
for name,value in arrays:
 pad=(-offset)%8
 if pad:parts.append(b'\0'*pad);offset+=pad
 raw=value.tobytes();layout[name]=dict(dtype=value.dtype.str,shape=list(value.shape),offset=offset,bytes=len(raw));parts.append(raw);offset+=len(raw)
raw=b''.join(parts);assert len(raw)==offset and offset<256*1024*1024
metadata=dict(format='standin.exact-female-prototype.v1',rules=sb.VERSION,source=source,sourceReferenceGraphSha256=mesh['lod_reference_graph_sha256'],referenceVertices=N,referenceFaces=len(s['faces']),vertexCount=nv,faceCount=len(preview_faces),controls=controls,presets=presets,aliases={'height':'height_cm'},landmarks=json.loads(props['landmarks']),deltaNames=delta_names,offsetNames=offset_names,limbs=limbs,forearms=forearms,arrays=layout,rawBytes=len(raw),rawSha256=hashlib.sha256(raw).hexdigest())
metadata.update(format='standin-shape-runtime.v2',encoding='female-reference-v4-zlib',dataVersion=1,asset='female-v4-exact-reference')
header=json.dumps(metadata,separators=(',',':')).encode();header+=b' '*((-len(header))%4);compressed=zlib.compress(raw,9);data=b'SSR2'+struct.pack('<I',len(header))+header+compressed;assert len(data)<=32*1024*1024
Path(args.out).write_bytes(data)
report=dict(runtimeSha256=hashlib.sha256(data).hexdigest(),bytes=len(data),rawBytes=len(raw),rawSha256=metadata['rawSha256'],source=source,rules=sb.VERSION,sourceReferenceGraphSha256=metadata['sourceReferenceGraphSha256'],catalogInputSha256=args.catalog_sha256,presets=list(presets),sourceArraysAndGeometryUnmodified=True,algorithmSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),independentApproval=False)
Path(args.report).write_text(json.dumps(report,indent=2)+'\n');print('BUILT_REFERENCE_SSR2',json.dumps(report),flush=True)
