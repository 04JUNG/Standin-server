"""Independent .01 posed contact path through the newly corrected fat/axilla case."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from verify import load,pairs

def compare(a,meta):
 old=json.loads((a.candidate/"parent-contact-pairs.json").read_text())
 new=json.loads((a.candidate/"candidate-contact-pairs.json").read_text())
 if old["source"]!=meta["parent"] or new["source"]!=meta["source"]:raise ValueError("contact source mismatch")
 rows=[]
 for row in new["rows"]:
  key=str(min(row["size"],1.));ref={tuple(x) for x in old["pairs"][key]}
  pp={tuple(x) for x in row.pop("pairs_data")}
  rows.append({**row,"new_pairs":len(pp-ref),"pairs":len(pp)})
 result={"complete":len(rows)==301,"passed":len(rows)==301 and all(r["finite"] and r["new_pairs"]==0 for r in rows),"rows":rows}
 (a.candidate/"posed-contact-path.json").write_text(json.dumps(result,indent=2))
 if not result["passed"]:raise ValueError("posed contact path failed")
 print("CONTACT_PATH_COMPLETE",len(rows),flush=True)

def main(a):
 meta=json.loads((a.candidate/'candidate.json').read_text());baseline={};rows=[]
 for key,path in [('blend_sha256',a.parent_blend),('controller_sha256',a.parent_controller)]:
  if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['parent'][key]:raise ValueError('baseline pin mismatch')
 if a.phase=='compare':return compare(a,meta)
 controls={'physique':1.4,'fat_gain':2.,'breast_spacing':-1.,'breast_projection':1.}
 for label,blend,controller in [('old',a.parent_blend,a.parent_controller),('new',Path(meta['source']['blend']),Path(meta['source']['controller']))]:
  if a.phase not in ['both',label]:continue
  sb=load(blend,controller,label);mesh,arm=sb.objects();s=sb.source();defaults=sb.defaults(mesh)
  for n in range(-100,201):
   value=n/100.;reference=min(value,1.)
   if label=='old' and reference in baseline:continue
   sb.set_controls({**defaults,**controls,'breast_size':reference if label=='old' else value},correct=False)
   sb.apply_pose('ARMS_DOWN');q=sb.evaluated(mesh);pp=pairs(q,s['faces'])
   if label=='old':baseline[reference]=pp
   else:
    extra=pp-baseline[reference] if baseline else set();guard=json.loads(mesh.get('breast_contact_qa','{}')) if value<0 or value>1 else {}
    row={'size':value,'finite':bool(np.isfinite(q).all()),'new_pairs':len(extra),'pairs':len(pp),'guard':guard,'pairs_data':[list(x) for x in sorted(pp)]}
    rows.append(row)
    (a.candidate/'posed-contact-path.partial.json').write_text(json.dumps({'passed':all(r['finite'] and r['new_pairs']==0 for r in rows),'complete':n==200,'rows':rows},indent=2))
   if n%25==0:print('CONTACT_PATH',label,value,len(pp),flush=True)
  sb.unregister()
  if label=='old':
   (a.candidate/'parent-contact-pairs.json').write_text(json.dumps({'source':meta['parent'],'pairs':{str(k):[list(x) for x in sorted(v)] for k,v in baseline.items()}}))
  else:
   (a.candidate/'candidate-contact-pairs.json').write_text(json.dumps({'source':meta['source'],'rows':rows}))
 if a.phase=='both':compare(a,meta)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['old','new','both','compare'],default='both');ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--parent-blend',type=Path,required=True);ap.add_argument('--parent-controller',type=Path,required=True)
 main(ap.parse_args(sys.argv[sys.argv.index('--')+1:]))
