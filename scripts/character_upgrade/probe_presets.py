"""Measure existing style presets on the immutable approved anatomy sources."""
import argparse,ast,importlib.util,json,sys,time
from pathlib import Path
import bpy,numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--approved',required=True);ap.add_argument('--out',required=True);ap.add_argument('--gender',choices=['male','female'],required=True)
args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);source=Path(args.approved)/'authoring'/args.gender;out=Path(args.out)/args.gender;out.mkdir(parents=True,exist_ok=True)
controller=source/'standin_body.py';text=controller.read_text(encoding='utf-8')
presets=next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PRESETS' for t in n.targets) and 'IDEAL83' in ast.literal_eval(n.value))
# Named head-count styles must measure as named; no extra head shrink multiplier.
for name in ('IDEAL83','IDEAL70','TOON64'):presets[name][1]['head_scale']=1.
bpy.ops.wm.open_mainfile(filepath=str(source/'model.blend'))
spec=importlib.util.spec_from_file_location('presets_controller',controller);sb=importlib.util.module_from_spec(spec);spec.loader.exec_module(sb);sb.register()
m,rig=sb.objects();base=sb.normalize(sb.defaults(m));report={'gender':args.gender,'rules':sb.VERSION,'source':str(source),'fields':sb.FIELDS,'presets':{}}
for name,(label,edits) in presets.items():
    start=time.time();values=sb.normalize({**base,**edits})
    metric=sb.validate_geometry(values);preview=sb.preview_compute(values)
    if not np.isfinite(preview).all():raise ValueError('nonfinite preview')
    np.save(out/(name+'.npy'),preview.astype(np.float32))
    report['presets'][name]={'label':label,'controls':values,'geometry':metric}
    (out/'probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PRESET_PROBE',args.gender,name,round(time.time()-start,2),json.dumps(metric),flush=True)
print('PRESETS_PROBED',args.gender,len(report['presets']),flush=True)
