"""Independently compare real Converter finals and fail closed when requested."""
import argparse,json,sys
from pathlib import Path
import bpy,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from converter.shape_bake import import_full_character
from converter import retarget as rt
from scripts.character_lod.quality import dist
from scripts.character_lod.region_policy import REGIONS
from scripts.character_lod.contract import final_geometry_pass
from scripts.character_lod.correspondence import load as load_correspondence


def read(path):
    rt.reset_scene();arm,meshes=import_full_character(path);m=meshes[0]
    co=np.array([list(m.matrix_world@v.co) for v in m.data.vertices])
    bones={b.name:np.array(arm.matrix_world@b.matrix_local) for b in arm.data.bones}
    names={g.index:g.name for g in m.vertex_groups}
    groups={name:[] for name in names.values()};hand=np.zeros(len(co),bool)
    for v in m.data.vertices:
        hand[v.index]=sum(g.weight for g in v.groups if 'Hand' in names[g.group])>.01
        for g in v.groups:
            if g.weight>.2:groups[names[g.group]].append(v.index)
    masks={name:np.isin(np.arange(len(co)),indices) for name,indices in groups.items()}
    return co,bones,masks,hand,np.array([p.vertices[:] for p in m.data.polygons],np.int32)


def main(a):
    root=Path(a.assets);rows=json.loads((root/'benchmark/results.json').read_text())
    registry=json.loads((root/'delivery-package/shape-bakes.host.json').read_text());results=[]
    if a.require_pass and (len(rows)!=72 or not all(r['ok'] for r in rows)):
        raise ValueError('final acceptance requires the full successful 72-conversion matrix')
    for sex in ('male','female'):
        for case in ('base','preset','custom'):
            for skin in ('default','flesh-guide'):
                pair={v:next(r for r in rows if r['gender']==sex and r['case']==case and r['skin']==skin and r['variant']==v and r['trial']==0) for v in ('original','candidate')}
                one,two=[read(Path(pair[v]['file'].replace('/work',str(root),1))) for v in ('original','candidate')]
                cid=pair['candidate']['character'];folder=Path(registry['sources'][registry['characters'][cid]['rules']]['blend']).parent
                mapping=load_correspondence(folder);d=np.linalg.norm(mapping.apply(one[0])-two[0],axis=1)*1000
                protected=mapping.exact_mask(np.load(folder/'protected-source.npy'))
                result=dict(gender=sex,case=case,skin=skin,correspondence='projected source triangles' if (folder/'source-projection.npz').exists() else 'source vertex subset',retained_world_max_mm=float(d.max()),retained_world_p99_mm=float(np.percentile(d,99)),retained_world_mean_mm=float(d.mean()),protected_max_mm=float(d[protected].max()),hand_max_mm=float(d[mapping.membership(one[3])].max()),rig_head_max_mm=max(float(np.linalg.norm(one[1][name][:3,3]-two[1][name][:3,3])*1000) for name in one[1]))
                result['bone_regions']={name:{'max_mm':float(d[mapping.membership(mask)].max()),'p99_mm':float(np.percentile(d[mapping.membership(mask)],99))} for name,mask in one[2].items() if mapping.membership(mask).any()}
                labels_path=folder/'source-regions.npy'
                if labels_path.exists():
                    labels=np.load(labels_path);body=np.isin(labels,[REGIONS.index(x) for x in ('torso','arms','legs')]);surface=dist(one[0][body],two[0],two[4]);result['body_source_to_candidate']=surface;result['body_surface_p99_mm']=surface['p99_mm']
                result['additional_final_correspondence_screen_pass']=final_geometry_pass(result) if labels_path.exists() else result['retained_world_p99_mm']<=.5 and result['protected_max_mm']<=.05
                result['numeric_screen_pass']=result['additional_final_correspondence_screen_pass']
                results.append(result);print('FINAL_PAIR',sex,case,skin,'hand max',result['hand_max_mm'],'body p99',result.get('body_surface_p99_mm'),'pass',result['additional_final_correspondence_screen_pass'],flush=True)
    (root/'benchmark/final-pairs.json').write_text(json.dumps(results,indent=2)+'\n')
    if a.require_pass and not all(r['additional_final_correspondence_screen_pass'] for r in results):
        raise ValueError('actual final geometry failed; no promotion')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--require-pass',action='store_true');main(p.parse_args(sys.argv[sys.argv.index('--')+1:]))
