"""Opt-in rough-slot semantic candidate execution over current, verified BVHs.

Uses the production pinned E5 encoder, but an isolated experimental member index.
No exact claims, camera fitting, composition or refine authorization.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
from ..pose_quarantine import load_pose_quarantine, pose_quarantine_sha256
from ..semantic_embedding import OnnxE5Encoder, load_embedding_profile, model_directory

VERSION = 'rough-semantic-members-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atom_text(atom):
    """Literal observed relations only. Never invent action or ground contact tags."""
    words = [atom.get(k, '') for k in ('subject','value','relation','object','bucket')]
    return ' '.join(str(w).replace('_',' ') for w in words if w)


def documents_for(row):
    atoms = row['posecode']['observed_atoms']
    for region in ('full','upper','lower'):
        selected=[]
        for atom in atoms:
            subject=atom.get('subject','')
            lower=any(s in subject for s in ('leg','knee','ankle','foot','feet'))
            if region=='full' or (region=='lower' and lower) or (region=='upper' and not lower):
                selected.append(atom_text(atom))
        if selected:
            yield dict(pose_id=row['pose_id'],region=region,bvh_sha256=row['bvh_sha256'],
                       text=region+' body pose: '+ '; '.join(dict.fromkeys(selected)))


class RoughSemanticSearch:
    version = VERSION

    def make_documents(self, row, path):
        return documents_for(row)

    def extension_identity(self):
        return {}

    def __init__(self, *, facts_path, geometry_db, build_dir,
                 profile_path=Path('config/semantic_embedding.e5-small.v1.json'), models_root=Path('data/models')):
        self.geometry_db=Path(geometry_db);self.build_dir=Path(build_dir)
        profile=load_embedding_profile(Path(profile_path))
        self.encoder=OnnxE5Encoder(profile,model_directory(profile,Path(models_root)))
        self.db_hash=sha(self.geometry_db);self.quarantine_hash=pose_quarantine_sha256()
        quarantine=load_pose_quarantine()
        with sqlite3.connect(f'file:{self.geometry_db.resolve()}?mode=ro',uri=True) as con:
            paths=dict(con.execute('SELECT pose_id,bvh_path FROM poses'))
        rows=[json.loads(x) for x in Path(facts_path).read_text().splitlines() if x.strip()]
        members={};self.paths={}
        for row in rows:
            pid=row['pose_id']
            if pid in quarantine:continue
            if pid in members:raise ValueError('duplicate fact member: '+pid)
            if row['geometry_db_sha256'].removeprefix('sha256:')!=self.db_hash:
                raise ValueError('facts geometry DB mismatch')
            path=Path(paths[pid]);expected=row['bvh_sha256'].removeprefix('sha256:')
            if sha(path)!=expected:raise ValueError('facts BVH mismatch: '+pid)
            members[pid]=row;self.paths[pid]=path
        if set(members)!=set(paths)-set(quarantine):raise ValueError('facts do not cover current eligible library')
        self.documents=[]
        # Split at complete atom boundaries; never truncate a relation mid-sentence.
        for pid in sorted(members):
            for doc in self.make_documents(members[pid], self.paths[pid]):
                prefix=doc['region']+' body pose: '
                parts=doc['text'][len(prefix):].split('; ');current=[];chunk=0
                for part in parts:
                    trial=prefix+'; '.join(current+[part])
                    encoded=self.encoder.tokenizer.encode(profile['encoding']['passage_prefix']+trial)
                    if encoded.overflowing:
                        if not current:raise ValueError('single atom exceeds encoder limit')
                        self.documents.append({**doc,'text':prefix+'; '.join(current),'chunk':chunk})
                        current=[];chunk+=1
                    current.append(part)
                if current:self.documents.append({**doc,'text':prefix+'; '.join(current),'chunk':chunk})
        identity=dict(version=self.version,geometry_db_sha256=self.db_hash,
            facts_sha256=sha(facts_path),quarantine_sha256=self.quarantine_hash,
            encoder_profile_sha256=sha(profile_path),encoder_version=self.encoder.embedding_version,
            implementation_sha256=sha(__file__),extension_identity=self.extension_identity(),document_count=len(self.documents),member_count=len(members),
            documents_sha256=hashlib.sha256(json.dumps(self.documents,sort_keys=True).encode()).hexdigest(),
            production_ready=False)
        manifest=self.build_dir/'manifest.json'
        if self.build_dir.exists():
            stored=json.loads(manifest.read_text())
            if stored['identity']!=identity or sha(self.build_dir/'vectors.npy')!=stored['vectors_sha256']:
                raise ValueError('experimental semantic build mismatch')
            self.vectors=np.load(self.build_dir/'vectors.npy',allow_pickle=False)
        else:
            vectors,stats=self.encoder.encode([d['text'] for d in self.documents],kind='passage')
            self.build_dir.mkdir(parents=True,exist_ok=False)
            np.save(self.build_dir/'vectors.npy',vectors,allow_pickle=False)
            (self.build_dir/'documents.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in self.documents))
            manifest.write_text(json.dumps(dict(identity=identity,encoding_stats=stats,
                vectors_sha256=sha(self.build_dir/'vectors.npy')),indent=2)+'\n')
            self.vectors=vectors
        if self.vectors.shape!=(len(self.documents),self.encoder.dimension) or not np.isfinite(self.vectors).all() or not np.allclose(np.linalg.norm(self.vectors,axis=1),1,atol=1e-5):
            raise ValueError('invalid experimental embeddings')
        self.build_id=sha(manifest)
        self._queries={}

    def search(self,query,*,region='full',top_k=5):
        if region not in ('full','upper','lower') or not 1<=top_k<=100:raise ValueError('invalid search request')
        query=' '.join(query.split())
        if not query:raise ValueError('empty semantic query')
        if sha(self.geometry_db)!=self.db_hash or pose_quarantine_sha256()!=self.quarantine_hash:
            raise ValueError('live library changed')
        if query not in self._queries:
            self._queries[query]=self.encoder.encode([query],kind='query')[0][0]
        scores=self.vectors@self._queries[query]
        indices=[i for i,d in enumerate(self.documents) if d['region']==region]
        ranked=sorted(indices,key=lambda i:(-float(scores[i]),self.documents[i]['pose_id']))
        ordered=[];seen=set()
        for i in ranked:
            pid=self.documents[i]['pose_id']
            if pid in seen:continue
            seen.add(pid);ordered.append(i)
            if len(ordered)>=top_k:break
        results=[]
        for i in ordered:
            d=self.documents[i];path=self.paths[d['pose_id']]
            if sha(path)!=d['bvh_sha256'].removeprefix('sha256:'):raise ValueError('candidate BVH changed')
            results.append(dict(pose_id=d['pose_id'],score=float(scores[i]),bvh_path=str(path.resolve()),
                bvh_sha256=d['bvh_sha256'],document=d['text'],query_region=region,
                candidate_scope='whole_pose_member',match_source='semantic_rough',
                exact_match_status='not_evaluated',refine_allowed=False))
        return dict(query=query,region=region,status='candidates' if results else 'empty',
                    semantic_build_id=self.build_id,results=results)


def execute_semantic_plan(plan, runtime, *, top_k=5):
    """Run every planned S query; leave G/F/D and composition status explicit."""
    row=plan.to_dict() if hasattr(plan,'to_dict') else plan
    results=[]
    for channel in row['channels']:
        if channel['kind']!='S':continue
        for query in dict.fromkeys(channel['queries']):
            results.append(runtime.search(query,region=channel['region'],top_k=top_k))
    return dict(person_index=row['person_index'],route=row['route'],semantic_requests=results,
                executed_query_count=len(results),composition_executed=False,
                geometry_executed=False,facts_rerank_executed=False,refine_allowed=False)
