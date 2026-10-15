"""Experimental semantic-first upper/lower retrieval. No geometric query routing.

Library text is a separate, fully DB/BVH-bound draft sidecar. Broad groups are
soft context, never hard filters. Missing visibility is not a neutral-pose label.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from time import perf_counter
import unicodedata

import numpy as np

VERSION='natural-parts-v1'
UPPER={
 'arms_down':'팔 내림', 'arms_front_bent':'몸 앞에서 팔 굽힘',
 'arms_forward':'앞으로 뻗음', 'arms_sideways':'옆으로 벌림',
 'arms_raised':'팔 들어 올림', 'hands_near_face_head':'얼굴·머리 근처',
 'hands_waist_back':'허리·등 뒤', 'arms_supporting_body':'팔로 몸 지지',
 'other':'목록 밖', 'unknown':'판단 못 함'}
LOWER={'standing':'서기','locomotion':'이동형','chair_sitting':'의자 앉기형',
 'floor_sitting':'바닥 앉기형','kneeling_crouching':'무릎·쭈그림',
 'lying':'눕기형','off_ground':'공중·매달림형','other':'목록 밖','unknown':'판단 못 함'}
GROUPS={'upper':UPPER,'lower':LOWER}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_part(raw,part,*,rough=False):
    if not isinstance(raw,dict):raise ValueError('part must be object')
    if raw.get('group') not in GROUPS[part]:raise ValueError('unknown part group')
    if raw.get('confidence') not in ('high','medium','low','unknown'):raise ValueError('invalid confidence')
    observations=raw.get('observations')
    if not isinstance(observations,list) or len(observations)>8 or any(not isinstance(t,str) or len(t)>240 for t in observations):
        raise ValueError('invalid observations')
    text=raw.get('description')
    if not isinstance(text,str) or len(text)>1000:raise ValueError('invalid description')
    if rough:
        if type(raw.get('visible')) is not bool:raise ValueError('visibility required')
        if not raw['visible'] and (text.strip() or observations or raw['group']!='unknown'):
            raise ValueError('unseen part must abstain, not invent neutral posture')
    if (not rough or raw['visible']) and not text.strip():raise ValueError('empty visible/library description')
    return dict(raw)


def semantic_text(part):
    return ' '.join([part['description'],*part['observations']])


def grams(text):
    s=re.sub(r'[^a-z0-9가-힣]+','',unicodedata.normalize('NFKC',text).lower())
    return set(s[i:i+n] for n in (2,3) for i in range(max(0,len(s)-n+1)))


class NaturalPartIndex:
    def __init__(self,bundle,tags_path,encoder,*,excluded=(),side_conditions=True):
        self.bundle=Path(bundle).resolve();self.tags_path=Path(tags_path)
        tagged=json.loads(self.tags_path.read_text())
        if tagged.get('version')!=VERSION:raise ValueError('part tags version mismatch')
        if tagged['source_db_sha256']!=digest(self.bundle/'poses.db'):raise ValueError('part tags bound to another DB')
        with sqlite3.connect((self.bundle/'poses.db').as_uri()+'?mode=ro',uri=True) as conn:
            db={pid:json.loads(meta or '{}') for pid,meta in conn.execute('select pose_id,meta_json from poses')}
        records=tagged['poses']
        if len(records)!=len(db) or {r['pose_id'] for r in records}!=set(db):raise ValueError('part tags coverage mismatch')
        for row in records:
            path=self.source(row['pose_id'])
            if digest(path)!=row['bvh_sha256'] or row['bvh_sha256']!=db[row['pose_id']].get('bvh_sha256'):
                raise ValueError('part tag/BVH/DB identity mismatch')
            for part in GROUPS:validate_part(row[part],part)
        excluded=set(excluded)
        self.records=[r for r in records if r['pose_id'] not in excluded]
        if not self.records:raise ValueError('no eligible natural part records')
        self.encoder=encoder;self.metadata=db
        self.side_conditions=side_conditions
        from .part_laterality import MirrorSelector,VERSION as LATERAL_VERSION
        self.mirror_selector=MirrorSelector(self) if side_conditions else None
        docs=[semantic_text(r[p]) for p in GROUPS for r in self.records]
        matrix=encoder.fit(docs)
        if matrix.ndim!=2 or matrix.shape[0]!=len(docs) or not np.isfinite(matrix).all():raise ValueError('invalid embedding matrix')
        n=len(self.records);self.vectors={p:matrix[i*n:(i+1)*n] for i,p in enumerate(GROUPS)}
        self.grams={p:[grams(semantic_text(r[p])) for r in self.records] for p in GROUPS}
        self.identity=dict(version=VERSION,library_version=tagged['library_version'],
            tags_sha256=digest(self.tags_path),source_db_sha256=tagged['source_db_sha256'],
            total=len(records),eligible=len(self.records),excluded=len(records)-len(self.records),
            source_binding_verified=True,new_tag_review_status='draft_unreviewed',
            model=encoder.name,model_sha256=encoder.model_sha256,
            tokenizer_sha256=encoder.tokenizer_sha256,weights={'dense_cosine':.8,'character_overlap':.2},
            group_hard_filter=False,skeleton_bonus=False,not_calibrated_accuracy=True)
        self.identity.update(side_conditions=side_conditions,side_rule_version=LATERAL_VERSION if side_conditions else None)

    def source(self,pose_id):
        path=(self.bundle/'bvh'/(pose_id+'.bvh')).resolve()
        if path.parent!=(self.bundle/'bvh').resolve() or not path.is_file():raise ValueError('invalid source pose')
        return path

    def search(self,description,part,top_k=5):
        from .search import pose_family_id
        if top_k<1:raise ValueError('top_k must be positive')
        validate_part(description,part,rough=True)
        if not description['visible'] or description.get('searchable') is False:
            return dict(status='abstained',hits=[],reason='not_visible' if not description['visible']
                else 'visible_but_insufficient_part_detail',part=part)
        started=perf_counter();text=semantic_text(description);vector=self.encoder.encode(text)[0]
        if vector.shape!=(self.vectors[part].shape[1],) or not np.isfinite(vector).all():
            raise ValueError('invalid query embedding')
        dense=self.vectors[part]@vector;q=grams(text)
        overlap=np.array([len(q&g)/max(1,len(q)) for g in self.grams[part]])
        scores=.8*dense+.2*overlap
        from .part_laterality import explicit_sides
        query_sides=explicit_sides(description,part)
        by_id={r['pose_id']:i for i,r in enumerate(self.records)}
        hits=[];seen=set()
        for i in np.argsort(-scores,kind='stable'):
            row=self.records[int(i)];pid=row['pose_id'];family=pose_family_id(pid,self.metadata[pid])
            if family in seen:continue
            seen.add(family)
            seed_i=int(i);lateral=None
            if self.mirror_selector:
                pid,lateral=self.mirror_selector.choose(pid,part,query_sides)
                row=self.records[by_id[pid]];i=by_id[pid]
            hits.append(dict(pose_id=pid,part=part,score=float(scores[seed_i]),dense_cosine=float(dense[i]),
                character_overlap=float(overlap[i]),variant_text_score=float(scores[i]),
                family_retrieval_score=float(scores[seed_i]),lateral=lateral,
                tag=row[part],family=family,kind=part+'_reference',
                bvh_sha256=row['bvh_sha256'],refine_allowed=False,release_allowed=False))
            if len(hits)>=top_k:break
        return dict(status='ok' if hits else 'no_result',hits=hits,part=part,
                    backend=VERSION+'-e5+char',elapsed_ms=(perf_counter()-started)*1000,
                    lateral_query=query_sides,side_conditions_enabled=self.side_conditions,
                    semantic_match_verified=False)
