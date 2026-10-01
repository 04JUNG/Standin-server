"""Opt-in multi-view text enrichment of the existing rough semantic ranker.

Images are captioned OFFLINE. Query-time VLM slots are reused. E5 remains a text
encoder; this is not image/text joint embedding training.
"""
from pathlib import Path
import json
import numpy as np
from .rough_semantic import sha
from .rough_semantic_v2 import StructuredRoughSemanticSearch
from . import pose_captioning as captioning
from . import semantic_pose_structure as structure
from ..pose_quarantine import pose_quarantine_sha256


class MultiviewRoughSemanticSearch(StructuredRoughSemanticSearch):
    version='rough-semantic-multiview-v3'

    def __init__(self,*,captions_path,**kwargs):
        self.captions_path=Path(captions_path);self.captions_hash=sha(self.captions_path)
        artifact=json.loads(self.captions_path.read_text())
        if artifact.get('schema_version')!=captioning.VERSION:raise ValueError('caption schema mismatch')
        if artifact['geometry_db_sha256']!=sha(kwargs['geometry_db']):raise ValueError('caption DB mismatch')
        if artifact['quarantine_sha256']!=pose_quarantine_sha256():raise ValueError('caption quarantine mismatch')
        if artifact['review_implementation_sha256']!=sha(captioning.__file__) or artifact['structure_implementation_sha256']!=sha(structure.__file__):
            raise ValueError('caption review version mismatch')
        self.captions={r['pose_id']:r for r in artifact['poses']}
        if len(self.captions)!=len(artifact['poses']):raise ValueError('duplicate caption member')
        super().__init__(**kwargs)
        if set(self.captions)!=set(self.paths):raise ValueError('captions must cover all eligible members')
        self.groups={}
        for i,d in enumerate(self.documents):
            self.groups.setdefault((d['pose_id'],d['region']),{}).setdefault(d.get('source','structure'),[]).append(i)

    def extension_identity(self):
        return {**super().extension_identity(),'multiview_implementation':sha(__file__),
            'caption_review_implementation':sha(captioning.__file__),'captions_sha256':self.captions_hash,
            'source_weights':{'structure':.65,'visual':.25,'usage':.10}}

    def make_documents(self,row,path):
        base=list(super().make_documents(row,path));pid=row['pose_id']
        if pid not in self.captions:raise ValueError('missing caption member: '+pid)
        c=self.captions[pid]
        if c['bvh_sha256']!=sha(path):raise ValueError('caption BVH mismatch')
        if {x['view'] for x in c['images']}!=set(captioning.VIEWS):raise ValueError('caption views missing')
        for im in c['images']:
            if sha(im['path'])!=im['sha256']:raise ValueError('caption image changed')
        if sha(c['response_path'])!=c['response_sha256']:raise ValueError('caption response changed')
        raw=json.loads(Path(c['response_path']).read_text())
        if raw.get('mock') is not False or raw.get('status')!='ok' or raw.get('pose_id')!=pid or raw.get('bvh_sha256')!=c['bvh_sha256']:
            raise ValueError('invalid caption provenance')
        review=captioning.review_response(raw['raw'],self.structures[pid])
        if any(c[k]!=review[k] for k in ('observations','usages')):raise ValueError('caption review changed')
        yield from ({**d,'source':'structure'} for d in base)
        for key,source in [('observations','visual'),('usages','usage')]:
            for claim in c[key]:
                if not claim['accepted']:continue
                # Full-body matching can see part captions; a part query never
                # inherits whole-body support or the opposite limb's meaning.
                regions=['full'] if claim['region']=='full' else [claim['region'],'full']
                for region in regions:
                    yield {'pose_id':pid,'region':region,'bvh_sha256':row['bvh_sha256'],
                        'source':source,'text':region+' body pose: '+claim['text'],
                        'review_status':claim['review_status'],'evidence_views':claim['evidence_views']}

    def _dense_scores(self,clauses):
        raw=np.stack([self.vectors @ self._encode_query(c) for c in clauses])
        dense=np.zeros(len(self.documents),dtype=float)
        self._source_scores={}
        for key,groups in self.groups.items():
            # Best passage per clause/source, then mean over clauses. Documents
            # never add votes; repeating captions cannot increase a source weight.
            scores={s:float(raw[:,ids].max(axis=1).mean()) for s,ids in groups.items()}
            base=scores['structure']
            fused=.65*base+.25*scores.get('visual',base)+.10*scores.get('usage',base)
            for ids in groups.values():dense[ids]=fused
            self._source_scores[key]=scores
        return dense

    def _candidate_metadata(self,pid,region):
        c=self.captions[pid]
        def selected(name):return [x for x in c[name] if x['accepted'] and (region=='full' or x['region']==region)]
        return {'semantic_enrichment':{'observations':selected('observations'),'usage_hypotheses':selected('usages'),
            'source_scores':self._source_scores.get((pid,region),{}),'model_version':c['model_version'],
            'views':list(captioning.VIEWS),'human_reviewed':False,'intent_verified':False}}

    def search(self,*args,**kwargs):
        if sha(self.captions_path)!=self.captions_hash:raise ValueError('live caption artifact changed')
        return super().search(*args,**kwargs)
