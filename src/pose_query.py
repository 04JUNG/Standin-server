"""Preserve VLM-observed held objects and body relations through retrieval.

Only structured, named handheld objects enable this layer. Library tags are
search evidence, not proof of a physical prop or a correct skin mesh. Absent
hand/position descriptions stay unknown; known conflicts are ranked later.
"""
import re
from .pose_relations import relations_for_query, relation_evidence

VERSION = 'observed-action-conditions-v3'
HANDHELD = frozenset({'phone','cup','bottle','book','bag','sword','knife','dagger',
    'pistol','rifle','gun','spear','bow','shield','hammer','axe','bat','ball',
    'umbrella','camera','microphone','guitar','saxophone','violin','flute','pen',
    'chopsticks','spoon','fork','plate','tray','box','flowers','flower','towel',
    'crossbow','staff','rope','racket','disc','dumbbell','trumpet','lantern','torch',
    'basket','broom','mop','fishing_rod','rake','shovel','stone','pillow'})
_ALIASES = {'phone':r'휴대폰|스마트폰|핸드폰|전화기',
            'cup':r'텀블러|머그(?:컵)?|컵|잔', 'book':r'책|서적',
            'bottle':r'물병|병', 'sword':r'검|칼'}
_CONTEXT = (
    r'미소(?:를)?\s*(?:지으며|짓고|짓는|지은|짓으며)(?:\s*있다)?',
    r'웃(?:으며|는|고)(?:\s*있다)?',
    r'대화(?:하는|하며|하고|중인|중)',
    r'(?:상체|상반신)만\s*(?:화면에\s*)?(?:드러나|보이|나와)[^.,;]*',
    r'(?:하반신|하체|다리)(?:은|는|가)?\s*(?:화면에\s*)?보이지[^.,;]*',
)


def _clean(text):
    removed=[]
    for pattern in _CONTEXT:
        text=re.sub(pattern,lambda m:(removed.append(m.group(0)) or ' '),text)
    text=re.sub(r'손에\s*들고', '손으로 쥐고', text)
    text=re.sub(r'\s*(?:있으며|있다)[.\s]*$', '', text)
    return re.sub(r'\s+',' ',text).strip(' .,'),removed


def _object_pattern(objects, lexicon):
    names=[]
    for obj in objects:
        names.append(_ALIASES.get(obj,re.escape(lexicon.labels.get(('object',obj),obj))))
    return '(?:'+'|'.join(names)+')'


def _hand_count(text, noun):
    found=set()
    pattern=rf'(양\s*손|두\s*손|한\s*손|한쪽\s*손|오른손|왼손|다른\s*손)([^.,;]{{0,22}})({noun})'
    for m in re.finditer(pattern,text):
        # Do not attach an earlier hand across a later "other hand" clause.
        if re.search(r'손|팔',m.group(2)): continue
        if re.search(r'않|아니|안\s*보|불명|추정',m.group(2)): continue
        found.add('two' if re.match(r'양|두',m.group(1)) else 'one')
    return next(iter(found)) if len(found)==1 else None


def _position(text):
    if re.search(r'귀(?:에|옆).{0,12}(?:대|통화)|통화.{0,12}귀',text): return 'at_ear'
    if re.search(r'턱\s*(?:밑|아래)',text): return 'under_chin'
    if re.search(r'가슴\s*앞',text): return 'chest_front'
    return None


def validated_object_hands(value, objects):
    """Anatomical sides only; absent, invalid and ambiguous values abstain."""
    if not isinstance(value, dict): return {}
    return {obj: side for obj, side in value.items()
            if obj in objects and obj in HANDHELD
            and isinstance(side, str) and side in ('left', 'right', 'both', 'unknown')}


def prepare_pose_query(query,lexicon):
    required=relations_for_query(query)
    objects=sorted(set(query.objects)&HANDHELD)
    physical=' '.join((query.body_details or '',query.text or ''))
    objects=[o for o in objects if ('object',o) in lexicon.labels and (
        re.search(_object_pattern([o],lexicon),physical) or o in query.held_object_hands)]
    base=dict(version=VERSION,enabled=bool(objects or required),original_text=query.text,
              core_text=query.text,tags=[],objects=objects,hand_count=None,
              hand_position=None,held_object_hands={},removed_context=[],auxiliary_actions=[],
              action_relations=required)
    if required:
        base['core_text']='. '.join(dict.fromkeys(filter(None,[query.body_details,query.text])))
    if not objects: return base
    noun=_object_pattern(objects,lexicon)
    details=query.body_details or ''
    combined=details+'. '+query.text
    base['hand_count']=_hand_count(combined,noun)
    base['hand_position']=_position(combined)
    base['held_object_hands']=validated_object_hands(
        getattr(query,'held_object_hands',{}),objects)
    text,removed=_clean(query.text)
    body,body_removed=_clean(details)
    # Use the physical description when the short sentence dropped a relation.
    lost_count=base['hand_count'] is not None and _hand_count(query.text,noun) is None
    lost_position=base['hand_position'] is not None and _position(query.text) is None
    if body and (lost_count or lost_position):
        text=body;removed+=body_removed
    actions=list(query.actions)
    auxiliary={'laugh','talk','listen','wait','rest'}
    # A phone explicitly at the ear is a physical action, not social context.
    if base['hand_position']=='at_ear': auxiliary.discard('talk')
    primary=[a for a in actions if a not in auxiliary]
    base.update(core_text=text,removed_context=removed,
                auxiliary_actions=[a for a in actions if a in auxiliary])
    base['tags']=[f'object:{o}' for o in objects]+[
        f'action:{a}' for a in primary if ('action',a) in lexicon.labels]
    if required:base['core_text']='. '.join(dict.fromkeys(filter(None,[query.body_details,query.text])))
    return base


def candidate_conditions(prepared,record,lexicon,body=None,measured=None):
    """Three-state evidence on named objects and explicitly stated relations."""
    if not prepared['enabled']: return None
    groups=record.get('groups') or record.get('tags') or {}
    known={t.get('value') for items in groups.values() for t in items
           if t.get('field')=='object'} & HANDHELD
    requested=set(prepared['objects'])
    matched=bool(known&requested)
    state='match' if matched else 'conflict' if known and requested else 'unknown'
    description=record.get('description') or ''
    if isinstance(description,dict): description=description.get('text') or ''
    noun=_object_pattern(prepared['objects'],lexicon) if requested else r'(?!)'
    relations={}
    for field,value in [('hand_count',_hand_count(description,noun)),('hand_position',_position(description))]:
        wanted=prepared.get(field)
        if wanted is None: continue
        # A relation without the requested object is not evidence for that prop.
        actual=value if matched else None
        status='unknown' if actual is None else 'match' if wanted==actual else 'conflict'
        relations[field]=dict(requested=wanted,candidate=actual,status=status)
    # The authored active hand is already swapped in each mirrored BVH's
    # metadata. Do not swap it again or infer handedness from a filename.
    # Bind it to the offline BVH measurement before using it for ranking.
    meta=record.get('meta') or {}
    parts=meta.get('body_parts') or {}
    bound=bool(body and meta.get('bvh_sha256')
               and meta['bvh_sha256']==body.get('source_sha256'))
    hands=parts.get('hands') or {}
    if bound and matched and prepared.get('hand_count'):
        counts={len([s for s in ('left','right') if hands.get(s)==o]) for o in requested & known}
        counts.discard(0)
        if len(counts)==1:
            actual='two' if next(iter(counts))==2 else 'one'
            wanted=prepared['hand_count']
            relations['hand_count']=dict(requested=wanted,candidate=actual,
                status='match' if wanted==actual else 'conflict',source='BVH_bound_object_specific_hands')
    for obj,wanted in prepared.get('held_object_hands',{}).items():
        if wanted=='unknown': continue
        sides=[side for side in ('left','right') if hands.get(side)==obj]
        actual=('both' if len(sides)==2 else sides[0] if sides else None) if bound and obj in known else None
        # Explicit per-object hands avoid assigning another prop's active hand.
        status='unknown' if actual is None else 'match' if wanted==actual else 'conflict'
        relations['held_hand:'+obj]=dict(requested=wanted,candidate=actual,status=status,
            source='bvh_bound_authored_hands' if actual else 'unknown',
            anatomical_side=True,is_mirror=meta.get('is_mirror'))
    conflicts=[k for k,v in relations.items() if v['status']=='conflict']
    if state=='conflict': conflicts.insert(0,'held_object')
    relation_matches=sum(v['status']=='match' for v in relations.values())
    tier=3 if conflicts else 0 if matched and relation_matches else 1 if matched else 2
    action=relation_evidence(prepared.get('action_relations',[]),record,measured,bound)
    relations.update(action)
    # Main observed action precedes incidental props, without deleting unknowns.
    action_states=[v['status'] for v in action.values()]
    priority=[tier,-relation_matches]
    if action_states:
        action_tier=2 if 'conflict' in action_states else 0 if all(s=='match' for s in action_states) else 1
        priority=[action_tier,tier,-relation_matches]
        conflicts.extend(k for k,v in action.items() if v['status']=='conflict')
    return dict(version=VERSION,requested_objects=sorted(requested),candidate_objects=sorted(known),
                object_state=state,relations=relations,conflicts=conflicts,
                priority=priority,evidence_source='library_tags_and_description',
                unknown_is_neutral=True)
