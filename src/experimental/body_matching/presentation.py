"""Fictional character design presentation, separate from shape and actual identity."""
from __future__ import annotations

STYLES = ('feminine', 'masculine', 'androgynous')
CUES = ('face_design', 'body_contour', 'hair_design', 'costume_design')


def unknown_presentation(reason='not_observed'):
    return dict(value=None, visibility='unknown', cues=[], evidence=reason)


def parse_presentation(raw, *, ambiguous=False):
    if not isinstance(raw,dict) or set(raw)!={'value','visibility','cues','evidence'}:
        raise ValueError('invalid_presentation_fields')
    if raw['value'] is not None and raw['value'] not in STYLES:raise ValueError('invalid_presentation_value')
    if raw['visibility'] not in {'visible','uncertain','unknown'}:raise ValueError('invalid_presentation_visibility')
    cues=raw['cues']
    if not isinstance(cues,list) or any(not isinstance(x,str) or x not in CUES for x in cues) or len(set(cues))!=len(cues):
        raise ValueError('invalid_presentation_cues')
    if not isinstance(raw['evidence'],str) or len(raw['evidence'])>1000:raise ValueError('invalid_presentation_evidence')
    if ambiguous:return unknown_presentation('ownership_ambiguous')
    if raw['value'] is None or raw['visibility']=='unknown' or not cues or not raw['evidence'].strip():
        return unknown_presentation(raw['evidence'])
    result=dict(raw,cues=list(cues))
    # Hair/costume alone cannot block opposite-style assets.
    if not set(cues)&{'face_design','body_contour'}:result['visibility']='uncertain'
    return result


def presentation_schema():
    return {'type':'object','additionalProperties':False,'required':['value','visibility','cues','evidence'],
        'properties':{'value':{'anyOf':[{'type':'string','enum':list(STYLES)},{'type':'null'}]},
            'visibility':{'type':'string','enum':['visible','uncertain','unknown']},
            'cues':{'type':'array','items':{'type':'string','enum':list(CUES)}},
            'evidence':{'type':'string','maxLength':240}}}


def asset_style(asset):
    return asset.get('metadata',{}).get('presentation_style','unspecified')


def observation_style(observation):
    raw=observation.get('presentation')
    return parse_presentation(raw,ambiguous=observation.get('ownership_ambiguous',False)) if raw else unknown_presentation()


def presentation_candidates(candidates, observation, default_body_id, defaults):
    """Filter only strong authored presentation evidence, after asset QA checks.

    Unisex assets remain available for shape matching; missing metadata never counts
    as matching. Weak clues only resolve a geometry/attribute score tie.
    """
    p=observation_style(observation);style=p['value']
    trace=dict(observed=style,visibility=p['visibility'],mode='unresolved',reason_codes=[],
               eligible_before=len(candidates),eligible_after=len(candidates))
    if style not in {'feminine','masculine'} or p['visibility']=='unknown':
        trace['reason_codes'].append('presentation_unresolved_default_is_not_gender_detection')
        return candidates,default_body_id,trace
    matches=[a for a in candidates if asset_style(a)==style]
    if p['visibility']=='visible' and matches:
        candidates=[a for a in candidates if asset_style(a) in {style,'unisex'}]
        trace.update(mode='compatible_candidates',eligible_after=len(candidates))
        trace['reason_codes'].append('presentation_opposite_style_excluded')
        proposed=defaults.get(style)
        default_body_id=proposed if any(a['body_id']==proposed for a in matches) else min(
            matches,key=lambda a:(a['body_id']!=default_body_id,-a.get('quality_priority',0),a['body_id']))['body_id']
    elif p['visibility']=='visible':
        trace['mode']='matching_style_unavailable'
        trace['reason_codes'].append('presentation_matching_asset_unavailable')
    else:
        trace['mode']='tie_or_default'
        trace['reason_codes'].append('presentation_uncertain_tie_or_default')
        if matches:
            proposed=defaults.get(style)
            default_body_id=proposed if any(a['body_id']==proposed for a in matches) else min(matches,key=lambda a:(-a.get('quality_priority',0),a['body_id']))['body_id']
    return candidates,default_body_id,trace


def presentation_tie_rank(asset, observation):
    p=observation_style(observation)
    if p['value'] not in {'feminine','masculine'} or p['visibility']=='unknown':return 0
    style=asset_style(asset)
    return 0 if style==p['value'] else 1 if style=='unisex' else 2 if style=='unspecified' else 3
