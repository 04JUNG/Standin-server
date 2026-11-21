"""Detect a waist hand paired with a lowered arm."""


def uses_pair(person):
    search = person['dynamic_search']
    evidence = search['description_evidence']
    tags = ((evidence.get('description') or {}).get('body_details') or '').split(' · ')
    return (search['route'] in ('hybrid', 'semantic') and evidence['status'] == 'valid'
            and {'한 손 허리', '한 팔 내리기'} <= set(tags))
