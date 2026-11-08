"""Diagnostic native-skin line search; this triage cannot certify full collision safety."""
def skin_nonregression(baseline,candidate):
    def pairs(report):return {tuple(sorted(c['regions'])):c for c in report['collisions']}
    before=pairs(baseline);reasons=[]
    if candidate.get('region_closure_failures')!=baseline.get('region_closure_failures'):reasons.append('skin_region_closure_changed')
    for pair,c in pairs(candidate).items():
        b=before.get(pair)
        if b is None:reasons.append('new_skin_penetration:'+':'.join(pair));continue
        if c['max_depth_mm']>b['max_depth_mm']+.1:reasons.append('deeper_skin_penetration:'+':'.join(pair))
        if c['penetrating_vertices']>b['penetrating_vertices']+max(3,.05*b['penetrating_vertices']):reasons.append('larger_skin_penetration:'+':'.join(pair))
    return reasons


COLLISION_REASONS = {'new_skin_penetration', 'deeper_skin_penetration', 'larger_skin_penetration'}


def edit_region(name):
    """Group the connected bones of a planned edit so a limb is retained as a unit."""
    if name in ('Head', 'Neck'):
        return 'head'
    for side in ('Left', 'Right'):
        if not name.startswith(side):
            continue
        suffix = name[len(side):]
        if suffix in ('Arm', 'ForeArm', 'Hand', 'UpperArm'):
            return side.lower() + '_arm'
        if suffix in ('UpLeg', 'Leg', 'Foot', 'ToeBase', 'Thigh', 'Shin'):
            return side.lower() + '_leg'
    return None


def independent_edits(edits, attempts, *, retry_implicated=False):
    """Return editable groups, with groups implicated in a regression ordered last."""
    grouped = {}
    for name, value in edits.items():
        region = edit_region(name)
        if region is None:
            return {}, []
        grouped.setdefault(region, {})[name] = value
    implicated = set()
    for attempt in attempts:
        for reason in attempt['reasons']:
            kind, _, parts = reason.partition(':')
            if kind in COLLISION_REASONS:
                implicated.update(edit_region(part) for part in parts.split(':'))
    blocked = sorted(set(grouped) & implicated)
    groups = {key: grouped[key] for key in sorted(grouped) if key not in implicated}
    if retry_implicated:
        groups.update({key: grouped[key] for key in blocked})
    return groups, blocked
