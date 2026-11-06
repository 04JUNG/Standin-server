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
