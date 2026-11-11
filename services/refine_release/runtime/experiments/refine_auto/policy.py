"""Evidence-based stage policy; unavailable regions do not block unrelated ones."""
def upper_action(policy,observed):
 if policy.get('upper_body_fit_verified',False):return 'verified_anchor'
 if {'LeftArm','RightArm'}<=set(observed):return 'bounded_joint_repair'
 return 'insufficient_upper_evidence_preserve_without_certifying'

def lower_available(observed,unobserved=()):
 usable=set(observed)-set(unobserved)
 return bool({'LeftLeg','RightLeg'} & usable)
