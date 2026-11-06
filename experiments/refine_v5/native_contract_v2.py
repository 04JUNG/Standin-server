"""Pure independent structural/geometry guard for diagnostic native roundtrips."""
import numpy as np

def compare_native(expected_heads, actual_heads, expected_parents, actual_parents, expected_lengths, actual_lengths, tolerance=1e-05):
    names = set(expected_heads)
    if any((set(x) != names for x in (actual_heads, expected_parents, actual_parents, expected_lengths, actual_lengths))):
        raise ValueError('native bone inventory changed')
    if expected_parents != actual_parents:
        raise ValueError('native hierarchy changed')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid tolerance')
    eh = np.array([expected_heads[n] for n in sorted(names)], float)
    ah = np.array([actual_heads[n] for n in sorted(names)], float)
    el = np.array([expected_lengths[n] for n in sorted(names)], float)
    al = np.array([actual_lengths[n] for n in sorted(names)], float)
    if eh.shape != (len(names), 3) or ah.shape != eh.shape or (not all((np.isfinite(x).all() for x in (eh, ah, el, al)))):
        raise ValueError('invalid native geometry')
    if (el <= 0).any() or (al <= 0).any():
        raise ValueError('invalid native length')
    position = float(np.linalg.norm(eh - ah, axis=1).max())
    length = float(np.max(np.abs(el - al)))
    if position > tolerance or length > tolerance:
        raise ValueError('native geometry regression')
    return dict(max_head_error_m=position, max_length_error_m=length)
