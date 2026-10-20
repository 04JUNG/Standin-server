"""Head-only work omission, never a hand-safety certificate or release grant."""
from __future__ import annotations


_HEAD_SOURCES = frozenset({'vlm_person', 'vlm_description', 'legacy_shot'})


def is_head_only_output(scope, source) -> bool:
    """Only an explicit, owned head framing can omit inference hand work.

    Do not infer this from missing joints, prose alone, or another tight crop.
    User framing overrides are resolved independently by the BFF at export.
    """
    return (isinstance(scope, str) and scope == 'head'
            and isinstance(source, str) and source in _HEAD_SOURCES)


def head_hand_omission() -> dict:
    return dict(status='not_applicable', method='head-output-hand-omission-v1',
                reason='head_only_output', scope='head',
                complete_collision_certificate=False)


def head_hand_summary() -> dict:
    return dict(**head_hand_omission(), checks=0, elapsed_ms=0.,
                budget_seconds=0., max_checks=0, rejected=[], unverified=[],
                checks_diagnostics=[])


def mirror_hand_evidence_allowed(report, output_scope, original, mirror) -> bool:
    """Transport guard: N/A is valid only for this exact head-only report.

    An omission marker cannot become a clear receipt by rewriting its status.
    Existing screened pairs retain their original transport contract.
    """
    if not isinstance(original, dict) or not isinstance(mirror, dict):
        return False
    marker = head_hand_omission()
    if any(value.get('method') == marker['method'] for value in (original, mirror)):
        return (output_scope == 'head'
                and report.get('hand_skin_screen') == head_hand_summary()
                and original == marker and mirror == marker)
    return original.get('status') == mirror.get('status') == 'clear_screen'
