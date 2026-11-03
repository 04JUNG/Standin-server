"""Named, bounded repairs. Opt-in recipes are still subject to visual approval."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactProfile:
    guide: str = "surface"
    palms_up: bool = False
    palms_down: bool = False
    palms_in: bool = False
    forward: float = 0.0
    upward: float = 0.0
    outward: float = 0.0


PROFILES = {
    "surface": ContactProfile(),
    "palm-support": ContactProfile(palms_up=True),
    "carry-forward": ContactProfile(forward=0.12, upward=0.04),
    "lap-pillow": ContactProfile(forward=0.06),
    "offered-blade": ContactProfile(guide="offered-blade", palms_up=True),
    "shoe-contact": ContactProfile(upward=-0.015, outward=0.02),
    "knee-contact": ContactProfile(palms_down=True, upward=0.04, forward=0.02),
    "cup-room": ContactProfile(palms_in=True, outward=0.04),
    "pistol-cradle": ContactProfile(guide="pistol-cradle"),
    "bow-inspection": ContactProfile(guide="bow-inspection", forward=0.08),
}


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError("unknown scene contact profile: " + str(name)) from None
