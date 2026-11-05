"""Opt-in contact repairs, dispatched by object family, with legacy fallback."""

from converter.scene_contact_profiles import get_profile
from converter.scene_placement import hand_fit
from converter.scene_contact_carry import carry, reading, phone
from converter.scene_contact_tools import thin_tool, bow
from converter.scene_contact_fabric import flexible
from converter import scene_contact_special as special


def add_guides(arm, spec, support="", *, profile, mirror=False, bow_state="drawn"):
    from converter.scene_guides import add_guides as original

    if not spec:
        return original(arm, spec, support, mirror=mirror, bow_state=bow_state)
    kind, grip, *_ = spec.split(":")
    config = get_profile(profile)

    def point(name):
        return arm.matrix_world @ arm.pose.bones["mixamorig:" + name].head

    chain = [point(n) for n in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head")]
    scale = sum((b - a).length for a, b in zip(chain, chain[1:])) / 0.65
    fits = {s: hand_fit(arm, n) for s, n in (("L", "Left"), ("R", "Right"))}
    if kind == "lantern" and grip != "B":
        from converter.scene_contact_hanging import lantern

        lantern(arm, grip, fits, scale)
    elif config.guide == "offered-blade":
        special.offered_blade(arm, kind, fits, scale)
    elif config.guide == "bow-inspection":
        if mirror:
            fits = {"L": fits["R"], "R": fits["L"]}
        special.inspection_bow(arm, fits, scale)
    elif kind == "pickaxe" or kind == "key" and grip == "B":
        special.shared_tool(arm, kind, fits, scale)
    elif kind == "gem":
        special.small_gem(arm, grip, fits, scale)
    elif kind == "goblet" and grip == "B":
        special.goblet(arm, fits, scale)
    elif kind == "crown":
        special.hollow_crown(arm, fits, scale)
    elif kind == "bouquet":
        special.bouquet(arm, fits, scale)
    elif kind in {"spoon", "knife", "comb", "key"} and grip != "B":
        thin_tool(arm, kind, grip, fits, scale)
    elif kind in {"towel", "scarf", "handkerchief", "rope"}:
        flexible(arm, kind, grip, fits, scale)
    elif kind == "bow" and grip != "B":
        bow(arm, grip, fits, scale, bow_state == "undrawn")
    elif kind in {
        "tray",
        "plate",
        "cake",
        "ringbox",
        "boulder",
        "orb",
        "gift",
        "pillow",
        "laundry",
        "teddy",
        "crate",
        "bag",
    }:
        carry(arm, kind, grip, fits, scale)
    elif kind in {"book", "letter", "photo"}:
        reading(arm, kind, grip, fits, scale)
    elif kind == "phone":
        phone(arm, grip, fits, scale)
    else:
        return original(arm, spec, support, mirror=mirror, bow_state=bow_state)
    return {
        "count": 1,
        "specification": spec,
        "contact_profile": profile,
        "quality": "primitive-placement-guide",
        "collision_certified": False,
    }
