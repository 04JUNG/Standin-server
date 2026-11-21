"""Loss-checked solid-material mapping for the pinned Blender FBX adapter.

Blender 5.2 exports SpecularColor=BaseColor and SpecularFactor=IORLevel/2.
Only source RGB values representable by that mapping are admitted, never averaged.
"""
import math

from converter.local.contract import LocalError


def specular_ior_level(diffuse, specular):
    def valid_rgb(value):
        return (isinstance(value, (list, tuple)) and len(value) == 3
                and all(isinstance(c, (int, float)) and not isinstance(c, bool)
                        and math.isfinite(c) and 0 <= c <= 1 for c in value))
    if not valid_rgb(diffuse) or not valid_rgb(specular):
        raise LocalError("UNSUPPORTED_MATERIAL", "Invalid solid CS3C material colors")
    if all(c == 0 for c in specular):
        return 0.0
    anchor = max(range(3), key=lambda i: diffuse[i])
    if diffuse[anchor] == 0:
        raise LocalError("UNSUPPORTED_MATERIAL", "Specular color cannot be represented by this FBX material mapping")
    factor = specular[anchor] / diffuse[anchor]
    # The pinned FBX importer clamps Specular IOR Level to [0,1].
    if not 0 <= factor <= .5 or max(abs(s-factor*d) for d,s in zip(diffuse,specular)) > 1e-6:
        raise LocalError("UNSUPPORTED_MATERIAL", "Specular color cannot be preserved by this FBX material mapping")
    return 2 * factor


def compare_materials(expected, actual):
    """Compare original effective FBX colors/parameters, not just Blender nodes."""
    if expected.keys() != actual.keys():
        raise LocalError("ROUNDTRIP_FAILED", "FBX material identities changed")
    errors = {"maxDiffuseError": 0.0, "maxSpecularError": 0.0,
              "maxShininessError": 0.0, "maxOpacityError": 0.0}
    for name, before in expected.items():
        after = actual[name]
        for values in (before, after):
            numbers = [*values["diffuse"], *values["specular"], values["shininess"], values["opacity"]]
            if (len(values["diffuse"]) != 3 or len(values["specular"]) != 3
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers)):
                raise LocalError("ROUNDTRIP_FAILED", "Invalid FBX material parameters")
        for field, key in (("diffuse", "maxDiffuseError"), ("specular", "maxSpecularError")):
            errors[key] = max(errors[key], *(abs(a-b) for a,b in zip(before[field], after[field])))
        for field, key in (("shininess", "maxShininessError"), ("opacity", "maxOpacityError")):
            errors[key] = max(errors[key], abs(before[field]-after[field]))
    if any(not math.isfinite(v) or v > 1e-5 for v in errors.values()):
        raise LocalError("ROUNDTRIP_FAILED", "FBX changed original solid material parameters")
    return {"passed": True, "materialCount": len(expected), **errors}
