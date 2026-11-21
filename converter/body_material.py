"""Restore approved anatomical clay exports to the legacy neutral gray.

Change only known material properties in the typed FBX. Geometry, bind data,
normals and pose are not passed through another Blender export.
"""
from __future__ import annotations

from pathlib import Path
import re

from converter.outline_fbx import fingerprint, one, write

VERSION = "anatomy-clay-gray-v1"
COLOR = (0.48, 0.53, 0.59)
# Match the previous male preset's roughness (0.68) in FBX Phong terms.
PROPERTIES = {
    b"DiffuseColor": COLOR,
    b"SpecularColor": COLOR,
    b"ReflectionColor": COLOR,
    b"SpecularFactor": (0.25,),
    b"Shininess": (10.24,),
    b"ShininessExponent": (10.24,),
}
_NAME = re.compile(rb"Standin anatomical clay(?:\.\d+)?")


def _protected(root):
    """Everything except the explicitly allowed color/shine values."""
    result = []
    for node in root.elems:
        if node.id != b"Objects":
            result.append(fingerprint(node))
            continue
        for obj in node.elems:
            name = obj.props[1].split(b"\x00", 1)[0] if len(obj.props) > 1 and isinstance(obj.props[1], bytes) else b""
            if obj.id != b"Material" or not _NAME.fullmatch(name):
                result.append(fingerprint(obj))
                continue
            result.append((obj.id, tuple(obj.props), bytes(obj.props_type)))
            for child in obj.elems:
                if child.id != b"Properties70":
                    result.append(fingerprint(child))
                    continue
                for prop in child.elems:
                    if prop.props and prop.props[0] in PROPERTIES:
                        result.append((prop.id, tuple(prop.props[:4]), bytes(prop.props_type)))
                    else:
                        result.append(fingerprint(prop))
    return result


def restore_tree(root):
    objects = one(root, b"Objects")
    connections = one(root, b"Connections")
    texture_ids = {obj.props[0] for obj in objects.elems if obj.id in (b"Texture", b"LayeredTexture")}
    textured = {link.props[2] for link in connections.elems
                if len(link.props) > 2 and link.props[1] in texture_ids}
    before = _protected(root)
    changed = []
    for obj in objects.elems:
        if obj.id != b"Material" or obj.props[0] in textured:
            continue
        name = obj.props[1].split(b"\x00", 1)[0]
        if not _NAME.fullmatch(name):
            continue
        fields = {p.props[0]: p for p in one(obj, b"Properties70").elems if p.props}
        if not PROPERTIES.keys() <= fields.keys():
            raise ValueError("anatomical clay material is missing expected FBX properties")
        altered = False
        for key, values in PROPERTIES.items():
            prop = fields[key]
            if len(prop.props) != 4 + len(values) or bytes(prop.props_type[4:]) != b"D" * len(values):
                raise ValueError("unexpected anatomical clay FBX property type")
            if any(abs(float(a) - b) > 1e-7 for a, b in zip(prop.props[4:], values)):
                prop.props[4:] = values
                altered = True
        if altered:
            changed.append(name.decode("utf-8"))
    if before != _protected(root):
        raise RuntimeError("body material normalization changed protected FBX data")
    return {"version": VERSION, "changed_materials": changed, "color": list(COLOR)}


def restore_anatomical_clay(path):
    from io_scene_fbx import parse_fbx
    path = Path(path)
    root, version = parse_fbx.parse(str(path))
    protected = _protected(root)
    receipt = restore_tree(root)
    if not receipt["changed_materials"]:
        return receipt
    temporary = path.with_name(path.stem + ".clay-material.fbx")
    try:
        write(temporary, root, version)
        checked, checked_version = parse_fbx.parse(str(temporary))
        if checked_version != version or protected != _protected(checked):
            raise RuntimeError("material export changed protected FBX data")
        if restore_tree(checked)["changed_materials"]:
            raise RuntimeError("material values did not survive the FBX round trip")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt
