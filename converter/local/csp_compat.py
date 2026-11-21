"""Lossless final-FBX name adapter for CSP's root_group_ import recursion.

Only an exported LimbNode display name changes. FBX object IDs, links, skin,
transforms and geometry remain byte identical; equal-length names keep offsets
and footer untouched. Registration/recovery files retain the authored name.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct

from converter.local.contract import LIMITS, LocalError

VERSION = "personal-csp-names/1"
RESERVED = b"root_group_"
MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"


def require(value, message):
    if not value:
        raise LocalError("ROUNDTRIP_FAILED", message)


def plan_aliases(models):
    """models: (object ID, raw FBX name, FBX model type), from bounded parser."""
    names = [name.split(b"\x00\x01", 1)[0] for _, name, _ in models]
    targets = [(ident, name) for ident, name, kind in models
               if name == RESERVED + b"\x00\x01Model" and kind == b"LimbNode"]
    require(len(targets) <= 1, "Ambiguous CSP root name")
    if not targets:
        return []
    # Deterministic, equal-length and unique among every Model, including meshes.
    candidates = [b"StandinBase", *(f"Standin{i:04x}".encode() for i in range(LIMITS["bones"] + 1))]
    alias = next((n for n in candidates if n not in names), None)
    require(alias is not None, "No unique CSP export alias")
    return [{"objectId": targets[0][0], "original": RESERVED.decode(), "exported": alias.decode()}]


def patch_bytes(data, aliases):
    require(data.startswith(MAGIC) and 27 <= len(data) <= LIMITS["sourceBytes"],
            "Invalid final binary FBX for CSP adapter")
    for entry in aliases:
        original, alias = entry["original"].encode(), entry["exported"].encode()
        require(original == RESERVED and len(original) == len(alias) and original != alias,
                "Invalid CSP alias plan")
        original += b"\x00\x01Model"
        alias += b"\x00\x01Model"
        prop = b"S" + struct.pack("<I", len(original)) + original
        replacement = b"S" + struct.pack("<I", len(alias)) + alias
        # This is our own validated Blender export. Its uncompressed Model
        # string occurs exactly once. Refuse ambiguity rather than scanning or
        # rewriting arbitrary arrays or changing a second property.
        require(data.count(prop) == 1, "Ambiguous final FBX name property")
        data = data.replace(prop, replacement, 1)
    return data


def apply_final_aliases(path):
    from converter.local.inspection import bounded_fbx_parser
    path = Path(path)
    require(path.stat().st_size <= LIMITS["sourceBytes"], "Oversized final FBX")
    with bounded_fbx_parser() as parser:
        tree, _ = parser.parse(str(path))
    objects = next((n for n in tree.elems if n.id == b"Objects"), None)
    require(objects is not None, "Missing final FBX objects")
    models = [(n.props[0], n.props[1], n.props[2]) for n in objects.elems if n.id == b"Model"]
    aliases = plan_aliases(models)
    before = path.read_bytes()
    after = patch_bytes(before, aliases)
    if aliases:
        path.write_bytes(after)
    return {"version": VERSION, "applied": bool(aliases), "nameAliases": aliases,
            "allOtherBytesExact": True, "weightsTruncated": False,
            "beforeSha256": hashlib.sha256(before).hexdigest(),
            "outputSha256": hashlib.sha256(after).hexdigest()}
