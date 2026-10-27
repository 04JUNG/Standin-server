"""Recognize CS3C containers without shipping a proprietary Modeler dependency.

The experimental decoder needs bytes extracted from a specific Modeler EXE and
does not preserve materials. Until a redistributable decoder passes the same
character checks as FBX, this adapter must not create a successful model card.
"""
import sqlite3
import struct
from pathlib import Path

from converter.local.contract import LIMITS, LocalError


def inspect_container(path):
    try:
        with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, LIMITS["expandedBytes"])
            db.set_progress_handler(lambda: 1, 100_000)
            db.execute("PRAGMA trusted_schema=OFF")
            db.execute("PRAGMA query_only=ON")
            row = db.execute("SELECT catalog_data FROM character LIMIT 1").fetchone()
        blob = row[0] if row else b""
        if (not isinstance(blob, bytes) or len(blob) < 81
                or blob[:24] != struct.pack("<I", 20) + b"CLIP_STUDIO_3D_DATA2"
                or blob[24:45] != struct.pack("<I", 17) + b"catalog_character"):
            raise ValueError("unknown catalog")
        version, plain_size, cipher_size = struct.unpack_from("<III", blob, 61)
        if version != 2 or cipher_size % 8 or len(blob) != 73 + cipher_size:
            raise ValueError("unknown version")
        if plain_size > LIMITS["expandedBytes"]:
            raise LocalError("LIMIT_EXCEEDED", "CS3C expanded data exceeds the local limit")
    except (sqlite3.Error, ValueError, TypeError, IndexError):
        raise LocalError("UNSUPPORTED_CS3C_VERSION", "Unrecognized CS3C container or catalog version") from None
    return {"container": "SQLite3", "catalogVersion": version,
            "declaredPlainBytes": plain_size, "decoderAvailable": False}


def import_character(path):
    info = inspect_container(path)
    raise LocalError("CS3C_DECODER_UNAVAILABLE",
                     "This runtime does not yet contain a validated redistributable CS3C decoder", info)
