"""Build and validate the immutable staging SQLite semantic index."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

import numpy as np

from .semantic_embedding import (
    E5_ENCODER_IMPLEMENTATION_VERSION,
    OnnxE5Encoder,
    embedding_profile_fingerprint,
    encoder_artifact_fingerprint,
    load_embedding_profile,
    model_directory,
    sha256_file,
)


SEMANTIC_DB_SCHEMA_VERSION = 2
SEMANTIC_BUILD_MANIFEST_SCHEMA_VERSION = 2
SEMANTIC_INDEX_BUILDER_VERSION = 3


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _latest_proposals(path: Path) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        unit_id = row["semantic_unit_id"]
        previous = current.get(unit_id)
        revision = int(row.get("content_revision", 0))
        if previous is None or revision > int(previous.get("content_revision", 0)):
            current[unit_id] = row
    return current


def _member_posecodes(
    proposals_path: Path,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    current = _latest_proposals(proposals_path)
    active_units = {row["semantic_unit_id"] for row in rows}
    missing_units = sorted(active_units - set(current))
    if missing_units:
        raise ValueError(f"semantic proposal missing for active unit: {missing_units[:5]}")

    posecodes: dict[str, dict[str, Any]] = {}
    versions: set[int] = set()
    coordinate_profiles: set[str] = set()
    measurement_keys: set[tuple[str, ...]] = set()
    for row in rows:
        unit_id = row["semantic_unit_id"]
        proposal = current[unit_id]
        expected_members = {member["pose_id"] for member in row["members"]}
        proposed_members = set(proposal.get("member_posecodes", {}))
        if proposed_members != expected_members:
            raise ValueError(
                f"{unit_id}: member posecode mismatch: "
                f"expected={sorted(expected_members)}, actual={sorted(proposed_members)}"
            )
        for pose_id, posecode in proposal["member_posecodes"].items():
            measurements = posecode.get("measurements")
            if not isinstance(measurements, dict) or not measurements:
                raise ValueError(f"{pose_id}: posecode measurements missing")
            values = np.asarray(list(measurements.values()), dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{pose_id}: non-finite posecode measurement")
            posecode_version = int(posecode["posecode_version"])
            coordinate_profile = str(posecode["coordinate_profile"])
            versions.add(posecode_version)
            coordinate_profiles.add(coordinate_profile)
            measurement_keys.add(tuple(sorted(measurements)))
            posecodes[pose_id] = posecode
    if len(versions) != 1 or len(coordinate_profiles) != 1 or len(measurement_keys) != 1:
        raise ValueError(
            "active member posecode contract is not uniform: "
            f"versions={versions}, profiles={coordinate_profiles}, "
            f"measurement_schemas={len(measurement_keys)}"
        )
    return posecodes, {
        "posecode_version": next(iter(versions)),
        "coordinate_profile": next(iter(coordinate_profiles)),
        "measurement_keys": list(next(iter(measurement_keys))),
        "measurements_per_member": len(next(iter(measurement_keys))),
    }


def _validate_document_inputs(
    rows: list[dict[str, Any]], summary: dict[str, Any], profile: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    if len(rows) != summary["semantic_units"]:
        raise ValueError("semantic document/summary unit count mismatch")
    unit_ids = [row["semantic_unit_id"] for row in rows]
    document_sets = [row["document_set_id"] for row in rows]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("duplicate semantic unit in search documents")
    if len(document_sets) != len(set(document_sets)):
        raise ValueError("duplicate semantic document_set_id")
    if any(not row.get("searchable") for row in rows):
        raise ValueError("unsearchable unit cannot enter the dense index")

    documents: list[dict[str, Any]] = []
    document_ids: list[str] = []
    caps = profile["retrieval"]["document_type_caps"]
    for row in rows:
        counts: dict[str, int] = {}
        if len(row.get("members", [])) != 2:
            raise ValueError(f"{row['semantic_unit_id']}: incomplete mirror pair")
        if not row.get("observed_unit_atoms"):
            raise ValueError(f"{row['semantic_unit_id']}: observed atoms missing")
        for document in row["text_documents"]:
            document_ids.append(document["document_id"])
            if document["text_sha256"] != _sha256_text(document["text"]):
                raise ValueError(f"text hash mismatch: {document['document_id']}")
            if document["retrieval"]["dense_candidate"] is not True:
                raise ValueError(f"dense candidate disabled: {document['document_id']}")
            document_type = document["document_type"]
            counts[document_type] = counts.get(document_type, 0) + 1
            documents.append({**document, "semantic_unit_id": row["semantic_unit_id"]})
        if counts.get("posecode_render") != 2:
            raise ValueError(f"{row['semantic_unit_id']}: two posecode passages required")
        for document_type, count in counts.items():
            if document_type not in caps or count > int(caps[document_type]):
                raise ValueError(
                    f"{row['semantic_unit_id']}: document cap exceeded for {document_type}"
                )
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("duplicate semantic text document ID")
    if len(documents) != summary["text_documents"]:
        raise ValueError("semantic text document count mismatch")
    return documents, unit_ids


def _geometry_pose_ids(path: Path) -> set[str]:
    with sqlite3.connect(path) as connection:
        return {row[0] for row in connection.execute("SELECT pose_id FROM poses")}


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        f"""
        PRAGMA user_version={SEMANTIC_DB_SCHEMA_VERSION};
        PRAGMA foreign_keys=ON;
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE semantic_units (
            semantic_unit_id TEXT PRIMARY KEY,
            document_set_id TEXT NOT NULL UNIQUE,
            semantic_unit_type TEXT NOT NULL,
            source_clip_id TEXT NOT NULL,
            mapping_id TEXT NOT NULL,
            mapping_status TEXT NOT NULL,
            mapping_confidence TEXT NOT NULL,
            canonical_pose_id TEXT NOT NULL,
            mirrored_pose_id TEXT NOT NULL,
            mirror_group_id TEXT NOT NULL,
            mirror_validation_status TEXT NOT NULL,
            source_mapping_json TEXT NOT NULL,
            retrieval_policy_json TEXT NOT NULL,
            searchable INTEGER NOT NULL CHECK(searchable IN (0,1))
        );
        CREATE TABLE pose_semantic_members (
            semantic_unit_id TEXT NOT NULL REFERENCES semantic_units(semantic_unit_id),
            pose_id TEXT NOT NULL,
            variant_kind TEXT NOT NULL CHECK(variant_kind IN ('original','mirrored')),
            mirror_of TEXT,
            bvh_sha256 TEXT NOT NULL,
            posecode_version INTEGER NOT NULL,
            coordinate_profile TEXT NOT NULL,
            posecode_measurements_json TEXT NOT NULL,
            observed_atoms_json TEXT NOT NULL,
            PRIMARY KEY(semantic_unit_id, pose_id),
            UNIQUE(pose_id)
        );
        CREATE INDEX idx_semantic_members_pose ON pose_semantic_members(pose_id);
        CREATE TABLE semantic_text_documents (
            document_id TEXT PRIMARY KEY,
            semantic_unit_id TEXT NOT NULL REFERENCES semantic_units(semantic_unit_id),
            document_type TEXT NOT NULL,
            language TEXT NOT NULL,
            text TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            evidence_state TEXT NOT NULL,
            dense_candidate INTEGER NOT NULL CHECK(dense_candidate IN (0,1)),
            lexical_candidate INTEGER NOT NULL CHECK(lexical_candidate IN (0,1)),
            candidate_only INTEGER NOT NULL CHECK(candidate_only IN (0,1)),
            hard_filter_eligible INTEGER NOT NULL CHECK(hard_filter_eligible IN (0,1)),
            retrieval_weight REAL NOT NULL,
            provenance_json TEXT NOT NULL
        );
        CREATE INDEX idx_semantic_documents_unit
            ON semantic_text_documents(semantic_unit_id, document_type);
        CREATE VIRTUAL TABLE semantic_text_documents_fts USING fts5(
            document_id UNINDEXED,
            semantic_unit_id UNINDEXED,
            language UNINDEXED,
            text,
            tokenize='unicode61'
        );
        CREATE TABLE semantic_embeddings (
            document_id TEXT PRIMARY KEY
                REFERENCES semantic_text_documents(document_id),
            embedding_blob BLOB NOT NULL,
            embedding_version TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            dtype TEXT NOT NULL,
            l2_norm REAL NOT NULL,
            vector_sha256 TEXT NOT NULL
        );
        CREATE TABLE semantic_atoms (
            atom_id TEXT PRIMARY KEY,
            semantic_unit_id TEXT NOT NULL REFERENCES semantic_units(semantic_unit_id),
            scope TEXT NOT NULL,
            predicate TEXT NOT NULL,
            subject TEXT,
            relation TEXT,
            object TEXT,
            axis TEXT,
            value TEXT,
            bucket TEXT,
            polarity TEXT NOT NULL,
            evidence_state TEXT NOT NULL,
            provenance_json TEXT NOT NULL,
            atom_json TEXT NOT NULL
        );
        CREATE INDEX idx_semantic_atoms_match
            ON semantic_atoms(predicate, subject, relation, object, value, bucket);
        """
    )


def _populate_database(
    connection: sqlite3.Connection,
    *,
    rows: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    vectors: np.ndarray,
    embedding_version: str,
    member_posecodes: dict[str, dict[str, Any]],
    meta: dict[str, Any],
) -> str:
    connection.executemany(
        "INSERT INTO meta(key,value_json) VALUES(?,?)",
        [(key, _json(value)) for key, value in sorted(meta.items())],
    )
    for row in rows:
        source_mapping = row["source_mapping"]
        mirror = row["mirror"]
        connection.execute(
            """INSERT INTO semantic_units VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row["semantic_unit_id"],
                row["document_set_id"],
                row["semantic_unit_type"],
                row["source_clip_id"],
                source_mapping["mapping_id"],
                source_mapping["status"],
                source_mapping["confidence"],
                row["canonical_pose_id"],
                row["mirrored_pose_id"],
                mirror["mirror_group_id"],
                mirror["validation_status"],
                _json(source_mapping),
                _json(row["retrieval_policy"]),
                int(row["searchable"]),
            ),
        )
        connection.executemany(
            "INSERT INTO pose_semantic_members VALUES(?,?,?,?,?,?,?,?,?)",
            [
                (
                    row["semantic_unit_id"],
                    member["pose_id"],
                    member["variant_kind"],
                    member.get("mirror_of"),
                    member["bvh_sha256"],
                    int(member_posecodes[member["pose_id"]]["posecode_version"]),
                    member_posecodes[member["pose_id"]]["coordinate_profile"],
                    _json(member_posecodes[member["pose_id"]]["measurements"]),
                    _json(member_posecodes[member["pose_id"]]["observed_atoms"]),
                )
                for member in row["members"]
            ],
        )
        for index, atom in enumerate(row["observed_unit_atoms"]):
            atom_id = _sha256_json(
                {
                    "semantic_unit_id": row["semantic_unit_id"],
                    "ordinal": index,
                    "atom": atom,
                }
            )
            connection.execute(
                "INSERT INTO semantic_atoms VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    atom_id,
                    row["semantic_unit_id"],
                    atom["scope"],
                    atom["predicate"],
                    atom.get("subject"),
                    atom.get("relation"),
                    atom.get("object"),
                    atom.get("axis"),
                    atom.get("value"),
                    atom.get("bucket"),
                    atom["polarity"],
                    atom["evidence_state"],
                    _json(atom["provenance"]),
                    _json(atom),
                ),
            )

    vector_fingerprint_rows: list[tuple[str, str]] = []
    for document, vector in zip(documents, vectors, strict=True):
        retrieval = document["retrieval"]
        connection.execute(
            "INSERT INTO semantic_text_documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                document["document_id"],
                document["semantic_unit_id"],
                document["document_type"],
                document["language"],
                document["text"],
                document["text_sha256"],
                document["evidence_state"],
                int(retrieval["dense_candidate"]),
                int(retrieval["lexical_candidate"]),
                int(retrieval["candidate_only"]),
                int(retrieval["hard_filter_eligible"]),
                float(retrieval["weight"]),
                _json(document["provenance"]),
            ),
        )
        connection.execute(
            "INSERT INTO semantic_text_documents_fts VALUES(?,?,?,?)",
            (
                document["document_id"],
                document["semantic_unit_id"],
                document["language"],
                document["text"],
            ),
        )
        encoded = np.ascontiguousarray(vector, dtype="<f4").tobytes()
        vector_hash = "sha256:" + hashlib.sha256(encoded).hexdigest()
        norm = float(np.linalg.norm(vector))
        connection.execute(
            "INSERT INTO semantic_embeddings VALUES(?,?,?,?,?,?,?)",
            (
                document["document_id"],
                sqlite3.Binary(encoded),
                embedding_version,
                int(vector.shape[0]),
                "float32_le",
                norm,
                vector_hash,
            ),
        )
        vector_fingerprint_rows.append((document["document_id"], vector_hash))
    return _sha256_json(vector_fingerprint_rows)


def validate_semantic_index(
    db_path: Path,
    manifest_path: Path,
    *,
    profile_path: Path | None = None,
    documents_path: Path | None = None,
    inventory_path: Path | None = None,
    geometry_db_path: Path | None = None,
    proposals_path: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SEMANTIC_BUILD_MANIFEST_SCHEMA_VERSION:
        raise ValueError("semantic build manifest schema mismatch")
    if sha256_file(db_path) != manifest["artifacts"]["semantic_db_sha256"]:
        raise ValueError("semantic DB hash mismatch")
    if profile_path is not None:
        profile = load_embedding_profile(profile_path)
        if (
            embedding_profile_fingerprint(profile)
            != manifest["embedding"]["profile_fingerprint"]
        ):
            raise ValueError("semantic embedding profile mismatch")
    if documents_path is not None:
        if sha256_file(documents_path) != manifest["inputs"]["search_documents_sha256"]:
            raise ValueError("semantic search document artifact mismatch")
    if inventory_path is not None:
        if sha256_file(inventory_path) != manifest["inputs"]["geometry_manifest_sha256"]:
            raise ValueError("geometry inventory mismatch")
    if geometry_db_path is not None:
        if sha256_file(geometry_db_path) != manifest["inputs"]["geometry_db_sha256"]:
            raise ValueError("geometry DB mismatch")
    if proposals_path is not None:
        if sha256_file(proposals_path) != manifest["inputs"]["posecode_proposals_sha256"]:
            raise ValueError("posecode proposal artifact mismatch")

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"semantic DB integrity failure: {integrity}")
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise ValueError(f"semantic DB foreign key failure: {foreign_keys[:3]}")
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "semantic_units",
                "pose_semantic_members",
                "semantic_text_documents",
                "semantic_text_documents_fts",
                "semantic_embeddings",
                "semantic_atoms",
            )
        }
        expected = manifest["counts"]
        expected_counts = {
            "semantic_units": expected["semantic_units"],
            "pose_semantic_members": expected["pose_members"],
            "semantic_text_documents": expected["text_documents"],
            "semantic_text_documents_fts": expected["text_documents"],
            "semantic_embeddings": expected["embeddings"],
            "semantic_atoms": expected["observed_unit_atoms"],
        }
        if counts != expected_counts:
            raise ValueError(f"semantic DB count mismatch: {counts} != {expected_counts}")

        posecode_contract = manifest["posecode"]
        member_rows = connection.execute(
            """SELECT posecode_version, coordinate_profile,
                      posecode_measurements_json, observed_atoms_json
               FROM pose_semantic_members"""
        ).fetchall()
        measurement_keys = set(posecode_contract["measurement_keys"])
        for posecode_version, coordinate_profile, measurements_json, atoms_json in member_rows:
            if posecode_version != posecode_contract["posecode_version"]:
                raise ValueError("member posecode version mismatch")
            if coordinate_profile != posecode_contract["coordinate_profile"]:
                raise ValueError("member posecode coordinate profile mismatch")
            measurements = json.loads(measurements_json)
            atoms = json.loads(atoms_json)
            if set(measurements) != measurement_keys or not isinstance(atoms, list):
                raise ValueError("member posecode payload schema mismatch")
            values = np.asarray(list(measurements.values()), dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("member posecode contains non-finite measurement")

        version = manifest["embedding"]["embedding_version"]
        dimension = int(manifest["embedding"]["dimension"])
        rows = connection.execute(
            """SELECT embedding_blob, embedding_version, dimension, dtype,
                      l2_norm, vector_sha256
               FROM semantic_embeddings"""
        ).fetchall()
        for blob, row_version, row_dimension, dtype, stored_norm, vector_hash in rows:
            if row_version != version or row_dimension != dimension or dtype != "float32_le":
                raise ValueError("semantic embedding metadata mismatch")
            if len(blob) != dimension * 4:
                raise ValueError("semantic embedding byte length mismatch")
            if "sha256:" + hashlib.sha256(blob).hexdigest() != vector_hash:
                raise ValueError("semantic embedding vector hash mismatch")
            vector = np.frombuffer(blob, dtype="<f4")
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(vector).all() or abs(norm - 1.0) > 1e-5:
                raise ValueError("semantic embedding finite/norm validation failure")
            if abs(norm - stored_norm) > 1e-6:
                raise ValueError("semantic embedding stored norm mismatch")

        missing = connection.execute(
            """SELECT COUNT(*) FROM semantic_text_documents d
               LEFT JOIN semantic_embeddings e ON e.document_id=d.document_id
               WHERE e.document_id IS NULL"""
        ).fetchone()[0]
        if missing:
            raise ValueError(f"semantic documents without embeddings: {missing}")
        build_id = json.loads(
            connection.execute(
                "SELECT value_json FROM meta WHERE key='semantic_build_id'"
            ).fetchone()[0]
        )
        if build_id != manifest["semantic_build_id"]:
            raise ValueError("semantic build ID mismatch inside DB")
    return {
        "status": "pass",
        "semantic_build_id": manifest["semantic_build_id"],
        "counts": counts,
        "dimension": manifest["embedding"]["dimension"],
        "embedding_version": manifest["embedding"]["embedding_version"],
    }


def build_semantic_index(
    *,
    documents_path: Path,
    document_summary_path: Path,
    inventory_path: Path,
    geometry_db_path: Path,
    proposals_path: Path,
    profile_path: Path,
    models_root: Path,
    output_root: Path,
) -> tuple[Path, dict[str, Any], bool]:
    rows = _read_jsonl(documents_path)
    document_summary = json.loads(document_summary_path.read_text(encoding="utf-8"))
    profile = load_embedding_profile(profile_path)
    documents, _ = _validate_document_inputs(rows, document_summary, profile)
    member_posecodes, posecode_contract = _member_posecodes(proposals_path, rows)

    inventory_rows = _read_jsonl(inventory_path)
    if not inventory_rows or inventory_rows[0].get("record_type") != "inventory_header":
        raise ValueError("geometry inventory header missing")
    inventory_header = inventory_rows[0]
    if inventory_header["pose_library_version"] != document_summary["pose_library_version"]:
        raise ValueError("geometry inventory pose_library_version mismatch")
    inventory_members = {
        row["pose_id"]: row
        for row in inventory_rows[1:]
        if row.get("record_type") == "pose_member_inventory"
    }
    for row in rows:
        for member in row["members"]:
            inventory_member = inventory_members.get(member["pose_id"])
            if inventory_member is None:
                raise ValueError(f"semantic member absent from inventory: {member['pose_id']}")
            if inventory_member["bvh"]["sha256"] != member["bvh_sha256"]:
                raise ValueError(f"semantic/inventory BVH hash mismatch: {member['pose_id']}")

    model_dir = model_directory(profile, models_root)
    encoder = OnnxE5Encoder(profile, model_dir)
    geometry_ids = _geometry_pose_ids(geometry_db_path)
    member_ids = {
        member["pose_id"] for row in rows for member in row.get("members", [])
    }
    missing_geometry = sorted(member_ids - geometry_ids)
    if missing_geometry:
        raise ValueError(f"semantic member absent from geometry DB: {missing_geometry[:5]}")

    input_hashes = {
        "search_documents_sha256": sha256_file(documents_path),
        "search_document_summary_sha256": sha256_file(document_summary_path),
        "geometry_manifest_sha256": sha256_file(inventory_path),
        "geometry_db_sha256": sha256_file(geometry_db_path),
        "posecode_proposals_sha256": sha256_file(proposals_path),
    }
    profile_fingerprint = embedding_profile_fingerprint(profile)
    artifact_fingerprint = encoder_artifact_fingerprint(profile)
    build_payload = {
        "semantic_db_schema_version": SEMANTIC_DB_SCHEMA_VERSION,
        "semantic_index_builder_version": SEMANTIC_INDEX_BUILDER_VERSION,
        "e5_encoder_implementation_version": E5_ENCODER_IMPLEMENTATION_VERSION,
        "semantic_build_input_id": document_summary["semantic_build_input_id"],
        "pose_library_version": document_summary["pose_library_version"],
        "input_hashes": input_hashes,
        "embedding_profile_fingerprint": profile_fingerprint,
        "encoder_artifact_fingerprint": artifact_fingerprint,
        "passage_template_version": document_summary["passage_template_version"],
        "retrieval": profile["retrieval"],
    }
    semantic_build_id = _sha256_json(build_payload)
    build_name = semantic_build_id.removeprefix("sha256:")
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / build_name
    final_db = final_dir / "pose_semantics.db"
    final_manifest = final_dir / "semantic-build.json"
    if final_dir.exists():
        if not final_db.is_file() or not final_manifest.is_file():
            raise ValueError(f"incomplete semantic build directory already exists: {final_dir}")
        validation = validate_semantic_index(
            final_db,
            final_manifest,
            profile_path=profile_path,
            documents_path=documents_path,
            inventory_path=inventory_path,
            geometry_db_path=geometry_db_path,
            proposals_path=proposals_path,
        )
        manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
        manifest["validation"] = validation
        return final_dir, manifest, True

    vectors, encoding_stats = encoder.encode(
        [document["text"] for document in documents], kind="passage"
    )
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{build_name}.", suffix=".tmp", dir=output_root)
    )
    try:
        db_path = temporary_dir / "pose_semantics.db"
        with sqlite3.connect(db_path) as connection:
            _create_schema(connection)
            meta = {
                "artifact_type": "semantic_search_index",
                "semantic_db_schema_version": SEMANTIC_DB_SCHEMA_VERSION,
                "semantic_index_builder_version": SEMANTIC_INDEX_BUILDER_VERSION,
                "e5_encoder_implementation_version": E5_ENCODER_IMPLEMENTATION_VERSION,
                "semantic_build_id": semantic_build_id,
                "pose_library_version": document_summary["pose_library_version"],
                "semantic_build_input_id": document_summary["semantic_build_input_id"],
                "embedding_profile_id": profile["embedding_profile_id"],
                "embedding_profile_fingerprint": profile_fingerprint,
                "embedding_version": encoder.embedding_version,
                "passage_template_version": document_summary["passage_template_version"],
                "production_ready": False,
                "posecode_contract": posecode_contract,
            }
            matrix_fingerprint = _populate_database(
                connection,
                rows=rows,
                documents=documents,
                vectors=vectors,
                embedding_version=encoder.embedding_version,
                member_posecodes=member_posecodes,
                meta=meta,
            )
            connection.commit()
            connection.execute("PRAGMA optimize")
            connection.execute("VACUUM")

        manifest = {
            "artifact_type": "semantic_search_index_build",
            "schema_version": SEMANTIC_BUILD_MANIFEST_SCHEMA_VERSION,
            "semantic_db_schema_version": SEMANTIC_DB_SCHEMA_VERSION,
            "semantic_index_builder_version": SEMANTIC_INDEX_BUILDER_VERSION,
            "e5_encoder_implementation_version": E5_ENCODER_IMPLEMENTATION_VERSION,
            "semantic_build_id": semantic_build_id,
            "staging_ready": True,
            "production_ready": False,
            "promotion_blockers": [
                "semantic_holdout_not_passed",
                "semantic_release_bundle_not_promoted",
                "licenses_and_product_bvh_export_unresolved",
            ],
            "counts": {
                "semantic_units": len(rows),
                "pose_members": len(member_ids),
                "text_documents": len(documents),
                "embeddings": int(vectors.shape[0]),
                "observed_unit_atoms": sum(
                    len(row["observed_unit_atoms"]) for row in rows
                ),
            },
            "inputs": {
                **input_hashes,
                "semantic_build_input_id": document_summary["semantic_build_input_id"],
                "pose_library_version": document_summary["pose_library_version"],
                "passage_template_version": document_summary["passage_template_version"],
            },
            "posecode": posecode_contract,
            "embedding": {
                "profile_id": profile["embedding_profile_id"],
                "profile_fingerprint": profile_fingerprint,
                "embedding_version": encoder.embedding_version,
                "model_id": profile["model"]["id"],
                "revision": profile["model"]["revision"],
                "license": profile["model"]["license"],
                "encoder_artifact_fingerprint": artifact_fingerprint,
                "runtime": encoder.runtime_versions,
                **profile["encoding"],
                "encoding_stats": encoding_stats,
                "matrix_fingerprint": matrix_fingerprint,
            },
            "retrieval": profile["retrieval"],
            "artifacts": {
                "semantic_db": "pose_semantics.db",
                "semantic_db_sha256": sha256_file(db_path),
            },
        }
        manifest_path = temporary_dir / "semantic-build.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validation = validate_semantic_index(
            db_path,
            manifest_path,
            profile_path=profile_path,
            documents_path=documents_path,
            inventory_path=inventory_path,
            geometry_db_path=geometry_db_path,
            proposals_path=proposals_path,
        )
        manifest["validation"] = validation
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, final_dir)
    except Exception:
        for path in sorted(temporary_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        temporary_dir.rmdir()
        raise
    return final_dir, manifest, False
