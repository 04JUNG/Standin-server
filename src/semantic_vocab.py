"""Versioned controlled vocabulary for the offline semantic pose catalog.

This module is intentionally separate from :class:`src.schema.Action`.  The
runtime enum remains a small routing contract; this vocabulary describes the
library documents used by the future semantic-search index.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any


SEMANTIC_VOCAB_VERSION = 2
DEFAULT_VOCAB_PATH = Path(__file__).resolve().parents[1] / "config/semantic_vocab.v2.json"


class SemanticVocabularyError(ValueError):
    """Raised when the vocabulary or an annotation violates the v2 contract."""


def _normalized_aliases(row: dict[str, Any]) -> set[str]:
    values = [row["id"], row["ko"], row["en"]]
    values.extend(row.get("aliases_ko", []))
    values.extend(row.get("aliases_en", []))
    return {" ".join(value.casefold().split()) for value in values if value.strip()}


def validate_vocab_document(document: dict[str, Any]) -> None:
    if document.get("semantic_vocab_version") != SEMANTIC_VOCAB_VERSION:
        raise SemanticVocabularyError(
            f"expected semantic_vocab_version={SEMANTIC_VOCAB_VERSION}"
        )
    fields = document.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise SemanticVocabularyError("fields must be a non-empty object")
    required = {
        "action_domain", "action_ids", "posture", "motion_phase",
        "style_context", "intended_props", "interaction_kind",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise SemanticVocabularyError(f"required fields missing: {missing}")

    ids_by_field: dict[str, set[str]] = {}
    for field_name, field in fields.items():
        if field.get("cardinality") not in {"one", "many"}:
            raise SemanticVocabularyError(f"{field_name}: invalid cardinality")
        rows = field.get("values")
        if not isinstance(rows, list) or not rows:
            raise SemanticVocabularyError(f"{field_name}: values must be non-empty")
        ids = [row.get("id") for row in rows]
        if any(not isinstance(value, str) or not value for value in ids):
            raise SemanticVocabularyError(f"{field_name}: invalid canonical id")
        if len(ids) != len(set(ids)):
            raise SemanticVocabularyError(f"{field_name}: duplicate canonical id")
        ids_by_field[field_name] = set(ids)
        aliases: dict[str, str] = {}
        for row in rows:
            if not row.get("ko") or not row.get("en"):
                raise SemanticVocabularyError(f"{field_name}/{row['id']}: labels missing")
            for alias in _normalized_aliases(row):
                previous = aliases.setdefault(alias, row["id"])
                if previous != row["id"]:
                    raise SemanticVocabularyError(
                        f"{field_name}: alias {alias!r} maps to {previous!r} and {row['id']!r}"
                    )

    domains = ids_by_field["action_domain"]
    for row in fields["action_ids"]["values"]:
        if row.get("domain") not in domains - {"unknown"}:
            raise SemanticVocabularyError(
                f"action_ids/{row['id']}: unknown domain {row.get('domain')!r}"
            )


@lru_cache(maxsize=4)
def load_semantic_vocab(path: str | Path = DEFAULT_VOCAB_PATH) -> dict[str, Any]:
    resolved = Path(path).resolve()
    document = json.loads(resolved.read_text(encoding="utf-8"))
    validate_vocab_document(document)
    return document


def vocabulary_fingerprint(document: dict[str, Any] | None = None) -> str:
    value = document or load_semantic_vocab()
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_ids(field_name: str, document: dict[str, Any] | None = None) -> set[str]:
    value = document or load_semantic_vocab()
    try:
        return {row["id"] for row in value["fields"][field_name]["values"]}
    except KeyError as exc:
        raise SemanticVocabularyError(f"unknown semantic field: {field_name}") from exc


def resolve_exact_alias(
    field_name: str,
    text: str,
    document: dict[str, Any] | None = None,
) -> str | None:
    """Resolve an exact field-scoped alias; never guess from partial text."""
    value = document or load_semantic_vocab()
    normalized = " ".join(text.casefold().split())
    for row in value["fields"][field_name]["values"]:
        if normalized in _normalized_aliases(row):
            return row["id"]
    return None


def validate_semantic_annotation(
    annotation: dict[str, Any],
    document: dict[str, Any] | None = None,
) -> list[str]:
    """Return deterministic validation errors for a normalized annotation."""
    value = document or load_semantic_vocab()
    errors: list[str] = []
    multi_fields = (
        "action_domain", "action_ids", "posture", "style_context", "intended_props"
    )
    for field_name in multi_fields:
        raw = annotation.get(field_name, [])
        if not isinstance(raw, list):
            errors.append(f"{field_name}: expected list")
            continue
        allowed = canonical_ids(field_name, value)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            errors.append(f"{field_name}: unknown ids {unknown}")
        if "unknown" in raw and len(set(raw)) > 1:
            errors.append(f"{field_name}: unknown cannot coexist with concrete ids")

    phase = annotation.get("motion_phase", "unknown")
    if phase not in canonical_ids("motion_phase", value):
        errors.append(f"motion_phase: unknown id {phase!r}")
    interaction = annotation.get("interaction", {})
    interaction_kind = interaction.get("kind", "unknown") if isinstance(interaction, dict) else None
    if interaction_kind not in canonical_ids("interaction_kind", value):
        errors.append(f"interaction.kind: unknown id {interaction_kind!r}")

    action_domains = set(annotation.get("action_domain", []))
    action_rows = {
        row["id"]: row for row in value["fields"]["action_ids"]["values"]
    }
    for action_id in annotation.get("action_ids", []):
        row = action_rows.get(action_id)
        if row and action_domains and "unknown" not in action_domains:
            if row["domain"] not in action_domains:
                errors.append(
                    f"action_ids/{action_id}: domain {row['domain']!r} absent from action_domain"
                )
    return errors
