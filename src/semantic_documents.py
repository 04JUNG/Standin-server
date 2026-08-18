"""Deterministic semantic-search document sets built from reviewed catalog inputs.

The renderer keeps observed BVH facts separate from source-level contextual
signals.  Source names and canonical action mappings may retrieve candidates,
but they never become pose truth or hard filters.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .semantic_vocab import load_semantic_vocab, vocabulary_fingerprint


SEARCH_DOCUMENT_SCHEMA_VERSION = 1
SEARCH_CONTENT_VERSION = 2
PASSAGE_TEMPLATE_VERSION = 1
SEARCH_DOCUMENT_BUILDER_VERSION = 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _latest_proposals(path: Path) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("record_type") != "semantic_proposal":
            continue
        unit_id = row["semantic_unit_id"]
        previous = current.get(unit_id)
        if previous is None or int(row.get("content_revision", 0)) > int(
            previous.get("content_revision", 0)
        ):
            current[unit_id] = row
    return current


def _inventory(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = _read_jsonl(path)
    if not rows or rows[0].get("record_type") != "inventory_header":
        raise ValueError("semantic inventory header is missing")
    members = {
        row["pose_id"]: row
        for row in rows[1:]
        if row.get("record_type") == "pose_member_inventory"
    }
    if len(members) != len(rows) - 1:
        raise ValueError("semantic inventory has duplicate or invalid member rows")
    return rows[0], members


def _unique_by(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row[key]
        if value in output:
            raise ValueError(f"duplicate {label}: {value}")
        output[value] = row
    return output


_DIRECTION_EN = re.compile(r"\b(?:left|right)\b", re.IGNORECASE)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def direction_neutral_source_text(value: str) -> tuple[str, bool]:
    """Return a shared mirror-group search label without left/right claims."""
    text = _CAMEL_BOUNDARY.sub(" ", value).replace("_", " ")
    neutralized = bool(_DIRECTION_EN.search(text) or "왼쪽" in text or "오른쪽" in text)
    text = _DIRECTION_EN.sub("one side", text)
    text = text.replace("왼쪽", "한쪽").replace("오른쪽", "한쪽")
    return " ".join(text.split()), neutralized


def _vocab_labels(vocab: dict[str, Any]) -> dict[str, dict[str, dict[str, str]]]:
    return {
        field: {
            row["id"]: {"ko": row["ko"], "en": row["en"]}
            for row in definition["values"]
        }
        for field, definition in vocab["fields"].items()
    }


def _concrete(values: list[str]) -> list[str]:
    return [value for value in values if value != "unknown"]


def render_canonical_context(
    canonical: dict[str, Any],
    vocab: dict[str, Any],
) -> dict[str, str] | None:
    """Render only concrete source-context fields in a stable field order."""
    labels = _vocab_labels(vocab)
    fields = (
        ("source_action_ids", "action_ids", "원본 동작 문맥", "source action context"),
        ("action_domain", "action_domain", "행동 영역", "action domains"),
        ("posture_hints", "posture", "자세 힌트", "posture hints"),
        ("style_context", "style_context", "스타일 문맥", "style context"),
        ("intended_props", "intended_props", "소품 문맥", "intended props"),
    )
    ko_parts: list[str] = []
    en_parts: list[str] = []
    for source_field, vocab_field, ko_prefix, en_prefix in fields:
        values = _concrete(list(canonical.get(source_field, [])))
        if not values:
            continue
        ko_values = [labels[vocab_field][value]["ko"] for value in values]
        en_values = [labels[vocab_field][value]["en"] for value in values]
        ko_parts.append(f"{ko_prefix}: {', '.join(ko_values)}")
        en_parts.append(f"{en_prefix}: {', '.join(en_values)}")

    interaction = canonical.get("interaction", {}).get("kind", "unknown")
    if interaction not in {"unknown", "solo"}:
        ko_parts.append(f"상호작용 문맥: {labels['interaction_kind'][interaction]['ko']}")
        en_parts.append(f"interaction context: {labels['interaction_kind'][interaction]['en']}")
    phase = canonical.get("motion_phase", "unknown")
    if phase != "unknown":
        ko_parts.append(f"동작 단계: {labels['motion_phase'][phase]['ko']}")
        en_parts.append(f"motion phase: {labels['motion_phase'][phase]['en']}")

    if not ko_parts:
        return None
    return {"ko": ". ".join(ko_parts) + ".", "en": ". ".join(en_parts) + "."}


def _text_document(
    *,
    semantic_unit_id: str,
    document_type: str,
    language: str,
    text: str,
    evidence_state: str,
    provenance: dict[str, Any],
    retrieval_weight: float,
    candidate_only: bool,
) -> dict[str, Any]:
    document_id = (
        f"{semantic_unit_id}:{document_type}:{language}:passage-v{PASSAGE_TEMPLATE_VERSION}"
    )
    return {
        "document_id": document_id,
        "document_type": document_type,
        "language": language,
        "text": text,
        "text_sha256": _sha256_text(text),
        "evidence_state": evidence_state,
        "retrieval": {
            "dense_candidate": True,
            "lexical_candidate": True,
            "candidate_only": candidate_only,
            "hard_filter_eligible": False,
            "weight": retrieval_weight,
        },
        "provenance": provenance,
    }


def _member_contract(
    proposal: dict[str, Any],
    inventory_members: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str, str]:
    unit_id = proposal["semantic_unit_id"]
    members = []
    for pose_id in proposal["member_pose_ids"]:
        member = inventory_members.get(pose_id)
        if member is None:
            raise ValueError(f"{unit_id}: member absent from inventory: {pose_id}")
        if member["grouping"]["semantic_unit_id"] != unit_id:
            raise ValueError(f"{unit_id}: inventory semantic unit mismatch for {pose_id}")
        variant = member["grouping"]["variant"]
        if not variant.get("paired_member_present"):
            raise ValueError(f"{unit_id}: orphan mirror member {pose_id}")
        members.append(
            {
                "pose_id": pose_id,
                "variant_kind": variant["kind"],
                "mirror_of": variant.get("mirror_of"),
                "bvh_sha256": member["bvh"]["sha256"],
            }
        )
    original = [row for row in members if row["variant_kind"] == "original"]
    mirrored = [row for row in members if row["variant_kind"] == "mirrored"]
    if len(original) != 1 or len(mirrored) != 1 or len(members) != 2:
        raise ValueError(f"{unit_id}: expected one original and one mirrored member")
    mirror_group_ids = {
        inventory_members[row["pose_id"]]["grouping"]["mirror_group_id"] for row in members
    }
    if len(mirror_group_ids) != 1:
        raise ValueError(f"{unit_id}: members disagree on mirror_group_id")
    return members, original[0]["pose_id"], mirrored[0]["pose_id"], mirror_group_ids.pop()


def _mirror_resolution(proposal: dict[str, Any]) -> tuple[str, str]:
    report = proposal.get("mirror_validation") or {}
    if report.get("status") == "pass":
        return "pass", "direct_mirror_validation"
    resolution = report.get("resolution") or {}
    if resolution.get("status") == "canonicalized" and not resolution.get(
        "search_tag_review_required", True
    ):
        return "canonicalized", resolution.get("method", "canonicalized")
    raise ValueError(f"{proposal['semantic_unit_id']}: unresolved mirror validation")


def render_search_unit(
    proposal: dict[str, Any],
    mapping: dict[str, Any],
    inventory_members: dict[str, dict[str, Any]],
    vocab: dict[str, Any],
) -> dict[str, Any]:
    unit_id = proposal["semantic_unit_id"]
    if proposal.get("semantic_unit_type") != "pose_variant_group":
        raise ValueError(f"{unit_id}: unsupported semantic_unit_type")
    if proposal.get("validation", {}).get("errors"):
        raise ValueError(f"{unit_id}: proposal validation errors")
    priority = proposal.get("validation", {}).get("review_priority")
    if priority in {"P0", "PX"}:
        raise ValueError(f"{unit_id}: blocked proposal priority {priority}")
    if mapping["source_clip_id"] != proposal["source_clip_id"]:
        raise ValueError(f"{unit_id}: mapping/source mismatch")
    if unit_id not in mapping.get("search_coverage", {}).get("semantic_unit_ids", []):
        raise ValueError(f"{unit_id}: source mapping lacks unit search coverage")

    members, canonical_pose_id, mirrored_pose_id, mirror_group_id = _member_contract(
        proposal, inventory_members
    )
    mirror_status, mirror_method = _mirror_resolution(proposal)
    semantic = proposal["semantic"]
    atoms = semantic.get("unit_atoms", [])
    if not atoms or any(atom.get("evidence_state") != "observed" for atom in atoms):
        raise ValueError(f"{unit_id}: observed unit atoms missing or invalid")
    caption_ko = semantic.get("caption_ko", "").strip()
    caption_en = semantic.get("caption_en", "").strip()
    if not caption_ko or not caption_en:
        raise ValueError(f"{unit_id}: posecode passages missing")
    if _DIRECTION_EN.search(caption_en) or "왼쪽" in caption_ko or "오른쪽" in caption_ko:
        raise ValueError(f"{unit_id}: shared posecode passage is not direction neutral")

    observed_provenance = {
        "kind": "bvh_rule_group",
        "ref": unit_id,
        "version": proposal["generator"]["posecode_version"],
        "review_status": "auto_verified_observed_tags",
    }
    documents = [
        _text_document(
            semantic_unit_id=unit_id,
            document_type="posecode_render",
            language="ko",
            text=caption_ko,
            evidence_state="observed",
            provenance=observed_provenance,
            retrieval_weight=1.0,
            candidate_only=False,
        ),
        _text_document(
            semantic_unit_id=unit_id,
            document_type="posecode_render",
            language="en",
            text=caption_en,
            evidence_state="observed",
            provenance=observed_provenance,
            retrieval_weight=1.0,
            candidate_only=False,
        ),
    ]

    context_allowed = priority == "P2"
    canonical_text = render_canonical_context(mapping["canonical"], vocab)
    if context_allowed and canonical_text:
        context_weight = {"high": 0.85, "medium": 0.60, "low": 0.35}[
            mapping["mapping"]["confidence"]
        ]
        context_provenance = {
            "kind": "canonical_source_mapping",
            "ref": mapping["mapping_id"],
            "version": mapping["mapping_rules_version"],
            "review_status": "auto_mapped_context",
        }
        for language in ("ko", "en"):
            documents.append(
                _text_document(
                    semantic_unit_id=unit_id,
                    document_type="canonical_context",
                    language=language,
                    text=canonical_text[language],
                    evidence_state="contextual",
                    provenance=context_provenance,
                    retrieval_weight=context_weight,
                    candidate_only=True,
                )
            )

    raw_label = mapping.get("raw_action_label")
    raw_context = None
    raw_direction_neutralized = False
    if context_allowed and raw_label:
        raw_context, raw_direction_neutralized = direction_neutral_source_text(raw_label)
        documents.append(
            _text_document(
                semantic_unit_id=unit_id,
                document_type="source_context",
                language="multilingual",
                text=f"source motion context: {raw_context}",
                evidence_state="contextual",
                provenance={
                    "kind": mapping["raw_action_label_source"],
                    "ref": mapping["source_clip_id"],
                    "version": mapping["mapping_rules_version"],
                    "review_status": "generated_context_only",
                },
                retrieval_weight=0.35,
                candidate_only=True,
            )
        )

    document_ids = [document["document_id"] for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError(f"{unit_id}: duplicate text document IDs")
    payload = {
        "semantic_unit_id": unit_id,
        "proposal_id": proposal["proposal_id"],
        "mapping_id": mapping["mapping_id"],
        "member_hashes": [(row["pose_id"], row["bvh_sha256"]) for row in members],
        "document_hashes": [document["text_sha256"] for document in documents],
        "passage_template_version": PASSAGE_TEMPLATE_VERSION,
    }
    return {
        "record_type": "semantic_search_document_set",
        "schema_version": SEARCH_DOCUMENT_SCHEMA_VERSION,
        "semantic_content_version": SEARCH_CONTENT_VERSION,
        "semantic_vocab_version": mapping["semantic_vocab_version"],
        "passage_template_version": PASSAGE_TEMPLATE_VERSION,
        "document_set_id": _sha256_json(payload),
        "semantic_unit_id": unit_id,
        "semantic_unit_type": proposal["semantic_unit_type"],
        "source_clip_id": proposal["source_clip_id"],
        "source_mapping": {
            "mapping_id": mapping["mapping_id"],
            "status": mapping["mapping"]["status"],
            "confidence": mapping["mapping"]["confidence"],
            "source_context_only": True,
            "canonical": mapping["canonical"],
            "raw_action_label": raw_label,
            "raw_search_text": raw_context,
            "raw_direction_neutralized": raw_direction_neutralized,
        },
        "members": members,
        "canonical_pose_id": canonical_pose_id,
        "mirrored_pose_id": mirrored_pose_id,
        "mirror": {
            "mirror_group_id": mirror_group_id,
            "validation_status": mirror_status,
            "resolution_method": mirror_method,
            "shared_passage_direction_neutral": True,
        },
        "observed_unit_atoms": atoms,
        "text_documents": documents,
        "retrieval_policy": {
            "action_id_absence_excludes_from_search": False,
            "context_is_candidate_only": True,
            "context_hard_filter_eligible": False,
            "observed_atoms_constraint_eligible": True,
            "unknown_is_not_negative": True,
        },
        "searchable": True,
        "generator": {
            "name": "scripts/build_semantic_documents.py",
            "version": SEARCH_DOCUMENT_BUILDER_VERSION,
        },
    }


def build_search_documents(
    *,
    inventory_path: Path,
    proposals_path: Path,
    mappings_path: Path,
    exclusions_path: Path,
    vocab_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory_header, inventory_members = _inventory(inventory_path)
    proposals = _latest_proposals(proposals_path)
    mappings = _unique_by(_read_jsonl(mappings_path), "source_clip_id", "source mapping")
    exclusions = json.loads(exclusions_path.read_text(encoding="utf-8"))
    excluded_sources = set(exclusions["source_clip_ids"])
    vocab = load_semantic_vocab(vocab_path)
    vocab_hash = vocabulary_fingerprint(vocab)

    expected_units = int(inventory_header["counts"]["semantic_units"])
    if len(proposals) != expected_units:
        raise ValueError(
            f"current proposal coverage mismatch: {len(proposals)} != {expected_units}"
        )
    excluded_units = {
        unit_id
        for unit_id, proposal in proposals.items()
        if proposal["source_clip_id"] in excluded_sources
    }
    active = {
        unit_id: proposal
        for unit_id, proposal in proposals.items()
        if proposal["source_clip_id"] not in excluded_sources
    }
    active_sources = {proposal["source_clip_id"] for proposal in active.values()}
    if active_sources != set(mappings):
        missing = sorted(active_sources - set(mappings))
        extra = sorted(set(mappings) - active_sources)
        raise ValueError(f"source mapping coverage mismatch: missing={missing[:5]}, extra={extra[:5]}")

    output = [
        render_search_unit(proposal, mappings[proposal["source_clip_id"]], inventory_members, vocab)
        for _, proposal in sorted(active.items())
    ]
    unit_ids = [row["semantic_unit_id"] for row in output]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("duplicate semantic search document set")
    document_ids = [
        document["document_id"] for row in output for document in row["text_documents"]
    ]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("duplicate semantic text document ID")
    if any(row["source_clip_id"] in excluded_sources for row in output):
        raise ValueError("excluded source leaked into semantic search documents")

    document_type_counts = Counter(
        document["document_type"] for row in output for document in row["text_documents"]
    )
    mapping_status_counts = Counter(row["source_mapping"]["status"] for row in output)
    unknown_units = [row for row in output if row["source_mapping"]["status"] == "unknown"]
    raw_neutralized = sum(row["source_mapping"]["raw_direction_neutralized"] for row in output)
    mirror_status_counts = Counter(row["mirror"]["validation_status"] for row in output)
    summary_payload = {
        "pose_library_version": inventory_header["pose_library_version"],
        "semantic_vocab_fingerprint": vocab_hash,
        "passage_template_version": PASSAGE_TEMPLATE_VERSION,
        "document_sets": [row["document_set_id"] for row in output],
    }
    summary = {
        "artifact_type": "semantic_search_document_build_summary",
        "schema_version": SEARCH_DOCUMENT_SCHEMA_VERSION,
        "semantic_content_version": SEARCH_CONTENT_VERSION,
        "semantic_vocab_version": vocab["semantic_vocab_version"],
        "semantic_vocab_fingerprint": vocab_hash,
        "passage_template_version": PASSAGE_TEMPLATE_VERSION,
        "builder_version": SEARCH_DOCUMENT_BUILDER_VERSION,
        "semantic_build_input_id": _sha256_json(summary_payload),
        "pose_library_version": inventory_header["pose_library_version"],
        "source_mappings": len(mappings),
        "semantic_units": len(output),
        "pose_members": sum(len(row["members"]) for row in output),
        "observed_unit_atoms": sum(len(row["observed_unit_atoms"]) for row in output),
        "text_documents": len(document_ids),
        "document_type_counts": dict(sorted(document_type_counts.items())),
        "mapping_status_unit_counts": dict(sorted(mapping_status_counts.items())),
        "unknown_action_units_searchable": sum(row["searchable"] for row in unknown_units),
        "unknown_action_units": len(unknown_units),
        "raw_context_direction_neutralized_units": raw_neutralized,
        "mirror_status_counts": dict(sorted(mirror_status_counts.items())),
        "excluded_source_clips": len(excluded_sources),
        "excluded_semantic_units": len(excluded_units),
        "excluded_units_in_output": 0,
        "searchable_units": sum(row["searchable"] for row in output),
        "unsearchable_units": sum(not row["searchable"] for row in output),
        "production_ready": False,
        "embedding_status": "not_built_pending_model_pin",
    }
    return output, summary
