"""Bounded, non-executable VLM semantic evidence sidecars.

All evidence is diagnostic_only. An independent geometry policy is responsible
for any later constraint admission. This module cannot create joint targets,
expand observation masks, select cameras, or schedule provider calls.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os

VERSION = "refine-evidence-v1"
SCHEMA_VERSION = "refine-evidence-schema-v2"
MAX_PEOPLE = 20
MAX_BYTES = 96_000
LIMITS = {"visibility": 16, "relations": 8, "contacts": 6, "preserve": 8}
BASE_REGIONS = {"head", "neck", "chest", "torso", "pelvis", "ground"}
SIDED = {"shoulder", "upper_arm", "elbow", "forearm", "wrist", "hand", "palm",
         "hand_back", "hip", "thigh", "knee", "shin", "ankle", "foot", "sole"}
REGIONS = BASE_REGIONS | {side + "_" + part for side in ("left", "right") for part in SIDED}
CONFIDENCE = {"high", "low", "unknown"}
EVIDENCE = {"direct_visible", "inferred", "unknown"}
SIDES = {"certain", "ambiguous", "unknown"}
SCOPES = {"full", "half", "bust", "shoulder_head", "head", "unknown"}
CONTACT_EFFECTORS = {"left_palm", "right_palm", "left_hand_back", "right_hand_back",
                     "left_sole", "right_sole", "left_knee", "right_knee", "pelvis"}


def enabled():
    return os.getenv("REFINE_V3_EVIDENCE_ENABLED", "0").strip().lower() in {"1", "true", "yes"}


def _freeze(value):
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                         for item in value):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    return value


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PersonEvidence:
    person_id: int
    body_scope: str
    visibility: tuple = ()
    relations: tuple = ()
    contacts: tuple = ()
    preserve: tuple = ()
    disposition: str = "diagnostic_only"

    def to_serializable(self):
        return {"person_id": self.person_id, "body_scope": self.body_scope,
                "disposition": self.disposition,
                **{key: [_thaw(item) for item in getattr(self, key)]
                   for key in LIMITS}}


@dataclass(frozen=True)
class EvidenceBundle:
    status: str
    people: tuple = ()
    reasons: tuple = ()
    provenance: tuple = ()
    version: str = VERSION

    def to_serializable(self):
        result = {"version": self.version, "status": self.status,
                  "people": [person.to_serializable() for person in self.people],
                  "reasons": list(self.reasons), "provenance": _thaw(self.provenance) or {}}
        result["evidence_sha"] = _sha(result)
        return result


def _valid_region(value, scope):
    if value not in REGIONS:
        return False
    if scope == "head":
        return value in {"head", "neck"}
    if scope == "shoulder_head":
        return value in {"head", "neck", "left_shoulder", "right_shoulder"}
    if scope == "bust":
        return value in {"head", "neck", "chest", "torso", "left_shoulder", "right_shoulder"}
    return True


def _entry(item, group, scope, ids):
    if not isinstance(item, dict):
        return None
    common = {"evidence": EVIDENCE, "confidence": CONFIDENCE}
    if group == "visibility":
        fields = {"region", "state", *common}
        enums = {**common, "state": {"visible", "occluded", "out_of_frame", "ambiguous"}}
        regions = ("region",)
    elif group == "relations":
        fields = {"id", "subject_region", "object_region", "kind", "frame", "side_certainty", *common}
        enums = {**common, "kind": {"above", "below", "in_front_of", "behind", "near", "bent", "extended"},
                 "frame": {"image", "camera_depth", "anatomical"}, "side_certainty": SIDES}
        regions = ("subject_region", "object_region")
    else:
        fields = {"effector_region", "target", "target_ref", "kind", "side_certainty", "chain_visible", *common}
        enums = {**common, "target": {"self_region", "ground", "external_object", "other_person", "unknown"},
                 "kind": {"touching", "grasping", "supporting", "unknown"}, "side_certainty": SIDES}
        regions = ("effector_region",)
    if set(item) != fields or any((not isinstance(item.get(key), str) or item.get(key) not in values) for key, values in enums.items()):
        return None
    if any(not isinstance(item.get(key), str) or not _valid_region(item[key], scope) for key in regions):
        return None
    if group == "relations":
        rid = item.get("id")
        if not isinstance(rid, str) or not rid.startswith("r") or not rid[1:].isdigit() or len(rid) > 4:
            return None
    if group == "contacts":
        if type(item["chain_visible"]) is not bool:
            return None
        if item["effector_region"] not in CONTACT_EFFECTORS:
            return None
        target, ref = item["target"], item["target_ref"]
        if target == "self_region" and (not isinstance(ref, str) or not _valid_region(ref, scope)):
            return None
        if target == "other_person" and (type(ref) is not int or ref not in ids):
            return None
        if target in {"ground", "external_object", "unknown"} and ref is not None:
            return None
    # A side ambiguity cannot accidentally become a usable side constraint.
    if item.get("side_certainty") != "certain" and group != "visibility":
        item = {**item, "evidence": "unknown", "confidence": "unknown"}
    return _freeze(item)


def parse_refine_evidence(raw, *, expected_person_ids, kind="observations", body_scopes=None, provenance=None):
    """Validate an entire ID set before accepting any person's optional bundle.

    Response IDs are request-local (explicit person_id), never array indexes.
    Excess/malformed groups are discarded, recorded as partial, and never silently
    truncated. Malformed optional data cannot invalidate the caller's core data.
    """
    provenance = _freeze(provenance or {})
    absent = EvidenceBundle("absent", provenance=provenance)
    if raw is None:
        return absent
    unavailable = lambda reason: EvidenceBundle("unavailable", reasons=(reason,), provenance=provenance)
    ids = tuple(expected_person_ids)
    if (kind not in {"observations", "contacts"} or len(ids) > MAX_PEOPLE
            or any(type(pid) is not int or pid < 0 for pid in ids) or len(set(ids)) != len(ids)):
        return unavailable("invalid_request_ids")
    try:
        if len(json.dumps(raw, allow_nan=False).encode()) > MAX_BYTES:
            return unavailable("evidence_size_limit")
    except (TypeError, ValueError, OverflowError, RecursionError):
        return unavailable("invalid_json_value")
    provenance = _freeze({**dict(provenance), "source_sha": _sha(raw), "schema_version": SCHEMA_VERSION})
    if not isinstance(raw, dict) or set(raw) != {"people"} or not isinstance(raw["people"], list):
        return unavailable("invalid_evidence_envelope")
    people = raw["people"]
    received = [p.get("person_id") if isinstance(p, dict) else None for p in people]
    if (len(received) != len(ids) or any(type(pid) is not int for pid in received)
            or len(set(received)) != len(received) or set(received) != set(ids)):
        return unavailable("person_id_set_mismatch")
    by_id = {p["person_id"]: p for p in people}
    reasons, parsed = [], []
    allowed_groups = ("visibility", "relations", "preserve") if kind == "observations" else ("contacts",)
    for pid in ids:
        person = by_id[pid]
        scope = person.get("body_scope", "unknown")
        expected_scope = (body_scopes or {}).get(pid)
        if hasattr(expected_scope, "value"):
            expected_scope = expected_scope.value
        if not isinstance(scope, str) or scope not in SCOPES or (expected_scope is not None and scope != expected_scope):
            reasons.append("scope_mismatch")
            continue
        if set(person) - {"person_id", "body_scope", *allowed_groups}:
            reasons.append("unexpected_person_fields")
            continue
        groups = {}
        for group in allowed_groups:
            items = person.get(group, [])
            if not isinstance(items, list) or len(items) > LIMITS[group]:
                reasons.append(group + "_limit_or_type")
                continue
            if group == "preserve":
                continue
            values = [_entry(item, group, scope, set(ids)) for item in items]
            if any(item is None for item in values):
                reasons.append(group + "_invalid_entry")
                continue
            if group == "relations":
                relation_ids = [dict(item)["id"] for item in values]
                if len(set(relation_ids)) != len(relation_ids):
                    reasons.append("duplicate_relation_ids")
                    continue
            groups[group] = tuple(values)
        relation_ids = {dict(item)["id"] for item in groups.get("relations", ())}
        if kind == "observations":
            preserve = person.get("preserve", [])
            if (isinstance(preserve, list) and len(preserve) <= LIMITS["preserve"]
                    and all(isinstance(rid, str) and rid in relation_ids for rid in preserve)
                    and len(set(preserve)) == len(preserve)):
                groups["preserve"] = tuple(preserve)
            elif preserve:
                reasons.append("preserve_invalid_reference")
        parsed.append(PersonEvidence(pid, scope, **groups))
    return EvidenceBundle("partial" if reasons else "valid", tuple(parsed), tuple(sorted(set(reasons))), provenance)


def bind_evidence_to_slots(bundle, slot_map, *, provenance):
    """Reconcile explicit request IDs to stable slots, with image/transform binding.

    Both analysis IDs and hand subset IDs require their own bijective slot_map.
    Detector-only slots have no entry and can never inherit an adjacent result.
    """
    if isinstance(bundle, dict):
        bundle = evidence_bundle_from_serializable(bundle)
    if bundle.status in {"absent", "unavailable"}:
        return bundle
    required = {"image_digest", "crop_transform", "slot_map_version"}
    ids = {person.person_id for person in bundle.people}
    if (not required.issubset(provenance) or any(not provenance[key] for key in required)
            or set(slot_map) != ids or any(type(value) is not int or value < 0 for value in slot_map.values())
            or len(set(slot_map.values())) != len(slot_map)):
        return EvidenceBundle("unavailable", reasons=("invalid_slot_binding",))
    people = []
    for person in bundle.people:
        contacts = []
        for entry in person.contacts:
            item = _thaw(entry)
            if item["target"] == "other_person":
                item["target_ref"] = slot_map[item["target_ref"]]
            contacts.append(_freeze(item))
        people.append(replace(person, person_id=slot_map[person.person_id], contacts=tuple(contacts)))
    merged = dict(bundle.provenance)
    merged.update(provenance)
    merged["slot_map"] = [[key, value] for key, value in sorted(slot_map.items())]
    return replace(bundle, people=tuple(people), provenance=_freeze(merged))


def response_trace(response, *, provider, model, logical_call_id, attempts=1):
    """Only actual metadata; unavailable provider token/image usage stays unknown."""
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    return {"provider": provider, "model": model, "prompt_version": VERSION,
            "schema_version": SCHEMA_VERSION, "logical_call_id": logical_call_id,
            "provider_response_id": getattr(response, "response_id", None),
            "logical_calls": 1, "provider_attempts": attempts, "additional_logical_calls": 0,
            "solver_logical_calls": 0,
            "image_inputs_per_attempt": 1,
            "image_attempts": attempts if type(attempts) is int else None,
            "token_usage_scope": "successful_response_only",
            "prompt_tokens": getattr(usage, "prompt_token_count", getattr(usage, "prompt_tokens", None)),
            "response_tokens": getattr(usage, "candidates_token_count", getattr(usage, "completion_tokens", None)),
            "total_tokens": getattr(usage, "total_token_count", getattr(usage, "total_tokens", None))}


def evidence_schema(kind="observations"):
    """Optional bounded JSON schema extension; core required fields are unchanged."""
    def enum(values):
        return {"type": "string", "enum": sorted(values)}
    def obj(properties, required=None):
        return {"type": "object", "properties": properties,
                "required": list(properties) if required is None else required, "additionalProperties": False}
    def arr(item, maximum):
        return {"type": "array", "items": item, "maxItems": maximum}
    common = {"evidence": enum(EVIDENCE), "confidence": enum(CONFIDENCE)}
    person = {"person_id": {"type": "integer", "minimum": 0, "maximum": 19}, "body_scope": enum(SCOPES)}
    if kind == "observations":
        person.update(visibility=arr(obj({"region": enum(REGIONS), "state": enum({"visible", "occluded", "out_of_frame", "ambiguous"}), **common}), 16),
                      relations=arr(obj({"id": {"type": "string", "maxLength": 4}, "subject_region": enum(REGIONS), "object_region": enum(REGIONS),
                                         "kind": enum({"above", "below", "in_front_of", "behind", "near", "bent", "extended"}),
                                         "frame": enum({"image", "camera_depth", "anatomical"}), "side_certainty": enum(SIDES), **common}), 8),
                      preserve=arr({"type": "string", "maxLength": 4}, 8))
    else:
        person["contacts"] = arr(obj({"effector_region": enum(CONTACT_EFFECTORS), "target": enum({"self_region", "ground", "external_object", "other_person", "unknown"}),
                                       "target_ref": {"type": ["string", "integer", "null"], "maxLength": 32},
                                       "kind": enum({"touching", "grasping", "supporting", "unknown"}), "side_certainty": enum(SIDES),
                                       "chain_visible": {"type": "boolean"}, **common}), 6)
    return obj({"people": arr(obj(person, ["person_id", "body_scope"]), MAX_PEOPLE)})


def extend_analysis_schema(schema):
    from copy import deepcopy
    schema = deepcopy(schema)
    schema["properties"]["refine_observations"] = provider_evidence_schema(evidence_schema())
    return schema


def provider_evidence_schema(value):
    """Keep Gemini's decoding grammar small; admission stays in the strict parser.

    The combined bounded evidence grammar rejected actual Gemini requests with
    400 INVALID_ARGUMENT. Only the optional evidence transport grammar omits
    decoding bounds. MAX_BYTES/MAX_PEOPLE/LIMITS, exact IDs/scopes,
    enum/field validation and diagnostic-only authority are unchanged.
    """
    if isinstance(value, dict):
        return {key: provider_evidence_schema(item) for key, item in value.items()
                if key not in {"minItems", "maxItems", "minimum", "maximum", "minLength", "maxLength"}}
    if isinstance(value, list):
        return [provider_evidence_schema(item) for item in value]
    return value


def hand_response_schema():
    hand = {"type": "object", "properties": {
        "visible": {"type": "boolean"}, "shape": {"type": ["string", "null"]},
        "alternative": {"type": ["string", "null"]}, "confidence": {"type": "string", "enum": ["high", "low"]}},
        "required": ["visible", "shape", "alternative", "confidence"], "additionalProperties": False}
    return {"type": "object", "properties": {
        "people": {"type": "array", "maxItems": MAX_PEOPLE, "items": {
            "type": "object", "properties": {"left": hand, "right": hand},
            "required": ["left", "right"], "additionalProperties": False}},
        "refine_contacts": provider_evidence_schema(evidence_schema("contacts"))},
        "required": ["people"], "additionalProperties": False}


def evidence_bundle_from_serializable(payload):
    """Revalidate a stored DTO before applying a request-subset reconciliation map.

    The checksum detects mutation/staleness; it is not an ownership certificate.
    Geometry admission is still exclusively the independent policy's job.
    """
    invalid = EvidenceBundle("unavailable", reasons=("invalid_stored_evidence",))
    if not isinstance(payload, dict):
        return invalid
    keys = ("version", "status", "people", "reasons", "provenance")
    if any(key not in payload for key in keys) or payload["version"] != VERSION:
        return invalid
    try:
        if _sha({key: payload[key] for key in keys}) != payload.get("evidence_sha"):
            return invalid
    except (TypeError, ValueError, OverflowError, RecursionError):
        return invalid
    status = payload["status"]
    if not isinstance(status, str) or status not in {"absent", "valid", "partial", "unavailable"}:
        return invalid
    people = payload["people"]
    if not isinstance(people, list) or len(people) > MAX_PEOPLE:
        return invalid
    ids = [person.get("person_id") if isinstance(person, dict) else None for person in people]
    observations, contacts = [], []
    for person in people:
        if not isinstance(person, dict) or person.get("disposition") != "diagnostic_only":
            return invalid
        observations.append({key: person[key] for key in ("person_id", "body_scope", "visibility", "relations", "preserve") if key in person})
        contacts.append({key: person[key] for key in ("person_id", "body_scope", "contacts") if key in person})
    observed = parse_refine_evidence({"people": observations}, expected_person_ids=ids)
    contacted = parse_refine_evidence({"people": contacts}, expected_person_ids=ids, kind="contacts")
    if observed.status != "valid" or contacted.status != "valid":
        return invalid
    # Original reasons are static parser enums, never copy arbitrary stored text.
    known_reasons = {"person_id_set_mismatch", "scope_mismatch", "unexpected_person_fields",
                     "duplicate_relation_ids", "preserve_invalid_reference"} | {
                         group + suffix for group in LIMITS for suffix in ("_invalid_entry", "_limit_or_type")}
    reasons = payload["reasons"]
    if (not isinstance(reasons, list) or any(not isinstance(reason, str) or reason not in known_reasons for reason in reasons)
            or not isinstance(payload["provenance"], dict)):
        return invalid
    parsed = tuple(replace(person, contacts=other.contacts) for person, other in zip(observed.people, contacted.people))
    provenance = dict(payload["provenance"])
    if isinstance(payload.get("call_trace"), dict):
        provenance["call_trace"] = payload["call_trace"]
    return EvidenceBundle(status, parsed, tuple(reasons), _freeze(provenance))


def _image_binding(image_digest, image_size):
    width, height = image_size
    return {"image_digest": image_digest,
            "crop_transform": {"version": "full-image-pixels-v1", "width": int(width), "height": int(height)},
            "slot_map_version": "vlm-reconciled-slots-v1"}


def bind_hand_contact_evidence(payload, person_indices, *, image_digest, image_size):
    """Join read_hands request-local IDs to the original VLM person subset.

    Call only when joining an already executed hand future. It never schedules a
    hand request for skipped/head-only people and never changes core hand presets.
    """
    if not enabled() or not isinstance(payload, dict) or not isinstance(payload.get("refine_evidence"), dict):
        return None
    indices = tuple(person_indices)
    bound = bind_evidence_to_slots(payload["refine_evidence"], dict(enumerate(indices)),
                                   provenance=_image_binding(image_digest, image_size))
    return bound.to_serializable()


def bind_analysis_evidence(vlm, slots, image_digest, image_size):
    """Return stable-slot sidecars after reconciliation and the optional hand join.

    Pipeline can assign result.get(slot.slot_id) to PersonDescriptor.refine_evidence.
    No detector-only slot inherits model metadata, even if sorting changed order.
    """
    if not enabled() or not isinstance(getattr(vlm, "refine_evidence", None), dict):
        return {}
    slot_map = {slot.slot_id: slot.slot_id for slot in slots if slot.slot_origin == "vlm"}
    if set(slot_map) != set(range(vlm.num_people)):
        return {}
    binding = _image_binding(image_digest, image_size)
    observed = bind_evidence_to_slots(vlm.refine_evidence, slot_map, provenance=binding)
    contacts = evidence_bundle_from_serializable(getattr(vlm, "_refine_contact_evidence", None))
    contacts_binding = _thaw(contacts.provenance) or {}
    if any(contacts_binding.get(key) != value for key, value in binding.items()):
        contacts = EvidenceBundle("unavailable", reasons=("contact_image_mismatch",))
    contacts_by_id = {person.person_id: person for person in contacts.people}
    output = {}
    for slot_id in slot_map.values():
        person = next((person for person in observed.people if person.person_id == slot_id), None)
        contact_person = contacts_by_id.get(slot_id)
        expected_scope = vlm.body_scopes[slot_id] if slot_id < len(vlm.body_scopes) else None
        expected_scope = getattr(expected_scope, "value", expected_scope)
        if contact_person and expected_scope is not None and contact_person.body_scope != expected_scope:
            contact_person = None
        if person is None and contact_person is None:
            continue
        observation = replace(observed, people=(person,)).to_serializable() if person is not None else None
        contact = replace(contacts, people=(contact_person,)).to_serializable() if contact_person is not None else None
        statuses = [part["status"] for part in (observation, contact) if part is not None]
        sidecar = {"version": VERSION, "status": "partial" if "partial" in statuses else "valid",
                   "person_id": slot_id, "observations": observation, "contacts": contact,
                   "disposition": "diagnostic_only", "additional_logical_calls": 0}
        sidecar["evidence_sha"] = _sha(sidecar)
        output[slot_id] = sidecar
    return output
