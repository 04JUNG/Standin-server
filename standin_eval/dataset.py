from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .schemas import (
    SCHEMA_VERSION,
    ValidationIssue,
    bbox,
    is_target_person,
    validate_cut_shape,
    validate_person_shape,
)
from .util import (
    REPO_ROOT,
    hash_jsonl,
    image_dimensions,
    read_json,
    read_jsonl,
    relative_to_repo,
    resolve_path,
    sha256_file,
    slug,
    utc_now,
    write_json,
    write_jsonl,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class EvalDataset:
    root: Path
    manifest: dict
    cuts: list[dict]
    persons: list[dict]

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    @property
    def cuts_by_id(self) -> dict[str, dict]:
        return {row["cut_id"]: row for row in self.cuts}

    @property
    def persons_by_cut(self) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in self.persons:
            grouped[row["cut_id"]].append(row)
        return dict(grouped)

    @property
    def target_persons(self) -> list[dict]:
        cuts = self.cuts_by_id
        return [
            row for row in self.persons
            if row.get("cut_id") in cuts and is_target_person(row, cuts[row["cut_id"]])
        ]

    @property
    def actual_cut_hash(self) -> str:
        return hash_jsonl(self.cuts)

    @property
    def actual_person_hash(self) -> str:
        return hash_jsonl(self.persons)


def dataset_root(identifier: str | Path, eval_root: str | Path = "evaluation") -> Path:
    direct = Path(identifier).expanduser()
    if direct.exists() or direct.is_absolute() or "/" in str(identifier):
        return resolve_path(direct)
    return resolve_path(Path(eval_root) / "datasets" / str(identifier))


def load_dataset(identifier: str | Path, eval_root: str | Path = "evaluation") -> EvalDataset:
    root = dataset_root(identifier, eval_root)
    manifest_path = root / "dataset.json"
    cuts_path = root / "cuts.jsonl"
    persons_path = root / "persons.jsonl"
    for path in (manifest_path, cuts_path, persons_path):
        if not path.exists():
            raise FileNotFoundError(f"dataset file missing: {path}")
    return EvalDataset(
        root=root,
        manifest=read_json(manifest_path),
        cuts=read_jsonl(cuts_path),
        persons=read_jsonl(persons_path),
    )


def _manifest(dataset_id: str, purpose: str, cuts: list[dict], persons: list[dict]) -> dict:
    scene_groups = {
        row["scene_group_id"] for row in cuts
        if row.get("scene_group_id") not in (None, "unknown")
        and not str(row.get("scene_group_id")).startswith("unresolved:")
    }
    cuts_by_id = {row["cut_id"]: row for row in cuts}
    targets = sum(
        1 for row in persons
        if row.get("cut_id") in cuts_by_id and is_target_person(row, cuts_by_id[row["cut_id"]])
    )
    aliases = sum(max(0, len(row.get("aliases", [])) - 1) for row in cuts)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "purpose": purpose,
        "created_at": utc_now(),
        "sealed_at": utc_now(),
        "cut_manifest_sha256": hash_jsonl(cuts),
        "person_gt_sha256": hash_jsonl(persons),
        "rubric_version": 1,
        "counts": {
            "files": len(cuts) + aliases,
            "unique_image_contents": len(cuts),
            "scene_groups": len(scene_groups) if scene_groups else None,
            "gt_persons": len(persons),
            "target_persons": targets,
        },
        "split_unit": ["artist_id", "project_id", "scene_group_id"],
    }


def init_dataset(
    name: str,
    roots: list[str | Path],
    eval_root: str | Path = "evaluation",
    purpose: str = "engineering",
) -> Path:
    output = resolve_path(Path(eval_root) / "datasets" / name)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"dataset already exists and is not empty: {output}")

    paths: list[Path] = []
    seen_paths: set[Path] = set()
    for root_value in roots:
        root = resolve_path(root_value)
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
            if root not in seen_paths:
                paths.append(root)
                seen_paths.add(root)
        elif root.is_dir():
            for item in sorted(root.rglob("*")):
                if (
                    item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
                    and item not in seen_paths
                ):
                    paths.append(item)
                    seen_paths.add(item)

    by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_hash[sha256_file(path)].append(path)

    cuts: list[dict] = []
    for image_hash, aliases in by_hash.items():
        # Root order is an explicit canonical-source preference.  For example,
        # pass the manually verified directory before a broad full-body set.
        canonical = aliases[0]
        width, height = image_dimensions(canonical)
        cut_id = f"{slug(canonical.stem)}-{image_hash[:8]}"
        cuts.append({
            "schema_version": SCHEMA_VERSION,
            "cut_id": cut_id,
            "image_path": relative_to_repo(canonical),
            "image_sha256": image_hash,
            "aliases": [relative_to_repo(path) for path in aliases],
            "image_width": width,
            "image_height": height,
            "scene_group_id": f"unresolved:{cut_id}",
            "artist_id": "unknown",
            "project_id": "unknown",
            "split": "engineering" if purpose == "engineering" else "calibration",
            "expected_route": "core",
            "num_people_gt": 0,
            "license_scope": "internal-eval",
        })

    persons: list[dict] = []
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "cuts.jsonl", cuts)
    write_jsonl(output / "persons.jsonl", persons)
    write_json(output / "dataset.json", _manifest(name, purpose, cuts, persons))
    return output


def seal_dataset(dataset: EvalDataset) -> dict:
    issues = validate_dataset(dataset, check_seal=False)
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        raise ValueError("dataset validation failed: " + "; ".join(item.message for item in errors))
    created = dataset.manifest.get("created_at") or utc_now()
    updated = _manifest(
        dataset.dataset_id,
        str(dataset.manifest.get("purpose", "engineering")),
        dataset.cuts,
        dataset.persons,
    )
    updated["created_at"] = created
    updated["rubric_version"] = int(dataset.manifest.get("rubric_version", 1))
    write_json(dataset.root / "dataset.json", updated)
    dataset.manifest = updated
    return updated


def validate_dataset(dataset: EvalDataset, check_seal: bool = True) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if dataset.manifest.get("schema_version") != SCHEMA_VERSION:
        issues.append(ValidationIssue(
            "error", "schema_version", f"unsupported dataset schema: {dataset.manifest.get('schema_version')}"
        ))
    if not isinstance(dataset.manifest.get("dataset_id"), str):
        issues.append(ValidationIssue("error", "dataset_id", "dataset_id is required"))

    if check_seal:
        expected = dataset.manifest.get("cut_manifest_sha256")
        if expected != dataset.actual_cut_hash:
            issues.append(ValidationIssue(
                "error", "cut_hash_mismatch",
                "cuts.jsonl changed after seal; run `dataset seal` after review",
            ))
        expected = dataset.manifest.get("person_gt_sha256")
        if expected != dataset.actual_person_hash:
            issues.append(ValidationIssue(
                "error", "person_hash_mismatch",
                "persons.jsonl changed after seal; run `dataset seal` after review",
            ))

    cut_ids: set[str] = set()
    image_hashes: dict[str, str] = {}
    for row in dataset.cuts:
        record_id = str(row.get("cut_id", "<missing>"))
        for message in validate_cut_shape(row):
            issues.append(ValidationIssue("error", "cut_shape", message, record_id))
        if record_id in cut_ids:
            issues.append(ValidationIssue("error", "duplicate_cut_id", record_id, record_id))
        cut_ids.add(record_id)
        image_path = resolve_path(str(row.get("image_path", "")), REPO_ROOT)
        if not image_path.exists():
            issues.append(ValidationIssue("error", "image_missing", str(image_path), record_id))
        elif image_path.is_file():
            actual_hash = sha256_file(image_path)
            if actual_hash != row.get("image_sha256"):
                issues.append(ValidationIssue(
                    "error", "image_hash_mismatch", f"expected {row.get('image_sha256')}, got {actual_hash}", record_id
                ))
        image_hash = str(row.get("image_sha256", ""))
        if image_hash in image_hashes:
            issues.append(ValidationIssue(
                "error", "duplicate_image_content",
                f"same image hash as {image_hashes[image_hash]}", record_id,
            ))
        image_hashes[image_hash] = record_id
        for field in ("artist_id", "project_id", "scene_group_id"):
            if str(row.get(field, "")).startswith(("unknown", "unresolved:")):
                issues.append(ValidationIssue(
                    "warning", f"unresolved_{field}", f"{field} is not resolved", record_id
                ))

    person_ids: set[str] = set()
    people_per_cut: Counter[str] = Counter()
    cuts = dataset.cuts_by_id
    for row in dataset.persons:
        record_id = str(row.get("person_id", "<missing>"))
        for message in validate_person_shape(row):
            issues.append(ValidationIssue("error", "person_shape", message, record_id))
        if record_id in person_ids:
            issues.append(ValidationIssue("error", "duplicate_person_id", record_id, record_id))
        person_ids.add(record_id)
        cut_id = str(row.get("cut_id", ""))
        people_per_cut[cut_id] += 1
        cut = cuts.get(cut_id)
        if cut is None:
            issues.append(ValidationIssue("error", "unknown_cut", cut_id, record_id))
            continue
        try:
            x1, y1, x2, y2 = bbox(row.get("bbox_xyxy"))
        except ValueError:
            continue
        width, height = cut.get("image_width"), cut.get("image_height")
        if isinstance(width, int) and isinstance(height, int):
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                issues.append(ValidationIssue(
                    "error", "bbox_out_of_bounds",
                    f"bbox {row.get('bbox_xyxy')} outside {width}x{height}", record_id,
                ))

    for cut in dataset.cuts:
        declared = cut.get("num_people_gt", 0)
        actual = people_per_cut[cut["cut_id"]]
        if declared != actual:
            level = "warning" if declared == 0 and actual > 0 else "error"
            issues.append(ValidationIssue(
                level, "person_count_mismatch",
                f"num_people_gt={declared}, person rows={actual}", cut["cut_id"],
            ))

    declared_targets = dataset.manifest.get("counts", {}).get("target_persons")
    if check_seal and declared_targets != len(dataset.target_persons):
        issues.append(ValidationIssue(
            "error", "target_count_mismatch",
            f"manifest target_persons={declared_targets}, actual={len(dataset.target_persons)}",
        ))
    return issues


def dataset_stats(dataset: EvalDataset) -> dict:
    cuts_by_id = dataset.cuts_by_id

    def resolved_count(field: str) -> int:
        values = {
            str(row.get(field)) for row in dataset.cuts
            if row.get(field) not in (None, "", "unknown")
            and not str(row.get(field)).startswith("unresolved:")
        }
        return len(values)

    return {
        "dataset_id": dataset.dataset_id,
        "root": str(dataset.root),
        "cuts": len(dataset.cuts),
        "persons": len(dataset.persons),
        "target_persons": len(dataset.target_persons),
        "artists": resolved_count("artist_id"),
        "projects": resolved_count("project_id"),
        "scene_groups": resolved_count("scene_group_id"),
        "splits": dict(Counter(row.get("split") for row in dataset.cuts)),
        "difficulty": dict(Counter(
            row.get("difficulty", "unknown") for row in dataset.persons
            if row.get("cut_id") in cuts_by_id
        )),
    }
