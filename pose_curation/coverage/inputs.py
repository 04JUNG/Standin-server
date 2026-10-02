"""Inventory and anonymous contact sheets, separate from model inference."""
from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

from PIL import Image, ImageDraw, ImageOps

from ..storage import sha256, write_json


def image_record(path: Path, identity: str, origin: str) -> dict:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        size = list(image.size)
    return {"id": identity, "path": path.resolve().as_posix(), "size": size,
            "sha256": sha256(path), "origin": origin}


def import_zip(archive: Path, destination: Path) -> list[dict]:
    """Extract images only; flattened names cannot escape the destination."""
    destination.mkdir(parents=True, exist_ok=True)
    records, names = [], set()
    with zipfile.ZipFile(archive) as source:
        if sum(item.file_size for item in source.infolist()) > 512 * 1024**2:
            raise ValueError("archive exceeds 512 MiB expanded limit")
        for item in source.infolist():
            name = Path(item.filename).name
            if item.is_dir() or item.filename.startswith("__MACOSX/"):
                continue
            if Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            if name in names or item.file_size > 32 * 1024**2:
                raise ValueError(f"duplicate or oversized image: {name}")
            names.add(name)
            path = destination / name
            path.write_bytes(source.read(item))
            records.append(image_record(path, path.stem, "supplied"))
    return sorted(records, key=lambda row: row["id"])


def import_private(directory: Path, destination: Path) -> list[dict]:
    """Deduplicate bytes and hide installation/job identifiers in review UI."""
    destination.mkdir(parents=True, exist_ok=True)
    by_hash = {}
    for path in sorted(directory.rglob("input.*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        digest = sha256(path)
        if digest in by_hash:
            by_hash[digest]["occurrences"] += 1
            continue
        identity = f"user_{digest[:12]}"
        target = destination / (identity + path.suffix.lower())
        shutil.copyfile(path, target)
        by_hash[digest] = {**image_record(target, identity, "user_upload"), "occurrences": 1}
    return sorted(by_hash.values(), key=lambda row: row["id"])


def contact_sheets(records: list[dict], destination: Path, prefix: str = "inputs") -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for page in range((len(records) + 39) // 40):
        canvas = Image.new("RGB", (1400, 1150), "#dddddd")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(records[page * 40:(page + 1) * 40]):
            with Image.open(row["path"]) as source:
                rgba = source.convert("RGBA")
                white = Image.new("RGBA", rgba.size, "white")
                white.alpha_composite(rgba)
                thumb = ImageOps.contain(white.convert("RGB"), (165, 200))
            x, y = slot % 8 * 175, slot // 8 * 230
            canvas.paste(thumb, (x + (175 - thumb.width) // 2, y))
            draw.text((x + 4, y + 205), row["id"], fill="black")
        canvas.save(destination / f"{prefix}-{page:02d}.jpg", quality=88)
