"""Install the pinned optional search models before constructing the inference pipeline."""
from __future__ import annotations

from hashlib import file_digest
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from tempfile import TemporaryDirectory

from .library_source import _download_s3


REQUIRED_FILES = frozenset({
    'body.onnx', 'depth.onnx', 'pose-model/manifest.json', 'pose-model/model.onnx',
    'pose-model/detector.onnx', 'multilingual-e5-small/model_quantized.onnx',
    'multilingual-e5-small/tokenizer.json', 'grounding-dino/config.json',
    'grounding-dino/model.safetensors', 'grounding-dino/tokenizer.json',
    'grounding-dino/preprocessor_config.json',
})
OPTIONAL_FILES = frozenset({
    'grounding-dino/README.md', 'grounding-dino/added_tokens.json',
    'grounding-dino/special_tokens_map.json', 'grounding-dino/tokenizer_config.json',
    'grounding-dino/vocab.txt',
})
MAX_ARCHIVE_BYTES = 2_000_000_000


def extract_archive(archive: Path, destination: Path) -> None:
    seen = set()
    total = 0
    with tarfile.open(archive, 'r:') as bundle:
        for item in bundle:
            name = item.name
            if name in seen or name not in REQUIRED_FILES | OPTIONAL_FILES or not item.isfile():
                raise ValueError('unexpected recall model archive entry')
            total += item.size
            if not 0 < item.size <= MAX_ARCHIVE_BYTES or total > MAX_ARCHIVE_BYTES:
                raise ValueError('invalid recall model archive size')
            seen.add(name)
            path = destination.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(item) as source, path.open('xb') as target:
                shutil.copyfileobj(source, target)
    if not REQUIRED_FILES <= seen:
        raise ValueError('incomplete recall model archive')


def ensure_recall_assets() -> Path | None:
    """Support an existing local directory, or provision it from the pinned S3 archive."""
    directory = os.getenv('ROUTING_RECALL_ASSETS_DIR')
    uri = os.getenv('ROUTING_RECALL_ASSETS_URI')
    digest = os.getenv('ROUTING_RECALL_ASSETS_SHA256')
    if not uri and not digest:
        return Path(directory) if directory else None
    if (not directory or not uri or not digest
            or not re.fullmatch(r'[0-9a-f]{64}', digest)
            or not re.fullmatch(r's3://[a-z0-9.-]+/pose-models/recall-90-v1/[a-z0-9.-]+\.tar', uri)):
        raise ValueError('recall models require a directory and a pinned S3 archive')
    destination = Path(directory).resolve()
    if destination.exists():
        marker = destination / '.archive-sha256'
        if (not marker.is_file() or marker.read_text(encoding='ascii') != digest
                or any(not (destination / name).is_file() for name in REQUIRED_FILES)):
            raise ValueError('existing recall model directory differs from the pinned archive')
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix='.recall-') as temporary:
        root = Path(temporary)
        archive = root / 'models.tar'
        _download_s3(uri, archive)
        if archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError('recall model archive exceeds size limit')
        with archive.open('rb') as source:
            if file_digest(source, 'sha256').hexdigest() != digest:
                raise ValueError('recall model archive checksum mismatch')
        assets = root / 'models'
        extract_archive(archive, assets)
        (assets / '.archive-sha256').write_text(digest, encoding='ascii')
        assets.rename(destination)
    return destination
