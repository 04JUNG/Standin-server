"""Acquire individual BVHs and trim metadata from the author's 100STYLE catalog."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from html.parser import HTMLParser
import io
import os
from pathlib import Path
import re
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..storage import read_json, sha256, utc_now, write_json

CATALOG_URL = "https://www.ianxmason.com/100style/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = "The 100STYLE Dataset - Ian Mason, Sebastian Starke and Taku Komura"


@dataclass(frozen=True)
class Clip:
    name: str
    style: str
    movement: str
    url: str
    start: int
    stop: int


class CatalogParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: dict[str, str] = {}
        self.href: str | None = None
        self.label: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.label = []

    def handle_data(self, text):
        if self.href:
            self.label.append(text)

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.links["".join(self.label).strip()] = self.href
            self.href = None


def parse_catalog(html: str, cuts: str) -> list[Clip]:
    parser = CatalogParser()
    parser.feed(html)
    rows = {r["STYLE_NAME"].strip(): r for r in csv.DictReader(io.StringIO(cuts))}
    clips = []
    for name, url in parser.links.items():
        match = re.fullmatch(r"([A-Za-z0-9]+)_(BR|BW|FR|FW|ID|SR|SW|TR[123])\.bvh", name)
        if not match:
            continue
        style, movement = match.groups()
        row = rows.get(style, {})
        start, stop = row.get(movement + "_START"), row.get(movement + "_STOP")
        if start in (None, "N/A") or stop in (None, "N/A"):
            continue
        if urlparse(url).hostname != "drive.google.com":
            raise ValueError(f"unexpected dataset download host: {url}")
        clips.append(Clip(name, style, movement, url, int(start), int(stop)))
    if not clips:
        raise ValueError("100STYLE catalog has no BVHs with valid trim metadata")
    return sorted(clips, key=lambda c: c.name)


def download(url: str, destination: Path, *, kind: str, attempts: int = 3) -> dict:
    """Bounded downloads, content checks, atomic commit, hash-verified local cache."""
    sidecar = destination.with_suffix(destination.suffix + ".source.json")
    if destination.is_file() and sidecar.is_file():
        record = read_json(sidecar)
        if record.get("url") == url and record.get("sha256") == sha256(destination):
            return record
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "Standin-Pose-Curation/1.0"})
            size = 0
            with urlopen(request, timeout=60) as response, temporary.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 64 * 1024 * 1024:
                        raise ValueError("individual download exceeds 64 MiB limit")
                    target.write(chunk)
            with temporary.open("rb") as stream:
                prefix = stream.read(1024).lstrip(b"\xef\xbb\xbf \r\n\t")
            if kind == "bvh" and not prefix.startswith(b"HIERARCHY"):
                raise ValueError("download is not BVH (possibly an HTML download/error page)")
            if kind == "csv" and not prefix.startswith(b"STYLE_NAME,"):
                raise ValueError("download is not the expected Frame_Cuts.csv")
            record = {"url": url, "sha256": sha256(temporary), "bytes": size, "downloaded_at": utc_now()}
            os.replace(temporary, destination)
            write_json(sidecar, record)
            return record
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def acquire_catalog(root: Path, *, offline: bool = False) -> list[Clip]:
    html_path, cuts_path = root / "catalog.html", root / "Frame_Cuts.csv"
    if not offline:
        download(CATALOG_URL, html_path, kind="html")
        parser = CatalogParser()
        parser.feed(html_path.read_text(encoding="utf-8"))
        url = parser.links.get("Frame_Cuts.csv")
        if not url or urlparse(url).hostname != "drive.google.com":
            raise ValueError("official catalog is missing its Frame_Cuts.csv link")
        download(url, cuts_path, kind="csv")
    return parse_catalog(html_path.read_text(encoding="utf-8"), cuts_path.read_text(encoding="utf-8-sig"))
