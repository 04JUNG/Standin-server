"""Opt-in scene preparation, independent of the normal analysis deadline.

The caller is the authenticated BFF, never an end-user path or Blender script.
Inputs are immutable reviewed artifacts. Model-derived changes require a second
scene review; a subsequent download releases exactly the pinned file.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import base64, hashlib, io, json, math, re, secrets, subprocess, time
from PIL import Image

VERSION = "scene-refinement-v4"
MAX_SOURCE = 32 * 1024 * 1024
MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode(value, limit):
    if not isinstance(value, str) or len(value) > 4 * ((limit + 2) // 3):
        raise ValueError("input exceeds size limit")
    data = base64.b64decode(value, validate=True)
    if not data or len(data) > limit:
        raise ValueError("invalid artifact size")
    return data


def fbx(value, sha):
    data = decode(value, MAX_SOURCE)
    if not data.startswith(MAGIC) or digest(data) != sha:
        raise ValueError("source FBX integrity mismatch")
    return data


@dataclass(frozen=True)
class Settings:
    root: Path
    blender: Path
    catalog: Path
    library: Path
    character: Path
    head_axes: Path
    character_sha256: str
    timeout: int = 300
    ttl: int = 1800


class Engine:
    def __init__(self, settings, reader, visual, *, parts_service=None):
        self.settings = settings
        self.reader = reader
        self.visual = visual
        self.parts_service = parts_service
        self.pending = {}
        self.retired = {}
        self.active = None
        if digest(settings.character.read_bytes()) != settings.character_sha256:
            raise ValueError("head-axis character binding mismatch")
        settings.root.mkdir(parents=True, exist_ok=True)

    def parts_capabilities(self):
        from src.refine_v4.service import CONTRACT
        from src.refine_v4 import VERSION as parts_version
        service = getattr(self, "parts_service", None)
        if service is None:
            return {"contract_version": CONTRACT, "engine_version": parts_version,
                    "available": False, "reason": "PARTS_DISABLED", "operations": [], "recipes": []}
        return service.capabilities()

    def refine_parts(self, request):
        from src.refine_v4.service import PartsError
        service = getattr(self, "parts_service", None)
        if service is None:
            raise PartsError("PARTS_DISABLED", 503)
        return service.refine(request)

    def stage(self, ctx, *stages):
        path = ctx.work.parent / "context.json"
        path.write_text(json.dumps({k: str(v) for k, v in vars(ctx).items()}))
        log = ctx.work.parent / ("-".join(stages) + ".log")
        with log.open("w") as stream:
            subprocess.run(
                [
                    str(self.settings.blender),
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name("worker.py")),
                    "--",
                    str(path),
                    *stages,
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=self.settings.timeout,
                check=True,
            )

    def retire(self, ctx):
        import shutil

        self.retired[ctx.work.parent] = time.monotonic()
        for path, stamp in list(self.retired.items()):
            if time.monotonic() - stamp > self.settings.ttl or len(self.retired) > 8:
                self.retired.pop(path)
                shutil.rmtree(path, ignore_errors=True)

    def prepare(self, request):
        self.active = None
        try:
            return self._prepare(request)
        finally:
            if self.active and all(
                v[1].work.parent != self.active.work.parent
                for v in self.pending.values()
            ):
                self.retire(self.active)
            self.active = None

    def _prepare(self, request):
        import shutil

        for key, (stamp, ctx, _, _) in list(self.pending.items()):
            if time.monotonic() - stamp > self.settings.ttl:
                self.pending.pop(key)
                shutil.rmtree(ctx.work.parent)
        if len(self.pending) >= 4:
            raise ValueError("scene preparation queue is full")
        image = decode(request["image_base64"], 20 * 1024 * 1024)
        if digest(image) != request["image_sha256"]:
            raise ValueError("original image changed")
        with Image.open(io.BytesIO(image)) as im:
            if (
                im.format not in ("PNG", "JPEG", "WEBP")
                or im.width * im.height > 24_000_000
            ):
                raise ValueError("unsupported image")
            width, height = im.size
        persons = request["persons"]
        if not isinstance(persons, list) or not 1 <= len(persons) <= 8:
            raise ValueError("invalid person count")
        ids = [p["person_index"] for p in persons]
        if any(type(i) != int or not 0 <= i <= 255 for i in ids) or len(
            set(ids)
        ) != len(ids):
            raise ValueError("invalid person identities")
        key = secrets.token_hex(24)
        root = self.settings.root / key
        work = root / "work"
        cut = "scene"
        folder = work / cut
        folder.mkdir(parents=True)
        ctx = SimpleNamespace(
            work=work,
            baseline=root / "baseline",
            catalog=self.settings.catalog,
            library=self.settings.library,
            character=self.settings.character,
            head_axes=self.settings.head_axes,
        )
        self.active = ctx
        (work / "original").write_bytes(image)
        descriptor = []
        for p in persons:
            i = p["person_index"]
            pose = p["pose_id"]
            obs = p["observation"]
            from .assets import valid_pose_id
            if not valid_pose_id(pose) or p["output_scope"] not in (
                "full",
                "half",
                "bust",
                "head",
                "shoulder_head",
            ):
                raise ValueError("invalid source pose")
            if not (self.settings.library / "bvh" / f"{pose}.bvh").is_file():
                raise ValueError("unregistered pose")
            box = obs["box"]
            kp = obs["keypoints"]
            scores = obs["scores"]
            if (
                len(box) != 4
                or len(kp) != 17
                or any(len(v) != 2 for v in kp)
                or len(scores) != 17
            ):
                raise ValueError("invalid observation dimensions")
            if not all(
                isinstance(x, (int, float)) and math.isfinite(x)
                for x in [*box, *scores, *(x for p in kp for x in p)]
            ):
                raise ValueError("nonfinite observation")
            if not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height:
                raise ValueError("invalid observation box")
            (folder / f"person_{i}-reviewed.fbx").write_bytes(
                fbx(p["fbx_base64"], p["fbx_sha256"])
            )
            descriptor.append(
                {
                    k: p[k]
                    for k in ("person_index", "pose_id", "output_scope", "observation")
                }
            )
        (work / "request.json").write_text(
            json.dumps(
                dict(
                    cut_id=cut,
                    image=dict(width=width, height=height),
                    persons=descriptor,
                )
            )
        )
        (work / "inputs.json").write_text(
            json.dumps(
                [
                    dict(
                        cut_id=cut,
                        image_path=str(work / "original"),
                        image_sha256=digest(image),
                    )
                ]
            )
        )
        self.stage(ctx, "baseline")
        from .evidence import run

        run(ctx, self.reader, self.visual)
        from .validation import validate_evidence

        validate_evidence(ctx, ids)
        from .plan import run as plan

        plan(ctx)
        selected = json.loads((folder / "selection.json").read_text())
        self.pending[key] = (time.monotonic(), ctx, selected, request)
        return dict(
            version=VERSION,
            plan_id=key,
            image_sha256=digest(image),
            character_sha256=self.settings.character_sha256,
            people=selected["people"],
        )

    def build(self, request):
        key = request.get("plan_id")
        pending = self.pending.get(key)
        try:
            return self._build(request)
        finally:
            if pending:
                self.pending.pop(key, None)
                self.retire(pending[1])

    def _build(self, request):
        artifact_only = request.get("artifact_only", False)
        if type(artifact_only) is not bool:
            raise ValueError("invalid artifact_only option")
        key = request["plan_id"]
        pending = self.pending.get(key)
        if not pending or time.monotonic() - pending[0] > self.settings.ttl:
            raise ValueError("scene plan expired")
        _, ctx, plan, source = pending
        folder = (
            ctx.work / json.loads((ctx.work / "inputs.json").read_text())[0]["cut_id"]
        )
        rows = request["persons"]
        wanted = {p["person_index"]: p for p in plan["people"]}
        if (
            len(rows) != len(wanted)
            or {p["person_index"] for p in rows} != wanted.keys()
        ):
            raise ValueError("prepared identities differ")
        for p in rows:
            i = p["person_index"]
            if (
                p["pose_id"] != wanted[i]["pose_id"]
                or p["character_sha256"] != self.settings.character_sha256
            ):
                raise ValueError("prepared pose/body lineage mismatch")
            (folder / f"person_{i}-prepared-full.fbx").write_bytes(
                fbx(p["fbx_base64"], p["fbx_sha256"])
            )
        self.stage(ctx, "people")
        from .layout import run as fit

        fit(ctx)
        self.stage(ctx, "arms")
        fit(ctx)
        if artifact_only:
            (folder / "artifact-only").touch()
        self.stage(ctx, "scenes")
        layout = json.loads((folder / "layout.json").read_text())
        result = json.loads((folder / "scene-result.json").read_text())
        from .validation import validate_layout

        validate_layout(layout)
        data = (folder / "scene.fbx").read_bytes()
        preview = b"" if artifact_only else (folder / "final.png").read_bytes()
        from converter.scene_camera_transport import camera_binding
        framing = result["results"]["final"]["verification"]["framing"]
        return dict(
            camera=camera_binding(framing, digest(data)),
            version=VERSION,
            plan_id=key,
            artifact_sha256=digest(data),
            **({"artifact_contract": "verified-scene-fbx-v1"} if artifact_only else {}),
            preview_sha256=digest(preview),
            fbx_base64=base64.b64encode(data).decode(),
            preview_base64=base64.b64encode(preview).decode(),
            people=plan["people"],
            layout=layout,
            verification=result,
            person_count=len(rows),
            image_sha256=source["image_sha256"],
        )
