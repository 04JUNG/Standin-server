"""Isolated, opt-in deployment adapter. No experiment outputs are writable inputs."""
from __future__ import annotations
import fcntl, hashlib, io, json, os, shutil, signal, subprocess, sys, uuid
from pathlib import Path
from PIL import Image
from . import VERSION

class Unavailable(RuntimeError): pass
class Busy(RuntimeError): pass

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_runtime():
    root = Path(__file__).parent
    manifest = json.loads((root / "runtime-manifest.json").read_text())
    actual = {str(p.relative_to(root / "runtime")) for p in (root / "runtime").rglob("*.py")}
    if actual != set(manifest["files"]): raise Unavailable("release runtime file inventory mismatch")
    for relative, digest in manifest["files"].items():
        if sha(inside(root / "runtime", relative)) != digest:
            raise Unavailable("release runtime hash mismatch")

def settings():
    required = ["REFINE_RELEASE_BODY", "REFINE_RELEASE_BODY_SHA256", "REFINE_RELEASE_STATE", "REFINE_EXTRACTOR_CHECKOUT", "REFINE_MODEL_MANIFEST", "REFINE_EXTRACTOR_REVISION", "REFINE_MODEL_MANIFEST_SHA256", "REFINE_EXTRACTOR_SOURCE_MANIFEST", "REFINE_EXTRACTOR_SOURCE_SHA256"]
    if os.getenv("REFINE_RELEASE_ENABLED") != "1": raise Unavailable("release refine is disabled")
    if any(not os.getenv(k) for k in required) or not os.getenv("OPENAI_API_KEY"):
        raise Unavailable("release configuration is incomplete")
    verify_runtime()
    cfg = {k: os.environ[k] for k in required}
    cfg["blender"] = shutil.which(os.getenv("REFINE_BLENDER", "blender"))
    if not cfg["blender"]: raise Unavailable("Blender is unavailable")
    revision = subprocess.check_output(["git", "-C", cfg["REFINE_EXTRACTOR_CHECKOUT"], "rev-parse", "HEAD"], text=True).strip()
    if revision != cfg["REFINE_EXTRACTOR_REVISION"] or sha(cfg["REFINE_MODEL_MANIFEST"]) != cfg["REFINE_MODEL_MANIFEST_SHA256"]:
        raise Unavailable("extractor/model manifest revision mismatch")
    source_manifest = Path(cfg["REFINE_EXTRACTOR_SOURCE_MANIFEST"])
    if sha(source_manifest) != cfg["REFINE_EXTRACTOR_SOURCE_SHA256"]:
        raise Unavailable("extractor source manifest mismatch")
    sources = json.loads(source_manifest.read_text())
    if "experiments/refine_v5/develop_skeleton_seeds.py" not in sources or "src/pose_humanart.py" not in sources:
        raise Unavailable("extractor source manifest is incomplete")
    for relative, digest in sources.items():
        if sha(inside(Path(cfg["REFINE_EXTRACTOR_CHECKOUT"]), relative)) != digest:
            raise Unavailable("extractor source changed")
    from .assets import configuration
    configuration()
    for module in ["numpy", "scipy", "cv2", "onnxruntime", "rtmlib"]:
        __import__(module)
    return cfg

def inside(root, relative):
    p = (root / relative).resolve()
    if not p.is_relative_to(root.resolve()): raise ValueError("asset path escapes registry root")
    return p

def prepare(registry, pose_id, image, destination, expected_bvh=None):
    entry = json.loads(Path(registry).read_text())[pose_id]
    source = Path(entry["assets"]).resolve(); case = entry["case"]
    if not case.isalnum(): raise ValueError("invalid registered case identifier")
    manifest_path = source / "input/manifest.json"
    if sha(manifest_path) != entry["manifest_sha256"]: raise ValueError("registered manifest changed")
    row = next(r for r in json.loads(manifest_path.read_text())["rows"] if r["case_id"] == case)
    row = json.loads(json.dumps(row))
    if row["selection"]["selected_pose_id"] != pose_id: raise ValueError("pose binding mismatch")
    rig_relative = f"input/p1-native/{case}-source.json"
    rig = json.loads(inside(source, rig_relative).read_text())
    relatives = [rig_relative, f"base-mesh/{case}.npz", "input/"+rig["source"], "input/"+row["selection"]["verified_selected_bvh"]]
    for relative in relatives:
        original = inside(source, relative)
        if entry["files"].get(relative) != sha(original): raise ValueError("registered asset hash mismatch")
        target = inside(destination, relative); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(original, target)
    if expected_bvh and sha(expected_bvh) != row["selection"]["selected_bvh_sha256"]:
        raise ValueError("database BVH differs from registered original")
    row["rough"] = f"rough/{case}.png"; row["rough_sha256"] = hashlib.sha256(image).hexdigest()
    rough = destination / "input" / row["rough"]; rough.parent.mkdir(parents=True, exist_ok=True); rough.write_bytes(image)
    (destination/"input/manifest.json").write_text(json.dumps({"rows":[row]}))
    return case

def execute(pose_id, image, expected_bvh=None, bvh_bytes=None):
    cfg = settings()
    if len(image) > 12*1024*1024: raise ValueError("image exceeds 12 MiB")
    try:
        with Image.open(io.BytesIO(image)) as im:
            if im.format != "PNG" or im.width*im.height > 16000000: raise ValueError("PNG up to 16 megapixels required")
            im.verify()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid image") from exc
    state = Path(cfg["REFINE_RELEASE_STATE"]).resolve(); state.mkdir(parents=True, exist_ok=True)
    with (state/"worker.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise Busy("release worker is busy")
        handle = uuid.uuid4().hex; job = state/handle; job.mkdir()
        from .assets import prepare as prepare_original
        if bvh_bytes is None:
            if expected_bvh is None: raise ValueError("library original BVH required")
            bvh_bytes = Path(expected_bvh).read_bytes()
        case = prepare_original(pose_id, bvh_bytes, image, job/"assets", cfg)
        command = [sys.executable, str(Path(__file__).parent/"runtime/run_minute.py"), "--assets", str(job/"assets"), "--out", str(job/"run"), "--case", case, "--extractor-checkout", cfg["REFINE_EXTRACTOR_CHECKOUT"], "--model-manifest", cfg["REFINE_MODEL_MANIFEST"]]
        env = dict(os.environ, REFINE_BLENDER=cfg["blender"], REFINE_MANAGED_PROCESS="1")
        with (job/"worker.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try: code = process.wait(timeout=float(os.getenv("REFINE_RELEASE_TIMEOUT", "300")))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(); raise Unavailable("release deadline exceeded; no artifact released")
            finally:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
        if code: raise Unavailable("release worker failed; no artifact released")
        decision = json.loads((job/"run/gate"/case/"decision.json").read_text())
        reason_codes = sorted({issue["kind"] for candidate in decision.get("candidates", []) for issue in candidate.get("issues", [])})
        if decision.get("reason"): reason_codes.append(decision["reason"])
        result = dict(version=VERSION, handle=handle, status="withheld", selected=decision.get("selected"), download_url=None, reason_codes=reason_codes)
        if decision.get("release_eligible") is True and decision.get("status") == "accepted_under_measured_policy" and decision.get("export", {}).get("full_matrix_roundtrip_passed") is True:
            outputs = list((job/"run/gate"/case/"release").glob("*.fbx"))
            if len(outputs)!=1: raise Unavailable("verified export is missing or ambiguous")
            shutil.copyfile(outputs[0], job/"result.fbx")
            camera = json.loads((job/"run/gate"/case/"release/camera.json").read_text())
            result.update(camera=camera, status="accepted", reason_codes=[], download_url=f"/refine/release/{handle}/fbx", artifact_sha256=sha(job/"result.fbx"))
        result['source'] = json.loads((job/'assets/original-provenance.json').read_text())
        (job/"result.json").write_text(json.dumps(result))
        return result

def artifact(handle):
    cfg = settings()
    if len(handle)!=32 or any(c not in "0123456789abcdef" for c in handle): raise FileNotFoundError()
    job = Path(cfg["REFINE_RELEASE_STATE"])/handle
    record = json.loads((job/"result.json").read_text())
    path = job/"result.fbx"
    if record["status"]!="accepted" or sha(path)!=record["artifact_sha256"]: raise FileNotFoundError()
    return path
