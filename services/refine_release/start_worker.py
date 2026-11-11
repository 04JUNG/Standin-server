"""Load non-secret, pinned deployment paths before starting the private worker."""
import json, os
from pathlib import Path
from .service import settings
allowed={"REFINE_RELEASE_REGISTRY","REFINE_EXTRACTOR_CHECKOUT","REFINE_EXTRACTOR_REVISION","REFINE_MODEL_MANIFEST","REFINE_MODEL_MANIFEST_SHA256","REFINE_EXTRACTOR_SOURCE_MANIFEST","REFINE_EXTRACTOR_SOURCE_SHA256"}
config=json.loads(Path('/opt/rc4/env.json').read_text())
if set(config)!=allowed or not all(isinstance(v,str) for v in config.values()):raise ValueError('invalid worker configuration')
os.environ.update(config)
settings()
os.execvp('uvicorn',['uvicorn','services.refine_release.worker:app','--host','127.0.0.1','--port','8002'])
