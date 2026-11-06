"""Fail-closed local artifact gate. No v5 profile is approved for service yet."""
import hashlib,shutil
from pathlib import Path
REGISTERED_PROFILE_DIGESTS=frozenset()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finalize(source,candidate,destination,audit,profile_digest):
    source,candidate,destination=map(Path,(source,candidate,destination))
    if destination.resolve() in (source.resolve(),candidate.resolve()):raise ValueError('destination must be a new artifact')
    if destination.exists():raise FileExistsError(destination)
    source_sha=sha(source);reasons=[]
    if profile_digest not in REGISTERED_PROFILE_DIGESTS:reasons.append('range_and_skin_profile_uncertified')
    if audit.get('source_sha256')!=source_sha:reasons.append('stale_source')
    if audit.get('candidate_sha256')!=sha(candidate):reasons.append('stale_candidate')
    if audit.get('native_validated') is not True:reasons.append('native_unverified')
    if audit.get('skin',{}).get('complete_collision_certificate') is not True:reasons.append('skin_unverified')
    chosen=source if reasons else candidate
    shutil.copyfile(chosen,destination)
    if sha(destination)!=sha(chosen) or sha(source)!=source_sha:raise ValueError('artifact integrity failure')
    return dict(status='fallback_exact_B0' if reasons else 'accepted',reasons=reasons,sha256=sha(destination),source_sha256=source_sha,path=str(destination))
