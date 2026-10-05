"""Local cached-query evaluation. Projection fit is not a visual accuracy claim."""
import argparse
from pathlib import Path
import time

import numpy as np

from ..orientation import Orientation
from ..review.catalog import Catalog
from ..review.store import ReviewStore
from ..storage import write_json, utc_now
from .index import ScopedIndex
from .queries import RoughQueries
from .matching import match, observed, projection


def fixed_error(bodies, points, scores, image_size=None):
    mask = observed(points, scores, image_size)
    target = np.asarray(points)[mask]
    weights = np.clip(np.asarray(scores)[mask], .3, 1)
    weights /= weights.sum()
    y = target - (target * weights[:, None]).sum(0)
    best = float('inf')
    for yaw in [0,45,90,180]:
        xy = projection(bodies[:, mask], Orientation(yaw))
        x = xy - (xy * weights[:,None]).sum(-2)[:,None,:]
        scale = np.maximum(0, (x*y*weights[:,None]).sum((-2,-1)) / np.maximum((x*x*weights[:,None]).sum((-2,-1)),1e-12))
        error = np.sqrt(((scale[:,None,None]*x-y)**2 * weights[:,None]).sum((-2,-1)) / (y*y*weights[:,None]).sum())
        best = min(best,float(error.min()))
    return best


def run(data, curation, output):
    index = ScopedIndex(curation, Catalog(data,curation), ReviewStore(curation/'reviews.sqlite'))
    manifest, arrays = index.read()
    queries = RoughQueries(curation)
    rows=[]
    eligible=list(queries.rows.values())
    # Fixed, evenly spaced subset to keep the local verification reproducible.
    selected=[eligible[i] for i in np.linspace(0,len(eligible)-1,min(48,len(eligible)),dtype=int)]
    if not selected:
        raise ValueError('No sufficiently observed arms for evaluation')
    for query in selected:
        query = queries.get(query['key'])
        start=time.monotonic()
        reduced=match(manifest,arrays,query['keypoints'],query['scores'],image_size=query['size'])
        elapsed=time.monotonic()-start
        full=match(manifest,arrays,query['keypoints'],query['scores'],pool='all',image_size=query['size'])
        fixed=fixed_error(arrays['body'],query['keypoints'],query['scores'],query['size'])
        rows.append({'query':query['key'],'fixed_four_error':fixed,
                     'representative_with_fallback_error':reduced['candidates'][0]['error'],
                     'all_continuous_error':full['candidates'][0]['error'],
                     'seconds':elapsed,'fallback':reduced['fallback_to_all'],
                     'candidate':reduced['candidates'][0]})
        print(f"[{len(rows)}] {query['key']}: {fixed:.3f} -> {rows[-1]['representative_with_fallback_error']:.3f}",flush=True)
    report={'created_at':utc_now(),'library_revision':manifest['revision'],'queries':len(rows),
            'eligible_queries':len(eligible),'upper_only_queries':sum(q['upper_only'] for q in eligible),
            'policy':'48 evenly spaced eligible real cached observations; visible upper-body fit regardless of image framing; no ownership re-inference',
            'limitations':['No ground-truth 3D orientation','Not a visual accuracy or held-out quality evaluation',
                           'No head/face matching; no perspective fitting; no joint deformation'],
            'summary':{'fixed_four_median':float(np.median([r['fixed_four_error'] for r in rows])),
                       'reduced_continuous_median':float(np.median([r['representative_with_fallback_error'] for r in rows])),
                       'all_continuous_median':float(np.median([r['all_continuous_error'] for r in rows])),
                       'median_seconds':float(np.median([r['seconds'] for r in rows])),
                       'fallback_count':sum(r['fallback'] for r in rows),
                       'worse_than_all_by_over_002':sum(r['representative_with_fallback_error']>r['all_continuous_error']+.02 for r in rows)},
            'rows':rows}
    write_json(output,report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--curation',type=Path,default=Path('data/curation'))
    p.add_argument('--output',type=Path,default=Path('data/dev/scoped-library/evaluation.json'))
    a=p.parse_args();print(run(a.data,a.curation,a.output)['summary'])
