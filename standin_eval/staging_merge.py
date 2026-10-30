"""Merge complete disjoint staging shards before publishing any fixed-set baseline."""
from pathlib import Path
import shutil
from .util import read_json, read_jsonl, write_json, write_jsonl
from .rubric_v32_io import score_bundle_v32
from .staging_compare import report


def merge_fragments(fragments, output, previous=None):
    output = Path(output)
    values = [read_json(p/'result.json') for p in fragments]
    first = values[0]
    ids = [r['case_id'] for v in values for r in v['arms']['search']]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError('requires exactly 50 disjoint cases')
    if any(v['revision'] != first['revision'] or v['contract'] != first['contract'] for v in values):
        raise ValueError('shard evaluation identities differ')
    if any(not read_json(p/'completion.json')['complete'] for p in fragments):
        raise ValueError('incomplete shard cannot publish a baseline')
    output.mkdir(parents=True, exist_ok=False)
    for p in fragments:
        for directory in ('inputs','images'):
            if (p/directory).exists():
                shutil.copytree(p/directory,output/directory,dirs_exist_ok=True)
    arms, summaries = {}, {}
    for arm in ('search','refine'):
        manifests = [read_json(p/arm/'dataset-manifest.json') for p in fragments]
        rows = [r for p in fragments for r in read_jsonl(p/arm/'reviews.jsonl')]
        rows.sort(key=lambda r:r['case_id'])
        manifest = {'cases':sorted([c for m in manifests for c in m['cases']],key=lambda c:c['case_id'])}
        config = read_json(fragments[0]/arm/'run-config.json')
        write_json(output/f'{arm}-manifest.json',manifest)
        write_json(output/f'{arm}-config.json',config)
        write_jsonl(output/f'{arm}-reviews.jsonl',rows)
        summaries[arm] = score_bundle_v32(output/f'{arm}-manifest.json',output/f'{arm}-reviews.jsonl',
            output/f'{arm}-config.json',output/arm,output)
        from .rubric_package import review_queue
        review_queue(output/arm,output/arm/'final-review.jsonl')
        arms[arm] = read_json(output/arm/'computed-scores.json')
    targets = {}
    for value in values:
        for target in value['targets']:
            key = (target['case_id'],target['reference_person_id'])
            if key in targets and targets[key]['target_sha256'] != target['target_sha256']:
                raise ValueError('shards disagree on frozen target')
            targets[key] = target
    current = {**first, 'arms':arms, 'summaries':summaries, 'targets':list(targets.values()),
        'operations':{k:sum(v['operations'][k] for v in values) for k in first['operations']}, 'shard_count':len(fragments)}
    write_json(output/'result.json',current)
    write_json(output/'completion.json',{'complete':True,'cases':50,'shards':len(fragments)})
    report(output,current,previous)
    return current
