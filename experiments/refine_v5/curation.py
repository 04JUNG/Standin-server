"""Agent review with provenance; exploratory grouping never certifies authors."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

PARTS = ('head torso pelvis left_upper_arm left_forearm left_hand right_upper_arm right_forearm right_hand left_thigh left_shin left_foot right_thigh right_shin right_foot').split()
VISIBILITY = {'visible', 'partial', 'occluded', 'cropped', 'unknown'}
FAMILIES = [(['001', '002', '003'], 'same mannequin drawing family'),
            (['005', '006'], 'same orange rendered reference session'),
            (['008', '009'], 'same photographed person and setting'),
            (['014', '015'], 'same illustrated character; conservative source risk'),
            (['021', '022'], 'similar fine-line drawing source; conservative risk'),
            (['065', '084'], 'same faint bag-carrying reference family')]
EXPOSED = {f'{i:03}' for i in range(1, 11)} | {'018', '026', '041', '074', '087'}

def review_mask(row):
    """Conservative bilateral mask until anatomical sides receive human review."""
    if row['target_identity'] == 'ambiguous' or row['scope'] in {'face', 'ambiguous'}:
        return [False] * 4
    upper = all(row['part_visibility'][s + '_upper_arm'] == 'visible' for s in ('left', 'right'))
    fore = upper and all(row['part_visibility'][s + '_forearm'] == 'visible' for s in ('left', 'right'))
    return [upper, fore, upper, fore]

def validate_review(inventory, review):
    expected = {r['case_id']: r for r in inventory}
    rows = review['records']
    if len(rows) != len(expected) or {r['case_id'] for r in rows} != set(expected):
        raise ValueError('review inventory mismatch or duplicates')
    if review.get('human_verified') is not False:
        raise ValueError('agent review cannot claim human verification')
    for row in rows:
        if row['image_sha256'] != expected[row['case_id']]['files']['image']['sha256']:
            raise ValueError('image hash mismatch')
        if set(row['part_visibility']) != set(PARTS) or not set(row['part_visibility'].values()) <= VISIBILITY:
            raise ValueError('invalid visibility')
        if row.get('human_verified') is not False or row.get('pose_authority') is not False:
            raise ValueError('unsupported human/pose authority')

def subset(groups, size):
    choices = {0: []}
    for group, cuts in groups:
        for n, chosen in list(choices.items())[::-1]:
            k = n + len(cuts)
            if k <= size and k not in choices:
                choices[k] = chosen + [group]
    if size not in choices:
        raise ValueError('group-preserving split size unavailable')
    return set(choices[size])

def curate(inventory, review):
    validate_review(inventory, review)
    parent = {r['original_cut_id']: r['original_cut_id'] for r in inventory}
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for cuts, reason in FAMILIES:
        for cut in cuts[1:]:
            parent[find(cut)] = find(cuts[0])
    groups = {}
    for cut in parent:
        groups.setdefault(find(cut), set()).add(cut)
    exposed = EXPOSED | {r['original_cut_id'] for r in review['records'] if r.get('prior_model_evaluation_sources')}
    unexposed = [(g, cuts) for g, cuts in groups.items() if not cuts & exposed]
    unexposed.sort(key=lambda x: hashlib.sha256(('v5-group-v2:' + x[0]).encode()).hexdigest())
    test = subset(unexposed, 20)
    val = subset([(g, cuts) for g, cuts in unexposed if g not in test], 20)
    result = []
    for row in review['records']:
        r = dict(row)
        group = find(r['original_cut_id'])
        r.update(source_risk_group=group, split='test' if group in test else 'validation' if group in val else 'development',
                 split_certification='exploratory_not_artist_blind', anatomical_side_status='agent_inferred_not_verified',
                 observed_arm_mask=review_mask(r))
        result.append(r)
    return dict(schema='v5-curated-exploratory-v2', human_verified=False, artist_disjoint_verified=False,
                independent_blind_evaluation=False, family_links=FAMILIES, records=result)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--review', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    inv = json.loads((Path(a.data) / 'inventory.json').read_text())
    review = json.loads(Path(a.review).read_text())
    result = curate(inv, review)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(dict(records=len(result['records']), rows_by_split=dict(Counter(r['split'] for r in result['records'])),
               cuts_by_split={s: len({r['original_cut_id'] for r in result['records'] if r['split'] == s}) for s in ('development', 'validation', 'test')},
               representations=dict(Counter(r['representation'] for r in result['records']))))
if __name__ == '__main__':
    main()
