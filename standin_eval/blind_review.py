"""Conservative aggregation of explicitly supplied human relevance ratings."""
ARMS = ('anchor_4', 'anchor_117', 'pooled_117')


def summarize(public, key, submission):
    if submission.get('study_id') != public['study_id'] or key.get('study_id') != public['study_id']:
        raise ValueError('study mismatch')
    digest = public['public_data_sha256']
    if submission.get('public_data_sha256') != digest or key.get('public_data_sha256') != digest:
        raise ValueError('review version mismatch')
    if not isinstance(submission.get('rater'), str) or not submission['rater'].strip():
        raise ValueError('rater alias required')
    if submission.get('rater_origin') != 'human-self-report':
        raise ValueError('explicit human self-report required; this is not identity verification')
    ratings = submission.get('ratings', {})
    expected = set(public['candidate_ids'])
    if not isinstance(ratings, dict) or not set(ratings) <= expected:
        raise ValueError('unknown candidate ratings')
    if any(v not in ('yes', 'no', 'uncertain') for v in ratings.values()):
        raise ValueError('invalid rating')
    base = {'study_id':public['study_id'], 'rater':submission['rater'].strip(),
            'rating_origin':'human self-report, not independently authenticated',
            'rated':len(ratings), 'expected':len(expected),
            'metric_kind':'human judged relevance, not original-pose-ID accuracy'}
    if set(ratings) != expected:
        return {**base,'status':'incomplete','metrics':None}
    records = []
    for query in public['queries']:
        qkey=key['queries'][query['id']]
        if set(qkey['slots'].values()) != set(ARMS):
            raise ValueError('invalid blinding assignment')
        if bool(query['failed']) != bool(qkey['failed']):
            raise ValueError('failure status mismatch')
        for slot, arm in qkey['slots'].items():
            if query['failed']:
                top1=top5=False
            else:
                groups=[g for g in query['groups'] if g['slot']==slot]
                if len(groups)!=1 or len(groups[0]['candidates'])!=5:
                    raise ValueError('five candidates per method required')
                values=[ratings[c['id']] for c in groups[0]['candidates']]
                top1=True if values[0]=='yes' else False if values[0]=='no' else None
                top5=True if 'yes' in values else False if all(v=='no' for v in values) else None
            records.append({'query':query['id'],'arm':arm,'failed':query['failed'],'top1':top1,'top5':top5})
    n=len(public['queries']);metrics={}
    for arm in ARMS:
        metrics[arm]={}
        for k in ('top1','top5'):
            values=[r[k] for r in records if r['arm']==arm]
            assert len(values)==n
            yes=sum(v is True for v in values);unknown=sum(v is None for v in values)
            metrics[arm][k]={'n':n,'confirmed_useful':yes,'uncertain':unknown,
                'not_useful_or_extraction_failure':n-yes-unknown,
                'lower_rate':yes/n,'upper_rate':(yes+unknown)/n}
    return {**base,'status':'complete','extraction_failures':sum(q['failed'] for q in public['queries']),
            'metrics':metrics,'records':records}
