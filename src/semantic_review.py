"""로컬 100문장 시각 검수. 모델 평가와 사람 평가를 별도 보존한다."""
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field


class ReviewFeedback(BaseModel):
    review_id: str = Field(min_length=1, max_length=64)
    query_id: str = Field(pattern=r'^Q\d{3}$')
    verdict: Literal['match', 'partial', 'wrong', 'deferred']
    selected_pose_id: Optional[str] = Field(default=None, max_length=250)
    note: str = Field(default='', max_length=2000)


class ReviewStore:
    def __init__(self, report_path, feedback_path, index):
        self.report_path = Path(report_path)
        self.feedback_path = Path(feedback_path)
        self.index = index
        self.lock = threading.Lock()

    def dataset(self):
        if not self.report_path.exists():
            raise HTTPException(404, '검수용 100문장 결과 파일을 찾을 수 없습니다.')
        raw = self.report_path.read_bytes()
        report = json.loads(raw)
        review_id = hashlib.sha256(raw).hexdigest()[:20]
        rows = []
        allowed = {d.pose_id for d in self.index.docs}
        for item in report['rows']:
            def candidates(top):
                out = []
                for pid, score, description in top:
                    if pid not in allowed:
                        continue
                    pose = self.index.pose_json(pid)
                    if pose:
                        out.append({'pose_id': pid, 'score': score, 'description': description,
                                    'thumbnails': pose['thumbnail_urls'], 'tags': pose['tags']})
                return out
            current = candidates(item['top'])
            previous = candidates(item.get('baseline_top', []))
            rows.append({'id': item['id'], 'category': item['category'], 'text': item['text'],
                         'current': current, 'previous': previous,
                         'ai_verdict': item['final_verdict'], 'ai_note': item['visual_note']})
        return {'review_id': review_id, 'rows': rows, 'total': len(rows)}

    def _latest(self, review_id):
        latest = {}
        if self.feedback_path.exists():
            for line in self.feedback_path.read_text().splitlines():
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and item.get('review_id') == review_id:
                        latest[item['query_id']] = item
                except (ValueError, KeyError):
                    continue  # 손상된 한 줄이 다른 검수 내역을 가리지 않는다.
        return latest

    def state(self, review_id):
        with self.lock:
            return self._latest(review_id)

    def save(self, item):
        data = self.dataset()
        if item.review_id != data['review_id']:
            raise HTTPException(409, '실험 결과 버전이 바뀌었습니다. 새로고침 후 저장하세요.')
        q = next((q for q in data['rows'] if q['id'] == item.query_id), None)
        if q is None:
            raise HTTPException(422, '알 수 없는 문장 ID입니다.')
        valid = {c['pose_id'] for c in q['current']}
        if item.selected_pose_id is not None and item.selected_pose_id not in valid:
            raise HTTPException(422, '선택한 포즈가 이 문장의 개선 후 후보에 없습니다.')
        if item.verdict in ('match', 'partial') and item.selected_pose_id is None:
            raise HTTPException(422, '맞음/부분 일치 평가에는 선택한 포즈가 필요합니다.')
        rank = next((n+1 for n,c in enumerate(q['current']) if c['pose_id'] == item.selected_pose_id), None)
        row = {'created_at': datetime.now(timezone.utc).isoformat(), **item.model_dump(),
               'query': q['text'], 'selected_rank': rank, 'source': 'human-review-ui'}
        with self.lock:
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with self.feedback_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
        return row


def install_review_routes(app, report_path, feedback_path, index):
    store = ReviewStore(report_path, feedback_path, index)

    @app.get('/review-data')
    def data():
        payload = store.dataset()
        payload['reviews'] = store.state(payload['review_id'])
        return payload

    @app.post('/review-feedback')
    def feedback(item: ReviewFeedback):
        return {'saved': True, 'review': store.save(item)}

    return store
