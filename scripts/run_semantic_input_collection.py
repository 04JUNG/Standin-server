"""정답을 가정하지 않고 입력집의 실제 검색 응답을 기록한다."""
import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library-dir', required=True)
    p.add_argument('--queries', required=True)
    p.add_argument('--dense-model-dir', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from scripts.eval_semantic_metrics import load_index
    from src.config import CFG
    from src.semantic_search import semantic_search, settings_from_config
    index = load_index(Path(a.library_dir), dense_model_dir=a.dense_model_dir)
    settings = replace(settings_from_config(CFG), weight_dense=.5, dense_lo=.72, dense_hi=.88)
    inputs = json.loads(Path(a.queries).read_text())['queries']
    rows = []
    for q in inputs:
        result = semantic_search(index, q['text'], source='user', top_k=5, settings=settings)
        rows.append({**q, 'status': result['status'], 'returned': len(result['candidates']),
                     'elapsed_ms': result['elapsed_ms'],
                     'top': [(c['pose_id'], c['score'], c['evidence']['description']) for c in result['candidates']],
                     'result': result})
        print(q['id'], result['status'], rows[-1]['top'][:1], flush=True)
    summary = {'queries': len(rows), 'returned': sum(r['status']=='ok' for r in rows),
               'no_result': sum(r['status']=='no_result' for r in rows),
               'errors': sum(r['status']=='error' for r in rows),
               'latency_p50_ms': statistics.median(r['elapsed_ms'] for r in rows),
               'latency_max_ms': max(r['elapsed_ms'] for r in rows),
               'warning': '응답 상태는 정확도가 아니다. 시각 검토를 별도 기록한다.'}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'summary': summary, 'index': index.identity,
              'input_sha256': hashlib.sha256(Path(a.queries).read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256((ROOT/'src/semantic_search.py').read_bytes()).hexdigest(),
              'settings': vars(settings), 'rows': rows}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(summary)


if __name__ == '__main__':
    main()
