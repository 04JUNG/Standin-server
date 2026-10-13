"""Local direct-text search and review only. No VLM or /analyze routing."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-dir', required=True)
    parser.add_argument('--dense-model-dir', default='')
    parser.add_argument('--port', type=int, default=4180)
    args = parser.parse_args()
    from dataclasses import replace
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    import uvicorn
    from scripts.eval_semantic_metrics import load_index
    from src.config import CFG
    from src.semantic_search import SemanticSearchService, settings_from_config
    from src.semantic_review import install_review_routes
    from api import app as api
    from api.models import SemanticSearchResponse

    library = Path(args.library_dir).resolve()
    settings = settings_from_config(CFG)
    if args.dense_model_dir:
        settings = replace(settings, weight_dense=.5, dense_lo=.72, dense_hi=.88)
    service = SemanticSearchService(load_index(library, dense_model_dir=args.dense_model_dir), settings)
    api.STATE['semantic'] = service
    api.STATE['db_path'] = str(library / 'source_bundle/poses.db')
    CFG.data_dir = str(library / 'source_bundle')
    app = FastAPI(title='Standin Direct Semantic Search Lab')
    app.add_api_route('/semantic-search', api.semantic_search_route, methods=['POST'],
                      response_model=SemanticSearchResponse, response_model_by_alias=True)
    app.add_api_route('/pose/{pose_id}/thumbnail', api.get_pose_thumbnail, methods=['GET'])
    app.mount('/lab-static', StaticFiles(directory=str(ROOT/'scripts/semantic-lab')), name='lab-static')

    @app.get('/')
    def page():
        return FileResponse(ROOT/'scripts/semantic-lab/index.html')

    @app.get('/review')
    def review():
        return FileResponse(ROOT/'scripts/semantic-lab/review.html')

    install_review_routes(app, ROOT/'evaluation/semantic/natural-inputs-100-20261013.review.json',
                          ROOT/'out/semantic-feedback/review-100.jsonl', service.index)

    @app.get('/lab-info')
    def info():
        return {'index':service.index.identity, 'dense':bool(args.dense_model_dir)}

    class Feedback(BaseModel):
        query: str = Field(min_length=1,max_length=500)
        pose_id: str = Field(min_length=1,max_length=250)
        verdict: str
        note: str = Field(default='',max_length=1000)

    @app.post('/feedback')
    def feedback(item:Feedback):
        if item.verdict not in ('relevant','irrelevant','expected'):
            raise HTTPException(422,'invalid verdict')
        if item.pose_id not in {d.pose_id for d in service.index.docs}:
            raise HTTPException(422,'unknown or excluded pose_id')
        row={'created_at':datetime.now(timezone.utc).isoformat(),**item.model_dump(),
             'index':service.index.identity}
        directory=ROOT/'out/semantic-feedback'
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'feedback.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        return {'saved':True}

    print(f'Local: http://127.0.0.1:{args.port}/',flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port)


if __name__=='__main__':
    main()
