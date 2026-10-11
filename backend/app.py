from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.pipeline import get_engine

app = FastAPI(title='ESG / Safety Incident Intelligence Dashboard')
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'


class Query(BaseModel):
    text: str = ''


@app.get('/api/demo')
def demo():
    return {
        'case_study': 'ESG / Safety Incident Intelligence Dashboard',
        'status': 'running',
        'expected_outcome': 'A dashboard that classifies incident type, severity, recurring patterns, and recommended preventive actions using NLP and summarization logic.',
        'endpoints': ['/api/summary', '/api/incidents', '/api/patterns', '/api/metrics', 'POST /api/run'],
    }


@app.get('/api/health')
def health():
    return {'status': 'ok', 'incidents_loaded': len(get_engine().df)}


@app.get('/api/summary')
def summary():
    return get_engine().summary()


@app.get('/api/incidents')
def incidents(limit: int = 50):
    return get_engine().incidents(limit)


@app.get('/api/patterns')
def patterns(min_count: int = 3):
    return get_engine().patterns(min_count)


@app.get('/api/metrics')
def metrics():
    return get_engine().metrics()


@app.post('/api/run')
def run_case(query: Query):
    return get_engine().analyze(query.text)


# Must stay LAST: the mount at "/" serves the frontend and would shadow later routes.
app.mount('/', StaticFiles(directory=BASE / 'frontend', html=True), name='frontend')
