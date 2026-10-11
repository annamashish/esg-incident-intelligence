# ESG / Safety Incident Intelligence Dashboard

Turns free-text safety and ESG incident narratives into management-ready insight: incident type, severity, risk tags, recurring patterns, a summary and recommended preventive actions.

## Run locally

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn backend.app:app --reload --port 8000
```

Open http://127.0.0.1:8000 (dashboard) and http://127.0.0.1:8000/docs (API docs).
Run these from the folder that contains `backend/`, `data/` and `frontend/`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/summary` | KPIs, distributions, top risk themes, management summary |
| GET | `/api/incidents?limit=N` | Register with predicted type, severity and tags |
| GET | `/api/patterns?min_count=3` | Recurring asset x risk combinations and monthly trend |
| GET | `/api/metrics` | Evaluation results |
| POST | `/api/run` | Analyse one narrative: `{"text": "..."}` |
| GET | `/api/demo`, `/api/health` | Status checks |

## Design and model logic

1. **Cleaning**: lowercase and strip punctuation.
2. **Classification**: TF-IDF (word 1-2 grams, English stop words removed) with Logistic Regression, one model for incident type (safety / environment / governance) and one for severity (low / medium / high). Class weights are balanced.
3. **Severity safety floor**: if the text contains critical terms (fire, fracture, lost time, hospital, SF6, arc fault, expired consent, falsified record, ...), severity is never below *high*, whatever the model says. The response reports when this rule fired.
4. **Risk tagging**: transparent regex rules (fire, electrical, work at height, vehicle movement, slip/trip, PPE, injury, near miss, oil/chemical spill, water use, biodiversity, waste, regulatory, documentation, heat stress). Rules are used because they are auditable by safety staff.
5. **Retrieval evidence**: cosine-similarity nearest past incidents, plus the terms that pushed the model towards its label.
6. **Pattern detection**: counts recurring (asset, risk tag) pairs, ranks by high-severity count, and builds a monthly severity trend.
7. **Summarisation**: extractive and template-driven. A single incident gets a one-line structured summary; the portfolio summary names the biggest category, top risk themes and the high-severity case closest to the centroid of all high-severity narratives. No external LLM is needed, so it runs offline.
8. **Recommendations**: a rule table maps each risk tag and category to preventive actions; high severity adds an escalation step.

## Data and assumptions

- `data/safety_esg_incidents.csv`: the small bundled set (a handful of rows).
- `data/incidents_extended.csv`: **synthetic** rows produced by `scripts/generate_data.py` from templates (marked `source = synthetic`), because the bundled set is too small to train or validate anything.
- The synthetic `site` and `date` columns are random and unrelated to the narrative text. Site lists and monthly trends are therefore illustrative only.
- Category labels `environment` and `safety` come from the bundled data; `governance` is added to cover the ESG "G" dimension.

## Evaluation (test run)

| Task | Accuracy | Macro-F1 | Method |
|---|---|---|---|
| Incident type | 1.00 | 1.00 | 5-fold stratified CV, 215 rows |
| Severity | 0.995 | 0.996 | 5-fold stratified CV, 215 rows (1 error) |
| Real-row holdout | 1.00 / 1.00 | n/a | Train on synthetic, test on the bundled rows (n is only 5-6) |

Live numbers and confusion matrices are at `/api/metrics` and on the dashboard.

**These scores are optimistic and should not be read as production accuracy.** Synthetic rows come from a small set of templates, so near-duplicates appear in both training and test folds. The real-row holdout is far too small to be meaningful.

## Limitations and failure cases

- Very small real dataset; the model has mostly learned template vocabulary.
- Narratives using unfamiliar wording get low confidence (flagged in the UI) and may be misclassified.
- Keyword rules can false-positive (for example, "transformer oil leak" is tagged both electrical and spill) and miss paraphrases or negation ("no fire occurred" still matches "fire").
- Asset detection is a keyword lookup; unknown assets are grouped as "other" and excluded from pattern counts.
- Severity here follows the labels in the data, not a formal risk matrix (likelihood x consequence).
- English only. No authentication.

## Production-readiness gaps and next steps

- Collect and label real incidents; hold out a time-based test set.
- Replace TF-IDF with sentence embeddings or a fine-tuned transformer; add an LLM for abstractive summaries with citations back to source incidents.
- Add a human-in-the-loop review queue, an audit trail, and model versioning and monitoring (drift, calibration).
- Persist data in a database instead of CSV; add authentication and role-based access.
- Integrate with the EHS / permit-to-work system and add CAPA closure tracking.
- Add automated tests and CI.

## Deploy to Railway

1. Push this folder to GitHub (the folder containing `Procfile`, `backend/`, `frontend/`, `data/`, `requirements.txt`).
2. In Railway: New Project, Deploy from GitHub repo, select the repo.
3. Railway installs `requirements.txt` and starts the `Procfile` command. `.python-version` pins Python 3.12.
4. Open Settings, Networking, Generate Domain, then visit the URL. Check `/api/health`.
