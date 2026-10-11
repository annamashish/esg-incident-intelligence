"""NLP pipeline for ESG / safety incident narratives.

Components
- clean_text:        lowercasing / normalisation
- risk tagging:      transparent keyword rules (fire, electrical, height, ...)
- classifiers:       TF-IDF (word 1-2 grams) + Logistic Regression for
                     incident category and severity
- severity guard:    rule-based floor so critical keywords are never under-rated
- retrieval:         cosine-similarity nearest incidents as evidence
- patterns:          recurring (asset, risk tag) combinations and monthly trend
- summarisation:     extractive, template-driven (no external LLM needed)
- recommendations:   rule table mapping tags / category to preventive actions
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEV_ORDER = {"low": 0, "medium": 1, "high": 2}
SEV_NAMES = ["low", "medium", "high"]

# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
RISK_RULES = {
    "fire": r"\b(fire|flame|smoke|burn|burnt|arc fault|flash)\b",
    "electrical": r"\b(electric|electrical|live panel|arc|flash|shock|energis|lock-?out|loto|transformer)\b",
    "work_at_height": r"\b(height|fall|fell|ladder|harness|nacelle|climb|dropped object)\b",
    "vehicle_movement": r"\b(vehicle|reversing|truck|forklift|crane)\b",
    "slip_trip": r"\b(slip|slipped|trip|tripped|wet|housekeeping)\b",
    "ppe": r"\b(ppe|helmet|gloves|harness|goggles)\b",
    "injury": r"\b(injury|injured|fracture|fractured|hospital|first aid|lost time|burn)\b",
    "near_miss": r"\bnear[- ]miss\b",
    "oil_chemical_spill": r"\b(oil|spill|seepage|leak|leaking|sf6|electrolyte|diesel|hazardous|contaminat)\w*",
    "water_use": r"\b(water usage|wastewater|overflow|valve leakage)\b",
    "biodiversity": r"\b(bird|species|vegetation|wildlife)\b",
    "waste": r"\b(waste|scrap|manifest|packaging)\b",
    "regulatory": r"\b(consent|permit|statutory|authority|regulatory|expired|overdue|unreported|not reported)\b",
    "documentation": r"\b(record|register|checklist|late|missing|signature|falsified|not uploaded|incomplete)\b",
    "heat_stress": r"\b(heat exhaustion|heat stress|heatstroke)\b",
}
TAG_RE = {k: re.compile(v, re.I) for k, v in RISK_RULES.items()}

# If any of these appear, severity can never be rated lower than "high".
CRITICAL_RE = re.compile(
    r"\b(fatal|fatality|lost time|fracture\w*|hospital|fire|evacuat\w*|arc fault|sf6|"
    r"electrical flash|fall from height|expired for|falsified|not reported to authority)\b",
    re.I,
)

ACTIONS = {
    "fire": "Inspect and test fire detection/suppression; run a fire drill; review hot-work and electrical housekeeping.",
    "electrical": "Re-verify lock-out/tag-out and arc-flash PPE; limit live work to authorised staff; retrain on energisation permits.",
    "work_at_height": "Enforce 100% harness tie-off; audit anchor points; add exclusion zones below overhead work.",
    "vehicle_movement": "Separate pedestrian and vehicle routes; use banksmen and reversing alarms; restrict night-shift vehicle movement near trenches.",
    "slip_trip": "Improve housekeeping and drainage on walkways; use anti-slip surfaces and cable covers; add daily walk-down checks.",
    "ppe": "Run a PPE compliance audit; supervisor spot checks; stop-work authority for non-compliance.",
    "injury": "Complete a root-cause investigation; share lessons learned in toolbox talks; review medical response readiness.",
    "near_miss": "Log and investigate near misses with the same rigour as injuries; track closure of corrective actions.",
    "oil_chemical_spill": "Provide secondary containment and spill kits; inspect seals/valves on a schedule; clean up and test the soil.",
    "water_use": "Fix leaking valves; meter water use per cleaning cycle; set alert thresholds for abnormal consumption.",
    "biodiversity": "Review the biodiversity management plan; consider curtailment or deterrents during sensitive periods.",
    "waste": "Segregate and label waste; maintain manifests; use only authorised disposal vendors.",
    "regulatory": "Escalate to compliance owner; renew consents/permits immediately; add expiry reminders to the compliance calendar.",
    "documentation": "Digitise records with mandatory fields; add reviewer sign-off and periodic audit sampling.",
    "heat_stress": "Introduce hydration and rest breaks; schedule heavy work outside peak heat; train supervisors on heat-illness signs.",
}
CATEGORY_DEFAULT_ACTION = {
    "safety": "Reinforce the safe-work procedure and supervision for this activity.",
    "environment": "Contain the release, document the impact and review the environmental controls.",
    "governance": "Review the control gap and assign an owner and deadline for closure.",
}


def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tag_risks(text: str) -> list[str]:
    return [tag for tag, rx in TAG_RE.items() if rx.search(text or "")]


def recommend(category: str, tags: list[str], severity: str) -> list[str]:
    actions = [ACTIONS[t] for t in tags if t in ACTIONS][:4]
    actions.append(CATEGORY_DEFAULT_ACTION.get(category, "Review controls and assign an owner."))
    if severity == "high":
        actions.insert(0, "Escalate to site head within 24 hours and start a formal investigation.")
    return list(dict.fromkeys(actions))


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    frames = []
    base = DATA_DIR / "safety_esg_incidents.csv"
    if base.exists():
        b = pd.read_csv(base)
        b["source"] = "bundled"
        frames.append(b)
    ext = DATA_DIR / "incidents_extended.csv"
    if ext.exists():
        frames.append(pd.read_csv(ext))
    df = pd.concat(frames, ignore_index=True)
    for col in ("site", "date"):
        if col not in df:
            df[col] = None
    df["site"] = df["site"].fillna("Unknown")
    df["narrative"] = df["narrative"].astype(str)
    df["clean"] = df["narrative"].map(clean_text)
    return df.dropna(subset=["target_category", "target_severity"]).reset_index(drop=True)


def _asset_of(text: str) -> str:
    assets = ["inverter station", "wind turbine", "cable trench", "transformer yard",
              "module cleaning", "gearbox", "battery", "control room", "access road",
              "diesel generator", "substation", "walkway", "turbine", "panel"]
    t = (text or "").lower()
    for a in assets:
        if a in t:
            return a
    return "other"


def _make_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000, C=10, class_weight="balanced")),
    ])


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class IncidentEngine:
    def __init__(self) -> None:
        self.df = load_data()
        self.df["tags"] = self.df["narrative"].map(tag_risks)
        self.df["asset"] = self.df["narrative"].map(_asset_of)
        self.cat_model = _make_model().fit(self.df["clean"], self.df["target_category"])
        self.sev_model = _make_model().fit(self.df["clean"], self.df["target_severity"])
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(self.df["clean"])
        self.matrix = self.vec.transform(self.df["clean"])
        self.df["pred_category"] = self.cat_model.predict(self.df["clean"])
        self.df["pred_severity"] = [
            self._guard(t, s) for t, s in zip(self.df["narrative"], self.sev_model.predict(self.df["clean"]))
        ]
        self._metrics = self._evaluate()

    # ---- inference ----------------------------------------------------
    @staticmethod
    def _guard(text: str, severity: str) -> str:
        if CRITICAL_RE.search(text or "") and SEV_ORDER[severity] < SEV_ORDER["high"]:
            return "high"
        return severity

    def _top_terms(self, model: Pipeline, cleaned: str, label: str, k: int = 5) -> list[str]:
        tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
        vec = tfidf.transform([cleaned])
        idx = list(clf.classes_).index(label)
        coefs = clf.coef_[idx] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        contrib = vec.multiply(coefs).toarray()[0]
        names = tfidf.get_feature_names_out()
        return [names[i] for i in np.argsort(contrib)[::-1][:k] if contrib[i] > 0]

    def similar(self, text: str, k: int = 3) -> list[dict]:
        sims = cosine_similarity(self.vec.transform([clean_text(text)]), self.matrix)[0]
        out = []
        for i in np.argsort(sims)[::-1][:k]:
            r = self.df.iloc[i]
            out.append({"incident_id": r["incident_id"], "narrative": r["narrative"],
                        "category": r["target_category"], "severity": r["target_severity"],
                        "similarity": round(float(sims[i]), 3)})
        return out

    def analyze(self, text: str) -> dict:
        cleaned = clean_text(text)
        if not cleaned:
            return {"error": "Empty narrative"}
        cat_proba = self.cat_model.predict_proba([cleaned])[0]
        sev_proba = self.sev_model.predict_proba([cleaned])[0]
        cat = self.cat_model.classes_[int(cat_proba.argmax())]
        raw_sev = self.sev_model.classes_[int(sev_proba.argmax())]
        sev = self._guard(text, raw_sev)
        tags = tag_risks(text)
        return {
            "input": text,
            "category": cat,
            "category_confidence": round(float(cat_proba.max()), 3),
            "severity": sev,
            "severity_model_output": raw_sev,
            "severity_rule_override": sev != raw_sev,
            "severity_confidence": round(float(sev_proba.max()), 3),
            "risk_tags": tags,
            "summary": self.summarise_one(text, cat, sev, tags),
            "recommended_actions": recommend(cat, tags, sev),
            "evidence": {
                "key_terms_category": self._top_terms(self.cat_model, cleaned, cat),
                "key_terms_severity": self._top_terms(self.sev_model, cleaned, raw_sev),
                "similar_incidents": self.similar(text),
            },
            "low_confidence": bool(cat_proba.max() < 0.5 or sev_proba.max() < 0.5),
            "limitations": "Trained on a very small bundled set plus synthetic data; treat as decision support, not a final judgement.",
        }

    # ---- summarisation --------------------------------------------------
    @staticmethod
    def summarise_one(text: str, cat: str, sev: str, tags: list[str]) -> str:
        first = re.split(r"[.;]", text.strip())[0].strip()
        risks = ", ".join(t.replace("_", " ") for t in tags[:3]) or "no specific risk keyword"
        return f"[{sev.upper()} {cat}] {first}. Key risks: {risks}."

    def management_summary(self) -> str:
        d = self.df
        tag_counts = Counter(t for ts in d["tags"] for t in ts)
        top_tags = ", ".join(f"{t.replace('_', ' ')} ({n})" for t, n in tag_counts.most_common(3))
        high = int((d["pred_severity"] == "high").sum())
        top_cat = d["pred_category"].value_counts().idxmax()
        worst = d[d["pred_severity"] == "high"]
        # extractive: the high-severity narrative closest to the centroid of high-severity narratives
        example = ""
        if len(worst):
            idx = worst.index
            centroid = np.asarray(self.matrix[idx].mean(axis=0))
            sims = cosine_similarity(centroid, self.matrix[idx])[0]
            example = worst.iloc[int(sims.argmax())]["narrative"]
        return (f"{len(d)} incidents analysed; {high} rated high severity. "
                f"'{top_cat}' is the largest category. Most frequent risk themes: {top_tags}. "
                f"Representative high-severity case: \"{example}\".")

    # ---- analytics ------------------------------------------------------
    def summary(self) -> dict:
        d = self.df
        tag_counts = Counter(t for ts in d["tags"] for t in ts)
        return {
            "total": int(len(d)),
            "by_category": d["pred_category"].value_counts().to_dict(),
            "by_severity": {s: int((d["pred_severity"] == s).sum()) for s in SEV_NAMES},
            "by_source": d["source"].value_counts().to_dict(),
            "top_risk_tags": [{"tag": t, "count": n} for t, n in tag_counts.most_common(8)],
            "management_summary": self.management_summary(),
        }

    def patterns(self, min_count: int = 3) -> dict:
        d = self.df.copy()
        rows = []
        for _, r in d.iterrows():
            for t in r["tags"] or ["none"]:
                rows.append((r["asset"], t, r["site"], r["pred_severity"], r["incident_id"]))
        p = pd.DataFrame(rows, columns=["asset", "tag", "site", "severity", "incident_id"])
        p = p[(p["tag"] != "none") & (p["asset"] != "other")]
        grp = p.groupby(["asset", "tag"]).agg(
            count=("incident_id", "nunique"),
            high=("severity", lambda s: int((s == "high").sum())),
            sites=("site", lambda s: sorted(set(s))[:4]),
            examples=("incident_id", lambda s: sorted(set(s))[:3]),
        ).reset_index()
        grp = grp[grp["count"] >= min_count].sort_values(["high", "count"], ascending=False).head(10)
        recurring = [
            {"asset": r["asset"], "risk": r["tag"], "count": int(r["count"]),
             "high_severity": int(r["high"]), "sites": r["sites"],
             "example_ids": r["examples"], "suggestion": ACTIONS.get(r["tag"], "")}
            for _, r in grp.iterrows()
        ]
        dates = pd.to_datetime(self.df["date"], errors="coerce")
        trend = (self.df.assign(month=dates.dt.strftime("%Y-%m")).dropna(subset=["month"])
                 .groupby(["month", "pred_severity"]).size().unstack(fill_value=0)
                 .reindex(columns=SEV_NAMES, fill_value=0))
        return {"recurring": recurring,
                "monthly_trend": [{"month": m, **{s: int(trend.loc[m, s]) for s in SEV_NAMES}}
                                  for m in trend.index]}

    def incidents(self, limit: int = 50) -> list[dict]:
        cols = ["incident_id", "narrative", "site", "date", "pred_category", "pred_severity",
                "target_category", "target_severity", "tags", "source"]
        out = self.df[cols].head(limit).to_dict("records")
        for r in out:
            r["date"] = None if pd.isna(r["date"]) else r["date"]
        return out

    # ---- evaluation -------------------------------------------------------
    def _evaluate(self) -> dict:
        d = self.df
        result: dict = {}
        for name, col in [("category", "target_category"), ("severity", "target_severity")]:
            y = d[col]
            n_splits = int(max(2, min(5, y.value_counts().min())))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
            pred = cross_val_predict(_make_model(), d["clean"], y, cv=cv)
            if name == "severity":
                pred = [self._guard(t, p) for t, p in zip(d["narrative"], pred)]
            labels = sorted(y.unique(), key=lambda s: SEV_ORDER.get(s, 0))
            result[name] = {
                "cv_folds": n_splits,
                "accuracy": round(float(accuracy_score(y, pred)), 3),
                "macro_f1": round(float(f1_score(y, pred, average="macro")), 3),
                "labels": labels,
                "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
            }
        # Harder test: train on synthetic only, test on the real bundled rows.
        real, syn = d[d["source"] == "bundled"], d[d["source"] == "synthetic"]
        if len(real) and len(syn):
            held = {}
            for name, col in [("category", "target_category"), ("severity", "target_severity")]:
                m = _make_model().fit(syn["clean"], syn[col])
                pred = m.predict(real["clean"])
                if name == "severity":
                    pred = [self._guard(t, p) for t, p in zip(real["narrative"], pred)]
                held[name] = {"n": int(len(real)), "accuracy": round(float(accuracy_score(real[col], pred)), 3)}
            result["real_holdout_train_on_synthetic"] = held
        result["notes"] = [
            "Cross-validation is optimistic: synthetic rows come from templates, so near-duplicates leak across folds.",
            "The real-holdout figure uses only the few bundled rows and is a noisy, small-sample estimate.",
            "Severity includes a rule floor for critical keywords (fatal, fire, lost time, ...).",
        ]
        return result

    def metrics(self) -> dict:
        return self._metrics


_engine: IncidentEngine | None = None


def get_engine() -> IncidentEngine:
    global _engine
    if _engine is None:
        _engine = IncidentEngine()
    return _engine
