"""
High-Precision FastAPI Server for VeritasAI Fake News Detection.
Combines Deep Learning Neural Networks (TensorFlow/Keras) with N-Gram Classifiers,
Token Saliency Explainability, and Instant LAN/Mobile QR Sharing.

v3.0 Changes:
  - Batch endpoint now uses IDENTICAL ensemble weights as single predict
  - Heuristic replaced with continuous sigmoid-based signal (no more 3-state cliff)
  - FastAPI lifespan replaces deprecated on_event("startup")
  - Input validation: min word count check + uncertainty_flag in response
  - is_grey_zone forwarded from explainer into response
"""

import os
import io
import json
import base64
import socket
import re
import time
import pickle
import math
import qrcode
import numpy as np
import tensorflow as tf
from tensorflow import keras
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

from explainer import analyze_linguistics, compute_token_saliency

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_fake_news_model.keras")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.json")
TFIDF_VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
TFIDF_MOD_PATH = os.path.join(MODELS_DIR, "tfidf_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
DATASET_STATS_PATH = os.path.join(DATA_DIR, "dataset_stats.json")

PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")

# Ensemble weights (SINGLE source of truth — used by BOTH endpoints)
W_DEEP  = 0.55   # Deep neural network weight
W_TFIDF = 0.35   # TF-IDF N-gram classifier weight
W_HEUR  = 0.10   # Heuristic / linguistic signal weight
MIN_WORD_COUNT = 10  # Below this → warn + lower confidence

# ──────────────────────────────────────────────────────────────
# Resources (module-level singletons)
# ──────────────────────────────────────────────────────────────
tf_model       = None
vectorizer     = None
tfidf_vectorizer = None
tfidf_classifier = None
metrics_data   = None
dataset_stats  = None


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def load_resources():
    global tf_model, vectorizer, tfidf_vectorizer, tfidf_classifier, metrics_data, dataset_stats
    print("Initializing VeritasAI Neural Engines...")

    # 1. Keras TextVectorization layer
    if os.path.exists(VOCAB_PATH) and os.path.exists(BEST_MODEL_PATH):
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        vec_layer = keras.layers.TextVectorization(
            max_tokens=len(vocab),
            output_mode="int",
            output_sequence_length=220,
            standardize="lower_and_strip_punctuation"
        )
        vec_layer.adapt(tf.constant(["warmup"]))
        vec_layer.set_vocabulary(vocab)
        vectorizer = vec_layer
        print(f"Keras Vectorizer loaded with {len(vocab)} tokens.")

    # 2. TensorFlow deep model
    if os.path.exists(BEST_MODEL_PATH):
        tf_model = keras.models.load_model(BEST_MODEL_PATH)
        print("TensorFlow Deep Learning model loaded.")

    # 3. TF-IDF N-gram Calibrated Classifier
    if os.path.exists(TFIDF_VEC_PATH) and os.path.exists(TFIDF_MOD_PATH):
        with open(TFIDF_VEC_PATH, "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        with open(TFIDF_MOD_PATH, "rb") as f:
            tfidf_classifier = pickle.load(f)
        print("TF-IDF N-Gram calibrated classifier loaded.")

    # 4. Metrics & Dataset Stats
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    if os.path.exists(DATASET_STATS_PATH):
        with open(DATASET_STATS_PATH, "r", encoding="utf-8") as f:
            dataset_stats = json.load(f)


# ──────────────────────────────────────────────────────────────
# Lifespan (replaces deprecated on_event)
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield

# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="VeritasAI - High-Accuracy Fake News Detection Engine",
    description="Multi-Model Deep Learning NLP System for Misinformation Identification",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS Policy
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1048576:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"detail": "Payload too large. Maximum allowed size is 1MB."}
        )
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Static Files
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ──────────────────────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────────────────────
class NewsItem(BaseModel):
    title: str = Field(..., max_length=500, description="Headline or title")
    text: str = Field(..., max_length=15000, description="Article body text")
    source: Optional[str] = Field(None, max_length=200)

    @field_validator("title", "text", mode="before")
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v


class BatchNewsRequest(BaseModel):
    articles: List[NewsItem] = Field(..., max_length=50)


# ──────────────────────────────────────────────────────────────
# Ensemble helper — continuous heuristic (replaces 3-state cliff)
# ──────────────────────────────────────────────────────────────
def _compute_heuristic(sens_score: float, cred_score: float) -> float:
    """
    Continuous heuristic probability based on sensationalism vs credibility scores.
    Uses a sigmoid-like blend so there's no abrupt cliff between states.
    Range: [0.05, 0.95]
    """
    net = sens_score - cred_score          # positive = more fake
    # Sigmoid: 1 / (1 + e^(-net/20))  scaled to [0.05, 0.95]
    raw = 1.0 / (1.0 + math.exp(-net / 20.0))
    return 0.05 + 0.90 * raw              # clamp to [0.05, 0.95]


def _ensemble(tf_raw: float, tfidf_raw: float, heuristic: float) -> float:
    """Single source-of-truth ensemble formula used by ALL endpoints."""
    p = W_DEEP * tf_raw + W_TFIDF * tfidf_raw + W_HEUR * heuristic
    return max(0.001, min(0.999, p))


def _risk_level(fake_prob: float) -> tuple[str, str]:
    if fake_prob >= 0.80:
        return "CRITICAL", "High probability of deceptive misinformation, fabricated claims, or conspiracy propaganda."
    elif fake_prob >= 0.65:
        return "HIGH", "Elevated risk of sensationalism, unverified claims, or manipulative framing."
    elif fake_prob >= 0.50:
        return "MODERATE", "Moderate risk — some sensational language or unverified claims present."
    elif fake_prob >= 0.25:
        return "LOW", "Predominantly factual reporting with standard journalistic framing."
    else:
        return "MINIMAL", "High consistency with empirical evidence, peer review, and institutional attribution."


# ──────────────────────────────────────────────────────────────
# Curated Samples
# ──────────────────────────────────────────────────────────────
SAMPLE_ARTICLES = [
    {
        "id": "sample-1",
        "title": "NASA James Webb Space Telescope Identifies Earliest Known Galaxy Clusters in Deep Field Survey",
        "text": "Astronomers utilizing infrared spectroscopy aboard the James Webb Space Telescope have confirmed the discovery of high-redshift galaxy clusters formed approximately 350 million years after the Big Bang. The international research team, led by astrophysicists from NASA, ESA, and CSA, published their peer-reviewed findings in the Astrophysical Journal following rigorous photometric calibration and spectral verification.",
        "category": "Science & Astronomy",
        "expected": "REAL",
        "badge": "Verified Real"
    },
    {
        "id": "sample-2",
        "title": "SHOCKING PROOF: Secret 5G Towers Are Broadcasting Mind-Control Frequencies to Enslave Citizens!",
        "text": "BOMBSHELL report reveals that 5G cellular antennas are not for high-speed internet at all, but are military-grade electromagnetic mind manipulation devices engineered by shadow globalist elites. Whistleblowers claim secret frequencies cause sudden docility, memory wipeouts, and obedience. Share this everywhere before the deep state deletes this video!",
        "category": "Conspiracy / Hoax",
        "expected": "FAKE",
        "badge": "5G Conspiracy"
    },
    {
        "id": "sample-3",
        "title": "Federal Reserve Holds Benchmark Interest Rates at 5.25 Percent Amid Moderating Core Inflation",
        "text": "The Federal Open Market Committee announced Wednesday it will maintain the federal funds target range at 5.25 to 5.50 percent. Chair Jerome Powell stated during the press conference that consumer price inflation has slowed to 2.8 percent annualized, while labor market participation remains stable at 63.3 percent.",
        "category": "Finance & Economics",
        "expected": "REAL",
        "badge": "Verified Real"
    },
    {
        "id": "sample-4",
        "title": "MIRACLE CURE: Doctors Are BANNED from Telling You This One Kitchen Spice Instantly Destroys All Cancers!",
        "text": "Big Pharma is panicking! A revolutionary secret discovered in ancient Himalayan caves reveals that mixing organic turmeric with crushed apple seeds cures 100 percent of terminal stage-4 cancers within 48 hours guaranteed. Corrupt medical boards are threatening any honest doctor who speaks the truth with immediate jail time. Order the miracle tincture now!",
        "category": "Health / Pseudo-Cure",
        "expected": "FAKE",
        "badge": "Health Hoax"
    },
    {
        "id": "sample-5",
        "title": "FDA Grants Full Approval to Novel Monoclonal Antibody for Early-Stage Alzheimer's Disease",
        "text": "The U.S. Food and Drug Administration (FDA) has granted traditional approval for lecanemab, a monoclonal antibody treatment targeting amyloid-beta plaques in adults with mild cognitive impairment. In a randomized, double-blind Phase III clinical trial involving 1,795 participants over 18 months, the therapeutic demonstrated a statistically significant 27 percent reduction in clinical cognitive decline compared to placebo.",
        "category": "Medicine / Biotech",
        "expected": "REAL",
        "badge": "Verified Real"
    },
    {
        "id": "sample-6",
        "title": "LEAK: Deep State Planning Worldwide 30-Day Grid Blackout to Confiscate All Cash and Enforce Digital Token!",
        "text": "An anonymous high-level intelligence officer has warned that a planned global power grid shutdown is scheduled for next month. During the blackout, all physical currencies will be declared null and void, replaced with a mandatory digital biometric CBDC chip. Stock up on canned beans and silver coins immediately!",
        "category": "Financial Panic",
        "expected": "FAKE",
        "badge": "Financial Scam"
    }
]


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>VeritasAI Backend Running.</h1>")


@app.post("/api/predict")
async def predict_news(item: NewsItem):
    """
    High-Precision Ensemble Prediction combining Deep Neural Activations,
    N-Gram Subword Classifiers, Continuous Linguistic Heuristics, and Token Saliency.
    """
    if not tf_model or not vectorizer:
        load_resources()
        if not tf_model or not vectorizer:
            raise HTTPException(status_code=503, detail="Neural models initializing. Please retry in a few seconds.")

    title = item.title.strip()
    text  = item.text.strip()

    if not title and not text:
        raise HTTPException(status_code=400, detail="Please provide headline or body content.")

    combined_text = f"{title} - {text}".strip()

    # Input quality check
    word_count = len(combined_text.split())
    too_short = word_count < MIN_WORD_COUNT

    start_time = time.time()

    # 1. Deep Neural Prediction (Keras/TensorFlow)
    seq = vectorizer(tf.constant([combined_text])).numpy()
    tf_pred_raw = float(tf_model.predict(seq, verbose=0)[0][0])

    # 2. N-Gram Calibrated Subword Classifier Prediction
    if tfidf_classifier and tfidf_vectorizer:
        tfidf_feat = tfidf_vectorizer.transform([combined_text])
        tfidf_pred_raw = float(tfidf_classifier.predict_proba(tfidf_feat)[0][1])
    else:
        tfidf_pred_raw = tf_pred_raw

    # 3. Continuous Linguistic Heuristic (replaces old 3-state cliff)
    linguistics = analyze_linguistics(title, text)
    sens_score = linguistics["sensationalism_score"]
    cred_score = linguistics["credibility_marker_score"]
    heuristic_prob = _compute_heuristic(sens_score, cred_score)

    # 4. Calibrated Ensemble Probability (SINGLE formula for all endpoints)
    fake_prob = _ensemble(tf_pred_raw, tfidf_pred_raw, heuristic_prob)
    real_prob = 1.0 - fake_prob

    inference_ms = (time.time() - start_time) * 1000

    # Verdict & Confidence
    is_fake = fake_prob >= 0.50
    verdict = "FAKE" if is_fake else "REAL"
    confidence = (fake_prob if is_fake else real_prob) * 100.0

    # Uncertainty flag (borderline predictions)
    uncertainty_flag = 40.0 <= (fake_prob * 100) <= 65.0

    # Degrade confidence for very short inputs
    if too_short:
        confidence = min(confidence, 65.0)

    # Risk Assessment
    risk_level, risk_description = _risk_level(fake_prob)

    # Saliency Tokens
    saliency = compute_token_saliency(combined_text, fake_prob)

    return {
        "verdict": verdict,
        "is_fake": is_fake,
        "fake_probability": round(fake_prob * 100, 2),
        "real_probability": round(real_prob * 100, 2),
        "confidence": round(confidence, 1),
        "uncertainty_flag": uncertainty_flag,
        "too_short_warning": too_short,
        "word_count": word_count,
        "risk_level": risk_level,
        "risk_description": risk_description,
        "inference_latency_ms": round(inference_ms, 2),
        "model_architecture": "Ensemble (Deep BiLSTM + Self-Attention + TF-IDF Trigram) v3",
        "linguistics": linguistics,
        "saliency_tokens": saliency,
        "component_scores": {
            "deep_neural": round(tf_pred_raw * 100, 1),
            "tfidf_ngram": round(tfidf_pred_raw * 100, 1),
            "linguistic_heuristic": round(heuristic_prob * 100, 1),
        }
    }


@app.post("/api/batch-predict")
async def batch_predict(batch: BatchNewsRequest):
    """
    Vectorized Parallel Batch Inference.
    Uses IDENTICAL ensemble weights as single /api/predict.
    """
    if not tf_model or not vectorizer:
        load_resources()
        if not tf_model or not vectorizer:
            raise HTTPException(status_code=503, detail="Neural models initializing.")

    items = batch.articles
    if not items:
        return {"results": [], "summary": {}}

    combined_texts = [f"{item.title} - {item.text}".strip() for item in items]
    start_time = time.time()

    seqs = vectorizer(tf.constant(combined_texts)).numpy()
    tf_preds = tf_model.predict(seqs, verbose=0).ravel()

    if tfidf_classifier and tfidf_vectorizer:
        tfidf_feats = tfidf_vectorizer.transform(combined_texts)
        tfidf_preds = tfidf_classifier.predict_proba(tfidf_feats)[:, 1]
    else:
        tfidf_preds = tf_preds

    total_time_ms = (time.time() - start_time) * 1000

    results = []
    fake_count = 0
    real_count = 0

    for i, item in enumerate(items):
        # Use SAME heuristic per article
        ling = analyze_linguistics(item.title, item.text)
        heur = _compute_heuristic(ling["sensationalism_score"], ling["credibility_marker_score"])

        f_p = _ensemble(float(tf_preds[i]), float(tfidf_preds[i]), heur)
        r_p = 1.0 - f_p
        is_f = f_p >= 0.50

        if is_f:
            fake_count += 1
        else:
            real_count += 1

        results.append({
            "title": item.title,
            "verdict": "FAKE" if is_f else "REAL",
            "fake_probability": round(f_p * 100, 1),
            "real_probability": round(r_p * 100, 1),
            "confidence": round((f_p if is_f else r_p) * 100, 1),
            "uncertainty_flag": 40.0 <= (f_p * 100) <= 65.0,
        })

    return {
        "results": results,
        "summary": {
            "total": len(items),
            "fake_count": fake_count,
            "real_count": real_count,
            "fake_percentage": round((fake_count / len(items)) * 100, 1),
            "batch_latency_ms": round(total_time_ms, 2),
            "avg_ms_per_article": round(total_time_ms / len(items), 2)
        }
    }


@app.get("/api/benchmark")
async def get_benchmark():
    global metrics_data
    if not metrics_data and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    return metrics_data


@app.get("/api/samples")
async def get_samples():
    return SAMPLE_ARTICLES


@app.get("/api/dataset-stats")
async def get_dataset_stats():
    global dataset_stats
    if not dataset_stats and os.path.exists(DATASET_STATS_PATH):
        with open(DATASET_STATS_PATH, "r", encoding="utf-8") as f:
            dataset_stats = json.load(f)
    return dataset_stats


@app.get("/api/share-info")
async def get_share_info():
    local_ip = get_local_ip()
    local_url = f"http://localhost:{PORT}"
    network_url = f"http://{local_ip}:{PORT}"

    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(network_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#0f172a", back_color="#ffffff")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "local_url": local_url,
        "network_url": network_url,
        "local_ip": local_ip,
        "port": PORT,
        "qr_code_base64": f"data:image/png;base64,{qr_base64}",
        "tunnel_instructions": {
            "ngrok": f"ngrok http {PORT}",
            "localtunnel": f"npx localtunnel --port {PORT}",
            "cloudflared": f"cloudflared tunnel --url {local_url}"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "tensorflow_loaded": tf_model is not None,
        "tfidf_loaded": tfidf_classifier is not None,
        "version": "3.0.0",
        "ensemble_weights": {"deep": W_DEEP, "tfidf": W_TFIDF, "heuristic": W_HEUR},
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
