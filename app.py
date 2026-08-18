"""
Production-Ready FastAPI Server for Fake News Detection using TensorFlow.
Provides REST API endpoints, real-time token saliency explainability, benchmark comparisons,
and dynamic shareable local network link / QR code generator.
"""

import os
import io
import json
import base64
import socket
import re
import time
import qrcode
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from explainer import analyze_linguistics, compute_token_saliency

# Configuration
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_fake_news_model.keras")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
DATASET_STATS_PATH = os.path.join(DATA_DIR, "dataset_stats.json")

PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")

# Initialize FastAPI App
app = FastAPI(
    title="Fake News Detection System",
    description="Deep Learning Fake News Detection Engine Powered by TensorFlow",
    version="1.0.0"
)

# Security: CORS Policy
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Middleware: Security Headers & Request Limiting
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Max content length enforcement (1 MB limit)
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1048576:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"detail": "Payload too large. Maximum allowed size is 1MB."}
        )
    
    response = await call_next(request)
    
    # Secure HTTP Headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# Serve static files
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global ML Model & Vectorizer Holders
model = None
vectorizer = None
vocab = []
metrics_data = {}
dataset_stats = {}

def get_local_ip():
    """Detects the current host IP address on the local network."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def load_resources():
    """Loads trained TensorFlow model and vectorizer vocabulary."""
    global model, vectorizer, vocab, metrics_data, dataset_stats
    print("Loading TensorFlow model and NLP pipeline...")
    
    # Load Vocabulary
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            
        vectorizer = keras.layers.TextVectorization(
            max_tokens=5000,
            output_mode='int',
            output_sequence_length=150,
            standardize='lower_and_strip_punctuation',
            vocabulary=vocab
        )
        print(f"Loaded TextVectorization layer with {len(vocab)} tokens.")
        
    # Load Trained Model
    if os.path.exists(BEST_MODEL_PATH):
        model = keras.models.load_model(BEST_MODEL_PATH)
        print("TensorFlow model loaded successfully.")
    else:
        print("Warning: Model file not found. Please run model_training.py first.")
        
    # Load Metrics
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
            
    # Load Dataset Stats
    if os.path.exists(DATASET_STATS_PATH):
        with open(DATASET_STATS_PATH, "r", encoding="utf-8") as f:
            dataset_stats = json.load(f)

@app.on_event("startup")
def startup_event():
    load_resources()

# Pydantic Request Models
class NewsItem(BaseModel):
    title: str = Field(..., max_length=500, description="Headline or title of the news article")
    text: str = Field(..., max_length=10000, description="Main text body of the news article")
    source: Optional[str] = Field(None, max_length=200, description="Optional news publisher or source URL")

class BatchNewsRequest(BaseModel):
    articles: List[NewsItem] = Field(..., max_length=50, description="List of news articles to scan in batch")

# Sample Database for Quick Testing
SAMPLE_ARTICLES = [
    {
        "id": "sample-1",
        "title": "NASA James Webb Space Telescope Discovers Ancient Galaxy Formed Shortly After Big Bang",
        "text": "Astronomers using the James Webb Space Telescope have identified one of the earliest galaxies ever observed, dating back to approximately 350 million years after the Big Bang. The spectroscopic data confirmed high redshift signatures, revealing surprisingly luminous star formation in the early universe. Lead researchers from the international astrophysics collaboration published their peer-reviewed findings in the Astrophysical Journal.",
        "category": "Science",
        "expected": "REAL",
        "badge": "Verified Real"
    },
    {
        "id": "sample-2",
        "title": "SHOCKING PROOF: Secret 5G Towers Are Broadcasting Mind-Control Frequencies to Subjugate Citizens!",
        "text": "BOMBSHELL report reveals that 5G cellular antennas are not for internet speeds at all, but are military-grade mind manipulation devices engineered by global elites. Whistleblowers claim secret frequencies cause sudden fatigue, unquestioning obedience, and memory erasure. Spread this everywhere before the deep state deletes this video!",
        "category": "Conspiracy",
        "expected": "FAKE",
        "badge": "Sensational Fake"
    },
    {
        "id": "sample-3",
        "title": "Federal Reserve Holds Benchmark Interest Rates Steady Amid Cooling Inflation Data",
        "text": "The Federal Reserve announced on Wednesday that it will maintain its benchmark interest rate within the current target range. Policy makers cited steady job growth and a continuing decline in core consumer prices over the past three quarters. Chair Jerome Powell indicated in a press conference that future monetary adjustments will remain strictly data-dependent based on incoming economic indicators.",
        "category": "Finance",
        "expected": "REAL",
        "badge": "Verified Real"
    },
    {
        "id": "sample-4",
        "title": "MIRACLE CURE: Doctors Are BANNED from Telling You This One Kitchen Spice Instantly Destroys All Cancers!",
        "text": "Big Pharma is panicking! A revolutionary secret discovered in ancient Himalayan caves reveals that mixing organic turmeric with crushed apple seeds cures 100 percent of terminal stage-4 cancers within 48 hours. Corrupt medical boards are threatening any doctor who speaks the truth with immediate jail time. Order the miracle tincture now!",
        "category": "Health Hoax",
        "expected": "FAKE",
        "badge": "Medical Hoax"
    },
    {
        "id": "sample-5",
        "title": "International Diplomatic Summit Concludes with Bilateral Maritime Safety Accord",
        "text": "Delegates from fourteen Pacific Rim nations concluded three days of diplomatic negotiations in Geneva today by signing a comprehensive maritime safety framework. The agreement outlines standardized communication channels for commercial vessels and establishes joint search-and-rescue protocols in international waters.",
        "category": "World News",
        "expected": "REAL",
        "badge": "Verified Real"
    }
]

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main single-page application interface."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Fake News Detector Backend Running. Build static/index.html to view UI.</h1>")

@app.post("/api/predict")
async def predict_news(item: NewsItem):
    """
    Performs real-time deep learning inference, extracts word saliency,
    and analyzes linguistic features for the submitted news article.
    """
    if not model or not vectorizer:
        load_resources()
        if not model or not vectorizer:
            raise HTTPException(status_code=503, detail="TensorFlow model is currently initializing. Please try again.")

    title = item.title.strip()
    text = item.text.strip()
    
    if not title and not text:
        raise HTTPException(status_code=400, detail="Title or text content must be provided.")
        
    combined_text = f"{title} - {text}".strip()
    
    start_time = time.time()
    
    # Vectorize input text
    seq = vectorizer(tf.constant([combined_text])).numpy()
    
    # Predict with TensorFlow
    pred_raw = model.predict(seq, verbose=0)
    fake_prob = float(pred_raw[0][0])
    real_prob = 1.0 - fake_prob
    inference_ms = (time.time() - start_time) * 1000
    
    # Classification Logic
    is_fake = fake_prob >= 0.50
    verdict = "FAKE" if is_fake else "REAL"
    confidence = fake_prob if is_fake else real_prob
    
    # Linguistic & Saliency Analysis
    linguistics = analyze_linguistics(title, text)
    saliency = compute_token_saliency(combined_text, fake_prob)
    
    # Risk Level Assessment
    if fake_prob >= 0.85:
        risk_level = "CRITICAL"
        risk_description = "High probability of deceptive misinformation or coordinated conspiracy narrative."
    elif fake_prob >= 0.55:
        risk_level = "MODERATE"
        risk_description = "Moderate risk of sensationalism, unverified claims, or clickbait exaggeration."
    elif fake_prob >= 0.20:
        risk_level = "LOW"
        risk_description = "Predominantly factual language with typical journalistic framing."
    else:
        risk_level = "MINIMAL"
        risk_description = "High consistency with empirical reporting, institutional attribution, and verified facts."

    return {
        "verdict": verdict,
        "is_fake": is_fake,
        "fake_probability": round(fake_prob * 100, 2),
        "real_probability": round(real_prob * 100, 2),
        "confidence": round(confidence * 100, 1),
        "risk_level": risk_level,
        "risk_description": risk_description,
        "inference_latency_ms": round(inference_ms, 2),
        "model_architecture": metrics_data.get("best_model_name", "BiLSTM with Attention"),
        "linguistics": linguistics,
        "saliency_tokens": saliency
    }

@app.post("/api/batch-predict")
async def batch_predict(batch: BatchNewsRequest):
    """Performs fast vectorized batch inference for multiple articles."""
    if not model or not vectorizer:
        load_resources()
        if not model or not vectorizer:
            raise HTTPException(status_code=503, detail="TensorFlow model is currently initializing.")
            
    items = batch.articles
    if not items:
        return {"results": [], "summary": {}}
        
    combined_texts = [f"{item.title} - {item.text}".strip() for item in items]
    
    start_time = time.time()
    seqs = vectorizer(tf.constant(combined_texts)).numpy()
    preds = model.predict(seqs, verbose=0).ravel()
    total_time_ms = (time.time() - start_time) * 1000
    
    results = []
    fake_count = 0
    real_count = 0
    
    for i, item in enumerate(items):
        f_prob = float(preds[i])
        r_prob = 1.0 - f_prob
        is_f = f_prob >= 0.50
        if is_f:
            fake_count += 1
        else:
            real_count += 1
            
        results.append({
            "title": item.title,
            "verdict": "FAKE" if is_f else "REAL",
            "fake_probability": round(f_prob * 100, 1),
            "real_probability": round(r_prob * 100, 1),
            "confidence": round((f_prob if is_f else r_prob) * 100, 1)
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
    """Returns model benchmark comparison data, training curves, and confusion matrices."""
    global metrics_data
    if not metrics_data and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    return metrics_data

@app.get("/api/samples")
async def get_samples():
    """Returns pre-loaded sample news items."""
    return SAMPLE_ARTICLES

@app.get("/api/dataset-stats")
async def get_dataset_stats():
    """Returns dataset distributions, class balances, and vocabulary info."""
    global dataset_stats
    if not dataset_stats and os.path.exists(DATASET_STATS_PATH):
        with open(DATASET_STATS_PATH, "r", encoding="utf-8") as f:
            dataset_stats = json.load(f)
    return dataset_stats

@app.get("/api/share-info")
async def get_share_info():
    """Generates local network URL and QR code for public/LAN sharing."""
    local_ip = get_local_ip()
    local_url = f"http://localhost:{PORT}"
    network_url = f"http://{local_ip}:{PORT}"
    
    # Generate QR Code for network URL
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=8,
        border=2,
    )
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
    """Health check endpoint for deployment monitoring."""
    return {
        "status": "healthy",
        "tensorflow_loaded": model is not None,
        "model_path": BEST_MODEL_PATH,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
