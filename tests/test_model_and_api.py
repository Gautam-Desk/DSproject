"""
Comprehensive Test Suite for Fake News Detection System.
Tests data integrity, TensorFlow model inferences, token saliency, and FastAPI REST endpoints.
"""

import sys
import os
import json
import pytest
import numpy as np
import pandas as pd

# Ensure root directory is on Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import tensorflow as tf
from tensorflow import keras
from fastapi.testclient import TestClient

from app import app, load_resources, get_local_ip
from explainer import analyze_linguistics, compute_token_saliency

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI TestClient and ensures model resources are loaded."""
    load_resources()
    with TestClient(app) as test_client:
        yield test_client

# ==========================================
# 1. Data Integrity Tests
# ==========================================
def test_dataset_files_exist():
    """Validates that all required dataset CSVs and metadata exist."""
    assert os.path.exists(os.path.join(DATA_DIR, "news_dataset.csv"))
    assert os.path.exists(os.path.join(DATA_DIR, "train.csv"))
    assert os.path.exists(os.path.join(DATA_DIR, "val.csv"))
    assert os.path.exists(os.path.join(DATA_DIR, "test.csv"))
    assert os.path.exists(os.path.join(DATA_DIR, "dataset_stats.json"))

def test_dataset_schema_and_distribution():
    """Validates dataset schema, non-null values, and balanced distribution."""
    df = pd.read_csv(os.path.join(DATA_DIR, "news_dataset.csv"))
    assert "title" in df.columns
    assert "text" in df.columns
    assert "label" in df.columns
    assert "cleaned_text" in df.columns
    assert df["label"].isin([0, 1]).all()
    assert df["cleaned_text"].isnull().sum() == 0
    assert len(df) >= 1000

    # Verify train/val/test split
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert abs((train_df["label"] == 0).mean() - 0.50) < 0.10

# ==========================================
# 2. Model & Serialization Tests
# ==========================================
def test_model_files_exist():
    """Checks that model weights, vocab, and metrics are saved."""
    assert os.path.exists(os.path.join(MODELS_DIR, "best_fake_news_model.keras"))
    assert os.path.exists(os.path.join(MODELS_DIR, "vocab.json"))
    assert os.path.exists(os.path.join(MODELS_DIR, "tokenizer_config.json"))
    assert os.path.exists(os.path.join(MODELS_DIR, "metrics.json"))

def test_model_loading_and_shape():
    """Loads the trained Keras model and validates input/output shapes."""
    model_path = os.path.join(MODELS_DIR, "best_fake_news_model.keras")
    model = keras.models.load_model(model_path)
    assert model is not None
    assert model.input_shape in [(None, 150), (None, 160), (None, 220)]
    assert model.output_shape == (None, 1)

def test_vocabulary_integrity():
    """Verifies vocabulary file contains expected tokens and length."""
    with open(os.path.join(MODELS_DIR, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    assert isinstance(vocab, list)
    assert len(vocab) > 500
    assert "" in vocab or "[UNK]" in vocab or "the" in vocab

# ==========================================
# 3. Model Inference & Robustness Tests
# ==========================================
def test_inference_on_real_news(client):
    """Tests model prediction on verified empirical news."""
    payload = {
        "title": "NASA James Webb Space Telescope Discovers Ancient Galaxy",
        "text": "Astronomers identified redshift signatures in the early universe, published in Astrophysical Journal."
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "REAL"
    assert data["real_probability"] > 80.0
    assert data["is_fake"] is False
    assert data["risk_level"] in ["MINIMAL", "LOW"]

def test_inference_on_fake_news(client):
    """Tests model prediction on sensational hoax news."""
    payload = {
        "title": "SHOCKING BOMBSHELL: Secret 5G Towers Mind Control Broadcast!",
        "text": "Secret frequencies cause sudden docility and memory wipeout. Whistleblowers expose deep state nanobots!"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "FAKE"
    assert data["fake_probability"] > 80.0
    assert data["is_fake"] is True
    assert data["risk_level"] in ["CRITICAL", "MODERATE"]

def test_inference_boundary_values(client):
    """Tests edge cases such as short text, numbers, and special characters."""
    payload = {
        "title": "Global Semiconductor Update 2026",
        "text": "Manufacturing facility reached 95% efficiency."
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["fake_probability"] <= 100.0
    assert 0.0 <= data["real_probability"] <= 100.0
    assert round(data["fake_probability"] + data["real_probability"], 1) == 100.0

# ==========================================
# 4. Explainability & Linguistic Engine Tests
# ==========================================
def test_linguistic_feature_extraction():
    """Verifies linguistic scores and trigger detection."""
    title = "BOMBSHELL ALERT: Miracle Cure Discovered!"
    text = "Doctors are banned from telling you this secret! Wake up now!"
    ling = analyze_linguistics(title, text)
    assert ling["sensationalism_score"] > 20.0
    assert ling["exclamation_count"] >= 2
    assert "bombshell" in ling["detected_sensational_terms"] or "miracle" in ling["detected_sensational_terms"]
    assert ling["word_count"] > 5

def test_token_saliency_computation():
    """Verifies token saliency attribution returns structured weights."""
    text = "NASA researchers published peer-reviewed findings in clinical trial"
    saliency = compute_token_saliency(text, fake_prob=0.05)
    assert len(saliency) > 0
    tokens = [item["token"] for item in saliency]
    assert "NASA" in tokens or "researchers" in tokens
    for item in saliency:
        assert -100.0 <= item["weight"] <= 100.0
        assert item["class"] in ["strongly-fake", "moderately-fake", "neutral", "moderately-real", "strongly-real"]

# ==========================================
# 5. REST API Endpoint Tests
# ==========================================
def test_api_health_endpoint(client):
    """Tests /health endpoint status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["tensorflow_loaded"] is True

def test_api_samples_endpoint(client):
    """Tests /api/samples returns benchmark articles."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) >= 5
    assert "title" in samples[0]
    assert "badge" in samples[0]

def test_api_benchmark_endpoint(client):
    """Tests /api/benchmark returns valid multi-model comparisons."""
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    benchmarks = response.json()
    assert "models" in benchmarks
    models = benchmarks["models"]
    assert "Self-Attention Transformer" in models or "BiLSTM with Attention" in models
    for m in models.values():
        assert m["accuracy"] >= 0.85
        assert m["f1_score"] >= 0.85
        assert "confusion_matrix" in m

def test_api_batch_predict_endpoint(client):
    """Tests /api/batch-predict endpoint with multiple articles."""
    payload = {
        "articles": [
            {
                "title": "Astronomers discover ancient stars with spectroscopy",
                "text": "The research was published after peer-reviewed investigation."
            },
            {
                "title": "BOMBSHELL: 5G mind control frequencies!",
                "text": "They don't want you to know the truth about the secret nanobots!"
            }
        ]
    }
    response = client.post("/api/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert "summary" in data
    assert data["summary"]["total"] == 2
    assert data["summary"]["real_count"] == 1
    assert data["summary"]["fake_count"] == 1

def test_api_share_info_endpoint(client):
    """Tests /api/share-info returns network IP and base64 QR code."""
    response = client.get("/api/share-info")
    assert response.status_code == 200
    data = response.json()
    assert "network_url" in data
    assert "local_url" in data
    assert "qr_code_base64" in data
    assert data["qr_code_base64"].startswith("data:image/png;base64,")

def test_api_validation_error_handling(client):
    """Tests rejection of empty payloads or invalid JSON."""
    response = client.post("/api/predict", json={"title": "", "text": ""})
    assert response.status_code == 400
