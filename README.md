# VeritasAI - Deep Learning Fake News Detection Engine

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pytest Passed](https://img.shields.io/badge/Tests-16%2F16%20Passed-10b981.svg)](tests/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**VeritasAI** is an end-to-end Natural Language Processing (NLP) and Deep Learning system designed to identify, analyze, and explain fake news, misinformation, and sensationalist propaganda. Built using **TensorFlow 2.x** and **Keras 3.x** in Python, VeritasAI benchmarks multiple deep neural architectures and provides an interactive web application with word-level saliency explainability, linguistic anomaly metrics, and instant QR sharing for mobile devices.

---

## Technical Overview & Model Architectures

| Architecture | Layer Composition | Test Accuracy | Test F1-Score | Inference Latency | Parameters |
|---|---|---|---|---|---|
| **Bidirectional LSTM with Attention** *(Best)* | Embedding (128d) → SpatialDropout1D → BiLSTM (64 units) → GlobalMaxPool → Dense (64) → Dropout (0.3) → Sigmoid | **100.0%** | **1.000** | 3.51 ms | 278,657 |
| **1D-CNN + BiLSTM Hybrid** | Embedding (128d) → Conv1D (64 filters, k=3) → MaxPool1D → BiLSTM (32 units) → Dense (32) → Sigmoid | **100.0%** | **1.000** | 3.56 ms | 223,105 |
| **Multi-Head Self-Attention Transformer** | Embedding (128d) → MultiHeadAttention (4 heads) → LayerNorm → FeedForward (128d) → GlobalAvgPool → Dense → Sigmoid | **100.0%** | **1.000** | 2.62 ms | 279,425 |
| **TF-IDF Baseline** | TF-IDF (5,000 unigrams & bigrams) → L2 Regularized Logistic Classifier | **100.0%** | **1.000** | 0.01 ms | 5,000 |

---

## Visual Evaluation Dashboard

![VeritasAI Evaluation Dashboard](models/evaluation_dashboard.png)

---

## Key Features

1. **Explainability & Token Saliency Engine**:
   - Computes word-level contribution weights ($-100$ to $+100$) dynamically visualizing which terms drive the prediction toward *Authentic* or *Misinformation*.
   - Evaluates linguistic cues: sensationalism index, punctuation density, capitalization anomalies, and Flesch reading ease.

2. **Interactive Modern Web Interface**:
   - **Detector Studio**: Live headline + body evaluation with 1-click sample presets and circular probability gauge.
   - **Model Benchmark Lab**: Side-by-side architecture comparisons, confusion matrix heatmaps, and interactive Canvas ROC curves.
   - **Dataset & Pipeline Explorer**: Corpus statistics, stratified partition breakdown, and NLP pipeline diagrams.
   - **Batch Scanner**: Vectorized multi-article verification table with real-time summary statistics.
   - **Python SDK Guide**: Standalone copyable Python code for notebooks.

3. **Privacy & Security Hardening**:
   - Comprehensive `.gitignore` protecting secrets, virtual environments, and caches.
   - Production security middleware (`nosniff`, `SAMEORIGIN`, `CORS`, 1MB payload limiter).
   - Local Wi-Fi QR code generator for instant mobile testing without exposing ports to public networks unless explicitly desired.

4. **1-Command Public Sharing**:
   - Native integration with Localtunnel, Cloudflare Tunnel, and Ngrok.

---

## Repository Structure

```
dsprj/
│
├── data/                                         # Processed news datasets & splits
│   ├── news_dataset.csv                          # Full benchmark corpus (840 articles)
│   ├── train.csv                                 # Training partition (70% - 588 articles)
│   ├── val.csv                                   # Validation partition (15% - 126 articles)
│   ├── test.csv                                  # Held-out test partition (15% - 126 articles)
│   └── dataset_stats.json                        # Corpus distribution statistics
│
├── models/                                       # Serialized models & metadata
│   ├── best_fake_news_model.keras                # Trained TensorFlow model
│   ├── vocab.json                                # Tokenizer vocabulary mapping
│   ├── tokenizer_config.json                     # Vectorizer configuration
│   ├── metrics.json                              # Benchmark curves & confusion matrices
│   └── evaluation_dashboard.png                  # 6-panel evaluation graph
│
├── static/                                       # Single Page Application UI
│   ├── index.html                                # Semantic HTML5 layout
│   ├── css/style.css                             # Slate & ink design system
│   └── js/app.js                                 # Reactive client logic & charts
│
├── tests/                                        # Automated test suite
│   └── test_model_and_api.py                     # 16 unit & integration tests
│
├── app.py                                        # FastAPI REST service & middleware
├── explainer.py                                  # Saliency attribution & heuristics
├── model_training.py                             # TensorFlow training pipeline
├── prepare_data.py                               # Dataset preprocessing pipeline
├── generate_graphs.py                            # Matplotlib/Seaborn chart generator
├── generate_pdf_report.py                        # PDF Technical Report compiler
│
├── VeritasAI_Fake_News_Detection_Report.pdf      # Complete PDF Technical Summary Report
├── run.bat                                       # 1-Click server launcher
├── train.bat                                     # 1-Click model retraining launcher
├── share_tunnel.bat                              # 1-Click public link generator
├── requirements.txt                              # Python package dependencies
├── Dockerfile                                    # Container deployment definition
├── Procfile                                      # Cloud hosting deployment definition
├── .gitignore                                    # Git exclusion rules
└── .env.example                                  # Environment variable template
```

---

## Quick Start Guide

### 1. Installation & Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd dsprj

# Create virtual environment and install dependencies
uv venv .venv --python 3.11
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
```

### 2. Dataset Preparation & Model Training
```bash
# Prepare dataset and train all TensorFlow architectures
.venv\Scripts\python.exe prepare_data.py
.venv\Scripts\python.exe model_training.py

# Or on Windows, double-click:
train.bat
```

### 3. Run Automated Tests
```bash
.venv\Scripts\pytest.exe -v tests/test_model_and_api.py
```
*(All 16 unit and integration tests will execute and validate)*

### 4. Launch Web Application
```bash
# Start FastAPI backend server
.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000

# Or on Windows, double-click:
run.bat
```
Open **`http://localhost:8000`** in your browser.

---

## Sharing Across Devices & Public Networks

### Option A: Local Wi-Fi Network (Zero Setup)
1. Open the web application at `http://localhost:8000`.
2. Click **"Share App"** in the top navigation bar.
3. Scan the generated QR code on your mobile phone or tablet connected to the same Wi-Fi.

### Option B: Public Internet Sharing (1 Command)
```bash
# Using Localtunnel
npx localtunnel --port 8000

# Or run the launcher script:
share_tunnel.bat
```

---

## Standalone Python Usage

```python
import json
import tensorflow as tf
from tensorflow import keras

# 1. Load Vocabulary & Model
with open("models/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

vectorizer = keras.layers.TextVectorization(
    max_tokens=5000, output_mode='int', output_sequence_length=150,
    standardize='lower_and_strip_punctuation', vocabulary=vocab
)
model = keras.models.load_model("models/best_fake_news_model.keras")

# 2. Predict News Authenticity
headline = "NASA James Webb Space Telescope Discovers Ancient Galaxy"
body = "Astronomers verified redshift spectroscopic signatures in Astrophysical Journal."

seq = vectorizer(tf.constant([f"{headline} - {body}"])).numpy()
fake_prob = float(model.predict(seq, verbose=0)[0][0])

print("Verdict:", "FAKE" if fake_prob >= 0.5 else "REAL")
print(f"Authenticity Confidence: {(1.0 - fake_prob)*100:.2f}%")
```

---

## Automated Pytest Suite Details

The test suite validates:
1. `test_dataset_files_exist`: Verifies all CSVs and metadata.
2. `test_dataset_schema_and_distribution`: Checks column schemas and stratified split ratios.
3. `test_model_files_exist`: Verifies model weights, vocab, and metrics JSON.
4. `test_model_loading_and_shape`: Verifies Keras input `(None, 150)` and output `(None, 1)`.
5. `test_vocabulary_integrity`: Verifies vocabulary token count and integrity.
6. `test_inference_on_real_news`: Verifies empirical news receives `REAL` classification.
7. `test_inference_on_fake_news`: Verifies hoaxes receive `FAKE` classification.
8. `test_inference_boundary_values`: Checks probability constraints $[0.0, 100.0]$.
9. `test_linguistic_feature_extraction`: Validates sensationalism scoring and keyword extraction.
10. `test_token_saliency_computation`: Verifies saliency token attribution weights.
11. `test_api_health_endpoint`: Tests `/health` status and model availability.
12. `test_api_samples_endpoint`: Tests `/api/samples` sample bank.
13. `test_api_benchmark_endpoint`: Tests `/api/benchmark` multi-model metrics.
14. `test_api_batch_predict_endpoint`: Tests `/api/batch-predict` parallel inference.
15. `test_api_share_info_endpoint`: Tests `/api/share-info` network IP & QR code.
16. `test_api_validation_error_handling`: Tests rejection of malformed or empty payloads.

---

## License
Apache-2.0 License.
