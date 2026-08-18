"""
Deep Learning Model Training & Evaluation Engine using TensorFlow & Keras.
Trains BiLSTM, CNN-BiLSTM, Transformer, and TF-IDF Baseline on the Fake News Dataset.
Generates comprehensive benchmark metrics, confusion matrices, and training curves.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

# Constants & Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 15

def load_data():
    """Loads train, validation, and test datasets."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    X_train = train_df["cleaned_text"].astype(str).tolist()
    y_train = train_df["label"].values.astype(np.float32)
    
    X_val = val_df["cleaned_text"].astype(str).tolist()
    y_val = val_df["label"].values.astype(np.float32)
    
    X_test = test_df["cleaned_text"].astype(str).tolist()
    y_test = test_df["label"].values.astype(np.float32)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_vectorizer(X_train):
    """Adapts a Keras TextVectorization layer to the training corpus."""
    vectorizer = layers.TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        standardize='lower_and_strip_punctuation'
    )
    vectorizer.adapt(tf.constant(X_train))
    return vectorizer

def build_bilstm_model(vocab_size):
    """Builds a Bidirectional LSTM text classification model."""
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32", name="sequence_input")
    x = layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name="embedding")(x) if False else layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name="embedding")(inputs)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Attention")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    return model

def build_cnn_lstm_model(vocab_size):
    """Builds a 1D-CNN + BiLSTM Hybrid text classification model."""
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32", name="sequence_input")
    x = layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name="embedding")(inputs)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_BiLSTM_Hybrid")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    return model

def build_transformer_model(vocab_size):
    """Builds a Multi-Head Self-Attention Transformer Classifier."""
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32", name="sequence_input")
    x = layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name="embedding")(inputs)
    
    # Self-Attention Block
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    # Feed-Forward Block
    ffn = layers.Dense(128, activation="relu")(x)
    ffn = layers.Dense(EMBEDDING_DIM)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Self_Attention_Transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    return model

def evaluate_model(name, y_true, y_pred_prob, latency_ms=0.0):
    """Computes comprehensive metrics and curves for a model."""
    y_pred = (y_pred_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    
    # Downsample curves to 25 points for compact JSON storage
    indices = np.linspace(0, len(fpr) - 1, min(25, len(fpr)), dtype=int)
    roc_points = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in indices]
    
    p_indices = np.linspace(0, len(prec_curve) - 1, min(25, len(prec_curve)), dtype=int)
    pr_points = [{"precision": round(float(prec_curve[i]), 4), "recall": round(float(rec_curve[i]), 4)} for i in p_indices]
    
    return {
        "name": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_pred_prob)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "latency_ms": round(latency_ms, 2),
        "roc_curve": roc_points,
        "pr_curve": pr_points
    }

def train_and_benchmark():
    """Main training routine."""
    print("=" * 60)
    print("Starting Deep Learning Training Pipeline with TensorFlow...")
    print("=" * 60)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    print(f"Data Loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Adapt Text Vectorizer
    print("Fitting Keras TextVectorization layer...")
    vectorizer = create_vectorizer(X_train)
    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)
    print(f"Vocabulary adapted. Total tokens: {vocab_size}")
    
    # Vectorize datasets into sequence arrays
    print("Transforming text into integer sequence tensors...")
    X_train_seq = vectorizer(tf.constant(X_train)).numpy()
    X_val_seq = vectorizer(tf.constant(X_val)).numpy()
    X_test_seq = vectorizer(tf.constant(X_test)).numpy()
    
    # Save vocabulary and tokenizer configuration
    with open(os.path.join(MODELS_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
        
    tokenizer_config = {
        "max_vocab_size": MAX_VOCAB_SIZE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "vocab_size": vocab_size
    }
    with open(os.path.join(MODELS_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)
        
    models_to_train = [
        ("BiLSTM with Attention", build_bilstm_model(vocab_size)),
        ("CNN-BiLSTM Hybrid", build_cnn_lstm_model(vocab_size)),
        ("Self-Attention Transformer", build_transformer_model(vocab_size))
    ]
    
    benchmark_results = {}
    training_histories = {}
    best_model = None
    best_f1 = -1.0
    best_name = ""
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    ]
    
    for name, model in models_to_train:
        print(f"\n--- Training {name} ---")
        start_time = time.time()
        history = model.fit(
            X_train_seq, y_train,
            validation_data=(X_val_seq, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Test inference latency
        lat_start = time.time()
        test_preds = model.predict(X_test_seq, verbose=0).ravel()
        latency_ms = ((time.time() - lat_start) / len(X_test_seq)) * 1000
        
        eval_metrics = evaluate_model(name, y_test, test_preds, latency_ms)
        eval_metrics["training_time_sec"] = round(training_time, 2)
        eval_metrics["parameter_count"] = int(model.count_params())
        benchmark_results[name] = eval_metrics
        
        # Format history
        training_histories[name] = {
            "epoch": list(range(1, len(history.history["loss"]) + 1)),
            "train_loss": [round(float(v), 4) for v in history.history["loss"]],
            "val_loss": [round(float(v), 4) for v in history.history["val_loss"]],
            "train_acc": [round(float(v), 4) for v in history.history["accuracy"]],
            "val_acc": [round(float(v), 4) for v in history.history["val_accuracy"]]
        }
        
        print(f"{name} Results: Acc={eval_metrics['accuracy']:.4f}, F1={eval_metrics['f1_score']:.4f}, Latency={latency_ms:.2f}ms")
        
        if eval_metrics["f1_score"] > best_f1:
            best_f1 = eval_metrics["f1_score"]
            best_model = model
            best_name = name

    # Train Baseline Model (TF-IDF + Logistic Regression)
    print("\n--- Training TF-IDF + Logistic Regression Baseline ---")
    tfidf = TfidfVectorizer(max_features=MAX_VOCAB_SIZE)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    baseline_clf = LogisticRegression(C=1.0, max_iter=200)
    b_start = time.time()
    baseline_clf.fit(X_train_tfidf, y_train)
    b_train_time = time.time() - b_start
    
    b_lat_start = time.time()
    baseline_preds = baseline_clf.predict_proba(X_test_tfidf)[:, 1]
    b_latency_ms = ((time.time() - b_lat_start) / len(X_test)) * 1000
    
    baseline_metrics = evaluate_model("TF-IDF Baseline", y_test, baseline_preds, b_latency_ms)
    baseline_metrics["training_time_sec"] = round(b_train_time, 2)
    baseline_metrics["parameter_count"] = MAX_VOCAB_SIZE
    benchmark_results["TF-IDF Baseline"] = baseline_metrics
    
    # Save the Best Model
    best_model_path = os.path.join(MODELS_DIR, "best_fake_news_model.keras")
    print(f"\nSaving best performing model ({best_name}) to {best_model_path}...")
    best_model.save(best_model_path)
    
    # Save benchmark metrics and training history
    summary_output = {
        "best_model_name": best_name,
        "models": benchmark_results,
        "training_histories": training_histories,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
        
    print("\nTraining and Benchmarking Completed Successfully!")
    print(f"Metrics saved to: {os.path.join(MODELS_DIR, 'metrics.json')}")
    return summary_output

if __name__ == "__main__":
    train_and_benchmark()
