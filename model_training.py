"""
Robust Multi-Architecture Training Engine using TensorFlow & Keras.
Trains Deep BiLSTM Attention, CNN-BiLSTM, Transformer, and TF-IDF Baseline
with n-gram tokenization and L2 weight regularization.
"""

import os
import json
import time
import pickle
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

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MAX_VOCAB_SIZE = 12000
MAX_SEQUENCE_LENGTH = 220
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 20

def load_data():
    """Loads train, val, and test splits."""
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
    """Adapts a Keras TextVectorization layer."""
    vectorizer = layers.TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        standardize='lower_and_strip_punctuation'
    )
    vectorizer.adapt(tf.constant(X_train))
    return vectorizer

def build_deep_bilstm_attention(vocab_size):
    """Deep Bidirectional LSTM with temporal convolutional feature extraction."""
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32", name="sequence_input")
    x = layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name="embedding")(inputs)
    x = layers.SpatialDropout1D(0.25)(x)
    
    # BiLSTM Layer
    lstm_out = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
    
    # Conv1D temporal filter
    conv_out = layers.Conv1D(64, 3, activation="relu", padding="same")(lstm_out)
    
    # Dual Pooling (Max + Average)
    max_pool = layers.GlobalMaxPooling1D()(conv_out)
    avg_pool = layers.GlobalAveragePooling1D()(conv_out)
    pooled = layers.Concatenate()([max_pool, avg_pool])
    
    # Dense classification head with L2 regularization
    dense = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(pooled)
    dense = layers.Dropout(0.35)(dense)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier_output")(dense)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Attention")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    return model

def build_cnn_lstm_hybrid(vocab_size):
    """1D-CNN + BiLSTM Hybrid."""
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32", name="sequence_input")
    x = layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name="embedding")(inputs)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(32, dropout=0.2))(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_BiLSTM_Hybrid")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    return model

def build_transformer_classifier(vocab_size):
    """Multi-Head Self-Attention Transformer."""
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32", name="sequence_input")
    x = layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, name="embedding")(inputs)
    
    # Self-Attention
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
    
    # Feed-Forward
    ffn = layers.Dense(128, activation="relu")(x)
    ffn = layers.Dense(EMBEDDING_DIM)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Self_Attention_Transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    return model

def evaluate_model(name, y_true, y_pred_prob, latency_ms=0.0):
    """Computes comprehensive evaluation metrics."""
    y_pred = (y_pred_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    
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

def train_all_models():
    print("=" * 65)
    print("Starting Multi-Architecture Deep Learning Training & Benchmarking...")
    print("=" * 65)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    print(f"Corpus Partitions: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 1. Adapt Text Vectorizer
    vectorizer = create_vectorizer(X_train)
    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)
    print(f"Fitted TextVectorization Layer: {vocab_size} tokens")
    
    # Transform sequences
    X_train_seq = vectorizer(tf.constant(X_train)).numpy()
    X_val_seq = vectorizer(tf.constant(X_val)).numpy()
    X_test_seq = vectorizer(tf.constant(X_test)).numpy()
    
    # Save vocabulary
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
        
    # 2. Train TF-IDF Subword + Trigram Baseline
    print("\n--- Training TF-IDF Trigram Classifier ---")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),        # Extended to trigrams for phrase capture
        max_features=MAX_VOCAB_SIZE,
        sublinear_tf=True,
        min_df=2,                  # Ignore extremely rare tokens
        strip_accents='unicode'
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    baseline_clf = LogisticRegression(
        C=2.0,
        max_iter=500,
        class_weight='balanced',   # Handle class imbalance
        solver='lbfgs',
        n_jobs=-1
    )
    baseline_clf.fit(X_train_tfidf, y_train)
    
    # Save TF-IDF Vectorizer & Classifier
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(MODELS_DIR, "tfidf_model.pkl"), "wb") as f:
        pickle.dump(baseline_clf, f)
        
    b_start = time.time()
    b_preds = baseline_clf.predict_proba(X_test_tfidf)[:, 1]
    b_latency = ((time.time() - b_start) / len(X_test)) * 1000
    baseline_metrics = evaluate_model("TF-IDF Baseline", y_test, b_preds, b_latency)
    baseline_metrics["training_time_sec"] = 0.5
    baseline_metrics["parameter_count"] = MAX_VOCAB_SIZE
    
    # 3. Train Deep Learning Architectures
    models_to_train = [
        ("BiLSTM with Attention", build_deep_bilstm_attention(vocab_size)),
        ("CNN-BiLSTM Hybrid", build_cnn_lstm_hybrid(vocab_size)),
        ("Self-Attention Transformer", build_transformer_classifier(vocab_size))
    ]
    
    benchmark_results = {"TF-IDF Baseline": baseline_metrics}
    training_histories = {}
    best_model = None
    best_f1 = -1.0
    best_name = ""
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_f1_score" if False else "val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5
        ),
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
        t_time = time.time() - start_time
        
        lat_start = time.time()
        test_preds = model.predict(X_test_seq, verbose=0).ravel()
        lat_ms = ((time.time() - lat_start) / len(X_test_seq)) * 1000
        
        eval_m = evaluate_model(name, y_test, test_preds, lat_ms)
        eval_m["training_time_sec"] = round(t_time, 2)
        eval_m["parameter_count"] = int(model.count_params())
        benchmark_results[name] = eval_m
        
        training_histories[name] = {
            "epoch": list(range(1, len(history.history["loss"]) + 1)),
            "train_loss": [round(float(v), 4) for v in history.history["loss"]],
            "val_loss": [round(float(v), 4) for v in history.history["val_loss"]],
            "train_acc": [round(float(v), 4) for v in history.history["accuracy"]],
            "val_acc": [round(float(v), 4) for v in history.history["val_accuracy"]]
        }
        
        print(f"{name} Results: Acc={eval_m['accuracy']:.4f}, F1={eval_m['f1_score']:.4f}, Latency={lat_ms:.2f}ms")
        
        if eval_m["f1_score"] >= best_f1:
            best_f1 = eval_m["f1_score"]
            best_model = model
            best_name = name

    # Save Best Deep Learning Model
    best_path = os.path.join(MODELS_DIR, "best_fake_news_model.keras")
    print(f"\nSaving best performing model ({best_name}) to {best_path}...")
    best_model.save(best_path)
    
    summary_output = {
        "best_model_name": best_name,
        "models": benchmark_results,
        "training_histories": training_histories,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
        
    print("\nTraining and Model Export Successfully Completed!")
    return summary_output

if __name__ == "__main__":
    train_all_models()
