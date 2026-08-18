"""
Visualization & Graph Generation Engine for Fake News Detection System.
Produces high-resolution evaluation charts, ROC/PR curves, training trajectories,
and confusion matrix heatmaps using Matplotlib and Seaborn.
"""

import os
import json
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
ARTIFACT_DIR = r"C:\Users\Gauta\.gemini\antigravity\brain\c49ef904-2c56-4509-8aa1-7936e1ba70ab"

def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_dashboard():
    data = load_metrics()
    models = data["models"]
    histories = data.get("training_histories", {})

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle('VeritasAI - Deep Learning Fake News Detection Evaluation Dashboard', 
                 fontsize=18, fontweight='bold', color='#f8fafc', y=0.98)

    palette = {
        'BiLSTM with Attention': '#38bdf8',
        'CNN-BiLSTM Hybrid': '#10b981',
        'Self-Attention Transformer': '#f59e0b',
        'TF-IDF Baseline': '#a855f7'
    }

    for ax in axes.flat:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#f8fafc')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    # 1. ROC Curves
    ax1 = axes[0, 0]
    ax1.set_title('ROC Curves (Receiver Operating Characteristic)', fontsize=12, fontweight='bold', pad=10)
    for name, m in models.items():
        if "roc_curve" in m and len(m["roc_curve"]) > 0:
            fpr = [pt["fpr"] for pt in m["roc_curve"]]
            tpr = [pt["tpr"] for pt in m["roc_curve"]]
            auc = m.get("roc_auc", 1.0)
            ax1.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=palette.get(name, '#38bdf8'), linewidth=2.2)
    ax1.plot([0, 1], [0, 1], '--', color='#64748b', alpha=0.6, label='Chance Baseline')
    ax1.set_xlabel('False Positive Rate (FPR)')
    ax1.set_ylabel('True Positive Rate (TPR)')
    ax1.legend(loc='lower right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])

    # 2. Precision-Recall Curves
    ax2 = axes[0, 1]
    ax2.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold', pad=10)
    for name, m in models.items():
        if "pr_curve" in m and len(m["pr_curve"]) > 0:
            rec = [pt["recall"] for pt in m["pr_curve"]]
            prec = [pt["precision"] for pt in m["pr_curve"]]
            ax2.plot(rec, prec, label=name, color=palette.get(name, '#38bdf8'), linewidth=2.2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(loc='lower left', facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([0.45, 1.05])

    # 3. Model Accuracy & F1 Comparison Bar Chart
    ax3 = axes[0, 2]
    ax3.set_title('Model Performance Benchmark (Held-out Test Set)', fontsize=12, fontweight='bold', pad=10)
    names = list(models.keys())
    accuracies = [models[n]["accuracy"] * 100 for n in names]
    f1_scores = [models[n]["f1_score"] * 100 for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    rects1 = ax3.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#38bdf8')
    rects2 = ax3.bar(x + width/2, f1_scores, width, label='F1-Score (%)', color='#10b981')
    
    ax3.set_ylabel('Score (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8.5)
    ax3.set_ylim([80, 105])
    ax3.legend(loc='lower right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5)

    # 4. Training & Validation Loss Trajectory
    ax4 = axes[1, 0]
    ax4.set_title('BiLSTM Training vs Validation Loss', fontsize=12, fontweight='bold', pad=10)
    bilstm_hist = histories.get("BiLSTM with Attention", {})
    if bilstm_hist:
        epochs = bilstm_hist["epoch"]
        ax4.plot(epochs, bilstm_hist["train_loss"], 'o-', color='#38bdf8', label='Train Loss', linewidth=2.0, markersize=4)
        ax4.plot(epochs, bilstm_hist["val_loss"], 's-', color='#f43f5e', label='Val Loss', linewidth=2.0, markersize=4)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Binary Crossentropy Loss')
    ax4.legend(loc='upper right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5)

    # 5. Training & Validation Accuracy Trajectory
    ax5 = axes[1, 1]
    ax5.set_title('BiLSTM Training vs Validation Accuracy', fontsize=12, fontweight='bold', pad=10)
    if bilstm_hist:
        epochs = bilstm_hist["epoch"]
        ax5.plot(epochs, [a * 100 for a in bilstm_hist["train_acc"]], 'o-', color='#38bdf8', label='Train Accuracy (%)', linewidth=2.0, markersize=4)
        ax5.plot(epochs, [a * 100 for a in bilstm_hist["val_acc"]], 's-', color='#10b981', label='Val Accuracy (%)', linewidth=2.0, markersize=4)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Accuracy (%)')
    ax5.legend(loc='lower right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5)

    # 6. Confusion Matrix Heatmap (BiLSTM Model)
    ax6 = axes[1, 2]
    ax6.set_title('BiLSTM Confusion Matrix (Test Set N=126)', fontsize=12, fontweight='bold', pad=10)
    cm_data = models["BiLSTM with Attention"]["confusion_matrix"]
    cm_matrix = np.array([
        [cm_data["true_negative"], cm_data["false_positive"]],
        [cm_data["false_negative"], cm_data["true_positive"]]
    ])
    
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax6,
                annot_kws={'size': 16, 'weight': 'bold'})
    ax6.set_xticklabels(['Pred: REAL', 'Pred: FAKE'], fontsize=10, color='#94a3b8')
    ax6.set_yticklabels(['Actual: REAL', 'Actual: FAKE'], fontsize=10, va='center', color='#94a3b8')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save outputs
    out_path = os.path.join(MODELS_DIR, "evaluation_dashboard.png")
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Evaluation dashboard saved to: {out_path}")

    # Copy to Artifacts directory
    if os.path.exists(ARTIFACT_DIR):
        artifact_img_path = os.path.join(ARTIFACT_DIR, "evaluation_dashboard.png")
        shutil.copy(out_path, artifact_img_path)
        print(f"Copied to artifact directory: {artifact_img_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_dashboard()
