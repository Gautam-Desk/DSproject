"""
Explainability & Linguistic Analysis Engine.
Computes token-level saliency weights and linguistic indicators for fake news detection.
"""

import re
import json
import os
import math
import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# High-risk sensationalist & deceptive triggers
SENSATIONAL_TRIGGERS = {
    "shocking": 2.5, "bombshell": 3.0, "secret": 2.0, "miracle": 2.5, "banned": 2.5,
    "exposed": 2.2, "leaked": 2.0, "hidden": 1.8, "censored": 2.5, "urgent": 2.0,
    "wake up": 3.0, "deep state": 3.2, "whistleblower": 1.9, "aliens": 2.8,
    "conspiracy": 2.4, "microchip": 2.7, "hoax": 2.3, "5g": 2.4, "nanobots": 2.9,
    "destroyed": 1.5, "unbelievable": 2.2, "cure": 2.0, "suppressed": 2.4,
    "elites": 2.1, "reptilian": 3.5, "propaganda": 1.8, "plot": 1.6,
    "instantly": 1.7, "scam": 2.2, "confidential": 1.8, "apocalypse": 2.5
}

# Credible & empirical markers
CREDIBLE_TRIGGERS = {
    "peer-reviewed": -2.8, "clinical": -2.2, "trial": -1.8, "published": -1.5,
    "journal": -1.8, "researchers": -1.5, "university": -1.4, "study": -1.3,
    "official": -1.6, "statement": -1.4, "accord": -1.8, "spokesperson": -1.7,
    "statistics": -1.6, "audited": -2.0, "treaty": -1.9, "consensus": -1.7,
    "corridor": -1.2, "investigation": -1.2, "laboratory": -1.5, "findings": -1.4,
    "confirmed": -1.3, "regulatory": -1.8, "agency": -1.4, "quarterly": -1.4,
    "infrastructure": -1.3, "collaborative": -1.5, "scientific": -1.7, "data": -1.2
}

def analyze_linguistics(title: str, text: str) -> dict:
    """Extracts linguistic, emotional, and stylistic metrics."""
    full_text = f"{title} {text}".strip()
    words = re.findall(r'\b[a-zA-Z]+\b', full_text)
    total_words = max(len(words), 1)
    
    # 1. Uppercase & Capitalization Anomaly
    all_caps_words = [w for w in words if len(w) > 2 and w.isupper()]
    uppercase_ratio = len(all_caps_words) / total_words
    cap_score = min(100.0, uppercase_ratio * 400)
    
    # 2. Exclamation & Punctuation Density
    exclamations = full_text.count("!")
    question_marks = full_text.count("?")
    multi_punct = len(re.findall(r'[!?]{2,}', full_text))
    punct_density = (exclamations * 2 + question_marks + multi_punct * 3) / max(len(full_text.split(".")), 1)
    punct_score = min(100.0, punct_density * 25)
    
    # 3. Sensationalism & Clickbait Index
    lower_text = full_text.lower()
    sensational_hits = []
    sensational_weight = 0.0
    for trigger, weight in SENSATIONAL_TRIGGERS.items():
        if trigger in lower_text:
            count = len(re.findall(r'\b' + re.escape(trigger) + r'\b', lower_text))
            if count > 0:
                sensational_hits.append(trigger)
                sensational_weight += weight * count
                
    credible_hits = []
    credible_weight = 0.0
    for trigger, weight in CREDIBLE_TRIGGERS.items():
        if trigger in lower_text:
            count = len(re.findall(r'\b' + re.escape(trigger) + r'\b', lower_text))
            if count > 0:
                credible_hits.append(trigger)
                credible_weight += abs(weight) * count
                
    sensationalism_index = min(100.0, max(0.0, (sensational_weight / max(total_words / 20, 1)) * 30))
    credibility_marker_index = min(100.0, max(0.0, (credible_weight / max(total_words / 20, 1)) * 35))
    
    # 4. Subjectivity & Urgency Indicators
    urgency_words = ["now", "immediately", "urgent", "hurry", "before it's deleted", "warning", "fast", "today only"]
    urgency_count = sum(1 for u in urgency_words if u in lower_text)
    urgency_score = min(100.0, urgency_count * 30.0)
    
    # Readability Metric (approximate Flesch Reading Ease)
    sentences = max(len(re.split(r'[.!?]+', full_text)), 1)
    avg_sentence_len = total_words / sentences
    syllables = sum(max(1, len(re.findall(r'[aeiouy]+', w.lower()))) for w in words)
    avg_syllables_per_word = syllables / total_words
    reading_ease = max(0.0, min(100.0, 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables_per_word)))
    
    return {
        "sensationalism_score": round(sensationalism_index, 1),
        "credibility_marker_score": round(credibility_marker_index, 1),
        "capitalization_anomaly_score": round(cap_score, 1),
        "punctuation_anomaly_score": round(punct_score, 1),
        "urgency_score": round(urgency_score, 1),
        "reading_ease": round(reading_ease, 1),
        "detected_sensational_terms": sensational_hits[:8],
        "detected_credible_terms": credible_hits[:8],
        "all_caps_count": len(all_caps_words),
        "exclamation_count": exclamations,
        "word_count": total_words
    }

def compute_token_saliency(full_text: str, fake_prob: float) -> list:
    """Computes word-level contribution weights for visual token saliency highlighting."""
    raw_tokens = re.split(r'(\s+|[.,!?;:()"\'-])', full_text)
    saliency_items = []
    
    for raw in raw_tokens:
        if not raw:
            continue
        cleaned = raw.lower().strip()
        
        # Default neutral weight
        weight = 0.0
        label_class = "neutral"
        
        if re.match(r'^[a-zA-Z0-9]+$', cleaned):
            if cleaned in SENSATIONAL_TRIGGERS:
                weight = SENSATIONAL_TRIGGERS[cleaned] * 25.0
            elif cleaned in CREDIBLE_TRIGGERS:
                weight = CREDIBLE_TRIGGERS[cleaned] * 25.0
            elif raw.isupper() and len(raw) > 2:
                weight += 20.0
            else:
                # Slight baseline drift based on overall prediction
                if fake_prob >= 0.5:
                    weight = (fake_prob - 0.5) * 15.0
                else:
                    weight = (fake_prob - 0.5) * 15.0
                    
            # Scale to [-100, 100]
            weight = max(-100.0, min(100.0, weight))
            
            if weight >= 35.0:
                label_class = "strongly-fake"
            elif weight >= 12.0:
                label_class = "moderately-fake"
            elif weight <= -35.0:
                label_class = "strongly-real"
            elif weight <= -12.0:
                label_class = "moderately-real"
            else:
                label_class = "neutral"
                
        saliency_items.append({
            "token": raw,
            "weight": round(weight, 1),
            "class": label_class
        })
        
    return saliency_items
