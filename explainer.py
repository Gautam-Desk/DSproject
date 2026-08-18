"""
High-Precision Explainability, Saliency Attribution, and Linguistic Reasoning Engine.
Computes token contribution weights, semantic credibility signals, sensationalism index,
and natural language explanation bullet points.
"""

import re
import json
import os
import math

BASE_DIR = os.path.dirname(__file__)

# High-impact sensationalist, panic, and conspiracy indicators
SENSATIONAL_TRIGGERS = {
    # Sensational Headlines & Clickbait
    "shocking": 3.0, "bombshell": 3.5, "unbelievable": 2.5, "banned": 3.0,
    "miracle": 3.2, "secret": 2.2, "exposed": 2.5, "censored": 3.0,
    "leaked": 2.2, "suppressed": 2.8, "urgent warning": 3.5, "wake up": 3.8,
    "they don't want you to know": 4.0, "they don't want you to see": 4.0,
    "must see": 2.8, "you won't believe": 3.5, "doctors are terrified": 3.8,
    "doctors are banned": 4.0, "cure all": 3.5, "cures cancer": 4.0,
    "instantly destroys": 3.5, "burns 30 pounds": 3.8, "overnight": 2.0,
    
    # Conspiracies & Pseudotext
    "5g": 2.8, "mind control": 3.8, "frequencies": 2.0, "deep state": 3.8,
    "shadow elites": 3.8, "globalists": 3.2, "nanobots": 3.5, "microchips": 3.2,
    "alien": 3.0, "mothership": 3.5, "extraterrestrial": 2.5, "soundstage": 3.2,
    "moon landing was staged": 4.2, "chemtrails": 3.8, "reptilian": 4.5,
    "weather machine": 3.8, "perpetual motion": 3.5, "free energy": 3.2,
    "time traveler": 3.8, "illuminati": 4.0, "new world order": 3.8,
    "depopulation": 3.8, "fake hurricane": 3.8, "antigravity drive": 3.2,
    
    # Financial Panic & Crypto Scams
    "total blackout": 3.2, "currency reset": 3.5, "all cash abolished": 3.8,
    "solar flare erase banks": 4.0, "withdraw all cash": 3.5, "guaranteed 500%": 4.0,
    "zero dollars electricity": 3.8, "scam": 2.0, "hoax": 2.2
}

# Empirical, institutional, verified, and journalistic attribution indicators
CREDIBLE_TRIGGERS = {
    # Academic & Peer-Reviewed Science
    "peer-reviewed": -3.5, "clinical trial": -3.2, "phase iii": -3.0, "double-blind": -3.2,
    "placebo": -2.5, "astrophysical journal": -3.5, "the lancet": -3.5, "nature astronomy": -3.5,
    "science": -2.0, "spectroscopy": -2.8, "redshift": -2.8, "telescope": -2.0,
    "researchers": -2.0, "astronomers": -2.2, "biologists": -2.2, "physicists": -2.2,
    "university": -2.0, "cern": -3.0, "nasa": -2.8, "esa": -2.8, "nih": -3.0,
    "monoclonal antibody": -3.2, "efficacy": -2.5, "statistically significant": -3.0,
    "data safety monitoring": -3.2,
    
    # Official Journalism & Press Wires
    "reuters": -3.5, "associated press": -3.5, "ap": -2.2, "bloomberg": -3.0,
    "according to": -2.5, "official report": -2.8, "spokesperson": -2.5,
    "press briefing": -2.8, "statement released": -2.5, "announced wednesday": -2.2,
    "announced today": -2.2, "in a formal statement": -2.8, "press conference": -2.4,
    
    # Governance & Economics
    "federal reserve": -3.2, "supreme court": -3.2, "clean air act": -3.0,
    "department of justice": -3.2, "bureau of labor statistics": -3.5, "nonfarm payroll": -3.0,
    "european central bank": -3.2, "international monetary fund": -3.2, "world health organization": -3.2,
    "who": -2.2, "fda": -2.8, "cisa": -3.0, "cybersecurity agency": -2.8,
    "sec filing": -3.2, "unemployment rate": -2.8, "inflation": -2.0, "statutory": -2.8,
    "antitrust": -2.5, "parliamentary": -2.5, "humanitarian corridor": -2.8,
    "red cross": -2.8, "treaty": -2.5, "maritime safety": -2.5
}

def analyze_linguistics(title: str, text: str) -> dict:
    """Extracts rich linguistic, rhetorical, and stylistic features."""
    full_text = f"{title} {text}".strip()
    words = re.findall(r'\b[a-zA-Z]+\b', full_text)
    total_words = max(len(words), 1)
    
    # 1. Uppercase & Capitalization Anomaly
    all_caps_words = [w for w in words if len(w) > 2 and w.isupper() and w not in ["NASA", "FDA", "WHO", "CERN", "CISA", "NIH", "ECB", "IMF", "RSV", "ESA", "NIST", "FIDO2"]]
    uppercase_ratio = len(all_caps_words) / total_words
    cap_score = min(100.0, uppercase_ratio * 450)
    
    # 2. Exclamation & Punctuation Density
    exclamations = full_text.count("!")
    question_marks = full_text.count("?")
    multi_punct = len(re.findall(r'[!?]{2,}', full_text))
    punct_density = (exclamations * 2.5 + question_marks * 1.5 + multi_punct * 4) / max(len(full_text.split(".")), 1)
    punct_score = min(100.0, punct_density * 28)
    
    # 3. Sensationalism & Credibility Matches
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
                
    sensationalism_index = min(100.0, max(0.0, (sensational_weight / max(total_words / 22, 1)) * 32))
    credibility_marker_index = min(100.0, max(0.0, (credible_weight / max(total_words / 22, 1)) * 35))
    
    # 4. Readability Score
    sentences = max(len(re.split(r'[.!?]+', full_text)), 1)
    avg_sentence_len = total_words / sentences
    syllables = sum(max(1, len(re.findall(r'[aeiouy]+', w.lower()))) for w in words)
    avg_syllables_per_word = syllables / total_words
    reading_ease = max(0.0, min(100.0, 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables_per_word)))
    
    # 5. Natural Language Reasoning Generation
    reasons = []
    if sensational_hits:
        reasons.append(f"Sensationalist / Clickbait vocabulary identified: {', '.join(sensational_hits[:4])}.")
    if credible_hits:
        reasons.append(f"Empirical & verified institutional markers found: {', '.join(credible_hits[:4])}.")
    if exclamations >= 2:
        reasons.append(f"Unusual exclamation frequency ({exclamations} occurrences) indicating urgent emotional appeal.")
    if len(all_caps_words) >= 2:
        reasons.append(f"Excessive ALL-CAPS words ({', '.join(all_caps_words[:4])}) typical of yellow journalism.")
    if not reasons:
        reasons.append("Standard neutral journalistic prose structure with measured syntax.")
        
    return {
        "sensationalism_score": round(sensationalism_index, 1),
        "credibility_marker_score": round(credibility_marker_index, 1),
        "capitalization_anomaly_score": round(cap_score, 1),
        "punctuation_anomaly_score": round(punct_score, 1),
        "reading_ease": round(reading_ease, 1),
        "detected_sensational_terms": sensational_hits[:8],
        "detected_credible_terms": credible_hits[:8],
        "all_caps_count": len(all_caps_words),
        "exclamation_count": exclamations,
        "word_count": total_words,
        "reasoning_bullets": reasons
    }

def compute_token_saliency(full_text: str, fake_prob: float) -> list:
    """Calculates granular token-by-token contribution weights."""
    raw_tokens = re.split(r'(\s+|[.,!?;:()"\'-])', full_text)
    saliency_items = []
    
    for raw in raw_tokens:
        if not raw:
            continue
        cleaned = raw.lower().strip()
        weight = 0.0
        label_class = "neutral"
        
        if re.match(r'^[a-zA-Z0-9]+$', cleaned):
            if cleaned in SENSATIONAL_TRIGGERS:
                weight = SENSATIONAL_TRIGGERS[cleaned] * 28.0
            elif cleaned in CREDIBLE_TRIGGERS:
                weight = CREDIBLE_TRIGGERS[cleaned] * 28.0
            elif raw.isupper() and len(raw) > 2 and raw not in ["NASA", "FDA", "WHO", "CERN", "CISA", "NIH"]:
                weight += 25.0
            else:
                weight = (fake_prob - 0.5) * 12.0
                
            weight = max(-100.0, min(100.0, weight))
            
            if weight >= 30.0:
                label_class = "strongly-fake"
            elif weight >= 10.0:
                label_class = "moderately-fake"
            elif weight <= -30.0:
                label_class = "strongly-real"
            elif weight <= -10.0:
                label_class = "moderately-real"
            else:
                label_class = "neutral"
                
        saliency_items.append({
            "token": raw,
            "weight": round(weight, 1),
            "class": label_class
        })
        
    return saliency_items
