"""
High-Precision Explainability, Saliency Attribution, and Linguistic Reasoning Engine.
Computes token contribution weights, semantic credibility signals, sensationalism index,
and natural language explanation bullet points.

FIX v3.0:
 - Multi-word triggers now use plain substring search (not \\b which breaks on spaces/apostrophes)
 - Single-word triggers still use word-boundary matching for precision
 - Massively expanded SENSATIONAL_TRIGGERS (+30 entries)
 - Massively expanded CREDIBLE_TRIGGERS (+25 entries)
 - Added uncertainty / grey-zone reasoning bullet
"""

import re
import json
import os
import math

BASE_DIR = os.path.dirname(__file__)

# ──────────────────────────────────────────────────────────────
# SENSATIONAL / FAKE-NEWS INDICATORS  (positive weight = more fake)
# Split into single-word vs multi-word so matching works correctly
# ──────────────────────────────────────────────────────────────
SENSATIONAL_SINGLE = {
    # Clickbait power-words
    "shocking": 3.0, "bombshell": 3.5, "unbelievable": 2.5, "banned": 3.0,
    "miracle": 3.2, "secret": 2.2, "exposed": 2.5, "censored": 3.0,
    "leaked": 2.2, "suppressed": 2.8, "urgent": 2.0, "hoax": 2.2,
    "scam": 2.0, "overnight": 2.0, "guaranteed": 2.8, "whistleblower": 2.0,
    "blackout": 2.5, "terrified": 3.0, "panicking": 3.0, "obliterates": 3.0,
    "obliterated": 3.0, "abolished": 2.5, "confiscate": 2.8, "enslaved": 3.0,
    # Conspiracies
    "5g": 2.8, "microchips": 3.2, "nanobots": 3.5, "chemtrails": 3.8,
    "reptilian": 4.5, "illuminati": 4.0, "depopulation": 3.8, "soundstage": 3.2,
    "alien": 3.0, "mothership": 3.5, "extraterrestrial": 2.5,
    # Medical quackery
    "detox": 1.8, "toxins": 1.8, "superfood": 2.0, "cures": 3.0,
    # Financial panic
    "cbdc": 2.2, "silver": 1.5, "collapse": 2.5, "hyperinflation": 2.2,
}

SENSATIONAL_PHRASES = {
    # Urgent calls to action
    "share this everywhere": 4.0, "share before deleted": 4.2,
    "share before they delete": 4.2, "forward this now": 3.8,
    "must see": 2.8, "wake up": 3.8, "wake up sheeple": 4.5,
    "you won't believe": 3.5, "you will not believe": 3.5,
    # Suppression narratives
    "they don't want you to know": 4.0, "they don't want you to see": 4.0,
    "they are hiding": 3.8, "government is hiding": 3.8,
    "big pharma is hiding": 4.0, "big pharma doesn't want": 4.0,
    "doctors are terrified": 3.8, "doctors are banned": 4.0,
    "doctors don't want you": 3.8, "media won't cover": 3.5,
    # Medical hoaxes
    "cure all": 3.5, "cures cancer": 4.0, "cures all cancers": 4.5,
    "instantly destroys": 3.5, "100 percent cure": 4.2, "100% cure": 4.2,
    "burns 30 pounds": 3.8, "miracle cure": 4.0, "ancient secret": 3.2,
    "ancient remedy": 2.8, "himalayan secret": 3.2, "forbidden cure": 4.0,
    # Conspiracy key phrases
    "moon landing was staged": 4.2, "moon landing fake": 4.0,
    "deep state": 3.8, "shadow government": 3.8, "shadow elites": 3.8,
    "globalist elite": 3.8, "globalists": 3.2, "new world order": 3.8,
    "mind control": 3.8, "weather machine": 3.8, "free energy": 3.2,
    "time traveler": 3.8, "fake hurricane": 3.8, "perpetual motion": 3.5,
    "antigravity drive": 3.2, "antigravity technology": 3.2,
    # Financial panic
    "total blackout": 3.2, "currency reset": 3.5, "all cash abolished": 3.8,
    "solar flare erase banks": 4.0, "withdraw all cash": 3.5,
    "guaranteed 500": 4.0, "zero dollars electricity": 3.8,
    "order now": 2.5, "order the miracle": 4.0, "secret tincture": 3.8,
    "biometric chip": 3.5, "mandatory chip": 3.8,
    # Emotional manipulation
    "urgent warning": 3.5, "emergency warning": 3.2, "breaking alert": 2.8,
    "this is not a drill": 3.5, "act now before": 3.0,
    "stock up": 2.5, "stock up now": 3.0,
}

# ──────────────────────────────────────────────────────────────
# CREDIBILITY / REAL-NEWS INDICATORS  (negative weight = more real)
# ──────────────────────────────────────────────────────────────
CREDIBLE_SINGLE = {
    # Scientific process
    "peer-reviewed": 3.5, "placebo": 2.5, "spectroscopy": 2.8,
    "redshift": 2.8, "telescope": 2.0, "researchers": 2.0,
    "astronomers": 2.2, "biologists": 2.2, "physicists": 2.2,
    "university": 2.0, "efficacy": 2.5, "cohort": 2.5, "longitudinal": 2.8,
    "statistically": 2.5, "randomized": 2.8, "blinded": 2.5,
    # Institutions
    "cern": 3.0, "nasa": 2.8, "esa": 2.8, "nih": 3.0, "fda": 2.8,
    "who": 2.2, "cisa": 3.0, "ecb": 2.8, "imf": 2.8, "sec": 2.5,
    "reuters": 3.5, "bloomberg": 3.0,
    # Legal / governance
    "statutory": 2.8, "antitrust": 2.5, "parliamentary": 2.5, "treaty": 2.5,
    "bipartisan": 2.5, "legislation": 2.2, "testimony": 2.5, "verdict": 2.2,
    "indictment": 2.5, "subpoena": 2.8, "injunction": 2.5,
    # Economics
    "inflation": 2.0, "unemployment": 2.5, "gdp": 2.5, "nonfarm": 3.0,
}

CREDIBLE_PHRASES = {
    # Peer review & science
    "clinical trial": 3.2, "phase iii": 3.0, "phase 3": 3.0,
    "double-blind": 3.2, "double blind": 3.2, "randomized controlled": 3.5,
    "astrophysical journal": 3.5, "the lancet": 3.5, "nature astronomy": 3.5,
    "nature medicine": 3.5, "new england journal": 3.8, "science journal": 3.0,
    "data safety monitoring": 3.2, "statistically significant": 3.0,
    "monoclonal antibody": 3.2, "peer review": 3.2, "meta-analysis": 3.5,
    "systematic review": 3.2, "confidence interval": 3.0,
    # Journalism attribution
    "associated press": 3.5, "according to": 2.5, "official report": 2.8,
    "spokesperson": 2.5, "press briefing": 2.8, "statement released": 2.5,
    "announced wednesday": 2.2, "announced today": 2.2,
    "in a formal statement": 2.8, "press conference": 2.4,
    "court filing": 3.0, "official transcript": 3.2, "committee hearing": 3.0,
    # Governance & economics
    "federal reserve": 3.2, "supreme court": 3.2, "clean air act": 3.0,
    "department of justice": 3.2, "bureau of labor statistics": 3.5,
    "european central bank": 3.2, "international monetary fund": 3.2,
    "world health organization": 3.2, "cybersecurity agency": 2.8,
    "sec filing": 3.2, "unemployment rate": 2.8, "nonfarm payroll": 3.0,
    "annual report": 2.5, "earnings report": 2.5,
    # Medical official
    "health authority": 2.8, "regulatory approval": 3.0, "fda approved": 3.2,
    "clinical evidence": 3.0, "treatment guideline": 2.8, "red cross": 2.8,
    "humanitarian corridor": 2.8, "maritime safety": 2.5,
}

# Combined for backward-compat lookups
SENSATIONAL_TRIGGERS = {**SENSATIONAL_SINGLE, **SENSATIONAL_PHRASES}
CREDIBLE_TRIGGERS = {**CREDIBLE_SINGLE, **CREDIBLE_PHRASES}


def _match_triggers(lower_text: str, single_dict: dict, phrase_dict: dict):
    """
    Match single-word triggers with word-boundary anchors (precise),
    and multi-word phrase triggers with plain substring search (avoids \\b breaking on spaces).
    Returns (hits list, total weight).
    """
    hits = []
    weight = 0.0

    # Single-word: word-boundary match
    for trigger, w in single_dict.items():
        pattern = r'\b' + re.escape(trigger) + r'\b'
        count = len(re.findall(pattern, lower_text))
        if count > 0:
            hits.append(trigger)
            weight += w * count

    # Phrases: plain substring (no boundary anchors needed)
    for trigger, w in phrase_dict.items():
        count = lower_text.count(trigger)
        if count > 0:
            hits.append(trigger)
            weight += w * count

    return hits, weight


def analyze_linguistics(title: str, text: str) -> dict:
    """Extracts rich linguistic, rhetorical, and stylistic features."""
    full_text = f"{title} {text}".strip()
    words = re.findall(r'\b[a-zA-Z]+\b', full_text)
    total_words = max(len(words), 1)

    # 1. Uppercase & Capitalization Anomaly
    KNOWN_ACRONYMS = {"NASA", "FDA", "WHO", "CERN", "CISA", "NIH", "ECB",
                      "IMF", "RSV", "ESA", "NIST", "FIDO2", "AP", "CDC",
                      "CIA", "FBI", "SEC", "GDP", "CBDC", "DNA", "RNA",
                      "AI", "UK", "US", "EU", "UN", "WTO", "OPEC"}
    all_caps_words = [w for w in words if len(w) > 2 and w.isupper() and w not in KNOWN_ACRONYMS]
    uppercase_ratio = len(all_caps_words) / total_words
    cap_score = min(100.0, uppercase_ratio * 450)

    # 2. Exclamation & Punctuation Density
    exclamations = full_text.count("!")
    question_marks = full_text.count("?")
    multi_punct = len(re.findall(r'[!?]{2,}', full_text))
    punct_density = (exclamations * 2.5 + question_marks * 1.5 + multi_punct * 4) / max(len(full_text.split(".")), 1)
    punct_score = min(100.0, punct_density * 28)

    # 3. Sensationalism & Credibility Matches (FIXED phrase matching)
    lower_text = full_text.lower()

    sensational_hits, sensational_weight = _match_triggers(
        lower_text, SENSATIONAL_SINGLE, SENSATIONAL_PHRASES
    )
    credible_hits, credible_weight = _match_triggers(
        lower_text, CREDIBLE_SINGLE, CREDIBLE_PHRASES
    )

    sensationalism_index = min(100.0, max(0.0, (sensational_weight / max(total_words / 22, 1)) * 32))
    credibility_marker_index = min(100.0, max(0.0, (credible_weight / max(total_words / 22, 1)) * 35))

    # 4. Readability Score (Flesch Reading Ease)
    sentences = max(len(re.split(r'[.!?]+', full_text)), 1)
    avg_sentence_len = total_words / sentences
    syllables = sum(max(1, len(re.findall(r'[aeiouy]+', w.lower()))) for w in words)
    avg_syllables_per_word = syllables / total_words
    reading_ease = max(0.0, min(100.0, 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables_per_word)))

    # 5. Grey-zone / satire detection
    satire_signals = sum(1 for w in ["satire", "parody", "opinion", "editorial", "satirical"] if w in lower_text)
    is_grey_zone = satire_signals > 0 or (20.0 < sensationalism_index < 45.0 and 20.0 < credibility_marker_index < 50.0)

    # 6. Natural Language Reasoning Generation
    reasons = []
    if sensational_hits:
        reasons.append(f"Sensationalist / clickbait vocabulary detected: {', '.join(sensational_hits[:5])}.")
    if credible_hits:
        reasons.append(f"Empirical & verified institutional markers found: {', '.join(credible_hits[:5])}.")
    if exclamations >= 2:
        reasons.append(f"Unusual exclamation frequency ({exclamations} occurrences) indicating urgent emotional appeal.")
    if len(all_caps_words) >= 2:
        reasons.append(f"Excessive ALL-CAPS words ({', '.join(all_caps_words[:4])}) typical of yellow journalism.")
    if is_grey_zone and not sensational_hits and not credible_hits:
        reasons.append("Content has mixed linguistic signals — model confidence may be moderate. Cross-check with a primary source.")
    if not reasons:
        reasons.append("Standard neutral journalistic prose structure with measured syntax and no extreme vocabulary.")

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
        "is_grey_zone": is_grey_zone,
        "reasoning_bullets": reasons
    }


def compute_token_saliency(full_text: str, fake_prob: float) -> list:
    """Calculates granular token-by-token contribution weights."""
    raw_tokens = re.split(r'(\s+|[.,!?;:()"\'\\-])', full_text)
    saliency_items = []

    for raw in raw_tokens:
        if not raw:
            continue
        cleaned = raw.lower().strip()
        weight = 0.0
        label_class = "neutral"

        if re.match(r'^[a-zA-Z0-9]+$', cleaned):
            # Check single-word triggers only (phrases can't match a single token)
            if cleaned in SENSATIONAL_SINGLE:
                weight = SENSATIONAL_SINGLE[cleaned] * 28.0
            elif cleaned in CREDIBLE_SINGLE:
                weight = -CREDIBLE_SINGLE[cleaned] * 28.0
            elif raw.isupper() and len(raw) > 2 and raw not in {
                "NASA", "FDA", "WHO", "CERN", "CISA", "NIH", "CDC", "CIA",
                "FBI", "SEC", "ECB", "IMF", "DNA", "RNA", "AI", "UK", "US", "EU"
            }:
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
