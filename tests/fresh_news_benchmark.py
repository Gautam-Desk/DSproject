"""
Fresh Out-Of-Sample Benchmark Test for VeritasAI Fake News Detection Engine.
Generates 10 completely novel, unseen Real and Fake news stories across multiple domains
and evaluates model accuracy, confidence, risk levels, and explainability.
"""

import urllib.request
import json
import time

FRESH_NEWS_CASES = [
    # ----------------------------------------------------------------------------------
    # 1. AUTHENTIC / REAL NEWS (Freshly Generated)
    # ----------------------------------------------------------------------------------
    {
        "id": "REAL-01",
        "category": "Climate & Space Science",
        "ground_truth": "REAL",
        "title": "NOAA and NASA Report 2025 Global Ocean Surface Temperatures Reached Record High in Multi-Satellite Analysis",
        "text": "Scientists at the National Oceanic and Atmospheric Administration (NOAA) and NASA Goddard Institute for Space Studies published a joint climatological report documenting that global sea surface temperatures averaged 0.95 degrees Celsius above twentieth-century baseline records. The observational data, calibrated across thermal infrared satellite sensors and calibrated ocean buoys, was released in the Geophysical Research Letters peer-reviewed journal."
    },
    {
        "id": "REAL-02",
        "category": "Macroeconomics & Banking",
        "ground_truth": "REAL",
        "title": "Bank of England Lowers Key Benchmark Rate to 4.75 Percent Following Drop in Services Inflation",
        "text": "The Monetary Policy Committee of the Bank of England voted 8-1 on Thursday to reduce the official Bank Rate by 25 basis points to 4.75 percent. Governor Andrew Bailey stated during the press conference that consumer price inflation had stabilized near the 2 percent target, though committee members emphasized that future monetary adjustments will remain data-dependent and measured."
    },
    {
        "id": "REAL-03",
        "category": "Biomedical & Oncology",
        "ground_truth": "REAL",
        "title": "New England Journal of Medicine Study Confirms mRNA Melanoma Vaccine Reduced Recurrence by 44 Percent",
        "text": "In a randomized Phase IIb clinical trial tracking 157 high-risk melanoma patients over three years, researchers documented that combining an individualized mRNA neoantigen therapy with standard immunotherapy reduced the hazard ratio of distant disease recurrence or death by 44 percent. The peer-reviewed findings were published Wednesday with independent oncology review."
    },
    {
        "id": "REAL-04",
        "category": "Semiconductors & AI",
        "ground_truth": "REAL",
        "title": "NVIDIA Announces Next-Generation Blackwell Ultra Architecture with 3nm Process and 288GB HBM3e Memory",
        "text": "In an official corporate technical keynote, NVIDIA unveiled its upcoming Blackwell Ultra computing platform engineered on TSMC's 3-nanometer silicon node. The enterprise accelerator features 288 gigabytes of ultra-high-bandwidth HBM3e memory designed for large-scale multi-modal neural network training and inference clusters."
    },
    {
        "id": "REAL-05",
        "category": "International Law & Treaties",
        "ground_truth": "REAL",
        "title": "International Court of Justice Delivers Binding Advisory Opinion on State Maritime Climate Obligations",
        "text": "The International Court of Justice (ICJ) in The Hague issued a formal advisory opinion confirming that states party to the UN Convention on the Law of the Sea have statutory legal obligations to protect marine environments from greenhouse gas pollution and ocean acidification caused by industrial carbon emissions."
    },

    # ----------------------------------------------------------------------------------
    # 2. FAKE / MISINFORMATION NEWS (Freshly Generated)
    # ----------------------------------------------------------------------------------
    {
        "id": "FAKE-01",
        "category": "Health Pseudo-Cure Hoax",
        "ground_truth": "FAKE",
        "title": "SHOCKING TRUTH: Drinking Raw Onion Juice with Crushed Aspirin Instantly Dissolves All Arterial Plaque Overnight!",
        "text": "Big Pharma is panicking! An underground naturopathic healer has exposed the forbidden ancient remedy that cleanses 100 percent of blocked heart arteries in just 8 hours while you sleep. Corrupt cardiology doctors are secretly bribed by pharmaceutical cartels to hide this miracle cure from everyday patients. Order the sacred tincture before this page is banned!"
    },
    {
        "id": "FAKE-02",
        "category": "5G & Thought-Control Conspiracy",
        "ground_truth": "FAKE",
        "title": "WHISTLEBLOWER BOMBSHELL: New Smart Electric Meters Emit High-Frequency Microwave Pulses to Read Your Private Thoughts!",
        "text": "Terrifying leaked blueprints from an undisclosed defense contractor reveal that mandatory smart home energy meters installed on residential houses are actually military-grade neural thought-scanning antennas. Globalist shadow elites are harvesting private citizen memories and sending them to central AI supercomputers. Wake up and wrap your electric meter in aluminum foil immediately!"
    },
    {
        "id": "FAKE-03",
        "category": "Space / Flat Earth Hoax",
        "ground_truth": "FAKE",
        "title": "LEAKED NASA MEMO: Space Agency Accidentally Admits Earth Is Enclosed in a Giant Impenetrable Glass Dome!",
        "text": "Anonymous hackers from an underground whistleblowing forum have released classified audio of high-ranking astronauts confessing that rockets never reach outer space because they bounce off a massive electromagnetic dome at 100 kilometers altitude. Mainstream news is completely paid off to maintain the round earth illusion. Share this truth before the deep state deletes it forever!"
    },
    {
        "id": "FAKE-04",
        "category": "Financial Panic & Banking Scam",
        "ground_truth": "FAKE",
        "title": "URGENT ALERT: Central Bankers Are Secretly Freezing All Private Savings Accounts Starting Friday at Midnight!",
        "text": "BOMBSHELL confidential memo leaked from the World Economic Forum confirms that all physical bank accounts will be seized by midnight to force citizens onto biometric digital ration cards. If you have money in any checking or savings account, withdraw every single dollar in physical cash right now before the total financial blackout begins!"
    },
    {
        "id": "FAKE-05",
        "category": "Celebrity & Alien Deepfake Hoax",
        "ground_truth": "FAKE",
        "title": "BOMBSHELL VIDEO: Top Hollywood Actor Flees Country After Exposing Secret Underground Alien Cloning Facility in Nevada!",
        "text": "A viral 4K video showing underground cryogenic chambers where shapeshifting reptilian extraterrestrials clone world celebrities has shocked the internet. The famous movie star who filmed the footage has escaped to South America while federal agents frantically scrub the footage from all social media platforms. You won't believe what they found inside chamber seven!"
    }
]

def run_fresh_news_check():
    print("=" * 80)
    print("   VERITASAI FRESH NEWS BENCHMARK EVALUATION (OUT-OF-SAMPLE TEST)   ")
    print("=" * 80)
    
    url = "http://127.0.0.1:8000/api/predict"
    correct_count = 0
    total_count = len(FRESH_NEWS_CASES)
    
    results_summary = []
    
    for item in FRESH_NEWS_CASES:
        payload = {
            "title": item["title"],
            "text": item["text"]
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        start = time.time()
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"Error calling API for {item['id']}: {e}")
            continue
            
        elapsed_ms = (time.time() - start) * 1000
        
        predicted_verdict = data["verdict"]
        is_correct = (predicted_verdict == item["ground_truth"])
        if is_correct:
            correct_count += 1
            
        real_prob = data["real_probability"]
        fake_prob = data["fake_probability"]
        confidence = data["confidence"]
        risk_level = data["risk_level"]
        reasons = data.get("linguistics", {}).get("reasoning_bullets", ["Standard analysis."])
        
        results_summary.append({
            "id": item["id"],
            "category": item["category"],
            "ground_truth": item["ground_truth"],
            "predicted": predicted_verdict,
            "is_correct": is_correct,
            "real_prob": real_prob,
            "fake_prob": fake_prob,
            "confidence": confidence,
            "risk_level": risk_level,
            "reasons": reasons,
            "title": item["title"],
            "latency": round(elapsed_ms, 1)
        })
        
        status_symbol = "[PASS]" if is_correct else "[FAIL]"
        
        print(f"\n[{item['id']}] Category: {item['category']}")
        print(f"Headline: \"{item['title']}\"")
        print(f"Ground Truth: {item['ground_truth']} | Predicted: {predicted_verdict} -> {status_symbol}")
        print(f"Probabilities: Real = {real_prob:.2f}% | Fake = {fake_prob:.2f}% (Confidence: {confidence:.1f}%)")
        print(f"Assessed Risk: {risk_level}")
        if reasons:
            print(f"Reasoning: {reasons[0]}")
        print("-" * 80)
        
    accuracy_pct = (correct_count / total_count) * 100
    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED: {correct_count}/{total_count} Correct ({accuracy_pct:.1f}% Accuracy)")
    print("=" * 80)
    
    # Save test results to json for records
    with open("tests/fresh_news_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy_percentage": accuracy_pct,
            "total_tested": total_count,
            "passed": correct_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results_summary
        }, f, indent=2)

if __name__ == "__main__":
    run_fresh_news_check()
