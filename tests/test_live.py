import urllib.request
import json

test_cases = [
    {
        "title": "Senate passes bipartisan foreign security assistance bill",
        "text": "The Senate voted 65-35 on Tuesday to approve a package providing critical aid according to Reuters reports."
    },
    {
        "title": "MIRACLE CURE: Doctors are banned from telling you this Himalayan root cures all cancer in 48 hours!",
        "text": "Big Pharma is hiding this ancient secret from you! Order the miracle drops now!"
    },
    {
        "title": "Apple announces new M4 chip with neural engine for Mac computers",
        "text": "In a press release from Cupertino, Apple unveiled its latest 3nm semiconductor processor with improved power efficiency."
    },
    {
        "title": "Secret 5G mind control frequencies activated across residential cities",
        "text": "Whistleblower leaks classified agenda to enslave citizens! Wake up before they delete this video!"
    }
]

print("=" * 65)
print("Evaluating Real-World News Samples on VeritasAI Engine:")
print("=" * 65)

for tc in test_cases:
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/predict",
        data=json.dumps(tc).encode(),
        headers={"Content-Type": "application/json"}
    )
    res = json.loads(urllib.request.urlopen(req).read().decode())
    print(f"\nHeadline: {tc['title']}")
    print(f"-> Verdict: {res['verdict']} | Real: {res['real_probability']}% | Fake: {res['fake_probability']}% | Confidence: {res['confidence']}%")
    print(f"-> Risk Level: {res['risk_level']}")
    if res.get("linguistics", {}).get("reasoning_bullets"):
        print(f"-> Reasoning: {res['linguistics']['reasoning_bullets'][0]}")

print("\nAll live real-world tests evaluated successfully!")
