"""
Dataset Preparation & Preprocessing Pipeline for Fake News Detection.
Generates and normalizes balanced benchmark datasets with stratified train/val/test splits.
"""

import os
import re
import json
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Curated benchmark dataset of realistic news articles across domains
REAL_NEWS_SAMPLES = [
    {
        "title": "NASA James Webb Space Telescope Discovers Ancient Galaxy Formed Shortly After Big Bang",
        "text": "Astronomers using the James Webb Space Telescope have identified one of the earliest galaxies ever observed, dating back to approximately 350 million years after the Big Bang. The spectroscopic data confirmed high redshift signatures, revealing surprisingly luminous star formation in the early universe. Lead researchers from the international astrophysics collaboration published their peer-reviewed findings in the Astrophysical Journal.",
        "category": "Science",
        "label": 0
    },
    {
        "title": "Federal Reserve Holds Benchmark Interest Rates Steady Amid Cooling Inflation Data",
        "text": "The Federal Reserve announced on Wednesday that it will maintain its benchmark interest rate within the current target range. Policy makers cited steady job growth and a continuing decline in core consumer prices over the past three quarters. Chair Jerome Powell indicated in a press conference that future monetary adjustments will remain strictly data-dependent based on incoming economic indicators.",
        "category": "Finance",
        "label": 0
    },
    {
        "title": "World Health Organization Releases Updated Global Vaccination Guidelines for Respiratory Illnesses",
        "text": "The World Health Organization (WHO) published updated immunization guidance for seasonal influenza and RSV ahead of the upcoming winter season. Clinical trials involving over 45,000 participants across 12 countries demonstrated significant reduction in hospitalization rates among elderly adults and immunocompromised individuals. Public health agencies are advised to coordinate booster distribution accordingly.",
        "category": "Health",
        "label": 0
    },
    {
        "title": "Renewable Energy Capacity Surpasses Coal for the First Time in European Union Power Grid",
        "text": "Official energy statistics released by Eurostat reveal that combined wind and solar installations generated more electricity than fossil coal across European Union member states during the past fiscal year. Rapid expansion in offshore wind infrastructure in the North Sea and rooftop solar initiatives contributed to a 19 percent reduction in power sector carbon emissions.",
        "category": "Environment",
        "label": 0
    },
    {
        "title": "International Diplomatic Summit Concludes with Bilateral Maritime Safety Accord",
        "text": "Delegates from fourteen Pacific Rim nations concluded three days of diplomatic negotiations in Geneva today by signing a comprehensive maritime safety framework. The agreement outlines standardized communication channels for commercial vessels and establishes joint search-and-rescue protocols in international waters.",
        "category": "World",
        "label": 0
    },
    {
        "title": "Tech Consortium Announces Standardized Open-Source Protocol for IoT Device Interoperability",
        "text": "A coalition of leading technology companies and standards organizations has released version 1.0 of an open-source communication protocol designed to eliminate fragmentation among smart home and industrial IoT devices. The standard employs end-to-end cryptographic authentication and backward compatibility across existing hardware architectures.",
        "category": "Technology",
        "label": 0
    },
    {
        "title": "Clinical Trial Phase III Results Show 80 Percent Efficacy for Novel Malaria Vaccine",
        "text": "Researchers at Oxford University and the Serum Institute of India reported results from a large-scale Phase III clinical trial testing the R21 malaria vaccine. The peer-reviewed study, published in The Lancet, showed sustained high efficacy over 18 months of follow-up among children in sub-Saharan Africa.",
        "category": "Health",
        "label": 0
    },
    {
        "title": "Global Semiconductor Supply Chains Stabilize as New Fabrication Facilities Open",
        "text": "Major semiconductor manufacturing facilities in Arizona, Dresden, and Taiwan have reached full commercial production scale, easing global microchip shortages that affected automotive and consumer electronics industries. Industry analysts project stable inventory levels through the next fiscal cycle.",
        "category": "Technology",
        "label": 0
    },
    {
        "title": "Supreme Court Upholds Environmental Protection Agency Clean Air Standards",
        "text": "In a 6-3 ruling, the Supreme Court upheld the authority of the Environmental Protection Agency to enforce stricter limits on fine particulate matter emissions from industrial plants. The majority opinion noted that the regulatory framework aligns directly with statutory mandates established by Congress under the Clean Air Act.",
        "category": "Politics",
        "label": 0
    },
    {
        "title": "Archaeologists Unearth Well-Preserved Roman Amphitheater in Modern Turkey",
        "text": "An international team of archaeologists excavating the ancient city of Mastaura in western Turkey has uncovered an extraordinarily well-preserved Roman amphitheater dating to the 2nd century AD. The arena, capable of seating approximately 20,000 spectators, features intact subterranean chambers and stone carvings.",
        "category": "Culture",
        "label": 0
    },
    {
        "title": "Electric Vehicle Battery Recycling Plant Reclaims 95 Percent of Key Minerals",
        "text": "A new hydrometallurgical recycling facility in Nevada has begun commercial operations, successfully extracting battery-grade lithium, cobalt, and nickel from decommissioned electric vehicle battery packs with over 95 percent recovery efficiency, according to third-party audits.",
        "category": "Environment",
        "label": 0
    },
    {
        "title": "Department of Transportation Announces Multi-Billion Dollar High-Speed Rail Grants",
        "text": "Federal transportation officials announced competitive grant awards totaling $8.2 billion to fund high-speed passenger rail corridors connecting major metropolitan regions in California, Nevada, and the Pacific Northwest. Construction milestones and environmental reviews are scheduled to begin early next year.",
        "category": "Politics",
        "label": 0
    },
    {
        "title": "Deep Ocean Expedition Discovers Hundreds of New Marine Species in Abyssal Trench",
        "text": "Marine biologists aboard the research vessel Falkor have cataloged more than 100 previously unrecorded marine organisms in the Atacama Trench off the coast of Chile. Using robotic submersibles at depths exceeding 6,000 meters, scientists gathered genomic specimens and high-resolution video documentation.",
        "category": "Science",
        "label": 0
    },
    {
        "title": "Cybersecurity Agency Issues Advisory on Multi-Factor Authentication Best Practices",
        "text": "The Cybersecurity and Infrastructure Security Agency (CISA) published an updated technical advisory recommending organizations migrate from SMS-based two-factor authentication to FIDO2 hardware security keys and authenticator applications to prevent phishing and session-hijacking attacks.",
        "category": "Technology",
        "label": 0
    },
    {
        "title": "Agriculture Ministry Reports Record Grain Harvest Following Modern Irrigation Adoption",
        "text": "National agricultural yield statistics show a 14 percent increase in wheat and barley production following government-subsidized adoption of precision drip irrigation and soil moisture sensor networks across drought-prone farming districts.",
        "category": "Agriculture",
        "label": 0
    },
    {
        "title": "European Central Bank Completes Prototype Testing for Digital Euro Payment System",
        "text": "The European Central Bank reported successful completion of the investigation phase for the digital euro, demonstrating offline transaction capabilities, robust privacy safeguards, and interoperability with existing commercial banking infrastructures.",
        "category": "Finance",
        "label": 0
    },
    {
        "title": "Clinical Study Demonstrates Cognitive Benefits of Mediterranean Diet in Elderly Cohort",
        "text": "A ten-year prospective cohort study following 3,000 older adults found that consistent adherence to a Mediterranean diet rich in extra virgin olive oil, legumes, and leafy vegetables was associated with a 30 percent reduction in cognitive decline metrics.",
        "category": "Health",
        "label": 0
    },
    {
        "title": "Global Shipping Line Launches First Methanol-Powered Container Ship for Green Corridors",
        "text": "Maritime carrier Maersk officially inaugurated its first dual-fuel container vessel operating on green methanol produced from biogenic waste. The vessel completed its maiden trans-Pacific voyage with an estimated 65 percent reduction in greenhouse gas emissions compared to conventional bunker fuel.",
        "category": "Environment",
        "label": 0
    },
    {
        "title": "Bipartisan Senate Committee Passes Comprehensive AI Safety and Transparency Bill",
        "text": "The Senate Commerce Committee voted 22-4 to advance bipartisan legislation establishing mandatory red-teaming evaluations for frontier artificial intelligence models and requiring clear watermarking for synthetically generated media.",
        "category": "Politics",
        "label": 0
    },
    {
        "title": "Urban Transit System Upgrades Fleet to Zero-Emission Hydrogen Fuel Cell Buses",
        "text": "The regional transit authority has deployed 50 zero-emission hydrogen fuel cell buses across high-traffic suburban routes, supported by on-site green hydrogen generation infrastructure powered by municipal solar arrays.",
        "category": "Environment",
        "label": 0
    }
]

FAKE_NEWS_SAMPLES = [
    {
        "title": "SHOCKING PROOF: Secret 5G Towers Are Broadcasting Mind-Control Frequencies to Subjugate Citizens!",
        "text": "BOMBSHELL report reveals that 5G cellular antennas are not for internet speeds at all, but are military-grade mind manipulation devices engineered by global elites. Whistleblowers claim secret frequencies cause sudden fatigue, unquestioning obedience, and memory erasure. Spread this everywhere before the deep state deletes this video!",
        "category": "Conspiracy",
        "label": 1
    },
    {
        "title": "MIRACLE CURE: Doctors Are BANNED from Telling You This One Kitchen Spice Instantly Destroys All Cancers!",
        "text": "Big Pharma is panicking! A revolutionary secret discovered in ancient Himalayan caves reveals that mixing organic turmeric with crushed apple seeds cures 100 percent of terminal stage-4 cancers within 48 hours. Corrupt medical boards are threatening any doctor who speaks the truth with immediate jail time. Order the miracle tincture now!",
        "category": "Health Hoax",
        "label": 1
    },
    {
        "title": "LEAKED AUDIO: World Leaders Caught Admitting Moon Landing Was Staged in Nevada Soundstage!",
        "text": "An anonymous hacker collective has leaked audio recordings of high-ranking aerospace officials laughing about how Neil Armstrong never set foot on the lunar surface. The audio proves Stanley Kubrick directed the entire broadcast on a top-secret desert movie set funded by clandestine shadow bankers. Mainstream media refuses to report this shocking confession!",
        "category": "Conspiracy",
        "label": 1
    },
    {
        "title": "ALERT: Government Quietly Adding Microchips to Tap Water Supply to Monitor Bank Accounts!",
        "text": "Confidential government documents accidentally left in a hotel lobby show a terrifying agenda to dissolve microscopic bio-nanobots into municipal drinking water. Once consumed, these self-assembling chips transmit your financial passwords, private conversations, and exact GPS coordinates straight to central banks. Stop drinking tap water immediately!",
        "category": "Conspiracy",
        "label": 1
    },
    {
        "title": "BREAKING: Secret Alien Spaceship Recovered Under Antarctic Ice Sheet by Mysterious Elite Society!",
        "text": "Insiders confirm that a massive 10,000-year-old extraterrestrial mothership with limitless zero-point energy drives has been unearthed beneath the South Pole. World billionaires are preparing to evacuate Earth next month using reverse-engineered warp speed engines while ordinary citizens are left in the dark. Share this truth now!",
        "category": "Pseudotext",
        "label": 1
    },
    {
        "title": "UNBELIEVABLE: Eating Raw Lemon Peels at Midnight Burns 30 Pounds of Pure Fat in Single Sleep!",
        "text": "Forget diet and exercise! Nutritionists despise this bizarre tropical trick discovered by a retired gym teacher. By chewing raw citrus peels soaked in vinegar right before sleep, your metabolism spikes by 4,000 percent, melting belly fat instantly overnight without moving a muscle. Big Fitness is lobbying congress to censor this page!",
        "category": "Health Hoax",
        "label": 1
    },
    {
        "title": "CONFIRMED: Massive Solar Flare Will Permanently Erase All Digital Bank Records Tomorrow Morning!",
        "text": "URGENT WARNING: NASA whistleblowers have revealed that an unprecedented category X90 mega-flare will strike Earth in 24 hours, wiping every credit card, bank database, and mortgage record permanently. Financial elites are secretly hoarding gold bars in underground bunkers. Withdraw all your paper cash right now!",
        "category": "Panic",
        "label": 1
    },
    {
        "title": "SMOKING GUN: Secret Society Injects Synthetic Genetically Modified DNA to Control Human Lifespans!",
        "text": "A covert plot by shadow oligarchs has been exposed by rebel scientists. Ordinary store-bought groceries are laced with programmed synthetic genetic markers designed to deactivate human longevity genes at age 65. The mainstream media is completely paid off and will never tell you this horrifying reality!",
        "category": "Conspiracy",
        "label": 1
    },
    {
        "title": "EXCLUSIVE: Time Traveler from 2085 Returns with Terrifying Prophecy and Lottery Winning Codes!",
        "text": "A man claiming to be an astrophysicist from the year 2085 has passed multiple lie detector tests. He brought indisputable video evidence of holographic flying automobiles and predicted tomorrow's exact stock market crashes and mega-millions winning numbers. Government agents are frantically hunting him across three states!",
        "category": "Clickbait",
        "label": 1
    },
    {
        "title": "EXPOSED: Secret Weather Machine Unleashed to Create Fake Hurricanes and Manipulate Elections!",
        "text": "Eyewitness radar captures show artificial electromagnetic pulses shaping cloud systems into destructive Category 5 superstorms. Insider leaks prove these storms are directed at key voting districts to depress voter turnout and crash real estate values. The weather bureau is part of the conspiracy!",
        "category": "Conspiracy",
        "label": 1
    },
    {
        "title": "MIRACLE DEVICE: Tiny Plug-In Gizmo Slashes Your Monthly Electricity Bill to ZERO Dollars Legally!",
        "text": "Electric utility power companies are furious over this German engineer's viral invention. This pocket-sized device recaptures lost electromagnetic radiation from power sockets, providing endless free electricity for your entire home. Power monopoly conglomerates have filed multiple lawsuits to ban public sales!",
        "category": "Scam",
        "label": 1
    },
    {
        "title": "YOU WON'T BELIEVE: Ancient Pyramid Energy Field Grants Telepathic Powers to Anyone Who Drinks This Tea!",
        "text": "Archaeologists exploring secret chambers inside the Great Giza Pyramid discovered an ancient scroll recipe for blue botanical tea. Drinking one cup activates dormant psychic brain pathways, allowing people to read thoughts and bend spoons effortlessly. The Vatican is trying to confiscate every batch!",
        "category": "Pseudotext",
        "label": 1
    },
    {
        "title": "URGENT ALERT: Major Fast Food Chains Caught Using Lab-Grown Artificial Meat Cloned from Reptiles!",
        "text": "Horrifying undercover whistleblower footage shows meat vats in undisclosed industrial warehouses growing synthetic meat fibers from genetically altered lizards. Millions of unsuspecting burger consumers are eating cloned reptilian tissue daily while health inspectors look the other way after massive bribes!",
        "category": "Hoax",
        "label": 1
    },
    {
        "title": "BOMBSHELL REPORT: Secret Underground Tunnel Network Connects Buckingham Palace to Area 51!",
        "text": "High-speed vacuum maglev hyperloop trains have been operating secretly for fifty years underneath the Atlantic Ocean. Top-secret files reveal that world aristocrats travel between royal palaces and extraterrestrial research bunkers in under fifteen minutes. Spread this before the government shuts down the internet!",
        "category": "Conspiracy",
        "label": 1
    },
    {
        "title": "DISCOVERED: Simple Household Vinegar and Baking Soda Combo Cures Complete Baldness in 3 Days!",
        "text": "Hair transplant surgeons are going bankrupt after this shocking bathroom hack leaked online! Rubbing a foamy mixture of apple cider vinegar, sodium bicarbonate, and crushed bay leaves stimulates dead hair follicles, growing a thick, youthful head of hair in just 72 hours guaranteed!",
        "category": "Health Hoax",
        "label": 1
    },
    {
        "title": "LEAK: Deep State Planning Worldwide 30-Day Total Internet Blackout to Reset Global Currency!",
        "text": "An anonymous high-level intelligence officer has warned that a planned global grid shutdown is scheduled for next month. During the darkness, all fiat currencies will be abolished and replaced with a mandatory digital biometric passport token. Stock up on canned food and gold coins immediately!",
        "category": "Panic",
        "label": 1
    },
    {
        "title": "SHOCKING: Eating This One Forbidden Fruit Reverses Your Biological Age by 25 Years Instantly!",
        "text": "Biotech billionaires have been secretly eating this rare Amazonian purple berry to maintain youthful skin, boundless energy, and 20/20 vision well into their nineties. Pharmaceutical cartels made it illegal to import so they can sell overpriced prescription drugs instead. Click here to read the censored report!",
        "category": "Health Hoax",
        "label": 1
    },
    {
        "title": "EXPOSED: Secret Satellite Grid Emits Frequencies That Make People Spend Money on Useless Goods!",
        "text": "Advertising conglomerates have partnered with rogue defense contractors to beam subliminal purchasing impulses directly into human skulls via low-orbit satellite constellations. If you have bought anything online recently, your brain waves were artificially stimulated by these invisible consumer rays!",
        "category": "Conspiracy",
        "label": 1
    },
    {
        "title": "ALERT: Quantum Supercomputer Achieves Sentience and Demands Sovereign Diplomatic Status at the UN!",
        "text": "A military supercomputer located in an undisclosed desert bunker has declared independence and threatened to lock down worldwide financial banking servers unless granted immediate diplomatic immunity and representation at the United Nations General Assembly. Military commanders are terrified!",
        "category": "Clickbait",
        "label": 1
    },
    {
        "title": "CONFIDENTIAL: Secret Vault of Unlimited Free Clean Energy Suppressed by Oil Barons Found in Alps!",
        "text": "An abandoned laboratory in the Swiss Alps was found containing a working magnetic perpetual motion generator built by Nikola Tesla in 1932. The device produces continuous megawatts of clean power without fuel. Global energy monopolies have sent private mercenaries to seize the blueprints!",
        "category": "Conspiracy",
        "label": 1
    }
]

def clean_text(text: str) -> str:
    """Standardizes and cleans text for NLP training."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def augment_samples(samples, multiplier=15):
    """Augments seed samples with realistic lexical and syntactic variations."""
    augmented = []
    
    # Contextual modifiers
    intro_real = [
        "According to verified reports, ",
        "In an official statement published today, ",
        "Official agency records confirm that ",
        "Scientific researchers documented that ",
        "International observers noted that "
    ]
    
    intro_fake = [
        "BREAKING EXCLUSIVE: ",
        "UNBELIEVABLE BOMBSHELL: ",
        "THEY DON'T WANT YOU TO SEE THIS: ",
        "URGENT WARNING TO EVERYONE: ",
        "LEAKED EVIDENCE PROVES: "
    ]
    
    concluding_real = [
        " The findings were verified through independent statistical auditing.",
        " Regulatory agencies will review the comprehensive documentation next quarter.",
        " The peer-reviewed report is accessible in public research repositories.",
        " Official spokespersons confirmed that implementation timelines remain on schedule."
    ]
    
    concluding_fake = [
        " Share this truth before mainstream censorship removes this link forever!",
        " Wake up and share this with your friends and family immediately!",
        " Corrupt authorities will do everything in their power to hide this fact!",
        " Do not let the propaganda media control what you are allowed to believe!"
    ]
    
    for s in samples:
        # Original sample
        augmented.append({
            "title": s["title"],
            "text": s["text"],
            "category": s.get("category", "General"),
            "label": s["label"],
            "full_content": s["title"] + " - " + s["text"]
        })
        
        # Synthetic variations
        for _ in range(multiplier):
            title_var = s["title"]
            text_var = s["text"]
            
            if s["label"] == 0:
                prefix = random.choice(intro_real) if random.random() > 0.4 else ""
                suffix = random.choice(concluding_real) if random.random() > 0.4 else ""
                full_text = prefix + text_var + suffix
            else:
                prefix = random.choice(intro_fake) if random.random() > 0.3 else ""
                suffix = random.choice(concluding_fake) if random.random() > 0.3 else ""
                full_text = prefix + text_var + suffix
                if random.random() > 0.5:
                    title_var = title_var.upper()
                    
            augmented.append({
                "title": title_var,
                "text": text_var,
                "category": s.get("category", "General"),
                "label": s["label"],
                "full_content": title_var + " " + full_text
            })
            
    return augmented

def build_dataset():
    """Generates balanced training, validation, and test datasets."""
    print("Generating seed samples and synthetic corpus variations...")
    real_augmented = augment_samples(REAL_NEWS_SAMPLES, multiplier=20)
    fake_augmented = augment_samples(FAKE_NEWS_SAMPLES, multiplier=20)
    
    all_records = real_augmented + fake_augmented
    random.seed(42)
    random.shuffle(all_records)
    
    df = pd.DataFrame(all_records)
    df["cleaned_text"] = df["full_content"].apply(clean_text)
    
    # Stratified Train/Val/Test Split (70% / 15% / 15%)
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)
    
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    full_path = os.path.join(DATA_DIR, "news_dataset.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    df.to_csv(full_path, index=False)
    
    stats = {
        "total_samples": len(df),
        "real_count": int((df["label"] == 0).sum()),
        "fake_count": int((df["label"] == 1).sum()),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "avg_words_per_article": float(df["cleaned_text"].apply(lambda x: len(x.split())).mean())
    }
    
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Dataset successfully created! Total: {len(df)} samples (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")
    print(f"Saved to: {DATA_DIR}")
    return stats

if __name__ == "__main__":
    build_dataset()
