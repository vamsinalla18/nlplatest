import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from symptom_patterns import SYMPTOM_PATTERNS, match_symptoms

os.makedirs("data", exist_ok=True)

print("Loading dataset...")

dataset = load_dataset(
    "gretelai/symptom_to_diagnosis",
    split="train"
)

# --- Per-sample extraction with frequency threshold ---
# For each disease, count in how many patient descriptions each symptom appears.
# A triple is included only if the symptom is mentioned in >= THRESHOLD of the
# disease's descriptions. This filters out single-sample false positives while
# preserving disease-specific rare-but-real symptoms at a lower threshold.

COMMON_THRESHOLD = 0.20   # generic symptoms need ≥20% occurrence
SPECIFIC_THRESHOLD = 0.12  # disease-specific clinical symptoms need ≥12%

# Diagnostically specific symptoms get the lower threshold.
SPECIFIC_SYMPTOMS = {
    "burning_urination", "blood_in_urine", "frequent_urination", "cloudy_urine",
    "dark_urine", "pelvic_pain", "yellow_skin", "retro_orbital_pain", "photophobia",
    "phonophobia", "blurred_vision", "wheezing", "blisters", "pus", "nail_changes",
    "visible_veins", "leg_cramps", "skin_discoloration", "excessive_thirst",
    "skin_scaling", "constipation", "throat_swelling", "joint_stiffness",
}

print("Extracting symptoms per disease from patient descriptions...")

disease_symptom_counts = defaultdict(lambda: defaultdict(int))
disease_sample_counts = defaultdict(int)
diseases = set()

for row in tqdm(dataset):
    disease = row["output_text"].strip().lower()
    text = row["input_text"]
    diseases.add(disease)
    disease_sample_counts[disease] += 1
    for sym in match_symptoms(text):
        disease_symptom_counts[disease][sym] += 1

# Build triples using thresholds
triples = set()
symptoms = set()

for disease in sorted(diseases):
    n = disease_sample_counts[disease]
    for sym, count in disease_symptom_counts[disease].items():
        threshold = SPECIFIC_THRESHOLD if sym in SPECIFIC_SYMPTOMS else COMMON_THRESHOLD
        if count / n >= threshold:
            triples.add((disease, "has_symptom", sym))
            symptoms.add(sym)

print("Saving triples...")

triples_list = sorted(triples)
triples_df = pd.DataFrame(triples_list, columns=["head", "relation", "tail"])
triples_df.to_csv("data/triples.csv", index=False, header=False)
print("Total unique triples:", len(triples_df))

# ---------- ENTITY TYPES ----------

types = []
for d in sorted(diseases):
    types.append((d, "Disease"))
for s in sorted(symptoms):
    types.append((s, "Symptom"))

types_df = pd.DataFrame(types)
types_df.to_csv("data/entity_types.csv", index=False, header=False)

print("Total entities:", len(types))
print("Diseases:", len(diseases))
print("Symptoms:", len(symptoms))
