"""
Generates all research-paper-quality plots:
  1. t-SNE embedding visualization
  2. Disease-Symptom affinity matrix
  3. Top-k accuracy (k=1,3,5)
  4. Disease confusion matrix
  5. Symptom discriminability
  6. Clinical sanity checks
  7. Symptoms per disease (data distribution)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import defaultdict
from scipy.stats import entropy as scipy_entropy

from tokenizers import Tokenizer
from model.multitask_model import MultiTaskModel

OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer.from_file("tokenizer/medical_tokenizer.json")
vocab_size  = tokenizer.get_vocab_size()
pad_id      = tokenizer.token_to_id("[PAD]")

types_df   = pd.read_csv("data/entity_types.csv", header=None)
triples_df = pd.read_csv("data/triples.csv", header=None)

num_types     = len(sorted(set(types_df[1].tolist())))
num_relations = len(set(triples_df[1].tolist()))

model = MultiTaskModel(vocab_size, num_types=num_types,
                       num_relations=num_relations, pad_id=pad_id).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device,
                                  weights_only=False))
model.eval()

diseases = [e for e, t in zip(types_df[0], types_df[1]) if t == "Disease"]
symptoms = [e for e, t in zip(types_df[0], types_df[1]) if t == "Symptom"]

# KG: disease → set of symptoms
disease_symptoms = defaultdict(set)
for _, row in triples_df.iterrows():
    disease_symptoms[row[0]].add(row[2])

MAX_LEN     = 16
relation_id = torch.tensor([0]).to(device)

def encode(name):
    ids = tokenizer.encode(name).ids[:MAX_LEN]
    ids = ids + [pad_id] * (MAX_LEN - len(ids))
    return torch.tensor(ids).unsqueeze(0).to(device)

def get_embedding(name):
    with torch.no_grad():
        t  = encode(name)
        sg = torch.zeros_like(t)
        return model.encoder(t, sg).mean(dim=1).squeeze().cpu().numpy()

def rel_score(disease, symptom):
    with torch.no_grad():
        return model.relation_score(encode(disease), relation_id,
                                    encode(symptom)).item()

# ── Pre-compute embeddings (used by plots 1) ──────────────────────────────────
print("Computing entity embeddings...")
all_entities = diseases + symptoms
embeddings   = np.array([get_embedding(e) for e in all_entities])
labels       = ["Disease"] * len(diseases) + ["Symptom"] * len(symptoms)

# ── Pre-compute affinity matrix (used by plots 2, 4, 5) ───────────────────────
print("Computing affinity matrix (54 diseases × 93 symptoms)...")
aff = np.zeros((len(diseases), len(symptoms)))
for i, d in enumerate(diseases):
    for j, s in enumerate(symptoms):
        aff[i, j] = rel_score(d, s)
# Lower L2 = more related → negate for display (higher = more related)
aff_sim = -aff


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — t-SNE Embedding Visualization
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 1: t-SNE...")
tsne   = TSNE(n_components=2, perplexity=20, random_state=42, max_iter=2000)
coords = tsne.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(14, 10))
colors  = {"Disease": "#e53e3e", "Symptom": "#3182ce"}
markers = {"Disease": "D",       "Symptom": "o"}
sizes   = {"Disease": 120,        "Symptom": 60}

for lbl in ["Disease", "Symptom"]:
    mask = [i for i, l in enumerate(labels) if l == lbl]
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=colors[lbl], marker=markers[lbl],
               s=sizes[lbl], alpha=0.85, label=lbl, edgecolors="white",
               linewidths=0.5, zorder=3)

# Label diseases + a few key symptoms
key_symptoms = {"fever", "chest_pain", "fatigue", "headache",
                "nausea", "joint_pain", "breathing_difficulty"}
for i, (name, lbl) in enumerate(zip(all_entities, labels)):
    if lbl == "Disease" or name in key_symptoms:
        display = name.replace("_", " ").title()
        ax.annotate(display, (coords[i, 0], coords[i, 1]),
                    fontsize=6.5 if lbl == "Symptom" else 8,
                    fontweight="bold" if lbl == "Disease" else "normal",
                    xytext=(4, 4), textcoords="offset points", alpha=0.9)

ax.legend(markerscale=1.3, fontsize=11)
ax.set_title("t-SNE Visualization of Learned Entity Embeddings", fontsize=14, fontweight="bold")
ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/1_tsne_embeddings.png", dpi=180)
plt.close()
print("  Saved 1_tsne_embeddings.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Disease-Symptom Affinity Matrix
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 2: Affinity matrix...")

# Only show symptoms that appear in at least one triple to reduce noise
active_syms = [s for s in symptoms if any(s in disease_symptoms[d] for d in diseases)]
sym_idx     = [symptoms.index(s) for s in active_syms]
aff_sub     = aff_sim[:, sym_idx]

dis_labels  = [d.title() for d in diseases]
sym_labels  = [s.replace("_", " ") for s in active_syms]

fig, ax = plt.subplots(figsize=(max(16, len(active_syms) * 0.22),
                                max(10, len(diseases) * 0.25)))
sns.heatmap(aff_sub, xticklabels=sym_labels, yticklabels=dis_labels,
            cmap="YlOrRd", ax=ax, linewidths=0.2,
            cbar_kws={"label": "Relatedness (higher = more related)"})
ax.set_title("Disease–Symptom Affinity Matrix (Learned KG Relation Scores)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Symptom", fontsize=10)
ax.set_ylabel("Disease", fontsize=10)
plt.xticks(fontsize=6.5, rotation=90)
plt.yticks(fontsize=7.5, rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_affinity_matrix.png", dpi=160)
plt.close()
print("  Saved 2_affinity_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Top-k Accuracy
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 3: Top-k accuracy...")

hits = {1: 0, 3: 0, 5: 0}
per_disease_rank = {}

for disease in diseases:
    known_syms = list(disease_symptoms.get(disease, []))
    if not known_syms:
        continue

    # Score all diseases against known symptoms
    scores = []
    for d in diseases:
        sym_scores = [rel_score(d, s) for s in known_syms]
        scores.append((d, sum(sym_scores) / len(sym_scores)))

    scores.sort(key=lambda x: x[1])  # lower = more related
    ranked = [d for d, _ in scores]
    rank   = ranked.index(disease) + 1
    per_disease_rank[disease] = rank

    for k in [1, 3, 5]:
        if rank <= k:
            hits[k] += 1

total = len(per_disease_rank)
acc   = {k: v / total for k, v in hits.items()}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: bar chart of hits@k
ks     = [1, 3, 5]
vals   = [acc[k] * 100 for k in ks]
colors = ["#e53e3e", "#dd6b20", "#38a169"]
bars   = axes[0].bar([f"Hits@{k}" for k in ks], vals, color=colors,
                     width=0.5, edgecolor="white")
for bar, v in zip(bars, vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{v:.1f}%", ha="center", fontweight="bold", fontsize=12)
axes[0].set_ylim(0, 110)
axes[0].set_ylabel("Accuracy (%)", fontsize=11)
axes[0].set_title("Top-k Disease Retrieval Accuracy", fontsize=12, fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)
axes[0].tick_params(labelsize=11)

# Right: rank distribution per disease
sorted_diseases = sorted(per_disease_rank, key=lambda d: per_disease_rank[d])
ranks = [per_disease_rank[d] for d in sorted_diseases]
bar_colors = ["#38a169" if r == 1 else "#dd6b20" if r <= 3 else "#e53e3e" for r in ranks]
axes[1].barh([d.title() for d in sorted_diseases], ranks,
             color=bar_colors, edgecolor="white")
axes[1].axvline(1, color="#38a169", linestyle="--", alpha=0.5, label="Rank 1")
axes[1].axvline(3, color="#dd6b20", linestyle="--", alpha=0.5, label="Rank 3")
axes[1].set_xlabel("Model Rank (lower = better)", fontsize=10)
axes[1].set_title("Per-Disease Rank", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].tick_params(labelsize=7)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/3_topk_accuracy.png", dpi=160)
plt.close()
print(f"  Hits@1={acc[1]:.1%}  Hits@3={acc[3]:.1%}  Hits@5={acc[5]:.1%}")
print("  Saved 3_topk_accuracy.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Disease Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 4: Confusion matrix...")

pred_map = {}
for disease in diseases:
    known_syms = list(disease_symptoms.get(disease, []))
    if not known_syms:
        continue
    scores = []
    for d in diseases:
        sym_scores = [rel_score(d, s) for s in known_syms]
        scores.append((d, sum(sym_scores) / len(sym_scores)))
    scores.sort(key=lambda x: x[1])
    pred_map[disease] = scores[0][0]  # top-1 prediction

conf = np.zeros((len(diseases), len(diseases)), dtype=int)
d_idx = {d: i for i, d in enumerate(diseases)}
for true_d, pred_d in pred_map.items():
    conf[d_idx[true_d], d_idx[pred_d]] += 1

dis_labels_short = [d.replace("gastroesophageal reflux disease", "GERD")
                     .replace("irritable bowel syndrome", "IBS")
                     .replace("chronic kidney disease", "CKD")
                     .replace("anxiety disorder", "anxiety")
                     .replace("rheumatoid arthritis", "RA")
                     .replace("osteoarthritis", "OA")
                     .replace("bronchial asthma", "asthma")
                     .title() for d in diseases]

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(conf, xticklabels=dis_labels_short, yticklabels=dis_labels_short,
            cmap="Blues", annot=True, fmt="d", ax=ax,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Count"})
ax.set_title("Disease Prediction Confusion Matrix (Top-1)", fontsize=13,
             fontweight="bold", pad=12)
ax.set_xlabel("Predicted Disease", fontsize=10)
ax.set_ylabel("True Disease", fontsize=10)
plt.xticks(fontsize=7, rotation=45, ha="right")
plt.yticks(fontsize=7, rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/4_confusion_matrix.png", dpi=160)
plt.close()
print("  Saved 4_confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Symptom Discriminability
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 5: Symptom discriminability...")

discrim = {}
for j, sym in enumerate(symptoms):
    scores = aff_sim[:, j]        # higher = more related
    probs  = np.exp(scores) / np.exp(scores).sum()
    ent    = scipy_entropy(probs)  # low entropy = discriminative
    discrim[sym] = ent

sorted_syms   = sorted(discrim, key=discrim.get)
top_n         = 30
top_syms      = sorted_syms[:top_n]
top_vals      = [discrim[s] for s in top_syms]
# Reverse so most discriminative (lowest entropy) is at the top
# RdYlGn: index 0 → red (least discriminative at bottom), index 29 → green (most at top)
top_syms_plot = top_syms[::-1]
top_vals_plot = top_vals[::-1]
bar_colors_d  = plt.cm.RdYlGn(np.linspace(0.1, 0.9, top_n))

fig, ax = plt.subplots(figsize=(10, 9))
bars = ax.barh([s.replace("_", " ") for s in top_syms_plot], top_vals_plot,
               color=bar_colors_d, edgecolor="white")
ax.set_xlabel("Entropy (lower = more discriminative)", fontsize=11)
ax.set_title(f"Top {top_n} Most Discriminative Symptoms", fontsize=13,
             fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/5_symptom_discriminability.png", dpi=160)
plt.close()
print("  Saved 5_symptom_discriminability.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Clinical Sanity Checks
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 6: Clinical sanity checks...")

CASES = {
    "Heart Attack":        ["chest_pain", "left_arm_pain", "jaw_pain", "cold_sweat", "nausea"],
    "Stroke":              ["facial_drooping", "slurred_speech", "weakness", "confusion", "vision_loss"],
    "COVID-19":            ["fever", "cough", "loss_of_smell", "loss_of_taste", "fatigue"],
    "Diabetes":            ["excessive_thirst", "frequent_urination", "blurred_vision", "slow_healing"],
    "UTI":                 ["burning_urination", "frequent_urination", "cloudy_urine", "pelvic_pain"],
    "Hypothyroidism":      ["fatigue", "weight_gain", "cold_intolerance", "hair_loss", "dry_skin"],
    "Migraine":            ["headache", "photophobia", "phonophobia", "nausea", "blurred_vision"],
    "Food Poisoning":      ["nausea", "vomiting", "diarrhea", "abdominal_pain", "fever"],
    "Meningitis":          ["headache", "neck_pain", "fever", "photophobia", "confusion"],
    "Anemia":              ["fatigue", "pallor", "cold_extremities", "breathing_difficulty", "dizziness"],
}

TOP_K = 3
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()

for idx, (case_name, case_syms) in enumerate(CASES.items()):
    valid_syms = [s for s in case_syms if s in symptoms]
    scores = []
    for d in diseases:
        sym_scores = [rel_score(d, s) for s in valid_syms]
        scores.append((d, sum(sym_scores) / len(sym_scores)))
    scores.sort(key=lambda x: x[1])
    top3 = scores[:TOP_K]

    names = [d.title() for d, _ in top3]
    raw   = np.array([-s for _, s in top3])
    probs = np.exp(raw) / np.exp(raw).sum()

    bar_c = ["#38a169" if n.lower() == case_name.lower() else "#3182ce" for n in names]
    axes[idx].barh(names[::-1], probs[::-1] * 100, color=bar_c[::-1], edgecolor="white")
    axes[idx].set_title(f"{case_name}", fontsize=9, fontweight="bold")
    axes[idx].set_xlabel("Score (%)", fontsize=8)
    axes[idx].tick_params(labelsize=8)
    axes[idx].grid(axis="x", alpha=0.3)

fig.suptitle("Clinical Sanity Checks — Top-3 Predictions for Classic Symptom Combinations",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/6_clinical_sanity_checks.png", dpi=160, bbox_inches="tight")
plt.close()
print("  Saved 6_clinical_sanity_checks.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Symptoms per Disease (Data Distribution)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 7: Data distribution...")

sym_counts = {d: len(disease_symptoms[d]) for d in diseases}
sorted_d   = sorted(sym_counts, key=sym_counts.get, reverse=True)
counts     = [sym_counts[d] for d in sorted_d]
labels_d   = [d.replace("gastroesophageal reflux disease", "GERD")
               .replace("irritable bowel syndrome", "IBS")
               .replace("chronic kidney disease", "CKD")
               .replace("anxiety disorder", "anxiety dis.")
               .replace("rheumatoid arthritis", "RA")
               .replace("osteoarthritis", "OA")
               .replace("bronchial asthma", "asthma")
               .title() for d in sorted_d]
bar_cols   = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_d)))[::-1]

fig, ax = plt.subplots(figsize=(12, 10))
bars = ax.barh(labels_d[::-1], counts[::-1], color=bar_cols, edgecolor="white")
for bar, cnt in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            str(cnt), va="center", fontsize=8)
ax.set_xlabel("Number of Symptoms", fontsize=11)
ax.set_title("Number of Symptoms per Disease in Knowledge Graph", fontsize=13,
             fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.tick_params(labelsize=8)
ax.set_xlim(0, max(counts) + 3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/7_symptoms_per_disease.png", dpi=160)
plt.close()
print("  Saved 7_symptoms_per_disease.png")

print("\n✓ All 7 research plots saved to analysis/")
