# Medical Embedding Schema using Transformer and Knowledge Graphs

A medical NLP system that learns structured disease-symptom embeddings by jointly training a custom BERT-like transformer encoder on three tasks: **Masked Language Modeling (MLM)**, **Knowledge Graph Relation Learning (TransE)**, and **Entity Type Classification**. The trained model powers an interactive web app that predicts diseases from symptoms.

---

## Project Overview

```
Medical Text Corpus (76,414 sentences)
        +
Knowledge Graph (405 triples, 54 diseases, 93 symptoms)
        ↓
Custom WordPiece Tokenizer (32k vocab)
        ↓
BERT Encoder (6 layers, 8 heads, hidden_dim=256)
        ↓
Multi-Task Training
  ├── MLM Loss        → contextual language understanding
  ├── TransE Loss     → disease-symptom relation learning
  └── Type Loss       → disease vs symptom classification
        ↓
Trained Model → Gradio Web App → Disease Predictions
```

---

## Results

| Metric | Value |
|--------|-------|
| Hits@1 (Disease Retrieval) | 92.6% |
| Hits@3 (Disease Retrieval) | 100.0% |
| Hits@5 (Disease Retrieval) | 100.0% |
| MRR (Link Prediction) | 0.349 |
| Hits@10 (Link Prediction) | 97% |
| Clinical Sanity Checks | 10/10 correct |

---

## Project Structure

```
├── app.py                          # Gradio web application
├── train.py                        # Main training entry point
├── evaluate.py                     # Link prediction + embedding evaluation
├── test.py                         # Smoke test (no model required)
├── test_disease_prediction.py      # CLI disease prediction test
├── symptom_patterns.py             # Natural language symptom matching (93 patterns)
├── requirements.txt
│
├── model/                          # Model architecture
│   ├── bert_encoder.py             # BERTEncoder (embeddings + transformer)
│   ├── embeddings.py               # Token + positional + segment embeddings
│   ├── transformer.py              # TransformerEncoder (6 layers)
│   └── multitask_model.py          # MultiTaskModel (MLM + Type + Relation heads)
│
├── training/                       # Training components
│   ├── trainer.py                  # Multitask Trainer with dynamic weights
│   ├── dataset.py                  # MLMDataset (masked language modeling)
│   ├── triples_dataset.py          # TriplesDataset (KG relation learning)
│   ├── type_dataset.py             # TypeDataset (entity type classification)
│   └── relation_loss.py            # Margin-based TransE loss
│
├── heads/                          # Task-specific output heads
│   ├── mlm_head.py                 # MLM projection head
│   └── type_head.py                # Entity type classification head
│
├── inference/                      # Inference pipeline
│   ├── disease_predictor.py        # DiseasePredictor class
│   └── build_disease_index.py      # Pre-compute disease embeddings
│
├── evaluation/                     # Evaluation metrics
│   ├── link_prediction.py          # MRR + Hits@10
│   └── visualize_embeddings.py     # t-SNE embedding plot
│
├── scripts/                        # Data preparation
│   ├── build_corpus.py             # Download MedQuad → data/corpus.txt
│   └── build_kg.py                 # Download symptom dataset → triples.csv
│
├── tokenizer/                      # WordPiece tokenizer
│   └── train_tokenizer.py          # Train tokenizer on corpus
│
├── data/
│   ├── triples.csv                 # KG triples: (disease, has_symptom, symptom)
│   └── entity_types.csv            # Entity type mapping: (entity, Disease|Symptom)
│
└── analysis/                       # Research plots and training logs
    ├── plot_training.py            # Training loss/weight plots
    ├── plot_research.py            # t-SNE, affinity matrix, top-k, sanity checks
    ├── run_training_analysis.py    # Run 5 epochs for logging (no model overwrite)
    └── training_logs.json          # Saved epoch-level training metrics
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/vamsinalla18/nlplatest.git
cd nlplatest

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preparation
```bash
python scripts/build_corpus.py        # downloads MedQuad → data/corpus.txt
python scripts/build_kg.py            # downloads symptom dataset → data/triples.csv
python tokenizer/train_tokenizer.py   # trains tokenizer → tokenizer/medical_tokenizer.json
```

### 2. Training
```bash
python train.py          # multitask training (5 epochs), saves model.pt
```

### 3. Evaluation
```bash
python evaluate.py                          # MRR, Hits@10, embedding plot
python inference/build_disease_index.py     # pre-compute disease index
```

### 4. Launch Web App
```bash
python app.py            # launches Gradio UI (public share URL)
```

### 5. Analysis Plots
```bash
python analysis/run_training_analysis.py    # run training for logs only
python analysis/plot_training.py            # generate training plots
python analysis/plot_research.py            # generate all research plots
```

---

## Model Architecture

- **Tokenizer**: WordPiece, 32k vocabulary, trained on medical corpus
- **Encoder**: 6-layer transformer, 8 attention heads, hidden dim 256, FFN dim 1024
- **Entity Embedding**: masked mean pooling over non-padding token positions
- **MLM Head**: linear projection → 32k vocab
- **Type Head**: linear projection → 2 classes (Disease / Symptom)
- **Relation Scoring**: TransE — `score(h,r,t) = ||e_h + r - e_t||_2`
- **Task Weights**: dynamically learned via softmax-parameterized 3-dim vector

---

## Web Application

The Gradio app lets patients:
1. Search and select symptoms from a dropdown (93 symptoms with friendly names)
2. Get related symptom suggestions based on KG co-occurrence
3. Click **Predict** to see top-5 ranked disease predictions with confidence scores

---

## Analysis Plots

| Plot | Description |
|------|-------------|
| `training_total_loss.png` | Total weighted loss per epoch |
| `training_task_losses.png` | Per-task loss (MLM / Relation / Type) |
| `training_task_weights.png` | Dynamic task weight evolution |
| `training_step_loss.png` | Step-level loss across all 4,255 steps |
| `1_tsne_embeddings.png` | t-SNE of all 147 entity embeddings |
| `2_affinity_matrix.png` | Disease × symptom affinity heatmap |
| `3_topk_accuracy.png` | Hits@1/3/5 + per-disease rank |
| `4_confusion_matrix.png` | 54×54 disease confusion matrix |
| `5_symptom_discriminability.png` | Top 30 most discriminative symptoms |
| `6_clinical_sanity_checks.png` | 10 classic clinical case predictions |
| `7_symptoms_per_disease.png` | Symptom count distribution |
