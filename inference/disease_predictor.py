import torch
import pandas as pd
from tokenizers import Tokenizer

from model.multitask_model import MultiTaskModel
from symptom_patterns import match_symptoms


def _load_model(model_path="model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.from_file("tokenizer/medical_tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")

    types_df = pd.read_csv("data/entity_types.csv", header=None)
    num_types = len(sorted(set(types_df[1].tolist())))

    triples_df = pd.read_csv("data/triples.csv", header=None)
    num_relations = len(set(triples_df[1].tolist()))

    model = MultiTaskModel(
        vocab_size,
        num_types=num_types,
        num_relations=num_relations,
        pad_id=pad_id,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device, tokenizer, pad_id


class DiseasePredictor:

    def __init__(self, model_path="model.pt"):

        self.model, self.device, self.tokenizer, self.pad_id = _load_model(model_path)

        types_df = pd.read_csv("data/entity_types.csv", header=None)
        all_entities = types_df[0].tolist()
        all_types = types_df[1].tolist()

        self.diseases = [e for e, t in zip(all_entities, all_types) if t == "Disease"]
        self.symptom_set = set(e for e, t in zip(all_entities, all_types) if t == "Symptom")

        self.max_len = 16
        self.relation_id = torch.tensor([0]).to(self.device)

    def _encode(self, text):
        ids = self.tokenizer.encode(text).ids[:self.max_len]
        ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids).unsqueeze(0).to(self.device)

    def _score_symptoms(self, symptom_list, top_k):
        disease_scores = []
        for disease in self.diseases:
            h = self._encode(disease)
            scores = []
            for symptom in symptom_list:
                t = self._encode(symptom)
                with torch.no_grad():
                    score = self.model.relation_score(h, self.relation_id, t)
                scores.append(score.item())
            avg_score = sum(scores) / len(scores)
            disease_scores.append((disease, avg_score))

        raw = torch.tensor([x[1] for x in disease_scores])
        probs = torch.softmax(-raw, dim=0)
        results = [(disease_scores[i][0], probs[i].item()) for i in range(len(disease_scores))]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def predict(self, symptoms_text, top_k=3):
        matched = match_symptoms(symptoms_text, valid_symptoms=self.symptom_set)
        if not matched:
            return []
        return self._score_symptoms(matched, top_k)

    def predict_from_list(self, canonical_symptoms, top_k=5):
        valid = [s for s in canonical_symptoms if s in self.symptom_set]
        if not valid:
            return []
        return self._score_symptoms(valid, top_k)
