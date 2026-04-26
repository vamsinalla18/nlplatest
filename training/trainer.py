import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import MLMDataset
from training.triples_dataset import TriplesDataset
from training.type_dataset import TypeDataset

from model.multitask_model import MultiTaskModel
from training.relation_loss import relation_loss


class Trainer:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer_path = "tokenizer/medical_tokenizer.json"

        ############################
        # A6000 OPTIMIZED SETTINGS
        ############################

        batch_size = 64
        num_workers = 8

        ################################
        # MLM DATASET
        ################################

        mlm_dataset = MLMDataset(
            corpus_file="data/corpus.txt",
            tokenizer_path=tokenizer_path
        )

        self.mlm_loader = DataLoader(
            mlm_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        vocab_size = mlm_dataset.tokenizer.get_vocab_size()
        pad_id = mlm_dataset.pad_token_id

        ################################
        # TRIPLES DATASET
        ################################

        triples_dataset = TriplesDataset(
            "data/triples.csv",
            tokenizer_path
        )

        self.triples_dataset = triples_dataset  # kept for negative sampling

        self.triples_loader = DataLoader(
            triples_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.triples_iter = iter(self.triples_loader)

        num_relations = len(triples_dataset.relation2id)

        # All known positive (head, tail) pairs — used to filter false negatives
        self.positive_pairs = {(h, t) for h, _, t in triples_dataset.triples}

        # Type-separated entity lists for constrained negative sampling
        self.disease_entities = [
            e for e in triples_dataset.entities
            if triples_dataset.entity_types.get(e) == "Disease"
        ]
        self.symptom_entities = [
            e for e in triples_dataset.entities
            if triples_dataset.entity_types.get(e) == "Symptom"
        ]

        ################################
        # TYPE DATASET
        ################################

        type_dataset = TypeDataset(
            "data/entity_types.csv",
            tokenizer_path
        )

        self.type_loader = DataLoader(
            type_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.type_iter = iter(self.type_loader)

        num_types = len(type_dataset.type2id)

        ################################
        # MODEL
        ################################

        self.model = MultiTaskModel(
            vocab_size,
            num_types=num_types,
            num_relations=num_relations,
            pad_id=pad_id,
        ).to(self.device)

        ################################
        # OPTIMIZER (better for transformers)
        ################################

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=0.01
        )

        ################################
        # LOSSES
        ################################

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.type_loss_fn = nn.CrossEntropyLoss()

        ################################
        # DYNAMIC TASK WEIGHTS
        ################################

        self.task_weights = torch.nn.Parameter(
            torch.ones(3).to(self.device)
        )

        self.weight_optimizer = torch.optim.Adam(
            [self.task_weights],
            lr=1e-3
        )

        ################################
        # MIXED PRECISION (A6000)
        ################################

        self.scaler = torch.amp.GradScaler("cuda")


    def get_triples_batch(self):

        try:
            batch = next(self.triples_iter)

        except StopIteration:

            self.triples_iter = iter(self.triples_loader)
            batch = next(self.triples_iter)

        return batch


    def get_type_batch(self):

        try:
            batch = next(self.type_iter)

        except StopIteration:

            self.type_iter = iter(self.type_loader)
            batch = next(self.type_iter)

        return batch


    def _sample_negative(self, h_name, t_name):
        """
        Return a (neg_h_name, neg_t_name) pair that is guaranteed NOT in the
        positive triple set. Type-constrained: tails are corrupted with a
        symptom, heads with a disease. Retries up to 50 times then falls back
        to an exhaustive pick so it never returns a false negative.
        """
        if random.random() < 0.5:
            # corrupt tail — pick a symptom that is not a known positive for h
            for _ in range(50):
                neg = random.choice(self.symptom_entities)
                if (h_name, neg) not in self.positive_pairs:
                    return h_name, neg
            # exhaustive fallback (always succeeds: not every symptom is linked to h)
            true_negs = [s for s in self.symptom_entities if (h_name, s) not in self.positive_pairs]
            return h_name, random.choice(true_negs)
        else:
            # corrupt head — pick a disease that is not a known positive for t
            for _ in range(50):
                neg = random.choice(self.disease_entities)
                if (neg, t_name) not in self.positive_pairs:
                    return neg, t_name
            true_negs = [d for d in self.disease_entities if (d, t_name) not in self.positive_pairs]
            return random.choice(true_negs), t_name


    def train_epoch(self, epoch_idx=0):

        self.model.train()

        total_loss = 0
        mlm_total = rel_total = type_total = 0
        step_losses = []

        for batch in tqdm(self.mlm_loader):

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            segment_ids = torch.zeros_like(input_ids).to(self.device)

            ################################
            # MIXED PRECISION FORWARD
            ################################

            with torch.amp.autocast("cuda"):

                # ---------- MLM ----------
                logits = self.model.forward_mlm(input_ids, segment_ids)

                mlm_loss = self.loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

                # ---------- RELATION ----------
                h_tokens, r, t_tokens, h_names, t_names = self.get_triples_batch()

                h_tokens = h_tokens.to(self.device)
                t_tokens = t_tokens.to(self.device)
                r = r.to(self.device)

                pos_score = self.model.relation_score(h_tokens, r, t_tokens)

                # One filtered negative per positive triple in the batch
                neg_h_list, neg_t_list = [], []
                for h_name, t_name in zip(h_names, t_names):
                    neg_h, neg_t = self._sample_negative(h_name, t_name)
                    neg_h_list.append(self.triples_dataset.encode_entity(neg_h))
                    neg_t_list.append(self.triples_dataset.encode_entity(neg_t))

                neg_h_tokens = torch.stack(neg_h_list).to(self.device)
                neg_t_tokens = torch.stack(neg_t_list).to(self.device)

                neg_score = self.model.relation_score(neg_h_tokens, r, neg_t_tokens)

                rel_loss = relation_loss(pos_score, neg_score)

                # ---------- TYPE ----------
                entity_tokens, type_labels = self.get_type_batch()

                entity_tokens = entity_tokens.to(self.device)
                type_labels = type_labels.to(self.device)

                type_logits = self.model.type_prediction(entity_tokens)

                type_loss = self.type_loss_fn(
                    type_logits, type_labels
                )

                ################################
                # TOTAL LOSS
                ################################

                weights = torch.softmax(self.task_weights, dim=0)

                loss = (
                    weights[0] * mlm_loss
                    + weights[1] * rel_loss
                    + weights[2] * type_loss
                )

            ################################
            # BACKPROP
            ################################

            self.optimizer.zero_grad()
            self.weight_optimizer.zero_grad()

            if torch.isnan(loss):

                print("Skipping batch due to NaN loss")
                continue

            self.scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0
            )

            self.scaler.step(self.optimizer)
            self.scaler.step(self.weight_optimizer)  # unscales task_weights grads too
            self.scaler.update()

            self.task_weights.data = torch.clamp(
                self.task_weights.data, -5, 5
            )

            total_loss += loss.item()
            mlm_total += mlm_loss.item()
            rel_total += rel_loss.item()
            type_total += type_loss.item()
            step_losses.append(loss.item())

        n = len(self.mlm_loader)
        weights = torch.softmax(self.task_weights, dim=0).detach().cpu().tolist()
        self._last_epoch_log = {
            "epoch": epoch_idx + 1,
            "total_loss": total_loss / n,
            "mlm_loss": mlm_total / n,
            "rel_loss": rel_total / n,
            "type_loss": type_total / n,
            "weight_mlm": weights[0],
            "weight_rel": weights[1],
            "weight_type": weights[2],
            "step_losses": step_losses,
        }
        return total_loss / n


    def train(self, epochs=5, save_model=True, log_path=None):
        import json

        all_logs = []

        for epoch in range(epochs):

            loss = self.train_epoch(epoch_idx=epoch)

            print(f"Epoch {epoch+1} Loss: {loss:.4f}")
            print(
                "Task weights:",
                torch.softmax(self.task_weights, dim=0).detach().cpu()
            )

            all_logs.append(self._last_epoch_log)

            if log_path:
                with open(log_path, "w") as f:
                    json.dump(all_logs, f, indent=2)

        if save_model:
            torch.save(self.model.state_dict(), "model.pt")
            print("Model saved to model.pt")

        return all_logs
