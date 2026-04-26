import torch
from tqdm import tqdm


def link_prediction(model, dataset, device):

    model.eval()

    entities = list(dataset.entities)
    entity_types = dataset.entity_types

    # Only consider symptom entities as candidates
    symptoms = [e for e in entities if entity_types.get(e) == "Symptom"]

    ranks = []
    hits10 = 0

    for h, r, t in tqdm(dataset.triples):

        h_tokens = dataset.encode_entity(h).unsqueeze(0).to(device)
        r_tensor = torch.tensor([r]).to(device)

        scores = []

        for candidate in symptoms:

            t_tokens = dataset.encode_entity(candidate).unsqueeze(0).to(device)

            with torch.no_grad():
                score = model.relation_score(
                    h_tokens,
                    r_tensor,
                    t_tokens
                )

            scores.append((candidate, score.item()))

        scores.sort(key=lambda x: x[1], reverse=False)  # lower L2 norm = more related

        ranked_entities = [x[0] for x in scores]

        if t not in ranked_entities:
            continue

        rank = ranked_entities.index(t) + 1
        ranks.append(rank)

        if rank <= 10:
            hits10 += 1

    if len(ranks) == 0:
        print("No valid triples evaluated")
        return

    mrr = sum(1.0 / r for r in ranks) / len(ranks)
    hits10 = hits10 / len(ranks)

    print("MRR:", mrr)
    print("Hits@10:", hits10)