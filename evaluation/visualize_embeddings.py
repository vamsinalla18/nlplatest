import torch
import umap
import matplotlib.pyplot as plt
import numpy as np
import random

from training.type_dataset import TypeDataset


def visualize_embeddings(model, tokenizer_path, device):

    dataset = TypeDataset(
        "data/entity_types.csv",
        tokenizer_path
    )

    model.eval()

    embeddings = []
    labels = []
    names = []

    for entity, label in zip(dataset.entities, dataset.types):

        tokens = dataset.encode_entity(entity).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.entity_embedding(tokens)

        embeddings.append(emb.cpu().numpy()[0])
        labels.append(label)
        names.append(entity)

    embeddings = np.array(embeddings)

    # remove NaNs
    valid = ~np.isnan(embeddings).any(axis=1)
    embeddings = embeddings[valid]
    labels = np.array(labels)[valid]
    names = np.array(names)[valid]

    # UMAP reduction
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=10,
        min_dist=0.3,
        random_state=42
    )

    coords = reducer.fit_transform(embeddings)

    # color mapping
    color_map = {
        "Disease": "red",
        "Symptom": "blue"
    }

    plt.figure(figsize=(12, 10))

    # plot points
    for label_type in set(labels):

        idx = labels == label_type

        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c=color_map.get(label_type, "gray"),
            label=label_type,
            alpha=0.7,
            s=40
        )

    # label only a few points to avoid clutter
    num_labels = min(20, len(coords))
    chosen = random.sample(range(len(coords)), num_labels)

    for i in chosen:
        x, y = coords[i]
        plt.text(
            x,
            y,
            names[i],
            fontsize=9,
            alpha=0.8
        )

    plt.legend()
    plt.title("Medical Entity Embeddings (UMAP Projection)")

    plt.tight_layout()

    plt.savefig("embedding_plot.png", dpi=300)
    print("Embedding plot saved as embedding_plot.png")