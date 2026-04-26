import torch


def relation_loss(pos_score, neg_score, margin=1.0):

    loss = torch.relu(margin + pos_score - neg_score)

    return loss.mean()