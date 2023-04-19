import torch
import torch.nn as nn
import torch.nn.functional as F


def hierarchical_cosine_loss(z: torch.tensor, z_hat: torch.tensor) -> torch.tensor:
    """ This function calculates the Hierarchical Cosine Loss for a given set of input tensors z and z_hat. To calculate
    this loss, the function performs a max-pooling operation on both z and z_hat, reducing the length of the sequence
    by half, and computes the cosine similarity between the resulting tensors.

    The loss is based on:
        - TS2Vec: Towards Universal Representation of Time Series (https://arxiv.org/pdf/2106.10466.pdf)

    Args:
        - z:        (batch_size, seq_len, output_dim)
        - z_hat:    (batch_size, seq_len, output_dim)
    Returns:
        - loss: torch.tensor
    """
    loss = torch.tensor(0., device=z.device)
    d = 0
    while z.size(1) > 1:
        loss += cosine_loss(z, z_hat)
        z = F.max_pool1d(z.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z_hat = F.max_pool1d(z_hat.transpose(1, 2), kernel_size=2).transpose(1, 2)
        d += 1
    return loss / d


def cosine_loss(z: torch.tensor, z_hat: torch.tensor) -> torch.tensor:
    """ This function calculates the Cosine Loss for a given set of input tensors z and z_hat. The Cosine Loss is
    defined as the negative mean of the cosine similarity between z and z_hat and aims to
    minimize the cosine distance between the two tensors z and z_hat, rather than maximizing their similarity.

    Args:
        - z:        (batch_size, seq_len, output_dim)
        - z_hat:    (batch_size, seq_len, output_dim)
    Returns:
        - loss: torch.tensor
    """
    cos_fn = nn.CosineSimilarity(dim=2).to(z.device)
    cos_sim = cos_fn(z, z_hat)
    loss = -torch.mean(cos_sim, dim=0).mean()
    return loss
