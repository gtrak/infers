"""Comparison functions: cosine similarity, L2 error, element-wise statistics."""

import torch


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened tensors."""
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    dot = (a_f * b_f).sum()
    na = a_f.norm()
    nb = b_f.norm()
    if na.item() == 0 or nb.item() == 0:
        return 0.0
    return (dot / (na * nb)).item()


def l2_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized L2 error ||a-b|| / ||a||."""
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    diff = (a_f - b_f).norm().item()
    norm = a_f.norm().item()
    return diff / (norm + 1e-30)


def element_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Element-wise absolute diff statistics.

    Returns dict with max, mean, and median absolute differences.
    """
    diff = (a.float() - b.float()).abs()
    return {
        "max": diff.max().item(),
        "mean": diff.mean().item(),
        "median": diff.median().item(),
    }
