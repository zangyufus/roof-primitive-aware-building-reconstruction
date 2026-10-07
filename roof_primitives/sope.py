import torch
from torch import nn


class SOPE(nn.Module):


    def __init__(self, feature_dim, hidden_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.boundary_score = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, point_features, descriptors, patch_ids, return_weights=False):
        if point_features.ndim != 2 or point_features.shape[1] != self.feature_dim:
            raise ValueError("point_features must have shape (N, feature_dim).")
        if descriptors.shape != (len(point_features), 3) or patch_ids.shape != (len(point_features),):
            raise ValueError("Descriptors and patch indices must align with point features.")
        if len(point_features) == 0 or patch_ids.dtype != torch.long or (patch_ids < 0).any():
            raise ValueError("Patch indices must be a nonempty, nonnegative int64 vector.")
        count = int(patch_ids.max().item()) + 1
        counts = point_features.new_zeros(count).scatter_add_(0, patch_ids, point_features.new_ones(len(patch_ids)))
        if (counts == 0).any():
            raise ValueError("Patch IDs must be contiguous; remove empty patches first.")
        indices = patch_ids[:, None].expand_as(point_features)
        mean = point_features.new_zeros(count, self.feature_dim).scatter_add_(0, indices, point_features)
        mean = mean / counts[:, None]
        logits = self.boundary_score(descriptors).squeeze(-1)
        maximum = logits.new_full((count,), -torch.inf)
        maximum.scatter_reduce_(0, patch_ids, logits, reduce="amax", include_self=True)
        exponential = (logits - maximum[patch_ids]).exp()
        denominator = logits.new_zeros(count).scatter_add_(0, patch_ids, exponential)
        weights = exponential / denominator[patch_ids].clamp_min(torch.finfo(logits.dtype).tiny)
        weighted = point_features.new_zeros(count, self.feature_dim)
        weighted.scatter_add_(0, indices, weights[:, None] * point_features)
        tokens = torch.cat([mean, weighted], dim=-1)
        return (tokens, weights) if return_weights else tokens
