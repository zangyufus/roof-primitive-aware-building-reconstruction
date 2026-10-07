import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def primitive_geometry(mask_logits, coordinates, padding_mask=None):

    weights = mask_logits.sigmoid()
    if padding_mask is not None:
        weights = weights.masked_fill(padding_mask[:, None], 0)
    total = weights.sum(-1).clamp_min(1e-8)
    mean = torch.einsum("bqn,bnd->bqd", weights, coordinates) / total[..., None]
    delta = coordinates[:, None] - mean[:, :, None]
    covariance = torch.einsum("bqn,bqni,bqnj->bqij", weights, delta, delta) / total[..., None, None]
    values, vectors = torch.linalg.eigh(covariance.float())
    values = values.clamp_min(0)
    normal = vectors[..., 0]
    normal = torch.where(normal[..., 2:3] < 0, -normal, normal)
    principal = vectors[..., 2]
    pivot = principal.gather(-1, principal.abs().argmax(-1, keepdim=True))
    principal = torch.where(pivot < 0, -principal, principal)
    largest = values[..., 2].clamp_min(1e-8)
    linearity = (values[..., 2] - values[..., 1]) / largest
    planarity = (values[..., 1] - values[..., 0]) / largest
    curvature = values[..., 0] / values.sum(-1).clamp_min(1e-8)
    selected = weights > 0.5

    selected.scatter_(-1, weights.argmax(-1, keepdim=True), True)
    if padding_mask is not None:
        selected &= ~padding_mask[:, None]
    expanded = coordinates[:, None].expand(-1, weights.shape[1], -1, -1)
    lower = expanded.masked_fill(~selected[..., None], torch.inf).amin(-2)
    upper = expanded.masked_fill(~selected[..., None], -torch.inf).amax(-2)
    boxes = torch.cat([(lower + upper) * 0.5, (upper - lower).clamp_min(1e-6)], dim=-1)
    effective_count = total.square() / weights.square().sum(-1).clamp_min(1e-8)
    valid = (effective_count >= 3) & (largest > 1e-8)
    shape = torch.stack([linearity, planarity, curvature], dim=-1)
    return boxes, normal.to(coordinates.dtype), principal.to(coordinates.dtype), shape.to(coordinates.dtype), valid


def relation_features(boxes, normal, principal, shape):

    center, size = boxes[..., :3], boxes[..., 3:]
    offset = (center[:, :, None] - center[:, None]).abs()
    offset = torch.log1p(offset / size[:, :, None].clamp_min(1e-6))
    ratio = torch.log(size[:, :, None].clamp_min(1e-6) / size[:, None].clamp_min(1e-6))
    box_relation = torch.cat([offset, ratio], dim=-1)
    direction_cosine = torch.einsum("bqd,bkd->bqk", principal, principal).clamp(-1, 1)
    normal_cosine = torch.einsum("bqd,bkd->bqk", normal, normal).clamp(-1, 1)
    shape_difference = (shape[:, :, None] - shape[:, None]).abs()
    geometry_relation = torch.cat([direction_cosine[..., None], normal_cosine[..., None], shape_difference], dim=-1)
    return box_relation, geometry_relation


class StructureAwareDecoderLayer(nn.Module):
    def __init__(self, dimension, heads, hidden_dim, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dimension, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dimension))
        self.norms = nn.ModuleList([nn.LayerNorm(dimension) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, memory, query_position, memory_position, bias, padding_mask):
        positioned = queries + query_position
        attended = self.self_attention(positioned, positioned, queries, attn_mask=bias, need_weights=False)[0]
        queries = self.norms[0](queries + self.dropout(attended))
        attended = self.cross_attention(queries + query_position, memory + memory_position, memory,
                                        key_padding_mask=padding_mask, need_weights=False)[0]
        queries = self.norms[1](queries + self.dropout(attended))
        return self.norms[2](queries + self.dropout(self.ffn(queries)))


class SAPD(nn.Module):


    def __init__(self, input_dim, dimension=256, heads=8, layers=6,
                 queries=100, hidden_dim=1024, dropout=0.0):
        super().__init__()
        if layers < 1 or queries < 1:
            raise ValueError("SAPD needs at least one decoder layer and one primitive query.")
        self.heads = heads
        self.token_projection = nn.Linear(input_dim, dimension)
        self.position_projection = nn.Sequential(nn.Linear(3, dimension), nn.ReLU(), nn.Linear(dimension, dimension))
        self.query_tokens = nn.Embedding(queries, dimension)
        self.query_position = nn.Embedding(queries, dimension)
        self.layers = nn.ModuleList([StructureAwareDecoderLayer(dimension, heads, hidden_dim, dropout) for _ in range(layers)])
        self.box_projection = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, heads))
        self.geometry_projection = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, heads))
        self.existence_head = nn.Linear(dimension, 1)
        self.mask_embedding = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, dimension))
        self.mask_features = nn.Linear(input_dim, dimension)

    def _predict(self, queries, mask_features):
        return {"existence_logits": self.existence_head(queries).squeeze(-1),
                "mask_logits": torch.einsum("bqd,bmd->bqm", self.mask_embedding(queries), mask_features)}

    def forward(self, patch_tokens, patch_coordinates, padding_mask=None):
        if patch_tokens.ndim != 3 or patch_coordinates.shape != (*patch_tokens.shape[:2], 3):
            raise ValueError("Patch tokens and coordinates must have matching (B, M) dimensions.")
        if patch_tokens.shape[1] == 0:
            raise ValueError("At least one patch is required.")
        if padding_mask is not None:
            if padding_mask.shape != patch_tokens.shape[:2] or padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must be a boolean (B, M) tensor.")
            if padding_mask.all(-1).any():
                raise ValueError("Every roof must contain at least one valid patch.")
        batch = patch_tokens.shape[0]
        memory = self.token_projection(patch_tokens)
        memory_position = self.position_projection(patch_coordinates)
        mask_features = self.mask_features(patch_tokens)
        queries = self.query_tokens.weight[None].expand(batch, -1, -1)
        query_position = self.query_position.weight[None].expand_as(queries)
        prediction = self._predict(queries, mask_features)
        intermediate = []
        for layer in self.layers:
            boxes, normal, principal, shape, valid = primitive_geometry(prediction["mask_logits"], patch_coordinates, padding_mask)
            box_features, geometry_features = relation_features(boxes, normal, principal, shape)
            pair_valid = valid[:, :, None] & valid[:, None]
            bias = self.box_projection(box_features) + self.geometry_projection(geometry_features) * pair_valid[..., None]
            bias = bias.permute(0, 3, 1, 2).reshape(batch * self.heads, queries.shape[1], queries.shape[1])
            queries = layer(queries, memory, query_position, memory_position, bias, padding_mask)
            prediction = self._predict(queries, mask_features)
            intermediate.append(prediction)
        if padding_mask is not None:
            prediction["mask_logits"] = prediction["mask_logits"].masked_fill(padding_mask[:, None], -torch.inf)
        return {**prediction, "intermediate": intermediate[:-1]}
