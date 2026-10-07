import torch
import torch.nn.functional as F
from torch import nn


def rotation_from_6d(values):

    first = F.normalize(values[..., :3], dim=-1, eps=1e-8)
    second_raw = values[..., 3:]
    second = F.normalize(second_raw - (first * second_raw).sum(-1, keepdim=True) * first,
                         dim=-1, eps=1e-8)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


class AffineTemplateAdaptation(nn.Module):


    def __init__(self, backbone, feature_dim=64, heads=4, latent_dim=256):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.cross_attention = nn.MultiheadAttention(feature_dim, heads, batch_first=True)
        self.condition_norm = nn.LayerNorm(feature_dim)
        self.point_fusion = nn.Sequential(nn.Linear(2 * feature_dim, 512), nn.GELU(), nn.Linear(512, 1024))
        self.shared_transform = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, latent_dim), nn.GELU())
        self.rotation_head = nn.Linear(latent_dim, 6)
        self.scale_head = nn.Linear(latent_dim, 3)
        self.translation_head = nn.Linear(latent_dim, 3)

        nn.init.zeros_(self.rotation_head.weight)
        with torch.no_grad():
            self.rotation_head.bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.translation_head.weight)
        nn.init.zeros_(self.translation_head.bias)

    def forward(self, source, target):
        if source.ndim != 3 or source.shape[-1] != 6 or target.shape != source.shape:
            raise ValueError("Source and target must share shape (B, N, 6), with XYZ and normals.")
        source_features, target_features = self.backbone(source), self.backbone(target)
        expected = (*source.shape[:2], self.feature_dim)
        if source_features.shape != expected or target_features.shape != expected:
            raise ValueError("The backbone must return point-aligned (B, N, feature_dim) features.")
        observed = self.cross_attention(source_features, target_features, target_features, need_weights=False)[0]
        conditioned_source = self.condition_norm(source_features + observed)
        fused_points = self.point_fusion(torch.cat([conditioned_source, target_features], dim=-1))
        descriptor = fused_points.amax(dim=1)
        latent = self.shared_transform(descriptor)
        rotation = rotation_from_6d(self.rotation_head(latent))
        scale = self.scale_head(latent).exp()
        translation = self.translation_head(latent)
        matrix = torch.eye(4, device=source.device, dtype=rotation.dtype)[None].repeat(len(source), 1, 1)
        matrix[:, :3, :3] = rotation @ torch.diag_embed(scale)
        matrix[:, :3, 3] = translation
        return {"affine_matrix": matrix, "rotation": rotation, "scale": scale,
                "translation": translation, "descriptor": descriptor}


def transform_template_vertices(vertices, affine_matrix):

    return vertices @ affine_matrix[..., :3, :3].transpose(-1, -2) + affine_matrix[..., None, :3, 3]
