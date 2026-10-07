import torch
from torch import nn
from .spectral_encoding import LaplacianSpectralEncoder, spectral_coordinates
from .structure_prioritized_masking import StructurePrioritizedMasking


class StructurePrioritizedPretraining(nn.Module):
    def __init__(self, encoder=None, mask_ratio=0.6, decoder_depth=4, heads=6,
                 geometry_config=None, seed=0):
        super().__init__()
        self.encoder = encoder if encoder is not None else LaplacianSpectralEncoder()
        self.masker = StructurePrioritizedMasking(mask_ratio, geometry_config, seed)
        self.mask_count = int(mask_ratio * self.encoder.grouping.groups)
        if not 0 < self.mask_count < self.encoder.grouping.groups:
            raise ValueError("Pretraining requires at least one visible and one masked group.")
        dimension = self.encoder.dimension
        self.mask_token = nn.Parameter(torch.empty(1, 1, dimension))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(dimension, heads, dimension * 4, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, decoder_depth, norm=nn.LayerNorm(dimension), enable_nested_tensor=False)
        self.reconstruction = nn.Linear(dimension, 3 * self.encoder.grouping.group_size)

    def forward(self, points, return_details=False):
        groups, centers = self.encoder.grouping(points)

        coordinates = spectral_coordinates(centers, self.encoder.neighbors, self.encoder.alpha)
        mask, details = self.masker(points, centers, return_details=True)
        batch, count, size, _ = groups.shape
        visible = count - self.mask_count
        visible_groups = groups[~mask].reshape(batch, visible, size, 3)
        visible_coordinates = coordinates[~mask].reshape(batch, visible, 3)
        encoded = self.encoder.encode_groups(visible_groups, visible_coordinates)
        decoder_tokens = self.mask_token.expand(batch, count, -1).clone()
        decoder_tokens[~mask] = encoded.reshape(-1, self.encoder.dimension)
        decoder_tokens = self.decoder(decoder_tokens + self.encoder.position(coordinates))
        reconstruction = self.reconstruction(decoder_tokens[mask]).reshape(-1, size, 3)
        original = groups[mask].reshape(-1, size, 3)
        squared = torch.cdist(reconstruction, original).square()
        loss = squared.amin(-1).mean() + squared.amin(-2).mean()
        if return_details:
            return {"loss": loss, "mask": mask, "reconstruction": reconstruction,
                    "centers": centers, "structures": details}
        return loss
