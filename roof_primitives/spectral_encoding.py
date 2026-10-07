import torch
from torch import nn


def weighted_knn_graph(centers, neighbors=20, alpha=1.0):
    if centers.ndim != 3 or centers.shape[-1] != 3 or centers.shape[1] < 4:
        raise ValueError("At least four group centers per roof are required.")
    if neighbors < 1 or alpha <= 0:
        raise ValueError("neighbors and alpha must be positive.")
    squared = torch.cdist(centers.float(), centers.float()).square()
    diagonal = torch.eye(centers.shape[1], device=centers.device, dtype=torch.bool)[None]
    squared = squared.masked_fill(diagonal, torch.inf)
    distance, indices = squared.topk(min(neighbors, centers.shape[1] - 1), dim=-1, largest=False)
    graph = squared.new_zeros(squared.shape)
    graph.scatter_(-1, indices, (-alpha * distance).exp())
    return torch.maximum(graph, graph.transpose(1, 2))


@torch.no_grad()
def spectral_coordinates(centers, neighbors=20, alpha=1.0, components=3, zero_tolerance=1e-6):
    graph = weighted_knn_graph(centers, neighbors, alpha)
    degree = graph.sum(-1).clamp_min(1e-12)
    inverse = degree.rsqrt()
    laplacian = torch.eye(graph.shape[-1], device=graph.device)[None] - graph * inverse[:, :, None] * inverse[:, None]
    values, vectors = torch.linalg.eigh(laplacian)
    valid = values > zero_tolerance
    if (valid.sum(-1) < components).any():
        raise ValueError("Insufficient nonzero Laplacian eigenvectors for spectral encoding.")
    indices = values.masked_fill(~valid, torch.inf).topk(components, dim=-1, largest=False).indices
    result = vectors.gather(-1, indices[:, None].expand(-1, centers.shape[1], -1))
    pivots = result.gather(1, result.abs().argmax(1, keepdim=True))
    result *= torch.where(pivots < 0, -1.0, 1.0)
    return result.to(centers.dtype)


class PointGrouping(nn.Module):
    def __init__(self, groups=64, group_size=32):
        super().__init__()
        self.groups, self.group_size = groups, group_size

    @torch.no_grad()
    def forward(self, points):
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < max(self.groups, self.group_size):
            raise ValueError("Input points must have shape (B, N, 3) with enough points for grouping.")
        batch, count, _ = points.shape
        indices = torch.empty(batch, self.groups, dtype=torch.long, device=points.device)
        distance = points.new_full((batch, count), torch.inf)
        farthest = (points - points.mean(1, keepdim=True)).square().sum(-1).argmax(-1)
        batch_ids = torch.arange(batch, device=points.device)
        for group in range(self.groups):
            indices[:, group] = farthest
            distance = torch.minimum(distance, (points - points[batch_ids, farthest][:, None]).square().sum(-1))
            distance.scatter_(1, indices[:, :group + 1], -1)
            farthest = distance.argmax(-1)
        centers = points[batch_ids[:, None], indices]
        neighbors = torch.cdist(centers, points).topk(self.group_size, dim=-1, largest=False).indices
        neighborhoods = points[batch_ids[:, None, None], neighbors] - centers[:, :, None]
        return neighborhoods, centers


class MiniPointNet(nn.Module):
    def __init__(self, dimension=384):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 256))
        self.fusion = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, dimension))

    def forward(self, neighborhoods):
        local = self.local(neighborhoods)
        context = local.amax(-2, keepdim=True).expand_as(local)
        return self.fusion(torch.cat([local, context], dim=-1)).amax(-2)


class LaplacianSpectralEncoder(nn.Module):
    def __init__(self, dimension=384, depth=12, heads=6, groups=64, group_size=32,
                 neighbors=20, alpha=1.0):
        super().__init__()
        self.dimension, self.neighbors, self.alpha = dimension, neighbors, alpha
        self.grouping = PointGrouping(groups, group_size)
        self.geometry = MiniPointNet(dimension)
        self.position = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, dimension))
        layer = nn.TransformerEncoderLayer(dimension, heads, dimension * 4, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dimension), enable_nested_tensor=False)

    def encode_groups(self, neighborhoods, coordinates):
        return self.transformer(self.geometry(neighborhoods) + self.position(coordinates))

    def forward(self, points):
        neighborhoods, centers = self.grouping(points)
        coordinates = spectral_coordinates(centers, self.neighbors, self.alpha)
        return self.encode_groups(neighborhoods, coordinates)


class PrimitiveRecognition(nn.Module):
    def __init__(self, encoder=None, classes=6):
        super().__init__()
        self.encoder = encoder if encoder is not None else LaplacianSpectralEncoder()
        dimension = self.encoder.dimension
        self.classifier = nn.Sequential(nn.Linear(2 * dimension, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, classes))

    def forward(self, points):
        tokens = self.encoder(points)
        return self.classifier(torch.cat([tokens.mean(1), tokens.amax(1)], dim=-1))
