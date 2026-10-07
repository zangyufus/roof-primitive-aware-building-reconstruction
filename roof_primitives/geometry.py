from dataclasses import dataclass
import numpy as np


def as_points(values, name="points"):
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain finite coordinates.")
    return points


@dataclass(frozen=True)
class RoofPlane:
    normal: np.ndarray
    offset: float
    indices: np.ndarray
    center: np.ndarray

    def signed_distance(self, points):
        return np.asarray(points) @ self.normal + self.offset

    def height(self, xy):
        return -(np.asarray(xy) @ self.normal[:2] + self.offset) / self.normal[2]


@dataclass(frozen=True)
class StructureLine:
    kind: str
    start: np.ndarray
    end: np.ndarray
    plane_ids: tuple
    support: float


@dataclass(frozen=True)
class GeometryConfig:
    plane_distance: float = 0.015
    ransac_iterations: int = 2000
    min_plane_points: int = 24
    min_plane_fraction: float = 0.03
    max_planes: int = 12
    min_normal_z: float = 0.2
    parallel_angle_degrees: float = 12.0
    min_step_height: float = 0.05
    adjacency_distance: float = 0.15
    structure_radius: float = 0.08
    max_pair_points: int = 256

    def __post_init__(self):
        for name in ("plane_distance", "min_step_height", "adjacency_distance", "structure_radius"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.ransac_iterations < 1 or self.min_plane_points < 3 or self.max_planes < 1:
            raise ValueError("RANSAC counts must be positive and planes need at least three points.")
        if not 0 < self.min_normal_z <= 1 or not 0 <= self.min_plane_fraction <= 1:
            raise ValueError("Invalid normal or plane-fraction threshold.")
        if not 0 < self.parallel_angle_degrees < 90 or self.max_pair_points < 3:
            raise ValueError("Invalid parallel-angle or pair-sampling setting.")


def _fit_plane(points):
    center = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    normal = vectors[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, -float(normal @ center), center


def detect_roof_planes(points, config=None, rng=None):

    points = as_points(points)
    config = config or GeometryConfig()
    rng = rng or np.random.default_rng()
    minimum = max(config.min_plane_points, int(np.ceil(config.min_plane_fraction * len(points))))
    remaining = np.arange(len(points))
    planes = []
    for _ in range(config.max_planes):
        if len(remaining) < minimum:
            break
        cloud = points[remaining]
        best = np.empty(0, dtype=np.int64)
        for _ in range(config.ransac_iterations):
            sample = cloud[rng.choice(len(cloud), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            length = np.linalg.norm(normal)
            if length <= 1e-12:
                continue
            normal /= length
            if abs(normal[2]) < config.min_normal_z:
                continue
            distances = np.abs((cloud - sample[0]) @ normal)
            inliers = np.flatnonzero(distances <= config.plane_distance)
            if len(inliers) > len(best):
                best = inliers
                if len(best) == len(cloud):
                    break
        if len(best) < minimum:
            break
        normal, offset, _ = _fit_plane(cloud[best])
        if normal[2] < config.min_normal_z:
            break
        refined = np.flatnonzero(np.abs(cloud @ normal + offset) <= config.plane_distance)
        if len(refined) < minimum:
            break
        normal, offset, center = _fit_plane(cloud[refined])
        if normal[2] < config.min_normal_z:
            break
        planes.append(RoofPlane(normal, offset, remaining[refined].copy(), center))
        remaining = np.delete(remaining, refined)
    return planes


def point_segment_distance(points, start, end):
    points = np.asarray(points, dtype=np.float64)
    vector = np.asarray(end) - np.asarray(start)
    length_sq = float(vector @ vector)
    if length_sq <= 1e-20:
        return np.linalg.norm(points - start, axis=-1)
    parameter = np.clip((points - start) @ vector / length_sq, 0.0, 1.0)
    return np.linalg.norm(points - (start + parameter[..., None] * vector), axis=-1)


def _pair_samples(points, limit):
    if len(points) <= limit:
        return points
    return points[np.linspace(0, len(points) - 1, limit, dtype=int)]


def detect_structure_lines(points, planes, config=None):

    points = as_points(points)
    config = config or GeometryConfig()
    cosine_limit = np.cos(np.deg2rad(config.parallel_angle_degrees))
    structures = []
    for i, first in enumerate(planes):
        a = _pair_samples(points[first.indices], config.max_pair_points)
        for j in range(i + 1, len(planes)):
            second = planes[j]
            b = _pair_samples(points[second.indices], config.max_pair_points)
            horizontal = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=-1)
            if horizontal.min() > config.adjacency_distance:
                continue
            ai, bi = np.unravel_index(horizontal.argmin(), horizontal.shape)
            support = min(len(first.indices), len(second.indices)) / max(len(points), 1)
            if first.normal @ second.normal >= cosine_limit:
                xy = (a[ai, :2] + b[bi, :2]) / 2
                height_a, height_b = first.height(xy), second.height(xy)
                if abs(height_a - height_b) < config.min_step_height:
                    continue
                lower_id = i if height_a < height_b else j
                lower = a if lower_id == i else b
                distances = horizontal if lower_id == i else horizontal.T
                near = lower[distances.min(axis=1) <= config.adjacency_distance]
                if len(near) < 3:
                    continue
                center_xy = near[:, :2].mean(axis=0)
                _, singular, vectors = np.linalg.svd(near[:, :2] - center_xy, full_matrices=False)
                if singular[0] <= 1e-10:
                    continue
                axis = vectors[0]
                coordinates = (near[:, :2] - center_xy) @ axis
                endpoints = center_xy + np.array([coordinates.min(), coordinates.max()])[:, None] * axis
                lower_plane = planes[lower_id]
                endpoints = np.column_stack([endpoints, lower_plane.height(endpoints)])
                if np.linalg.norm(endpoints[1] - endpoints[0]) < config.plane_distance:
                    continue
                structures.append(StructureLine("step", endpoints[0], endpoints[1], (lower_id,), support))
                continue

            if (first.signed_distance(second.center) <= config.plane_distance
                    or second.signed_distance(first.center) <= config.plane_distance):
                continue
            direction = np.cross(first.normal, second.normal)
            direction /= np.linalg.norm(direction)
            origin = np.linalg.lstsq(np.stack([first.normal, second.normal]),
                                     -np.array([first.offset, second.offset]), rcond=None)[0]
            intervals = []
            for cloud in (a, b):
                parameter = (cloud - origin) @ direction
                distance = np.linalg.norm(cloud - origin - parameter[:, None] * direction, axis=1)
                nearby = parameter[distance <= config.adjacency_distance]
                if len(nearby) < 3:
                    break
                intervals.append((nearby.min(), nearby.max()))
            if len(intervals) != 2:
                continue
            start = max(interval[0] for interval in intervals)
            end = min(interval[1] for interval in intervals)
            if end - start < config.plane_distance:
                continue
            structures.append(StructureLine("valley", origin + start * direction,
                                            origin + end * direction, (i, j), support))
    return structures


def assign_group_planes(centers, planes, max_distance):
    centers = as_points(centers, "centers")
    assignments = np.full(len(centers), -1, dtype=np.int64)
    if not planes:
        return assignments
    distances = np.stack([np.abs(plane.signed_distance(centers)) for plane in planes], axis=1)
    nearest = distances.argmin(axis=1)
    valid = distances[np.arange(len(centers)), nearest] <= max_distance
    assignments[valid] = nearest[valid]
    return assignments


def plane_grid_patches(points, planes, resolution=0.10, min_points=16):

    points = as_points(points)
    if not planes or resolution <= 0 or min_points < 1:
        raise ValueError("Detected planes, positive resolution, and min_points are required.")
    distances = np.stack([np.abs(plane.signed_distance(points)) for plane in planes], axis=1)
    assignment = distances.argmin(axis=1)
    for index, plane in enumerate(planes):
        assignment[plane.indices] = index
    patch_ids = np.full(len(points), -1, dtype=np.int64)
    patch_centers = []
    for plane_id, plane in enumerate(planes):
        indices = np.flatnonzero(assignment == plane_id)
        if not len(indices):
            continue
        cloud = points[indices]
        _, _, axes = np.linalg.svd(cloud - cloud.mean(0), full_matrices=False)
        uv = (cloud - plane.center) @ axes[:2].T
        cells = np.floor((uv - uv.min(0)) / resolution).astype(np.int64)
        _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        retained = np.flatnonzero(counts >= min_points)
        if not len(retained):
            patch_ids[indices] = len(patch_centers)
            patch_centers.append(cloud.mean(0))
            continue
        retained_centers = np.array([cloud[inverse == cell].mean(0) for cell in retained])
        start = len(patch_centers)
        for local_id, cell in enumerate(retained):
            patch_ids[indices[inverse == cell]] = start + local_id
            patch_centers.append(retained_centers[local_id])
        residual = indices[patch_ids[indices] < 0]
        if len(residual):
            distance = ((points[residual, None] - retained_centers[None]) ** 2).sum(-1)
            patch_ids[residual] = start + distance.argmin(axis=1)

    centers = np.array([points[patch_ids == patch].mean(0) for patch in range(len(patch_centers))])
    return patch_ids, centers


def local_roof_descriptors(points, neighbors=32):

    points = as_points(points)
    if len(points) < 3 or neighbors < 3:
        raise ValueError("At least three points and neighbors are required.")
    neighbors = min(neighbors, len(points))
    normals = np.empty_like(points)
    linearity = np.empty(len(points))
    curvature = np.empty(len(points))

    for begin in range(0, len(points), 256):
        query = points[begin:begin + 256]
        distances = ((query[:, None] - points[None]) ** 2).sum(axis=-1)
        indices = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
        groups = points[indices]
        groups -= groups.mean(axis=1, keepdims=True)
        covariance = np.einsum("bni,bnj->bij", groups, groups) / neighbors
        values, vectors = np.linalg.eigh(covariance)
        values = values.clip(min=0)
        normal = vectors[:, :, 0]
        normal[normal[:, 2] < 0] *= -1
        normals[begin:begin + len(query)] = normal
        linearity[begin:begin + len(query)] = (values[:, 2] - values[:, 1]) / (values[:, 2] + 1e-9)
        curvature[begin:begin + len(query)] = values[:, 0] / (values.sum(axis=1) + 1e-9)
    return normals, linearity, curvature


def _circumcircles(vertices, triangles):
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    ba, ca = b - a, c - a
    determinant = 2 * (ba[:, 0] * ca[:, 1] - ba[:, 1] * ca[:, 0])
    valid = np.abs(determinant) > 1e-14
    safe = np.where(valid, determinant, 1)
    ba_sq, ca_sq = (ba * ba).sum(1), (ca * ca).sum(1)
    center = a + np.column_stack([(ca[:, 1] * ba_sq - ba[:, 1] * ca_sq) / safe,
                                  (ba[:, 0] * ca_sq - ca[:, 0] * ba_sq) / safe])
    radius_sq = ((center - a) ** 2).sum(1)
    return center, radius_sq, valid


def alpha_shape_segments(xy, radius):

    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("xy must have finite shape (N, 2).")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("The alpha-shape radius must be positive and finite.")
    xy = np.unique(xy, axis=0)
    if len(xy) < 3 or np.linalg.matrix_rank(xy - xy.mean(0)) < 2:
        raise ValueError("An outline needs three noncollinear horizontal points.")
    center = (xy.min(0) + xy.max(0)) / 2
    scale = float(np.ptp(xy, axis=0).max())
    vertices = np.vstack([xy, center + scale * np.array([[-32, -16], [32, -16], [0, 32]])])
    triangles = np.array([[len(xy), len(xy) + 1, len(xy) + 2]], dtype=np.int64)
    for index in np.random.default_rng(0).permutation(len(xy)):
        circles, radii, valid = _circumcircles(vertices, triangles)
        bad = valid & (((circles - vertices[index]) ** 2).sum(1) <= radii + 1e-12 * scale ** 2)
        removed = triangles[bad]
        if not len(removed):
            raise RuntimeError("The horizontal triangulation became numerically degenerate.")
        edges = np.concatenate([removed[:, [0, 1]], removed[:, [1, 2]], removed[:, [2, 0]]])
        edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        boundary = edges[counts == 1]
        added = np.column_stack([boundary, np.full(len(boundary), index)])
        triangles = np.vstack([triangles[~bad], added])
    triangles = triangles[(triangles < len(xy)).all(axis=1)]
    _, radii, valid = _circumcircles(vertices, triangles)
    triangles = triangles[valid & (radii <= radius ** 2)]
    if not len(triangles):
        raise ValueError("The alpha-shape radius is too small for this horizontal sampling.")
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return xy[edges[counts == 1]]


def outline_proximity(points, alpha_radius, sigma=0.05):
    points = as_points(points)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite.")
    segments = alpha_shape_segments(points[:, :2], alpha_radius)
    distances = np.full(len(points), np.inf)
    for start, end in segments:
        distances = np.minimum(distances, point_segment_distance(points[:, :2], start, end))
    return np.exp(-distances ** 2 / (2 * sigma ** 2))
