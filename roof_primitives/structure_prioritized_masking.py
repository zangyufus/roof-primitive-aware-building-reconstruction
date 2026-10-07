import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .geometry import (GeometryConfig, as_points, detect_roof_planes,
                       detect_structure_lines, assign_group_planes,
                       point_segment_distance)


@dataclass
class MaskResult:
    mask: np.ndarray
    priorities: np.ndarray
    plane_ids: np.ndarray
    planes: list
    structures: list


def prioritize_groups(centers, plane_ids, structures, radius):

    centers = as_points(centers, "centers")
    plane_ids = np.asarray(plane_ids, dtype=np.int64)
    if plane_ids.shape != (len(centers),):
        raise ValueError("plane_ids must have one entry per group.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite.")
    priorities = np.zeros(len(centers), dtype=np.float64)
    for structure in structures:
        if structure.kind not in ("step", "valley"):
            raise ValueError("Only stepping edges and valley lines may be prioritized.")
        if structure.support <= 0 or not np.isfinite(structure.support):
            continue
        distance = point_segment_distance(centers, structure.start, structure.end)
        valid = np.isin(plane_ids, structure.plane_ids) & (distance <= radius)
        score = structure.support * np.exp(-0.5 * (distance / radius) ** 2)
        priorities = np.maximum(priorities, np.where(valid, score, 0.0))
    return priorities


def select_mask(priorities, mask_ratio=0.6, rng=None):

    priorities = np.asarray(priorities, dtype=np.float64)
    if priorities.ndim != 1 or not np.isfinite(priorities).all() or (priorities < 0).any():
        raise ValueError("priorities must be a finite, nonnegative vector.")
    if not np.isfinite(mask_ratio) or not 0 <= mask_ratio <= 1:
        raise ValueError("mask_ratio must be between zero and one.")
    rng = rng or np.random.default_rng()
    count = int(mask_ratio * len(priorities))
    mask = np.zeros(len(priorities), dtype=bool)
    candidates = np.flatnonzero(priorities > 0)

    candidates = rng.permutation(candidates)
    ranked = candidates[np.argsort(-priorities[candidates], kind="stable")]
    mask[ranked[:count]] = True
    remaining = count - int(mask.sum())
    if remaining:
        available = np.flatnonzero(~mask)
        mask[rng.choice(available, remaining, replace=False)] = True
    return mask


class StructurePrioritizedMasking:
    def __init__(self, mask_ratio=0.6, config=None, seed=0):
        if not np.isfinite(mask_ratio) or not 0 <= mask_ratio <= 1:
            raise ValueError("mask_ratio must be between zero and one.")
        self.mask_ratio = mask_ratio
        self.config = config or GeometryConfig()
        self.rng = np.random.default_rng(seed)

    def analyze(self, points, centers, planes=None, structures=None, plane_ids=None):
        points, centers = as_points(points), as_points(centers, "centers")
        if planes is None:
            planes = detect_roof_planes(points, self.config, self.rng)
        if structures is None:
            structures = detect_structure_lines(points, planes, self.config)
        if plane_ids is None:
            plane_ids = assign_group_planes(centers, planes, 3 * self.config.plane_distance)
        priorities = prioritize_groups(centers, plane_ids, structures, self.config.structure_radius)
        mask = select_mask(priorities, self.mask_ratio, self.rng)
        return MaskResult(mask, priorities, np.asarray(plane_ids), planes, structures)

    def __call__(self, points, centers, return_details=False):

        is_tensor = hasattr(points, "detach")
        if is_tensor != hasattr(centers, "detach"):
            raise TypeError("points and centers must use the same array backend.")
        if is_tensor:
            cloud = points.detach().cpu().numpy()
            groups = centers.detach().cpu().numpy()
        else:
            cloud, groups = np.asarray(points), np.asarray(centers)
        single = cloud.ndim == 2
        if single:
            cloud, groups = cloud[None], groups[None]
        if cloud.ndim != 3 or groups.ndim != 3 or cloud.shape[0] != groups.shape[0]:
            raise ValueError("points and centers must have matching batch dimensions.")
        if cloud.shape[0] == 0:
            raise ValueError("The batch must contain at least one roof.")
        details = [self.analyze(p, c) for p, c in zip(cloud, groups)]
        masks = np.stack([result.mask for result in details])
        if single:
            masks = masks[0]
        if is_tensor:
            import torch
            masks = torch.as_tensor(masks, dtype=torch.bool, device=centers.device)
        return (masks, details[0] if single else details) if return_details else masks


def farthest_group_centers(points, count=64):
    points = as_points(points)
    if count < 1 or count > len(points):
        raise ValueError("The group count must be between one and the point count.")
    selected = np.empty(count, dtype=np.int64)
    distance = np.full(len(points), np.inf)
    index = int(((points - points.mean(0)) ** 2).sum(1).argmax())
    for group in range(count):
        selected[group] = index
        distance = np.minimum(distance, ((points - points[index]) ** 2).sum(1))
        distance[selected[:group + 1]] = -1
        index = int(distance.argmax())
    return points[selected]


def _load_coordinates(path):
    path = Path(path)
    values = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("Input must have at least three coordinate columns.")
    return as_points(values[:, :3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, help="Roof XYZ text or NPY file.")
    parser.add_argument("--centers", help="Group-center XYZ text or NPY file; otherwise use FPS.")
    parser.add_argument("--output", required=True, help="Output NPZ file.")
    parser.add_argument("--num-groups", type=int, default=64)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    defaults = GeometryConfig()
    for name in defaults.__dataclass_fields__:
        value = getattr(defaults, name)
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    points = _load_coordinates(args.points)
    centers = _load_coordinates(args.centers) if args.centers else farthest_group_centers(points, args.num_groups)
    config = GeometryConfig(**{name: getattr(args, name) for name in defaults.__dataclass_fields__})
    result = StructurePrioritizedMasking(args.mask_ratio, config, args.seed).analyze(points, centers)
    output = Path(args.output)
    if output.suffix != ".npz":
        raise ValueError("The output path must end in .npz.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, mask=result.mask, centers=centers, priorities=result.priorities,
                        plane_ids=result.plane_ids,
                        line_kinds=np.asarray([line.kind for line in result.structures], dtype="U6"),
                        line_starts=np.asarray([line.start for line in result.structures]).reshape(-1, 3),
                        line_ends=np.asarray([line.end for line in result.structures]).reshape(-1, 3),
                        plane_equations=np.asarray([np.r_[plane.normal, plane.offset] for plane in result.planes]).reshape(-1, 4))
    print(f"Saved {int(result.mask.sum())}/{len(centers)} masked groups to {output}; "
          f"{len(result.planes)} planes, {len(result.structures)} structural lines, "
          f"{int((result.priorities > 0).sum())} prioritized candidates.")


if __name__ == "__main__":
    main()
