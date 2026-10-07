import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def decomposition_loss(predictions, target_masks, padding_mask=None,
                       existence_weight=1.0, patch_weight=1.0, epsilon=1e-6):

    existence = predictions["existence_logits"]
    masks = predictions["mask_logits"]
    if masks.ndim != 3 or existence.shape != masks.shape[:2] or len(target_masks) != len(masks):
        raise ValueError("Existence logits, masks, and targets must have matching batch dimensions.")
    if len(masks) == 0:
        raise ValueError("The batch must contain at least one roof.")
    object_losses, matched_dice = [], []
    for batch, targets in enumerate(target_masks):
        valid = torch.ones(masks.shape[-1], dtype=torch.bool, device=masks.device)
        if padding_mask is not None:
            valid = ~padding_mask[batch]
        if not valid.any():
            raise ValueError("Each roof must contain at least one valid patch.")
        if targets.ndim != 2 or targets.shape[1] != masks.shape[-1] or len(targets) > masks.shape[1]:
            raise ValueError("Targets must be (K, M), with at most Q ground-truth primitives.")
        targets = targets.to(device=masks.device, dtype=masks.dtype)[:, valid]
        logits = masks[batch][:, valid]
        probabilities = logits.sigmoid()
        object_targets = torch.zeros_like(existence[batch])
        if len(targets):
            overlap = probabilities @ targets.T
            dice_cost = 1 - (2 * overlap + epsilon) / (probabilities.sum(-1)[:, None] + targets.sum(-1)[None] + epsilon)
            object_cost = F.softplus(-existence[batch])[:, None]
            with torch.no_grad():
                cost = existence_weight * object_cost + patch_weight * dice_cost
                queries, ground_truth = linear_sum_assignment(cost.detach().float().cpu().numpy())
            queries = torch.as_tensor(queries, device=masks.device)
            ground_truth = torch.as_tensor(ground_truth, device=masks.device)
            object_targets[queries] = 1
            matched_dice.append(dice_cost[queries, ground_truth])
        object_losses.append(F.binary_cross_entropy_with_logits(existence[batch], object_targets))
    object_loss = torch.stack(object_losses).mean()
    patch_loss = torch.cat(matched_dice).mean() if matched_dice else masks[torch.isfinite(masks)].sum() * 0
    return {"loss": existence_weight * object_loss + patch_weight * patch_loss,
            "existence_loss": object_loss, "patch_loss": patch_loss}


def classification_focal_loss(logits, labels, gamma=2.0):

    log_probability = F.log_softmax(logits, dim=-1).gather(-1, labels[:, None]).squeeze(-1)
    return (-((1 - log_probability.exp()) ** gamma) * log_probability).mean()


def affine_adaptation_loss(prediction, target_rotation, target_scale, target_translation,
                           source_points, target_points, parameter_weight=1.0,
                           alignment_weight=1.0, epsilon=1e-6, distance_chunk=512):

    difference = prediction["rotation"].transpose(-1, -2) @ target_rotation
    cosine = (difference.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    rotation_loss = cosine.clamp(-1 + epsilon, 1 - epsilon).acos().mean()
    scale_loss = (prediction["scale"] - target_scale).abs().sum(-1).mean()
    translation_loss = (prediction["translation"] - target_translation).abs().sum(-1).mean()
    matrix = prediction["affine_matrix"]
    transformed = source_points[..., :3] @ matrix[:, :3, :3].transpose(-1, -2) + matrix[:, None, :3, 3]
    if target_points.shape[1] == 0 or source_points.shape[1] == 0 or distance_chunk < 1:
        raise ValueError("Nonempty source/target clouds and positive distance_chunk are required.")
    total_distance = transformed.new_zeros(())
    for start in range(0, target_points.shape[1], distance_chunk):
        observed = target_points[:, start:start + distance_chunk, :3]
        distance = torch.cdist(observed, transformed).square().amin(-1)
        total_distance = total_distance + distance.sum()
    alignment_loss = total_distance / (target_points.shape[0] * target_points.shape[1])
    parameter_loss = rotation_loss + scale_loss + translation_loss
    return {"loss": parameter_weight * parameter_loss + alignment_weight * alignment_loss,
            "rotation_loss": rotation_loss, "scale_loss": scale_loss,
            "translation_loss": translation_loss, "alignment_loss": alignment_loss}
