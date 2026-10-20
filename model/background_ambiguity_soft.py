"""Continuous PBA from two CAM views and segmentation confidence alone.

Only original background targets are relaxed. Geometric-mean CAM agreement
sets a continuous ambiguity weight, and detached student class probabilities
interpolate partial background/new CE with directional new-class CE. There is
no seed cutoff, candidate cutoff, confidence cutoff, minimum region size,
prototype input or additional temperature. The optional strength scales the
correction relative to ordinary CE for explicit weight ablations (default 1).
"""
import math
import torch
import torch.nn.functional as F


def continuous_background_loss(logits, labels, cams, cams_aux, image_tags, valid_mask,
                               old_classes, *, class_weight=None, ignore_index=255, strength=1.0):
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('PBA correction strength must be finite and between zero and one.')
    batch, classes, height, width = logits.shape
    if labels.shape != (batch, height, width) or valid_mask.shape != labels.shape:
        raise ValueError('Logits, target labels and validity must share a pixel grid.')
    if valid_mask.dtype != torch.bool:
        raise ValueError('Validity must be a boolean padding mask.')
    if cams.shape != cams_aux.shape or cams.shape[:2] != image_tags.shape:
        raise ValueError('Main and auxiliary CAMs must match foreground image tags.')
    if image_tags.shape != (batch, classes - 1) or not 0 < old_classes < classes - 1:
        raise ValueError('Require old and incoming foreground classes.')
    valid = valid_mask & (labels != ignore_index)
    targets = labels.masked_fill(~valid, ignore_index)
    original = F.cross_entropy(logits, targets, weight=class_weight,
                               ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        main = F.interpolate(cams[:, old_classes:].detach().float(), (height, width),
                              mode='bilinear', align_corners=False).clamp(0, 1)
        auxiliary = F.interpolate(cams_aux[:, old_classes:].detach().float(), (height, width),
                                   mode='bilinear', align_corners=False).clamp(0, 1)
        agreement = (main * auxiliary).sqrt()
        agreement *= image_tags[:, old_classes:].bool()[:, :, None, None]
        agreement *= valid[:, None]
        cam_ambiguity = agreement.amax(1)
        ambiguity = strength * cam_ambiguity
        mass = agreement.sum(1, keepdim=True)
        distribution = agreement / mass.clamp_min(torch.finfo(agreement.dtype).tiny)
    logp = F.log_softmax(logits, dim=1)
    incoming_logp = logp[:, old_classes + 1:]
    confidence = incoming_logp.detach().exp()
    partial = -torch.logaddexp(logp[:, :1], incoming_logp)
    directional = -incoming_logp
    if class_weight is not None:
        partial = partial * class_weight[0]
        directional = directional * class_weight[old_classes + 1:][None, :, None, None]
    candidate_loss = ((1 - confidence) * partial + confidence * directional)
    expected = (distribution * candidate_loss).sum(1)
    background_loss = (1 - ambiguity) * original + ambiguity * expected
    background = valid & (labels == 0)
    pixels = torch.where(background, background_loss, original)
    loss = pixels.sum() / valid.sum().clamp_min(1)
    background_count = background.sum().clamp_min(1)
    directional_mass = ambiguity * (distribution * confidence).sum(1)
    stats = dict(valid_pixels=int(valid.sum()), background_pixels=int(background.sum()),
                 pba_strength=float(strength),
                 mean_cam_ambiguity=float((cam_ambiguity * background).sum() / background_count),
                 mean_background_ambiguity=float((ambiguity * background).sum() / background_count),
                 mean_directional_weight=float((directional_mass * background).sum() / background_count),
                 ambiguity_mass=float((ambiguity * background).sum()),
                 thresholds_used=False, prototypes_used=False)
    return loss, stats
