"""Sliding-window probability TTA adapted from NIPS2026_SQ's VOC evaluator."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def sliding_window_tta(
    model,
    inputs,
    num_classes,
    scales=(0.5, 0.75, 1.0, 1.25, 1.5),
    crop_size=512,
    stride=384,
    max_size=1024,
):
    """Return probabilities in original image coordinates for one RGB image.

    Match Trainer_voc_anchor.validate: preserve aspect ratio, round dimensions
    to patch multiples, pad normalized images with zero, average overlapping
    crops and flipped views, then average probabilities across scales. The
    reference uses FP16 model inference and probability accumulation.
    """
    if inputs.ndim != 4 or inputs.shape[0] != 1:
        raise ValueError("Sliding-window TTA expects a single BCHW image")
    if not inputs.is_cuda:
        raise ValueError("Reference TTA uses CUDA FP16 inference")
    if crop_size < 16 or crop_size % 16 or not 0 < stride <= crop_size:
        raise ValueError("crop_size must be a multiple of 16 and stride must cover all pixels")
    if max_size < 16 or not scales or any(s <= 0 for s in scales):
        raise ValueError("max_size and scales must be positive")

    def starts(length):
        if length <= crop_size:
            return [0]
        values = list(range(0, length - crop_size + 1, stride))
        last = length - crop_size
        if values[-1] != last:
            values.append(last)
        return values

    def scaled_hw(h, w, short_side):
        ratio = float(short_side) / float(min(h, w))
        sh = max(16, int(round(h * ratio / 16.0)) * 16)
        sw = max(16, int(round(w * ratio / 16.0)) * 16)
        cap = max(sh, sw)
        if cap > max_size:
            ratio = float(max_size) / float(cap)
            sh = max(16, int(round(sh * ratio / 16.0)) * 16)
            sw = max(16, int(round(sw * ratio / 16.0)) * 16)
        return sh, sw

    h, w = inputs.shape[-2:]
    tta_prob_sum = None
    for scale in scales:
        sh, sw = scaled_hw(h, w, int(crop_size * scale))
        scaled = F.interpolate(inputs, size=(sh, sw), mode="bilinear", align_corners=False)
        canvas_h, canvas_w = max(sh, crop_size), max(sw, crop_size)
        padded = F.pad(scaled, (0, canvas_w - sw, 0, canvas_h - sh), value=0.0)
        ensemble = torch.zeros(
            (1, num_classes, canvas_h, canvas_w), device=inputs.device, dtype=torch.float16
        )
        votes = torch.zeros((1, 1, canvas_h, canvas_w), device=inputs.device, dtype=torch.float16)
        for y in starts(canvas_h):
            for x in starts(canvas_w):
                crop_img = padded[:, :, y:y + crop_size, x:x + crop_size]
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(crop_img)[1]
                prob = F.softmax(
                    F.interpolate(logits.float(), size=(crop_size, crop_size),
                                  mode="bilinear", align_corners=False), dim=1
                ).half()
                vh, vw = min(crop_size, sh - y), min(crop_size, sw - x)
                if vh > 0 and vw > 0:
                    ensemble[:, :, y:y + vh, x:x + vw] += prob[:, :, :vh, :vw]
                    votes[:, :, y:y + vh, x:x + vw] += 1
                flip_img = torch.flip(crop_img, dims=[3])
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    flip_logits = model(flip_img)[1]
                flip_prob = F.softmax(
                    F.interpolate(torch.flip(flip_logits.float(), dims=[3]),
                                  size=(crop_size, crop_size), mode="bilinear",
                                  align_corners=False), dim=1
                ).half()
                if vh > 0 and vw > 0:
                    ensemble[:, :, y:y + vh, x:x + vw] += flip_prob[:, :, :vh, :vw]
                    votes[:, :, y:y + vh, x:x + vw] += 1
        if sh != canvas_h or sw != canvas_w:
            ensemble = ensemble[:, :, :sh, :sw]
            votes = votes[:, :, :sh, :sw]
        if torch.any(votes == 0):
            raise RuntimeError("Sliding-window TTA left pixels uncovered")
        scale_prob = F.interpolate(
            (ensemble / votes.clamp_min(1)).float(), size=(h, w),
            mode="bilinear", align_corners=False,
        ).half()
        tta_prob_sum = scale_prob if tta_prob_sum is None else tta_prob_sum + scale_prob

    return tta_prob_sum / float(len(scales))
