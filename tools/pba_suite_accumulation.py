"""Loss reductions for one effective batch, including uneven ignored pixels.

DDP averages gradients over ranks. Every microbatch therefore contributes
world_size * numerator / the count from the entire optimizer update.
"""
import torch
import torch.nn.functional as F


def pixel_contribution(mean_loss, valid_count, global_count, world_size):
    return mean_loss * valid_count * world_size / global_count.clamp_min(1)


def ptc_contribution(features, mask, global_positive, global_negative,
                     world_size, sample_fraction):
    features = F.normalize(features.flatten(2), dim=1, eps=1e-8)
    similarity = (features.transpose(1, 2) @ features).abs()
    positive = ((mask == 1) * similarity).sum()
    negative = ((mask == 0) * similarity).sum()
    return (0.5 * sample_fraction
            - 0.5 * world_size * positive / (global_positive + 1)
            + 0.5 * world_size * negative / (global_negative + 1))
